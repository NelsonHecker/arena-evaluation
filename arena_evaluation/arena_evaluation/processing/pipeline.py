from __future__ import annotations

import pathlib
import datetime
import polars as pl

from ..storage.schemas import RobotParams, RunDescriptor, TopicBundle
from ..storage.folder_manager import FolderManager
from ..storage.manifest import MetadataWriter

from .mcap_reader import MCAPReader
from .topic_aligner import TopicAligner
from .episode_splitter import EpisodeSplitter
from .parquet_store import ParquetStore, TopicParquetStore
from .metrics.registry import MetricRegistry

import arena_evaluation

class ProcessingPipeline:
    """
    Orchestrates the data processing pipeline:
    MCAP -> Extract -> Align -> Split -> Metrics -> Parquet
    """
    def __init__(self, folder_manager: FolderManager):
        self.folder_manager = folder_manager
        
    def extract_run(self, run: RunDescriptor) -> TopicBundle | None:
        """
        Extract topics from MCAP and cache them as individual Parquet files.
        Returns the TopicBundle if successful, else None.
        """
        run_dir = pathlib.Path(run.run_dir)
        mcap_path = self.folder_manager.mcap_path(run_dir)
        extracted_dir = self.folder_manager.extracted_topics_path(run_dir)
        
        source_path = mcap_path
        if not source_path.exists():
            print(f"  [skip] No MCAP found for {run.planner}/{run.stage}")
            return None
                
        print(f"  Extracting {run.planner}/{run.stage} to {extracted_dir}...")
        reader = MCAPReader(source_path)
        bundle = reader.read()
        
        TopicParquetStore.write(bundle, extracted_dir)
        return bundle

    def process_run(self, run: RunDescriptor, force_extract: bool = False) -> pathlib.Path | None:
        """
        Process a single run described by a RunDescriptor and generate its metrics.parquet.
        Returns the path to the generated parquet file, or None if the run is skipped.
        """
        run_dir = pathlib.Path(run.run_dir)
        metrics_path = self.folder_manager.metrics_path(run_dir)
        extracted_dir = self.folder_manager.extracted_topics_path(run_dir)
        metadata_path = run_dir / "metadata.yaml"

        # 1. Read metadata
        metadata = MetadataWriter.read(metadata_path)

        # 2. Extract or Load Topics (now returns dict[str, TopicBundle])
        bundles = None
        if not force_extract:
            bundles = TopicParquetStore.read(extracted_dir)
            if bundles:
                print(f"  Loading cached topics for {run.planner}/{run.stage}...")
                
        if bundles is None:
            bundles = self.extract_run(run)
            
        if bundles is None or len(bundles) == 0:
            return None

        # Process each robot
        results = []
        total_episodes_valid = 0
        
        for robot_name, bundle in bundles.items():
            print(f"  [{robot_name}] Splitting into episodes...")
            aligner = TopicAligner()
            splitter = EpisodeSplitter(aligner)
            episodes = splitter.split(bundle, robot_name=robot_name)

            if not episodes:
                print(f"  [{robot_name}] [skip] No valid episodes found")
                continue

            # Load robot params (fallback to burger if no metadata or specific model unknown)
            # Since we now have multi-robot, we ideally know the model. 
            # If we don't map name to model cleanly here, fallback to burger.
            robot_model = metadata.robot_model[0] if metadata.robot_model else "turtlebot3_burger"
            robot_params = RobotParams.load(robot_model)

            print(f"  [{robot_name}] Computing metrics for {len(episodes)} episodes...")
            registry = MetricRegistry(robot_params)
            pedsim_avail = metadata.pedsim_available if metadata.pedsim_available is not None else False
            if bundle.peds is not None:
                pedsim_avail = True

            available_topics = set()
            for field in ("odom", "scan", "cmd_vel", "joint_states", "peds", "collision_events", "collision_monitor_state", "plan", "initialpose", "tf", "tf_static", "tf_gt"):
                if getattr(bundle, field, None) is not None:
                    available_topics.add(field)

            for ep in episodes:
                ep_metrics = registry.run(ep, pedsim_available=pedsim_avail, available_topics=available_topics)

                # Add identity columns
                ep_metrics["episode"] = ep.episode_id
                ep_metrics["planner"] = run.planner
                
                from ..presentation.dimension_detector import split_planner_name
                lp, ip = split_planner_name(run.planner)
                ep_metrics["local_planner"] = lp
                ep_metrics["inter_planner"] = ip
                
                # Use the exact namespace for the robot identity
                ep_metrics["robot"] = ep.robot_name or robot_name
                ep_metrics["map"] = metadata.map
                ep_metrics["stage"] = run.stage
                ep_metrics["benchmark_id"] = run.benchmark_id
                ep_metrics["start"] = ep.start_pos
                ep_metrics["goal"] = ep.goal_pos

                results.append(ep_metrics)
                
            total_episodes_valid = max(total_episodes_valid, len(episodes))

        if not results:
            print(f"  [skip] No valid episodes generated for {run.planner}/{run.stage}")
            return None

        # 6. Save to Parquet
        print(f"  Saving metrics to {metrics_path}...")
        df = pl.DataFrame(results)

        # Update metadata
        metadata.processing_completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata.episodes_valid = total_episodes_valid
        metadata.pipeline_version = arena_evaluation.__version__ if hasattr(arena_evaluation, "__version__") else "1.0.0"

        MetadataWriter.write(metadata, metadata_path)
        ParquetStore.write(df, metrics_path, metadata)

        print(f"  Done: {metrics_path}")
        return metrics_path

    def extract_run_dir(self, run_dir: pathlib.Path) -> TopicBundle | None:
        """
        Extract topics from a direct run directory.
        """
        run_dir = run_dir.resolve()
        extracted_dir = self.folder_manager.extracted_topics_path(run_dir)
        
        recording_subdir = run_dir / "recording"
        mcap_candidates = list(recording_subdir.glob("*.mcap")) if recording_subdir.exists() else []
        if not mcap_candidates:
            mcap_candidates = list(run_dir.glob("**/*.mcap"))
        if not mcap_candidates:
            print(f"Error: no .mcap file found under {run_dir}")
            return None
            
        source_path = sorted(mcap_candidates)[0]
        
        print(f"  Extracting ad-hoc run to {extracted_dir}...")
        reader = MCAPReader(source_path)
        bundle = reader.read()
        
        TopicParquetStore.write(bundle, extracted_dir)
        return bundle

    def process_run_dir(self, run_dir: pathlib.Path, force_extract: bool = False) -> pathlib.Path | None:
        """
        Process a single recording directory directly, without needing a benchmark structure.

        Use this when you have an ad-hoc recording at e.g.:
            /opt/arena_ws/data/recordings/20260528-215316/

        The planner and stage are inferred from the directory name or set to 'unknown'.
        Output: run_dir/metrics.parquet
        """
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            print(f"Error: run directory does not exist: {run_dir}")
            return None

        metadata_path = run_dir / "metadata.yaml"
        if not metadata_path.exists():
            print(f"Error: no metadata.yaml found in {run_dir}")
            return None

        # Build a minimal RunDescriptor from whatever we know
        metadata = MetadataWriter.read(metadata_path)
        descriptor = RunDescriptor(
            run_dir=str(run_dir),
            benchmark_id=getattr(metadata, "benchmark_id", "unknown"),
            planner=getattr(metadata, "planner", run_dir.name),
            stage=getattr(metadata, "stage", "unknown"),
        )
        
        extracted_dir = self.folder_manager.extracted_topics_path(run_dir)
        metrics_path = run_dir / "metrics.parquet"

        # 2. Extract or Load Topics
        bundles = None
        if not force_extract:
            bundles = TopicParquetStore.read(extracted_dir)
            if bundles:
                print(f"  Loading cached topics for ad-hoc run...")
                
        if bundles is None:
            bundles = self.extract_run_dir(run_dir)
            
        if bundles is None or len(bundles) == 0:
            return None

        results = []
        total_episodes_valid = 0
        
        for robot_name, bundle in bundles.items():
            print(f"  [{robot_name}] Splitting into episodes...")
            aligner = TopicAligner()
            splitter = EpisodeSplitter(aligner)
            episodes = splitter.split(bundle, robot_name=robot_name)

            if not episodes:
                print(f"  [{robot_name}] [skip] No valid episodes found")
                continue

            robot_model = metadata.robot_model[0] if metadata.robot_model else "turtlebot3_burger"
            robot_params = RobotParams.load(robot_model)

            print(f"  [{robot_name}] Computing metrics for {len(episodes)} episodes...")
            registry = MetricRegistry(robot_params)
            pedsim_avail = metadata.pedsim_available if metadata.pedsim_available is not None else False
            if bundle.peds is not None:
                pedsim_avail = True

            available_topics = set()
            for field in ("odom", "scan", "cmd_vel", "joint_states", "peds", "collision_events", "collision_monitor_state", "plan", "initialpose", "tf", "tf_static", "tf_gt"):
                if getattr(bundle, field, None) is not None:
                    available_topics.add(field)

            for ep in episodes:
                ep_metrics = registry.run(ep, pedsim_available=pedsim_avail, available_topics=available_topics)
                ep_metrics["episode"] = ep.episode_id
                ep_metrics["planner"] = descriptor.planner
                
                from ..presentation.dimension_detector import split_planner_name
                lp, ip = split_planner_name(descriptor.planner)
                ep_metrics["local_planner"] = lp
                ep_metrics["inter_planner"] = ip
                
                ep_metrics["robot"] = ep.robot_name or robot_name
                ep_metrics["map"] = metadata.map
                ep_metrics["stage"] = descriptor.stage
                ep_metrics["benchmark_id"] = descriptor.benchmark_id
                ep_metrics["start"] = ep.start_pos
                ep_metrics["goal"] = ep.goal_pos
                results.append(ep_metrics)
                
            total_episodes_valid = max(total_episodes_valid, len(episodes))

        if not results:
            print(f"  [skip] No valid episodes found")
            return None

        df = pl.DataFrame(results)
        metadata.processing_completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata.episodes_valid = total_episodes_valid
        metadata.pipeline_version = arena_evaluation.__version__ if hasattr(arena_evaluation, "__version__") else "1.0.0"

        MetadataWriter.write(metadata, metadata_path)
        ParquetStore.write(df, metrics_path, metadata)

        print(f"  Done: {metrics_path}")
        return metrics_path

    def extract_benchmark(self, benchmark_id: str) -> None:
        """
        Extract all runs in a benchmark.
        """
        runs = self.folder_manager.discover_runs(benchmark_id)
        if not runs:
            print(f"No runs found for benchmark '{benchmark_id}'")
            return
            
        print(f"Extracting benchmark {benchmark_id} ({len(runs)} runs)...")
        for i, run in enumerate(runs):
            print(f"[{i+1}/{len(runs)}] Extracting run {run.run_dir}...")
            self.extract_run(run)
        print("Done.")

    def process_benchmark(self, benchmark_id: str, force_extract: bool = False) -> None:
        """
        Process all runs in a benchmark and combine into a single parquet file.
        """
        runs = self.folder_manager.discover_runs(benchmark_id)
        if not runs:
            print(f"No runs found for benchmark '{benchmark_id}'")
            return
            
        parquet_files = []
        
        print(f"Processing benchmark {benchmark_id} ({len(runs)} runs)...")
        for i, run in enumerate(runs):
            print(f"[{i+1}/{len(runs)}] Processing run {run.run_dir}...")
            out_path = self.process_run(run, force_extract=force_extract)
            if out_path:
                parquet_files.append(out_path)
                
        if parquet_files:
            combined_path = self.folder_manager.combined_metrics_path(benchmark_id)
            print(f"Combining {len(parquet_files)} result files into {combined_path}...")
            ParquetStore.combine(parquet_files, combined_path)
            print("Done.")
        else:
            print("No valid results were generated.")
