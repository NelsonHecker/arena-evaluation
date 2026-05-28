from __future__ import annotations

import pathlib
import datetime
import polars as pl

from ..storage.schemas import RobotParams, RunDescriptor
from ..storage.folder_manager import FolderManager
from ..storage.manifest import MetadataWriter

from .mcap_reader import MCAPReader
from .topic_aligner import TopicAligner
from .episode_splitter import EpisodeSplitter
from .parquet_store import ParquetStore
from .metrics.registry import MetricRegistry

import arena_evaluation

class ProcessingPipeline:
    """
    Orchestrates the data processing pipeline:
    MCAP -> Align -> Split -> Metrics -> Parquet
    """
    def __init__(self, folder_manager: FolderManager):
        self.folder_manager = folder_manager
        
    def process_run(self, run: RunDescriptor) -> pathlib.Path | None:
        """
        Process a single run described by a RunDescriptor and generate its metrics.parquet.
        Returns the path to the generated parquet file, or None if the run is skipped.
        """
        run_dir = pathlib.Path(run.run_dir)
        mcap_path = self.folder_manager.mcap_path(run_dir)
        metrics_path = self.folder_manager.metrics_path(run_dir)
        metadata_path = run_dir / "metadata.yaml"

        # Determine source
        source_path = mcap_path
        if not source_path.exists():
            # Check legacy csv dir
            if (run_dir / "odom.csv").exists():
                source_path = run_dir
            else:
                print(f"  [skip] No MCAP or legacy CSV found for {run.planner}/{run.stage}")
                return None

        # 1. Read metadata
        metadata = MetadataWriter.read(metadata_path)

        # For robot params, assume single robot or take first if list
        robot_model = metadata.robot_model[0] if metadata.robot_model else "turtlebot3_burger"
        robot_params = RobotParams.load(robot_model)

        # 2. Read MCAP
        print(f"  Reading {run.planner}/{run.stage}...")
        reader = MCAPReader(source_path)
        bundle = reader.read()

        # 3. Align and Split
        print(f"  Splitting into episodes...")
        aligner = TopicAligner()
        splitter = EpisodeSplitter(aligner)
        episodes = splitter.split(bundle)

        if not episodes:
            print(f"  [skip] No valid episodes found for {run.planner}/{run.stage}")
            return None

        print(f"  Computing metrics for {len(episodes)} episodes...")

        # 4. Calculate Metrics
        registry = MetricRegistry(robot_params)
        pedsim_avail = metadata.pedsim_available if metadata.pedsim_available is not None else False

        results = []
        for ep in episodes:
            ep_metrics = registry.run(ep, pedsim_available=pedsim_avail)

            # 5. Add identity columns
            ep_metrics["episode"] = ep.episode_id
            ep_metrics["planner"] = run.planner
            ep_metrics["robot"] = robot_model
            ep_metrics["stage"] = run.stage
            ep_metrics["benchmark_id"] = run.benchmark_id

            results.append(ep_metrics)

        # 6. Save to Parquet
        print(f"  Saving metrics to {metrics_path}...")
        df = pl.DataFrame(results)

        # Update metadata
        metadata.processing_completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata.episodes_valid = len(episodes)
        metadata.pipeline_version = arena_evaluation.__version__ if hasattr(arena_evaluation, "__version__") else "1.0.0"

        MetadataWriter.write(metadata, metadata_path)
        ParquetStore.write(df, metrics_path, metadata)

        print(f"  Done: {metrics_path}")
        return metrics_path

    def process_run_dir(self, run_dir: pathlib.Path) -> pathlib.Path | None:
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

        # Resolve the MCAP path relative to the run dir
        # FolderManager expects the folder structure; bypass it for ad-hoc runs
        recording_subdir = run_dir / "recording"
        mcap_candidates = list(recording_subdir.glob("*.mcap")) if recording_subdir.exists() else []
        if not mcap_candidates:
            mcap_candidates = list(run_dir.glob("**/*.mcap"))
        if not mcap_candidates:
            print(f"Error: no .mcap file found under {run_dir}")
            return None

        source_path = sorted(mcap_candidates)[0]
        metrics_path = run_dir / "metrics.parquet"

        # Re-use the core logic
        robot_model = metadata.robot_model[0] if metadata.robot_model else "turtlebot3_burger"
        robot_params = RobotParams.load(robot_model)

        print(f"  Reading MCAP: {source_path}")
        reader = MCAPReader(source_path)
        bundle = reader.read()

        print(f"  Splitting into episodes...")
        aligner = TopicAligner()
        splitter = EpisodeSplitter(aligner)
        episodes = splitter.split(bundle)

        if not episodes:
            print(f"  [skip] No valid episodes found")
            return None

        print(f"  Computing metrics for {len(episodes)} episodes...")
        registry = MetricRegistry(robot_params)
        pedsim_avail = metadata.pedsim_available if metadata.pedsim_available is not None else False

        results = []
        for ep in episodes:
            ep_metrics = registry.run(ep, pedsim_available=pedsim_avail)
            ep_metrics["episode"] = ep.episode_id
            ep_metrics["planner"] = descriptor.planner
            ep_metrics["robot"] = robot_model
            ep_metrics["stage"] = descriptor.stage
            ep_metrics["benchmark_id"] = descriptor.benchmark_id
            results.append(ep_metrics)

        df = pl.DataFrame(results)
        metadata.processing_completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata.episodes_valid = len(episodes)
        metadata.pipeline_version = arena_evaluation.__version__ if hasattr(arena_evaluation, "__version__") else "1.0.0"

        MetadataWriter.write(metadata, metadata_path)
        ParquetStore.write(df, metrics_path, metadata)

        print(f"  Done: {metrics_path}")
        return metrics_path

    def process_benchmark(self, benchmark_id: str) -> None:
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
            out_path = self.process_run(run)
            if out_path:
                parquet_files.append(out_path)
                
        if parquet_files:
            combined_path = self.folder_manager.combined_metrics_path(benchmark_id)
            print(f"Combining {len(parquet_files)} result files into {combined_path}...")
            ParquetStore.combine(parquet_files, combined_path)
            print("Done.")
        else:
            print("No valid results were generated.")
