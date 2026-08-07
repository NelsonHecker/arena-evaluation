from __future__ import annotations

import pathlib
import datetime
import contextlib
import typing
import polars as pl
import os
import concurrent.futures

from ..storage.schemas import RobotParams, RunDescriptor, EpisodeDescriptor, TopicBundle
from ..storage.folder_manager import FolderManager

def _worker_init():
    import os
    os.environ["POLARS_MAX_THREADS"] = "1"

def _extract_worker(data_root_str: str, ep: EpisodeDescriptor, force_extract: bool) -> int:
    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager
    import pathlib
    fm = FolderManager(data_root=pathlib.Path(data_root_str))
    pipeline = ProcessingPipeline(fm, profiler=None, workers=1)
    pipeline.extract_episode(ep, force_extract=force_extract)
    return ep.episode_id

def _process_worker(data_root_str: str, ep: EpisodeDescriptor, force_extract: bool) -> typing.Tuple[int, typing.Any]:
    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager
    import pathlib
    fm = FolderManager(data_root=pathlib.Path(data_root_str))
    pipeline = ProcessingPipeline(fm, profiler=None, workers=1)
    result = pipeline.process_episode(ep, force_extract=force_extract)
    return ep.episode_id, result
from ..storage.manifest import MetadataWriter
from ..benchmark.profiler import PipelineProfiler

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
    def __init__(self, folder_manager: FolderManager, profiler: PipelineProfiler | None = None, workers: int | None = None):
        self.folder_manager = folder_manager
        self.profiler = profiler
        # -1 (or None) = auto-detect the CPU count; any value < 1 falls back to it.
        self.workers = (os.cpu_count() or 1) if (workers is None or workers < 1) else workers

    # ── New flat-episode API ───────────────────────────────────────────────────

    def extract_episode(self, ep: EpisodeDescriptor, force_extract: bool = False) -> dict[str, TopicBundle] | None:
        """Extract topics from a single episode MCAP into episode_dir/topics/."""
        episode_dir = pathlib.Path(ep.episode_dir)
        mcap_path = self.folder_manager.mcap_path_for_episode(episode_dir)
        topics_dir = self.folder_manager.extracted_topics_path_for_episode(episode_dir)

        if not mcap_path.exists():
            print(f"  [skip] No MCAP found for episode_{ep.episode_id:03d} ({ep.planner}/{ep.stage})")
            return None

        if not force_extract:
            bundles = TopicParquetStore.read(topics_dir)
            if bundles:
                print(f"  Loading cached topics for episode_{ep.episode_id:03d}...")
                return bundles

        _ctx = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx:
            print(f"  Extracting episode_{ep.episode_id:03d} ({ep.planner}/{ep.stage}) → {topics_dir}...")
            reader = MCAPReader(mcap_path)
            bundles = reader.read()
            TopicParquetStore.write(bundles, topics_dir)
            return bundles

    def process_episode(self, ep: EpisodeDescriptor, force_extract: bool = False) -> dict | None:
        """
        Process a single episode: extract MCAP, compute metrics.
        Returns a metrics dict (one row for combined_metrics.parquet) or None if skipped.
        """
        episode_dir = pathlib.Path(ep.episode_dir)

        # Load episode metadata
        yaml_path = episode_dir / f"{episode_dir.name}.yaml"
        if not yaml_path.exists():
            yaml_path = episode_dir / "metadata.yaml"

        metadata = None
        if yaml_path.exists():
            try:
                metadata = MetadataWriter.read(yaml_path)
            except Exception as e:
                print(f"  [warn] Could not read metadata for episode_{ep.episode_id:03d}: {e}")

        # Extract topics
        bundles = self.extract_episode(ep, force_extract=force_extract)
        if bundles is None or len(bundles) == 0:
            return None

        _ctx = self.profiler.phase("process") if self.profiler else contextlib.nullcontext()
        with _ctx:
            robot_model = "turtlebot3_burger"
            pedsim_avail = False
            if metadata is not None:
                robot_model = metadata.robot_model[0] if metadata.robot_model else robot_model
                pedsim_avail = metadata.pedsim_available or False

            robot_params = RobotParams.load(robot_model)
            registry = MetricRegistry(robot_params)

            all_results = []
            for robot_name, bundle in bundles.items():
                if bundle.peds is not None:
                    pedsim_avail = True

                available_topics = {
                    field for field in (
                        "odom", "scan", "cmd_vel", "joint_states", "peds",
                        "collision_events", "collision_monitor_state",
                        "power", "energy", "plan", "initialpose", "tf", "tf_static", "tf_gt"
                    )
                    if getattr(bundle, field, None) is not None
                }

                aligner = TopicAligner()
                splitter = EpisodeSplitter(aligner)
                episodes = splitter.split(bundle, robot_name=robot_name)

                if not episodes:
                    print(f"  [episode_{ep.episode_id:03d}] [{robot_name}] No valid episodes found in MCAP")
                    continue

                # Each per-episode MCAP should produce exactly 1 AlignedEpisodeBundle
                for aligned_ep in episodes:
                    ep_metrics = registry.run(aligned_ep, pedsim_available=pedsim_avail, available_topics=available_topics)
                    ep_metrics["episode"] = ep.episode_id
                    ep_metrics["planner"] = ep.planner
                    # Prefer the explicit planner identity written into the
                    # episode yaml at recording time (from the contestant's
                    # mobile.local_planner / mobile.inter_planner config);
                    # fall back to name parsing for legacy recordings.
                    if metadata is not None and metadata.local_planner:
                        ep_metrics["local_planner"] = metadata.local_planner
                        ep_metrics["inter_planner"] = metadata.inter_planner or ""
                    else:
                        from ..presentation.dimension_detector import split_planner_name
                        lp, ip = split_planner_name(ep.planner)
                        ep_metrics["local_planner"] = lp
                        ep_metrics["inter_planner"] = ip
                    ep_metrics["robot"] = aligned_ep.robot_name or robot_name
                    ep_metrics["map"] = ep.map
                    ep_metrics["stage"] = ep.stage
                    ep_metrics["benchmark_id"] = ep.benchmark_id
                    ep_metrics["start"] = aligned_ep.start_pos
                    ep_metrics["goal"] = aligned_ep.goal_pos
                    ep_metrics["is_reference"] = ep.is_reference
                    ep_metrics["reference_type"] = ep.reference_type
                    all_results.append(ep_metrics)

            if all_results:
                try:
                    df_ep = pl.DataFrame(all_results)
                    ParquetStore.write(df_ep, episode_dir / "metrics.parquet")
                except Exception:
                    pass

            return all_results[0] if len(all_results) == 1 else (all_results if all_results else None)

    def extract_benchmark(self, benchmark_id: str, force_extract: bool = False) -> None:
        """Extract all episodes in a benchmark."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)
        if not episodes:
            # Fall back to legacy discover_runs
            runs = self.folder_manager.discover_runs(benchmark_id)
            if runs:
                print(f"Extracting benchmark {benchmark_id} (legacy structure, {len(runs)} runs)...")
                for i, run in enumerate(runs):
                    print(f"[{i+1}/{len(runs)}] Extracting run {run.run_dir}...")
                    self.extract_run(run)
                print("Done.")
            else:
                print(f"No episodes or runs found for benchmark '{benchmark_id}'")
            return

        print(f"Extracting benchmark {benchmark_id} ({len(episodes)} episodes) with {self.workers} workers...")
        data_root_str = str(self.folder_manager.data_root)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers, initializer=_worker_init) as executor:
            futures = [
                executor.submit(_extract_worker, data_root_str, ep, force_extract)
                for ep in episodes
            ]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    ep_id = future.result()
                    print(f"[{i+1}/{len(episodes)}] Extracted episode_{ep_id:03d}")
                except Exception as e:
                    print(f"Error extracting episode: {e}")
        print("Done.")

    def process_benchmark(self, benchmark_id: str, force_extract: bool = False) -> None:
        """Process all episodes and write combined_metrics.parquet using a 2-Phase pipeline."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)

        if not episodes:
            # Fall back to legacy processing
            runs = self.folder_manager.discover_runs(benchmark_id)
            if runs:
                print(f"Processing benchmark {benchmark_id} (legacy structure, {len(runs)} runs)...")
                parquet_files = []
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
            else:
                print(f"No episodes or runs found for benchmark '{benchmark_id}'")
            return

        # ── Phase 1: Extract All Episodes ─────────────────────────────────────
        print(f"Phase 1: Extracting all episodes for benchmark {benchmark_id} ({len(episodes)} episodes)...")
        _ctx_extract = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx_extract:
            self.extract_benchmark(benchmark_id, force_extract=force_extract)

        # ── Phase 2: Calculate All Metrics (with cross-episode lookups) ────────
        print(f"Phase 2: Calculating metrics across {len(episodes)} episodes...")
        all_metrics: list[dict] = []
        episode_bundles: dict[int, dict[str, TopicBundle]] = {}

        # 1. Load bundles and process per-episode metrics
        data_root_str = str(self.folder_manager.data_root)
        _ctx_process = self.profiler.phase("process") if self.profiler else contextlib.nullcontext()
        with _ctx_process:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers, initializer=_worker_init) as executor:
                futures = {
                    executor.submit(_process_worker, data_root_str, ep, False): ep
                    for ep in episodes
                }
                
                completed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    ep = futures[future]
                    completed_count += 1
                    try:
                        ep_id, result = future.result()
                        if result is not None:
                            if isinstance(result, list):
                                all_metrics.extend(result)
                            else:
                                all_metrics.append(result)
                        
                        print(f"[{completed_count}/{len(episodes)}] Processed metrics for episode_{ep_id:03d}")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error processing episode {ep.episode_id}: {e}")

        if not all_metrics:
            print("No valid results were generated.")
            return

        # 2. Cross-episode reference matching (e.g. unhindered_peds matching for DTW deflection)
        ref_episodes_by_stage: dict[str, list[dict]] = {}
        for row in all_metrics:
            if row.get("is_reference") and row.get("reference_type") == "unhindered_peds":
                stage = row.get("stage")
                if stage not in ref_episodes_by_stage:
                    ref_episodes_by_stage[stage] = []
                ref_episodes_by_stage[stage].append(row)

        if ref_episodes_by_stage:
            import numpy as np
            for row in all_metrics:
                if not row.get("is_reference"):
                    stage = row.get("stage")
                    refs = ref_episodes_by_stage.get(stage, [])
                    if refs and "pedestrian_path" in row and row["pedestrian_path"] is not None:
                        actual_paths = row["pedestrian_path"]
                        # Match with first reference episode for the same stage
                        ref_paths = refs[0].get("pedestrian_path")
                        if actual_paths and ref_paths:
                            try:
                                actual_pts = np.concatenate([np.array(p) for p in actual_paths if len(p) > 0], axis=0) if isinstance(actual_paths, list) else np.array(actual_paths)
                                ref_pts = np.concatenate([np.array(p) for p in ref_paths if len(p) > 0], axis=0) if isinstance(ref_paths, list) else np.array(ref_paths)
                                if len(actual_pts) > 0 and len(ref_pts) > 0 and actual_pts.ndim == 2 and ref_pts.ndim == 2:
                                    from .metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
                                    deflect = PedestrianDisturbanceCalculator._compute_trajectory_deflection(actual_pts, ref_pts)
                                    row["ped_path_deflection_m"] = round(float(deflect), 3)
                            except Exception:
                                pass

        combined_path = self.folder_manager.combined_metrics_path(benchmark_id)
        df = pl.DataFrame(all_metrics)
        ParquetStore.write(df, combined_path)
        print(f"Done. {len(all_metrics)} episode rows → {combined_path}")

    # ── Legacy API (backward-compatible) ──────────────────────────────────────

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

        _ctx = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx:
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

        metadata = MetadataWriter.read(metadata_path)

        bundles = None
        if not force_extract:
            bundles = TopicParquetStore.read(extracted_dir)
            if bundles:
                print(f"  Loading cached topics for {run.planner}/{run.stage}...")

        if bundles is None:
            bundles = self.extract_run(run)

        if bundles is None or len(bundles) == 0:
            return None

        _ctx = self.profiler.phase("process") if self.profiler else contextlib.nullcontext()
        with _ctx:
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
                for field in ("odom", "scan", "cmd_vel", "joint_states", "peds", "collision_events", "collision_monitor_state", "power", "energy", "plan", "initialpose", "tf", "tf_static", "tf_gt"):
                    if getattr(bundle, field, None) is not None:
                        available_topics.add(field)

                for ep in episodes:
                    ep.run = run
                    ep.folder_manager = self.folder_manager
                    ep_metrics = registry.run(ep, pedsim_available=pedsim_avail, available_topics=available_topics)

                    ep_metrics["episode"] = ep.episode_id
                    ep_metrics["planner"] = run.planner

                    # Prefer explicit planner identity from metadata.yaml;
                    # fall back to name parsing for legacy recordings.
                    if metadata.local_planner:
                        ep_metrics["local_planner"] = metadata.local_planner
                        ep_metrics["inter_planner"] = metadata.inter_planner or ""
                    else:
                        from ..presentation.dimension_detector import split_planner_name
                        lp, ip = split_planner_name(run.planner)
                        ep_metrics["local_planner"] = lp
                        ep_metrics["inter_planner"] = ip

                    ep_metrics["robot"] = ep.robot_name or robot_name
                    ep_metrics["map"] = metadata.map
                    ep_metrics["stage"] = run.stage
                    ep_metrics["benchmark_id"] = run.benchmark_id
                    ep_metrics["start"] = ep.start_pos
                    ep_metrics["goal"] = ep.goal_pos

                    ep_metrics["is_reference"] = getattr(metadata, "is_reference", False) or False
                    ep_metrics["reference_type"] = getattr(metadata, "reference_type", None)
                    ep_metrics["parent_episode_id"] = getattr(metadata, "parent_episode_id", None)

                    results.append(ep_metrics)

                total_episodes_valid = max(total_episodes_valid, len(episodes))

            if not results:
                print(f"  [skip] No valid episodes generated for {run.planner}/{run.stage}")
                return None

            print(f"  Saving metrics to {metrics_path}...")
            df = pl.DataFrame(results)

            metadata.processing_completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            metadata.episodes_valid = total_episodes_valid
            metadata.pipeline_version = arena_evaluation.__version__ if hasattr(arena_evaluation, "__version__") else "1.0.0"

            MetadataWriter.write(metadata, metadata_path)
            ParquetStore.write(df, metrics_path, metadata)

            print(f"  Done: {metrics_path}")
            return metrics_path

    def extract_run_dir(self, run_dir: pathlib.Path) -> TopicBundle | None:
        """Extract topics from a direct run directory (legacy ad-hoc use)."""
        run_dir = run_dir.resolve()
        extracted_dir = self.folder_manager.extracted_topics_path(run_dir)

        mcap_candidates = list(run_dir.glob("*.mcap"))
        if not mcap_candidates:
            # Check for episode structure inside run_dir
            for ep_dir in sorted(run_dir.glob("episode_*")):
                mcap_candidates.extend(ep_dir.glob("*.mcap"))
        if not mcap_candidates:
            print(f"Error: no .mcap file found under {run_dir}")
            return None

        source_path = sorted(mcap_candidates)[0]

        _ctx = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx:
            print(f"  Extracting ad-hoc run to {extracted_dir}...")
            reader = MCAPReader(source_path)
            bundle = reader.read()

            TopicParquetStore.write(bundle, extracted_dir)
            return bundle

    def process_run_dir(self, run_dir: pathlib.Path, force_extract: bool = False) -> pathlib.Path | None:
        """
        Process a single recording directory directly, without needing a benchmark structure.
        Supports both episode directories (episode_XXX/) and legacy step directories.
        """
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            print(f"Error: run directory does not exist: {run_dir}")
            return None

        # Check if this looks like a single episode directory
        ep_yaml = run_dir / f"{run_dir.name}.yaml"
        meta_yaml = run_dir / "metadata.yaml"
        yaml_path = ep_yaml if ep_yaml.exists() else (meta_yaml if meta_yaml.exists() else None)

        if yaml_path is None:
            print(f"Error: no metadata yaml found in {run_dir}")
            return None

        metadata = MetadataWriter.read(yaml_path)

        # Try to parse episode_id from directory name
        ep_id = None
        if run_dir.name.startswith("episode_"):
            try:
                ep_id = int(run_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                pass

        ep_desc = EpisodeDescriptor(
            episode_dir=str(run_dir),
            benchmark_id=getattr(metadata, "benchmark_id", "unknown"),
            episode_id=ep_id or 0,
            planner=getattr(metadata, "planner", run_dir.name),
            stage=getattr(metadata, "stage", "unknown"),
            map=getattr(metadata, "map", "unknown"),
            is_reference=getattr(metadata, "is_reference", False),
            reference_type=getattr(metadata, "reference_type", None),
        )

        result = self.process_episode(ep_desc, force_extract=force_extract)
        if result is None:
            print(f"Processing failed for {run_dir} — see errors above.")
            return None

        rows = result if isinstance(result, list) else [result]
        df = pl.DataFrame(rows)
        out_path = run_dir / "metrics.parquet"
        ParquetStore.write(df, out_path, metadata)
        print(f"  Done: {out_path}")
        return out_path
