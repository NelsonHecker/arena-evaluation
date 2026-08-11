from __future__ import annotations

import pathlib
import datetime
import contextlib
import typing
import polars as pl
import os
import concurrent.futures

from ..storage.schemas import RobotParams, EpisodeDescriptor, TopicBundle, AlignedEpisodeBundle
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
            if topics_dir.exists():
                import shutil
                shutil.rmtree(topics_dir)
            reader = MCAPReader(mcap_path)
            bundles = reader.read(map_name_fallback=ep.map)
            TopicParquetStore.write(bundles, topics_dir, overwrite=True)
            return bundles

    def process_episode(self, ep: EpisodeDescriptor, force_extract: bool = False) -> dict | None:
        """Process a single episode: extract MCAP, compute metrics."""
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
                        "power", "energy", "acoustics", "plan", "initialpose",
                        "tf", "tf_static", "tf_gt", "semantic_snapshot", "characterization_phase"
                    )
                    if getattr(bundle, field, None) is not None
                }

                if bundle.odom is None:
                    continue
                    
                aligner = TopicAligner()
                aligned_df = aligner.align(bundle)
                if isinstance(aligned_df, pl.LazyFrame):
                    aligned_df = aligned_df.collect()
                
                if aligned_df is None or len(aligned_df) < 5:
                    print(f"  [episode_{ep.episode_id:03d}] [{robot_name}] No valid data found in MCAP after alignment")
                    continue
                
                start_pos, goal_pos = [], []
                first_row = aligned_df.row(0, named=True)
                last_row = aligned_df.row(-1, named=True)
                
                if "pos_x" in first_row and "pos_y" in first_row and first_row["pos_x"] is not None:
                    start_pos = [first_row["pos_x"], first_row["pos_y"], first_row.get("yaw", 0.0)]
                
                if "pos_x" in last_row and "pos_y" in last_row and last_row["pos_x"] is not None:
                    goal_pos = [last_row["pos_x"], last_row["pos_y"], last_row.get("yaw", 0.0)]
                
                plan_df = bundle.plan.collect() if isinstance(bundle.plan, pl.LazyFrame) else bundle.plan
                if plan_df is not None and len(plan_df) > 0:
                    row_plan = plan_df.row(-1, named=True)
                    if "poses_x" in row_plan and len(row_plan["poses_x"]) > 0:
                        goal_pos = [row_plan["poses_x"][-1], row_plan["poses_y"][-1], row_plan["poses_yaw"][-1]]
                
                num_pedestrians = 0
                if "num_pedestrians" in aligned_df.columns:
                    peds = aligned_df["num_pedestrians"].drop_nulls()
                    if len(peds) > 0:
                        num_pedestrians = int(peds.max())

                semantic_snapshot_df = bundle.semantic_snapshot
                if isinstance(semantic_snapshot_df, pl.LazyFrame):
                    semantic_snapshot_df = semantic_snapshot_df.collect()

                conditions = None
                if bundle.episode_record is not None:
                    er = bundle.episode_record
                    if isinstance(er, pl.LazyFrame):
                        er = er.collect()
                    if len(er) > 0:
                        import json
                        try:
                            cond_raw = er["conditions"][0] if "conditions" in er.columns else None
                            if cond_raw:
                                conditions = json.loads(cond_raw)
                        except Exception:
                            conditions = None

                aligned_ep = AlignedEpisodeBundle(
                    episode_id=ep.episode_id,
                    data=aligned_df,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    num_pedestrians=num_pedestrians,
                    robot_name=robot_name,
                    semantic_snapshot=semantic_snapshot_df,
                    conditions=conditions
                )
                episodes = [aligned_ep]

                # Each per-episode MCAP should produce exactly 1 AlignedEpisodeBundle
                for aligned_ep in episodes:
                    ep_metrics = registry.run(aligned_ep, pedsim_available=pedsim_avail, available_topics=available_topics)
                    ep_metrics["episode"] = ep.episode_id
                    ep_metrics["planner"] = ep.planner
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
            print(f"No episodes found for benchmark '{benchmark_id}'")
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
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return

        print(f"Phase 1: Extracting all episodes for benchmark {benchmark_id} ({len(episodes)} episodes)...")
        _ctx_extract = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx_extract:
            self.extract_benchmark(benchmark_id, force_extract=force_extract)

        print(f"Phase 2: Calculating metrics across {len(episodes)} episodes...")
        all_metrics: list[dict] = []
        episode_bundles: dict[int, dict[str, TopicBundle]] = {}

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
