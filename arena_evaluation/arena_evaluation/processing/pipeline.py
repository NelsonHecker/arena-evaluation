from __future__ import annotations

import pathlib
import datetime
import contextlib
import typing
import polars as pl
import os
import sys
import time
import shutil
import json
import traceback
import multiprocessing
import concurrent.futures

from ..storage.schemas import RobotParams, EpisodeDescriptor, TopicBundle, AlignedEpisodeBundle
from ..storage.folder_manager import FolderManager

def _worker_init():
    import os
    import signal
    os.environ["POLARS_MAX_THREADS"] = "1"
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def _shutdown_executor_cleanly(executor: concurrent.futures.ProcessPoolExecutor):
    """Force terminate all child worker processes without blocking on wait=True."""
    try:
        processes = list(getattr(executor, "_processes", {}).values())
        for proc in processes:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

def _extract_worker(data_root_str: str, ep: EpisodeDescriptor, force_extract: bool, status_dict: typing.Any = None) -> int:
    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager
    import pathlib
    import time
    if status_dict is not None:
        try:
            status_dict[ep.episode_id] = (ep.planner, ep.stage, "Extracting MCAP / Topics", 1, 1, time.perf_counter())
        except Exception:
            pass
    fm = FolderManager(data_root=pathlib.Path(data_root_str))
    pipeline = ProcessingPipeline(fm, profiler=None, workers=1)
    pipeline.extract_episode(ep, force_extract=force_extract)
    if status_dict is not None:
        try:
            status_dict.pop(ep.episode_id, None)
        except Exception:
            pass
    return ep.episode_id

def _process_worker(data_root_str: str, ep: EpisodeDescriptor, force_extract: bool, status_dict: typing.Any = None) -> typing.Tuple[int, typing.Any]:
    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager
    import pathlib
    import time
    if status_dict is not None:
        try:
            status_dict[ep.episode_id] = (ep.planner, ep.stage, "Loading Topics", 0, 18, time.perf_counter())
        except Exception:
            pass
    fm = FolderManager(data_root=pathlib.Path(data_root_str))
    pipeline = ProcessingPipeline(fm, profiler=None, workers=1)
    result = pipeline.process_episode(ep, force_extract=force_extract, status_dict=status_dict)
    if status_dict is not None:
        try:
            status_dict.pop(ep.episode_id, None)
        except Exception:
            pass
    return ep.episode_id, result


def _resolve_odom_frame(aligned_df) -> "pl.DataFrame | None":
    """Filter null poses and slice to longest consistent segment."""
    import polars as pl
    import numpy as np

    if aligned_df is None or len(aligned_df) == 0:
        return aligned_df

    if "pos_x_gt" in aligned_df.columns:
        aligned_df = aligned_df.filter(
            pl.col("pos_x_gt").is_not_null() & ~pl.col("pos_x_gt").is_nan()
            & pl.col("pos_y_gt").is_not_null() & ~pl.col("pos_y_gt").is_nan()
            & pl.col("yaw_gt").is_not_null() & ~pl.col("yaw_gt").is_nan()
        )
    else:
        aligned_df = aligned_df.filter(
            pl.col("pos_x").is_not_null() & ~pl.col("pos_x").is_nan()
            & pl.col("pos_y").is_not_null() & ~pl.col("pos_y").is_nan()
            & pl.col("yaw").is_not_null() & ~pl.col("yaw").is_nan()
        )
    if len(aligned_df) == 0:
        return aligned_df

    use_gt = "pos_x_gt" in aligned_df.columns
    odom_x = aligned_df["pos_x_gt" if use_gt else "pos_x"].to_numpy()
    odom_y = aligned_df["pos_y_gt" if use_gt else "pos_y"].to_numpy()
    if len(odom_x) > 1:
        dists = np.sqrt(np.diff(odom_x) ** 2 + np.diff(odom_y) ** 2)
        jumps = np.where(dists > 0.5)[0]
        if len(jumps) > 0:
            split_indices = jumps + 1
            segments_x = np.split(odom_x, split_indices)
            segments_y = np.split(odom_y, split_indices)
            best_seg_idx = -1
            best_len = -1.0
            for i in range(len(segments_x)):
                seg_x, seg_y = segments_x[i], segments_y[i]
                if len(seg_x) < 2:
                    seg_len = 0.0
                else:
                    seg_len = float(np.sum(np.sqrt(np.diff(seg_x) ** 2 + np.diff(seg_y) ** 2)))
                if seg_len >= 0.2 and seg_len > best_len:
                    best_len = seg_len
                    best_seg_idx = i
            if best_seg_idx != -1:
                start_idx = 0 if best_seg_idx == 0 else int(split_indices[best_seg_idx - 1])
                end_idx = int(split_indices[best_seg_idx]) if best_seg_idx < len(split_indices) - 1 else len(odom_x)
                aligned_df = aligned_df.slice(start_idx, end_idx - start_idx)
    return aligned_df


def _collect_native_topics(bundle) -> dict:
    """Raw native-rate topic frames keyed by topic name (collected)."""
    import polars as pl

    topics: dict = {}
    for field in (
        "odom", "scan", "cmd_vel", "joint_states", "peds",
        "collision_events", "collision_monitor_state", "power", "energy",
        "acoustics", "tf_gt",
    ):
        df = getattr(bundle, field, None)
        if df is None:
            continue
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        if df is not None and len(df) > 0:
            topics[field] = df
    return topics


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
                return bundles

        _ctx = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx:
            if topics_dir.exists():
                import shutil
                shutil.rmtree(topics_dir)
            reader = MCAPReader(mcap_path)
            bundles = reader.read(map_name_fallback=ep.map)
            TopicParquetStore.write(bundles, topics_dir, overwrite=True)
            return bundles

    def process_episode(
        self,
        ep: EpisodeDescriptor,
        force_extract: bool = False,
        status_dict: typing.Any = None,
    ) -> dict | None:
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

                # Resolve the pose frame once so all calculators share it
                aligned_df = _resolve_odom_frame(aligned_df)
                if aligned_df is None or len(aligned_df) < 5:
                    print(f"  [episode_{ep.episode_id:03d}] [{robot_name}] No valid pose data after resolution")
                    continue

                topics = _collect_native_topics(bundle)
                if len(aligned_df) > 0:
                    t0 = int(aligned_df["time_ns"][0])
                    t1 = int(aligned_df["time_ns"][-1])
                    tol = 100_000_000
                    topics = {
                        name: df.filter(
                            (pl.col("time_ns") >= t0 - tol) & (pl.col("time_ns") <= t1 + tol)
                        )
                        for name, df in topics.items()
                        if "time_ns" in df.columns
                    }

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
                    conditions=conditions,
                    map=ep.map,
                    topics=topics,
                )
                episodes = [aligned_ep]

                def on_calc_progress(calc_name: str, calc_idx: int, total_calcs: int):
                    if status_dict is not None:
                        try:
                            import time
                            status_dict[ep.episode_id] = (
                                ep.planner,
                                ep.stage,
                                calc_name,
                                calc_idx,
                                total_calcs,
                                time.perf_counter(),
                            )
                        except Exception:
                            pass

                for aligned_ep in episodes:
                    ep_metrics = registry.run(
                        aligned_ep,
                        pedsim_available=pedsim_avail,
                        available_topics=available_topics,
                        progress_callback=on_calc_progress,
                    )
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

        import multiprocessing
        from .progress_display import PipelineProgressDisplay

        manager = multiprocessing.Manager()
        status_dict = manager.dict()
        data_root_str = str(self.folder_manager.data_root)

        with PipelineProgressDisplay(
            f"Phase 1: Extracting MCAP topics for {benchmark_id}",
            len(episodes),
            self.workers,
            status_dict=status_dict,
        ) as display:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers, initializer=_worker_init
            )
            try:
                futures = {
                    executor.submit(
                        _extract_worker, data_root_str, ep, force_extract, status_dict
                    ): (ep, time.perf_counter())
                    for ep in episodes
                }
                for future in concurrent.futures.as_completed(futures):
                    ep, t_start = futures[future]
                    try:
                        ep_id = future.result()
                        elapsed = time.perf_counter() - t_start
                        display.log_completed(
                            ep_id, f"{ep.planner}/{ep.stage}", elapsed
                        )
                    except Exception as e:
                        display.log_error(ep.episode_id, str(e))
            except KeyboardInterrupt:
                _shutdown_executor_cleanly(executor)
                raise
            finally:
                _shutdown_executor_cleanly(executor)

    def process_benchmark(self, benchmark_id: str, force_extract: bool = False) -> None:
        """Process all episodes and write combined_metrics.parquet using a 2-Phase pipeline."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)

        if not episodes:
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return

        _ctx_extract = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx_extract:
            self.extract_benchmark(benchmark_id, force_extract=force_extract)

        all_metrics: list[dict] = []
        data_root_str = str(self.folder_manager.data_root)

        import multiprocessing
        from .progress_display import PipelineProgressDisplay

        manager = multiprocessing.Manager()
        status_dict = manager.dict()

        _ctx_process = self.profiler.phase("process") if self.profiler else contextlib.nullcontext()
        with _ctx_process:
            with PipelineProgressDisplay(
                f"Phase 2: Calculating metrics across {len(episodes)} episodes",
                len(episodes),
                self.workers,
                status_dict=status_dict,
            ) as display:
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.workers, initializer=_worker_init
                )
                try:
                    futures = {
                        executor.submit(
                            _process_worker, data_root_str, ep, False, status_dict
                        ): (ep, time.perf_counter())
                        for ep in episodes
                    }

                    for future in concurrent.futures.as_completed(futures):
                        ep, t_start = futures[future]
                        try:
                            ep_id, result = future.result()
                            elapsed = time.perf_counter() - t_start
                            if result is not None:
                                if isinstance(result, list):
                                    all_metrics.extend(result)
                                else:
                                    all_metrics.append(result)

                            display.log_completed(
                                ep_id, f"{ep.planner}/{ep.stage}", elapsed
                            )
                        except Exception as e:
                            display.log_error(ep.episode_id, str(e))
                except KeyboardInterrupt:
                    print("\n[KeyboardInterrupt] Metrics calculation interrupted by user. Terminating workers...", flush=True)
                    _shutdown_executor_cleanly(executor)
                    raise
                finally:
                    _shutdown_executor_cleanly(executor)

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

        ref_ped_by_stage: dict[str, list[dict]] = {}
        ref_robot_by_stage: dict[str, list[dict]] = {}
        for row in all_metrics:
            if not row.get("is_reference"):
                continue
            stage = row.get("stage")
            if row.get("reference_type") == "unhindered_peds":
                ref_ped_by_stage.setdefault(stage, []).append(row)
            elif row.get("reference_type") == "unobstructed_robot":
                ref_robot_by_stage.setdefault(stage, []).append(row)

        if ref_episodes_by_stage or ref_robot_by_stage:
            print(f"Phase 2 Post-Processing: Cross-referencing {len(all_metrics)} episodes against {len(ref_ped_by_stage)} ped references and {len(ref_robot_by_stage)} robot references...", flush=True)

        from .metrics.social.mutual_accommodation import MutualAccommodationCalculator

        for row in all_metrics:
            if row.get("is_reference"):
                continue

            stage = row.get("stage")
            ref_robots = ref_robot_by_stage.get(stage, [])
            ref_peds = ref_ped_by_stage.get(stage, [])

            ref_robot_row = ref_robots[0] if ref_robots else None
            ref_ped_row = ref_peds[0] if ref_peds else None

            reconciled = MutualAccommodationCalculator.reconcile_stage_references(
                dynamic_row=row,
                ref_robot_row=ref_robot_row,
                ref_ped_row=ref_ped_row,
            )
            row.update(reconciled)

        combined_path = self.folder_manager.combined_metrics_path(benchmark_id)
        df = pl.DataFrame(all_metrics)
        ParquetStore.write(df, combined_path)
        print(f"Done. {len(all_metrics)} episode rows -> {combined_path}")

