from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import multiprocessing
import os
import pathlib
import re
import shutil
import time
import typing

import polars as pl

if typing.TYPE_CHECKING:
    import numpy as np

    from arena_evaluation.processing.metrics.registry import MetricRegistry

from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import (
    AlignedEpisodeBundle,
    EpisodeDescriptor,
    RobotParams,
    RunMetadata,
    TopicBundle,
)

_log = logging.getLogger(__name__)

# Frames shared across robots or scoped to the whole episode, never per-robot sample data.
_SHARED_FRAMES = frozenset({"tf", "tf_static", "world_semantics", "episode_lifecycle", "episode_integrity"})
KEPT_EMPTY_TABLES = ("collision_contacts", "pedestrian_gestures")


def _episode_series_tables(
    registry: MetricRegistry,
    episode: AlignedEpisodeBundle,
    *,
    run_id: str,
    robot_id: str,
) -> dict[str, pl.DataFrame]:
    """1NF series frames the registered calculators publish for one robot episode, keyed by table stem."""
    frames: dict[str, pl.DataFrame] = {}
    for stem, calculator in registry.series_calculators():
        frame = calculator.series_table(episode, run_id=run_id, robot_id=robot_id)
        if frame is not None and frame.height > 0:
            frames[stem] = frame
    return frames


def _merge_series_frames(groups: list[dict[str, pl.DataFrame]]) -> dict[str, pl.DataFrame]:
    """Concatenate per-robot series frames by table stem."""
    parts: dict[str, list[pl.DataFrame]] = {}
    for group in groups:
        for stem, frame in group.items():
            parts.setdefault(stem, []).append(frame)
    return {stem: frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed") for stem, frames in parts.items()}


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
        processes = list(executor._processes.values())
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


def _extract_worker(
    data_root_str: str,
    ep: EpisodeDescriptor,
    force_extract: bool,
    status_dict: typing.MutableMapping[int, tuple[str, str, str, int, int, float]] | None = None,
) -> int:
    import pathlib
    import time

    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager

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


def _process_worker(
    data_root_str: str,
    ep: EpisodeDescriptor,
    force_extract: bool,
    status_dict: typing.MutableMapping[int, tuple[str, str, str, int, int, float]] | None = None,
) -> tuple[int, typing.Any]:
    import pathlib
    import time

    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager

    if status_dict is not None:
        try:
            status_dict[ep.episode_id] = (ep.planner, ep.stage, "Loading Topics", 0, 18, time.perf_counter())
        except Exception:
            pass
    fm = FolderManager(data_root=pathlib.Path(data_root_str))
    pipeline = ProcessingPipeline(fm, profiler=None, workers=1)
    try:
        result = pipeline.process_episode(ep, force_extract=force_extract, status_dict=status_dict)
    except Exception as e:
        _log.exception(f"episode_{ep.episode_id:03d}: metrics failed")
        result = [_status_row(ep, None, "", "error", repr(e))]
    if status_dict is not None:
        try:
            status_dict.pop(ep.episode_id, None)
        except Exception:
            pass
    return ep.episode_id, result


def _consistent_pose_segment(aligned_df: pl.DataFrame | None, ride_arrivals_ns: np.ndarray | None = None) -> pl.DataFrame | None:
    """The rows with a finite ground-truth pose, cut to the longest stretch free of teleport jumps (elevator rides and stair crossings excepted)."""
    import polars as pl

    from .pose_segments import consistent_segment

    if aligned_df is None or len(aligned_df) == 0:
        return aligned_df
    aligned_df = aligned_df.filter(pl.col("pos_x_gt").is_not_null() & ~pl.col("pos_x_gt").is_nan() & pl.col("pos_y_gt").is_not_null() & ~pl.col("pos_y_gt").is_nan() & pl.col("yaw_gt").is_not_null() & ~pl.col("yaw_gt").is_nan())
    if len(aligned_df) < 2:
        return aligned_df
    start, end = consistent_segment(aligned_df["pos_x_gt"].to_numpy(), aligned_df["pos_y_gt"].to_numpy(), aligned_df["time_ns"].to_numpy(), ride_arrivals_ns)
    return aligned_df.slice(start, end - start)


def _metric_world(metadata: RunMetadata | None, ep: EpisodeDescriptor) -> str | None:
    """World name the world-geometry metrics resolve their asset by, None when unknown."""
    if metadata is not None and metadata.map:
        return metadata.map
    if ep.map and ep.map != "unknown":
        return ep.map
    return None


_RECORD_RUNNING = 1  # arena_runtime_msgs/msg/EpisodeRecord.RUNNING
_RECORD_SUCCESS = 2  # arena_runtime_msgs/msg/EpisodeRecord.SUCCESS


def _episode_window(record: pl.DataFrame | pl.LazyFrame | None) -> tuple[int | None, int | None]:
    """[RUNNING stamp, terminal stamp] of the episode record, a bound is None when its row is missing."""
    if record is None:
        return None, None
    df = record.collect() if isinstance(record, pl.LazyFrame) else record
    if len(df) == 0 or "t_ns" not in df.columns:
        return None, None
    df = df.sort("t_ns")
    if "outcome_state" not in df.columns:
        return int(df["t_ns"][0]), int(df["t_ns"][-1])
    states = df["outcome_state"].to_list()
    stamps = df["t_ns"].to_list()
    start = next((int(t) for t, s in zip(stamps, states, strict=True) if s == _RECORD_RUNNING), None)
    end = None if states[-1] == _RECORD_RUNNING else int(stamps[-1])
    return start, end


def _episode_endpoints(
    aligned_df: pl.DataFrame,
    plan: pl.DataFrame | pl.LazyFrame | None,
    phases: dict | None = None,
) -> tuple[list[float], list[float]]:
    """Map-frame (x, y, yaw) of the first sample and of the goal: the last goto of the recorded phases, else the last planned pose, else the last sample."""
    first_row = aligned_df.row(0, named=True)
    last_row = aligned_df.row(-1, named=True)
    # Ground truth is map frame, odom starts wherever the robot's odometry happened to be.
    x, y, yaw = ("pos_x_gt", "pos_y_gt", "yaw_gt") if "pos_x_gt" in first_row else ("pos_x", "pos_y", "yaw")

    start_pos: list[float] = []
    goal_pos: list[float] = []
    if x in first_row and y in first_row and first_row[x] is not None:
        start_pos = [first_row[x], first_row[y], first_row.get(yaw, 0.0)]
    if x in last_row and y in last_row and last_row[x] is not None:
        goal_pos = [last_row[x], last_row[y], last_row.get(yaw, 0.0)]

    plan_df = plan.collect() if isinstance(plan, pl.LazyFrame) else plan
    if plan_df is not None and "t_ns" in plan_df.columns:
        # A recording can outlive its episode by a beat and catch the next episode's first plan.
        plan_df = plan_df.filter(pl.col("t_ns") <= aligned_df["time_ns"].max())
    if plan_df is not None and len(plan_df) > 0:
        latest_plan = plan_df.filter(pl.col("plan_id") == plan_df["plan_id"].max()).sort("waypoint_idx")
        goal_pos = [latest_plan["x"][-1], latest_plan["y"][-1], latest_plan["yaw"][-1]]

    gotos = [pose for pose in phases["map_poses"] if pose is not None] if phases else []
    if gotos:
        goal_pos = [float(v) for v in gotos[-1]]
    return start_pos, goal_pos


def _robot_phases(record: pl.DataFrame, robot_dir: str) -> dict | None:
    """This robot's {phases, conditions, map_poses, goal_inputs, instructions} from the last EpisodeRecord row that carries any, None without."""
    if "phases_yaml" not in record.columns:
        return None
    import json

    from arena_evaluation.processing.metrics.ecological.condition_metrics import bare_robot_name

    bare = bare_robot_name(robot_dir)
    for raw in reversed(record["phases_yaml"].to_list()):
        if not raw:
            continue
        try:
            entry = json.loads(raw).get(bare)
        except (ValueError, AttributeError):
            return None
        if entry is not None:
            return entry
    return None


def _fleet_poses(robots: dict[str, TopicBundle], window: tuple[int | None, int | None]) -> dict[str, pl.DataFrame]:
    """Recorded judge poses of every robot in the episode by bare name, cut to the window."""
    from arena_evaluation.processing.metrics.ecological.condition_metrics import bare_robot_name

    start_ns, end_ns = window
    fleet: dict[str, pl.DataFrame] = {}
    for name, bundle in robots.items():
        frame = bundle.robot_task_pose
        if frame is None:
            continue
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        if start_ns is not None:
            frame = frame.filter(pl.col("t_ns") >= start_ns - _ALIGN_TOLERANCE_NS)
        if end_ns is not None:
            frame = frame.filter(pl.col("t_ns") <= end_ns + _ALIGN_TOLERANCE_NS)
        if len(frame) > 0:
            fleet[bare_robot_name(name)] = frame
    return fleet


def _collect_native_topics(bundle: TopicBundle) -> dict[str, pl.DataFrame]:
    """Raw native-rate topic frames keyed by topic name (collected)."""
    topics: dict[str, pl.DataFrame] = {}
    for name in sorted(bundle.available() - _SHARED_FRAMES):
        df = getattr(bundle, name)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        if len(df) > 0 or name in KEPT_EMPTY_TABLES:
            topics[name] = df
    return topics


from arena_evaluation.benchmark.profiler import PipelineProfiler
from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.metrics.ecological.mechanism_common import robot_env_snapshot
from arena_evaluation.processing.parquet_store import ParquetStore, TopicParquetStore
from arena_evaluation.processing.pedestrian_frames import ALIGN_TOLERANCE_NS as _ALIGN_TOLERANCE_NS
from arena_evaluation.processing.pedestrian_frames import peak_pedestrian_count
from arena_evaluation.processing.pose_anchor import resolve_pose_source
from arena_evaluation.processing.pose_segments import ride_arrival_times
from arena_evaluation.processing.native_bundle import NativeBundle
from arena_evaluation.storage.manifest import MetadataWriter

# Columns whose values are all-None (no kind in the recording) or all-empty
# (no collisions) would otherwise infer as Null / List(Null) and clash with
# an Int64 / List(String) run when frames from several runs are combined.
_METRIC_DTYPES = {
    "collision_amount_wall": pl.Int64,
    "collision_amount_static": pl.Int64,
    "collision_amount_pedestrian": pl.Int64,
    "collision_obstacles": pl.List(pl.String),
    "pose_source": pl.String,
    "pose_samples": pl.Int64,
    "status": pl.String,
    "status_reason": pl.String,
    "result": pl.String,
    "outcome_info": pl.String,
    "integrity": pl.List(pl.String),
    "success": pl.Boolean,
    "ne": pl.Float64,
    "osr": pl.Boolean,
    "success_geodesic": pl.Boolean,
    "spl_geodesic": pl.Float64,
    "ndtw": pl.Float64,
    "sdtw": pl.Float64,
    "stuck": pl.Boolean,
    "start": pl.List(pl.Float64),
    "goal": pl.List(pl.Float64),
}

from .metrics.registry import MetricRegistry

# EpisodeRecord.outcome_state -> (result, success)
_OUTCOME_VERDICTS: dict[int | None, tuple[str, bool]] = {2: ("GOAL_REACHED", True), 3: ("FAILED", False), 4: ("CANCELLED", False), 5: ("FATAL", False)}
_STATUS_EVALUATED = "evaluated"


def _outcome_verdict(outcome_state: int | None, outcome_info: str | None) -> tuple[str, bool]:
    if outcome_info == "collision":
        return "COLLISION", False
    return _OUTCOME_VERDICTS.get(outcome_state, ("UNRESOLVED", False))


def _episode_rows(record: pl.DataFrame) -> pl.DataFrame:
    """The episode's own records in publish order, up to its first terminal record after it ran, without the next episode's queued records that share its last timestamp."""
    df = record.sort("t_ns", maintain_order=True)
    if "outcome_state" not in df.columns:
        return df
    states = df["outcome_state"].to_list()
    started = next((i for i, state in enumerate(states) if state == _RECORD_RUNNING), None)
    if started is None:
        return df
    end = next((i for i in range(started, len(states)) if states[i] is not None and states[i] >= _RECORD_SUCCESS), None)
    return df if end is None else df.head(end + 1)


def _record_human_params(record: pl.DataFrame | pl.LazyFrame | None) -> dict[str, typing.Any] | None:
    """Human parameters of the episode's own records, merged into one name -> value mapping (None when no record carried any)."""
    import json

    df = record.collect() if isinstance(record, pl.LazyFrame) else record
    if df is None or len(df) == 0:
        return None
    blobs = _episode_rows(df).select(pl.col("human_params").fill_null("")).filter(pl.col("human_params") != "").get_column("human_params")
    merged: dict[str, typing.Any] = {}
    for raw in reversed(blobs.to_numpy()):
        try:
            parsed = json.loads(str(raw))
        except ValueError:
            continue
        if isinstance(parsed, dict):
            for name, value in parsed.items():
                merged.setdefault(str(name), value)
    return merged or None


def _record_conditions(record: pl.DataFrame) -> list[dict] | None:
    """The episode's conditions: the last of its own records whose `conditions` JSON is a non-empty list."""
    import json

    if "conditions_yaml" not in record.columns or "t_ns" not in record.columns:
        return None
    for raw in reversed(_episode_rows(record)["conditions_yaml"].to_list()):
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            continue
        if parsed:
            return parsed
    return None


def _record_outcome(record: pl.DataFrame | pl.LazyFrame | None, metadata: RunMetadata | None) -> tuple[int | None, str | None]:
    """(outcome_state, outcome_info): the episode's own terminal record, else the runner's yaml outcome, else the last record."""
    df = record.collect() if isinstance(record, pl.LazyFrame) else record
    rows = df is not None and len(df) > 0 and "outcome_state" in df.columns and "t_ns" in df.columns
    if rows:
        assert df is not None
        df = _episode_rows(df)
        running = df.filter(pl.col("outcome_state") == _RECORD_RUNNING)
        after = df.filter(pl.col("t_ns") > running["t_ns"][0]) if len(running) else df
        terminal = after.filter(pl.col("outcome_state") != _RECORD_RUNNING)
        if len(terminal):
            last = terminal.row(-1, named=True)
            return int(last["outcome_state"]), last.get("outcome_info")
    if metadata is not None and metadata.outcome_state is not None:
        return metadata.outcome_state, metadata.outcome_info
    if rows:
        assert df is not None
        last = df.row(-1, named=True)
        return int(last["outcome_state"]), last.get("outcome_info")
    return None, None


def _record_integrity(
    record: pl.DataFrame | pl.LazyFrame | None,
    flags: pl.DataFrame | pl.LazyFrame | None,
    metadata: RunMetadata | None,
) -> list[str]:
    """Integrity reasons of the last recorded record, from the episode yaml when no record carries any."""
    df = record.collect() if isinstance(record, pl.LazyFrame) else record
    reasons = flags.collect() if isinstance(flags, pl.LazyFrame) else flags
    if df is not None and len(df) > 0 and "t_ns" in df.columns and reasons is not None and len(reasons) > 0:
        last_stamp = df["t_ns"].max()
        return reasons.filter(pl.col("t_ns") == last_stamp).sort("flag_idx")["flag"].to_list()
    if metadata is not None:
        return list(metadata.integrity)
    return []


def _planner_split(ep: EpisodeDescriptor, metadata: RunMetadata | None) -> tuple[str, str]:
    if metadata is not None and metadata.local_planner:
        return metadata.local_planner, metadata.inter_planner or ""
    from arena_evaluation.storage.planner_names import split_planner_name

    return split_planner_name(ep.planner)


def _status_row(
    ep: EpisodeDescriptor,
    metadata: RunMetadata | None,
    robot: str,
    status: str,
    reason: str,
    record: pl.DataFrame | pl.LazyFrame | None = None,
    flags: pl.DataFrame | pl.LazyFrame | None = None,
) -> dict:
    """Identity and runtime verdict of an episode without metrics."""
    local_planner, inter_planner = _planner_split(ep, metadata)
    outcome_state, outcome_info = _record_outcome(record, metadata)
    result, success = _outcome_verdict(outcome_state, outcome_info)
    return {
        "episode": ep.episode_id,
        "planner": ep.planner,
        "local_planner": local_planner,
        "inter_planner": inter_planner,
        "robot": robot,
        "map": ep.map,
        "stage": ep.stage,
        "benchmark_id": ep.benchmark_id,
        "start": [],
        "goal": [],
        "is_reference": ep.is_reference,
        "reference_type": ep.reference_type,
        "status": status,
        "status_reason": reason,
        "pose_source": None,
        "outcome_info": outcome_info,
        "integrity": _record_integrity(record, flags, metadata),
        "result": result,
        "success": success,
    }


def _print_status_summary(rows: list[dict]) -> None:
    """One line per status, then every row the pipeline could not fully evaluate."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("status") or "unknown"] = counts.get(row.get("status") or "unknown", 0) + 1
    print("Episode status: " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())), flush=True)
    flagged = [r for r in rows if r.get("status") != _STATUS_EVALUATED]
    for r in sorted(flagged, key=lambda r: (str(r.get("stage")), r.get("episode", 0))):
        print(f"  episode {r.get('episode')} {r.get('planner')}/{r.get('stage')} [{r.get('robot')}] {r.get('status')}: {r.get('status_reason')}", flush=True)


def _metadata_robot(metadata: RunMetadata | None, bundles: dict[str, TopicBundle] | None = None) -> str:
    """Bundle-style robot key (env_0_jackal) for a row without a robot bundle."""
    model = metadata.robot_model[0].partition("[")[0] if metadata is not None and metadata.robot_model else ""
    env = next((match.group(0) for name in (bundles or {}) if (match := re.match(r"env_\d+", name))), None)
    return f"{env}_{model}" if env and model else model


def _suite_metrics(episode_dir: pathlib.Path, data_root: pathlib.Path) -> dict[str, typing.Any]:
    """Suite-level metric overrides from the run's manifest.yaml, {} for bare run dirs and junk values."""
    import yaml

    for parent in episode_dir.parents:
        manifest_path = parent / "manifest.yaml"
        if manifest_path.exists():
            try:
                suite = yaml.safe_load(manifest_path.read_text()).get("suite") or {}
                raw = suite.get("metrics") or {}
            except Exception as e:
                _log.warning(f"{manifest_path}: unreadable, ignoring suite metrics: {e!r}")
                return {}
            metrics: dict[str, typing.Any] = {}
            v = raw.get("max_collisions")
            if v is not None:
                try:
                    metrics["max_collisions"] = int(v)
                except (TypeError, ValueError):
                    _log.warning(f"{manifest_path}: suite metrics.max_collisions {v!r} is not an int, ignoring")
            barn = raw.get("barn")
            if barn is not None:
                from .metrics.performance.barn_metrics import parse_reference

                try:
                    metrics["barn"] = parse_reference(barn)
                except ValueError as e:
                    _log.warning(f"{manifest_path}: suite metrics.barn ignored: {e}")
            return metrics
        if parent == data_root:
            break
    return {}


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

        if not force_extract and TopicParquetStore.is_current(topics_dir):
            bundles = TopicParquetStore.read(topics_dir)
            if bundles:
                return bundles

        _ctx = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx:
            if topics_dir.exists():
                shutil.rmtree(topics_dir)
            bundles = MCAPReader(mcap_path).read(topics_dir)
            TopicParquetStore.mark_complete(topics_dir)
            return bundles

    def _describe_run_dir(self, run_dir: pathlib.Path) -> EpisodeDescriptor:
        ep = self.folder_manager.describe_episode(pathlib.Path(run_dir))
        if ep is None:
            raise FileNotFoundError(f"{run_dir} is not an episode directory (episode_NNN with episode_NNN.yaml)")
        return ep

    def extract_run_dir(self, run_dir: pathlib.Path, force_extract: bool = False) -> dict[str, TopicBundle] | None:
        """Extract one `episode_NNN` directory into its topics/ folder."""
        return self.extract_episode(self._describe_run_dir(run_dir), force_extract=force_extract)

    def process_run_dir(self, run_dir: pathlib.Path, force_extract: bool = False) -> pathlib.Path | None:
        """Extract and evaluate one `episode_NNN` directory, returning its metrics.parquet path."""
        rows = self.process_episode(self._describe_run_dir(run_dir), force_extract=force_extract)
        if not rows:
            return None
        out = pathlib.Path(run_dir) / "metrics.parquet"
        return out if out.exists() else None

    def process_episode(
        self,
        ep: EpisodeDescriptor,
        force_extract: bool = False,
        status_dict: typing.MutableMapping[int, tuple[str, str, str, int, int, float]] | None = None,
    ) -> list[dict]:
        """Extract and evaluate one episode: one row per robot, a status row where metrics are impossible."""
        from arena_evaluation.processing.scoped_evaluator import ScopedEvaluator
        from arena_evaluation.storage.star_schema import (
            StarSchemaBuilder,
            StarSchemaTables,
            write_episode_star_tables,
        )

        episode_dir = pathlib.Path(ep.episode_dir)

        if not force_extract and (episode_dir / "metrics.parquet").exists() and (episode_dir / "metrics").exists():
            try:
                df, _ = ParquetStore.read(episode_dir / "metrics.parquet")
                if len(df) > 0:
                    return df.to_dicts()
            except Exception:
                pass

        # Load episode metadata
        yaml_path = episode_dir / f"{episode_dir.name}.yaml"
        if not yaml_path.exists():
            yaml_path = episode_dir / "metadata.yaml"

        metadata = None
        if yaml_path.exists():
            try:
                metadata = MetadataWriter.read(yaml_path)
            except Exception as e:
                _log.warning(f"episode_{ep.episode_id:03d}: could not read metadata: {e}")

        def finish(
            rows: list[dict],
            star_tables: list[StarSchemaTables] | None = None,
            series: list[dict[str, pl.DataFrame]] | None = None,
        ) -> list[dict]:
            try:
                ParquetStore.write_rows(rows, episode_dir / "metrics.parquet", schema_overrides=_METRIC_DTYPES)
            except Exception as e:
                _log.warning(f"episode_{ep.episode_id:03d}: metrics.parquet not written, the combined frame is the only copy: {e!r}")
            try:
                if star_tables:
                    merged_star = StarSchemaBuilder.concat_tables(star_tables)
                else:
                    robot_token = str(rows[0].get("robot") or "") if rows else ""
                    merged_star = StarSchemaBuilder().build_status_episode_tables(
                        run_id=ep.benchmark_id,
                        episode_id=ep.episode_id,
                        descriptor=ep,
                        metadata=metadata,
                        robot_id=robot_token,
                    )
                write_episode_star_tables(merged_star, episode_dir / "metrics")
            except Exception as e:
                _log.warning(f"episode_{ep.episode_id:03d}: scoped star tables not written: {e!r}")
            series_dir = episode_dir / "metrics"
            for stem, frame in _merge_series_frames(series or []).items():
                series_dir.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(series_dir / f"series_{stem}.parquet")
            return rows

        bundles = self.extract_episode(ep, force_extract=force_extract)
        if bundles is None or len(bundles) == 0:
            return finish([_status_row(ep, metadata, _metadata_robot(metadata), "no_recording", "no MCAP or empty extraction")])

        _ctx = self.profiler.phase("process") if self.profiler else contextlib.nullcontext()
        with _ctx:
            robot_model = "turtlebot3_burger"
            pedsim_avail = False
            if metadata is not None:
                robot_model = metadata.robot_model[0] if metadata.robot_model else robot_model
                pedsim_avail = metadata.pedsim_available or False

            robot_params = RobotParams.load(robot_model)
            suite_metrics_cfg = _suite_metrics(episode_dir, self.folder_manager.data_root)
            registry = MetricRegistry(robot_params, world=_metric_world(metadata, ep), metrics_config=suite_metrics_cfg)
            scoped_evaluator = ScopedEvaluator(robot_params, metrics_config=suite_metrics_cfg)
            star_builder = StarSchemaBuilder(robot_params=robot_params, metrics_config=suite_metrics_cfg)

            all_results: list[dict] = []
            episode_star_tables: list[StarSchemaTables] = []
            episode_series: list[dict[str, pl.DataFrame]] = []
            robots = {name: bundle for name, bundle in bundles.items() if bundle.robot_tracks is not None or bundle.robot_odom is not None}
            if not robots:
                any_record = next((b.episode_lifecycle for b in bundles.values() if b.episode_lifecycle is not None), None)
                return finish([_status_row(ep, metadata, _metadata_robot(metadata, bundles), "no_trajectory", "no tf ground-truth or odometry rows", any_record)])
            for robot_name, bundle in robots.items():
                window = _episode_window(bundle.episode_lifecycle)
                tracks, pose_source = resolve_pose_source(bundle, window)
                if tracks is None:
                    reason = "no tf ground-truth rows inside the episode window"
                    all_results.append(_status_row(ep, metadata, robot_name, "no_trajectory", reason, bundle.episode_lifecycle, bundle.episode_integrity))
                    continue

                if bundle.pedestrian_tracks is not None:
                    pedsim_avail = True
                robot_pedsim = pedsim_avail

                available_topics = bundle.available()

                aligned_df = NativeBundle().align(bundle, *window)
                if isinstance(aligned_df, pl.LazyFrame):
                    aligned_df = aligned_df.collect()

                if aligned_df is None or len(aligned_df) < 5:
                    reason = f"{0 if aligned_df is None else len(aligned_df)} ground-truth rows inside the episode window"
                    _log.warning(f"episode_{ep.episode_id:03d} [{robot_name}]: {reason}")
                    all_results.append(_status_row(ep, metadata, robot_name, "no_trajectory", reason, bundle.episode_lifecycle, bundle.episode_integrity))
                    continue

                # Resolve the pose frame once so all calculators share it
                aligned_df = _consistent_pose_segment(aligned_df, ride_arrival_times(robot_env_snapshot(bundle.world_semantics, robot_name)))
                if aligned_df is None or len(aligned_df) < 5:
                    reason = f"{0 if aligned_df is None else len(aligned_df)} pose rows after resolution"
                    _log.warning(f"episode_{ep.episode_id:03d} [{robot_name}]: {reason}")
                    all_results.append(_status_row(ep, metadata, robot_name, "no_trajectory", reason, bundle.episode_lifecycle, bundle.episode_integrity))
                    continue

                topics = _collect_native_topics(bundle)
                if len(aligned_df) > 0:
                    t0 = int(aligned_df["time_ns"][0])
                    t1 = int(aligned_df["time_ns"][-1])
                    topics = {name: df.filter((pl.col("t_ns") >= t0 - _ALIGN_TOLERANCE_NS) & (pl.col("t_ns") <= t1 + _ALIGN_TOLERANCE_NS)) if "t_ns" in df.columns else df for name, df in topics.items()}

                peds_df = bundle.pedestrian_tracks.collect() if isinstance(bundle.pedestrian_tracks, pl.LazyFrame) else bundle.pedestrian_tracks
                num_pedestrians = peak_pedestrian_count(peds_df, None if window[0] is None else window[0] - _ALIGN_TOLERANCE_NS, window[1])

                semantic_snapshot_df = bundle.world_semantics
                if isinstance(semantic_snapshot_df, pl.LazyFrame):
                    semantic_snapshot_df = semantic_snapshot_df.collect()

                conditions = None
                human_params = None
                phases = None
                outcome_state = None
                outcome_info = None
                if bundle.episode_lifecycle is not None:
                    er = bundle.episode_lifecycle
                    if isinstance(er, pl.LazyFrame):
                        er = er.collect()
                    if len(er) > 0:
                        conditions = _record_conditions(er)
                        human_params = _record_human_params(er)
                        if "outcome_state" in er.columns and "t_ns" in er.columns:
                            phases = _robot_phases(_episode_rows(er), robot_name)
                        outcome_state, outcome_info = _record_outcome(er, metadata)

                integrity = _record_integrity(bundle.episode_lifecycle, bundle.episode_integrity, metadata)
                start_pos, goal_pos = _episode_endpoints(aligned_df, bundle.robot_plans, phases)

                fleet = _fleet_poses(robots, window)

                aligned_ep = AlignedEpisodeBundle(
                    episode_id=ep.episode_id,
                    data=aligned_df,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    num_pedestrians=num_pedestrians,
                    robot_name=robot_name,
                    semantic_snapshot=semantic_snapshot_df,
                    conditions=conditions,
                    human_params=human_params,
                    phases=phases,
                    fleet=fleet,
                    outcome_state=outcome_state,
                    outcome_info=outcome_info,
                    run=metadata,
                    peds=peds_df,
                    map=ep.map,
                    topics=topics,
                    run_dir=episode_dir.parent.parent,
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
                        pedsim_available=robot_pedsim,
                        available_topics=available_topics,
                        progress_callback=on_calc_progress,
                    )
                    ep_metrics["episode"] = ep.episode_id
                    ep_metrics["planner"] = ep.planner
                    if metadata is not None and metadata.local_planner:
                        ep_metrics["local_planner"] = metadata.local_planner
                        ep_metrics["inter_planner"] = metadata.inter_planner or ""
                    else:
                        from arena_evaluation.storage.planner_names import split_planner_name

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
                    ep_metrics["integrity"] = integrity
                    ep_metrics["pose_source"] = pose_source.kind
                    ep_metrics["pose_samples"] = pose_source.samples
                    ep_metrics["status"] = _STATUS_EVALUATED
                    ep_metrics["status_reason"] = None
                    all_results.append(ep_metrics)

                    scoped_facts = scoped_evaluator.evaluate(
                        aligned_ep,
                        run_id=ep.benchmark_id,
                        episode_id=ep.episode_id,
                        episode_metrics=ep_metrics,
                    )
                    episode_star_tables.append(
                        star_builder.build_episode_tables(
                            aligned_ep,
                            scoped_facts,
                            run_id=ep.benchmark_id,
                            episode_id=ep.episode_id,
                            robot_params=robot_params,
                            descriptor=ep,
                            metadata=metadata,
                            metrics_config=suite_metrics_cfg,
                        )
                    )
                    episode_series.append(
                        _episode_series_tables(
                            registry,
                            aligned_ep,
                            run_id=ep.benchmark_id,
                            robot_id=str(ep_metrics.get("robot") or ""),
                        )
                    )

            return finish(all_results, episode_star_tables, episode_series)

    def extract_benchmark(self, benchmark_id: str, force_extract: bool = False) -> None:
        """Extract all episodes in a benchmark."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)
        if not episodes:
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return

        from .progress_display import PipelineProgressDisplay

        data_root_str = str(self.folder_manager.data_root)
        if self.workers <= 1:
            local_status: dict[int, tuple[str, str, str, int, int, float]] = {}
            with PipelineProgressDisplay(
                f"Phase 1: Extracting MCAP topics for {benchmark_id}",
                len(episodes),
                1,
                status_dict=local_status,
            ) as display:
                for ep in episodes:
                    t_start = time.perf_counter()
                    try:
                        ep_id = _extract_worker(data_root_str, ep, force_extract, local_status)
                        elapsed = time.perf_counter() - t_start
                        display.log_completed(ep_id, f"{ep.planner}/{ep.stage}", elapsed)
                    except Exception as e:
                        display.log_error(ep.episode_id, str(e))
            return

        mp_ctx = multiprocessing.get_context("spawn")
        manager = mp_ctx.Manager()
        status_dict = manager.dict()

        with PipelineProgressDisplay(
            f"Phase 1: Extracting MCAP topics for {benchmark_id}",
            len(episodes),
            self.workers,
            status_dict=status_dict,
        ) as display:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp_ctx,
                initializer=_worker_init,
            )
            try:
                futures = {executor.submit(_extract_worker, data_root_str, ep, force_extract, status_dict): (ep, time.perf_counter()) for ep in episodes}
                for future in concurrent.futures.as_completed(futures):
                    ep, t_start = futures[future]
                    try:
                        ep_id = future.result()
                        elapsed = time.perf_counter() - t_start
                        display.log_completed(ep_id, f"{ep.planner}/{ep.stage}", elapsed)
                    except Exception as e:
                        display.log_error(ep.episode_id, str(e))
            except KeyboardInterrupt:
                _shutdown_executor_cleanly(executor)
                raise
            finally:
                _shutdown_executor_cleanly(executor)

    def process_benchmark(self, benchmark_id: str, force_extract: bool = False) -> pathlib.Path | None:
        """Process all episodes and write combined_metrics.parquet using a 2-Phase pipeline, returning its path, None when nothing was written."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)

        if not episodes:
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return None

        _ctx_extract = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx_extract:
            self.extract_benchmark(benchmark_id, force_extract=force_extract)

        all_metrics: list[dict] = []
        data_root_str = str(self.folder_manager.data_root)

        from .progress_display import PipelineProgressDisplay

        _ctx_process = self.profiler.phase("process") if self.profiler else contextlib.nullcontext()
        with _ctx_process:
            if self.workers <= 1:
                local_status: dict[int, tuple[str, str, str, int, int, float]] = {}
                with PipelineProgressDisplay(
                    f"Phase 2: Calculating metrics across {len(episodes)} episodes",
                    len(episodes),
                    1,
                    status_dict=local_status,
                ) as display:
                    for ep in episodes:
                        t_start = time.perf_counter()
                        try:
                            ep_id, result = _process_worker(data_root_str, ep, False, local_status)
                            elapsed = time.perf_counter() - t_start
                            all_metrics.extend(result)
                            display.log_completed(ep_id, f"{ep.planner}/{ep.stage}", elapsed)
                        except Exception as e:
                            display.log_error(ep.episode_id, str(e))
                            all_metrics.append(_status_row(ep, None, "", "error", repr(e)))
            else:
                mp_ctx = multiprocessing.get_context("spawn")
                manager = mp_ctx.Manager()
                status_dict = manager.dict()
                with PipelineProgressDisplay(
                    f"Phase 2: Calculating metrics across {len(episodes)} episodes",
                    len(episodes),
                    self.workers,
                    status_dict=status_dict,
                ) as display:
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=self.workers,
                        mp_context=mp_ctx,
                        initializer=_worker_init,
                    )
                    try:
                        futures = {executor.submit(_process_worker, data_root_str, ep, False, status_dict): (ep, time.perf_counter()) for ep in episodes}

                        for future in concurrent.futures.as_completed(futures):
                            ep, t_start = futures[future]
                            try:
                                ep_id, result = future.result()
                                elapsed = time.perf_counter() - t_start
                                all_metrics.extend(result)
                                display.log_completed(ep_id, f"{ep.planner}/{ep.stage}", elapsed)
                            except Exception as e:
                                display.log_error(ep.episode_id, str(e))
                                all_metrics.append(_status_row(ep, None, "", "error", repr(e)))
                    except KeyboardInterrupt:
                        print("\n[KeyboardInterrupt] Metrics calculation interrupted by user. Terminating workers...", flush=True)
                        _shutdown_executor_cleanly(executor)
                        raise
                    finally:
                        _shutdown_executor_cleanly(executor)

        _print_status_summary(all_metrics)
        if not all_metrics:
            print("No valid results were generated.")
            return None

        ref_episodes_by_stage: dict[str | None, list[dict]] = {}
        for row in all_metrics:
            if row.get("is_reference") and row.get("reference_type") == "unhindered_peds":
                stage = row.get("stage")
                if stage not in ref_episodes_by_stage:
                    ref_episodes_by_stage[stage] = []
                ref_episodes_by_stage[stage].append(row)

        ref_ped_by_stage: dict[str | None, list[dict]] = {}
        ref_robot_by_stage: dict[str | None, list[dict]] = {}
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

            ref_robot_row = next((r for r in ref_robots if r.get("status") == _STATUS_EVALUATED), None)
            ref_ped_row = next((r for r in ref_peds if r.get("status") == _STATUS_EVALUATED), None)

            reconciled = MutualAccommodationCalculator.reconcile_stage_references(
                dynamic_row=row,
                ref_robot_row=ref_robot_row,
                ref_ped_row=ref_ped_row,
            )
            row.update(reconciled)

        combined_path = self.folder_manager.combined_metrics_path(benchmark_id)
        ParquetStore.write_rows(all_metrics, combined_path, schema_overrides=_METRIC_DTYPES)

        from arena_evaluation.storage.star_schema import (
            StarSchemaBuilder,
            StarSchemaTables,
            read_episode_star_tables,
            write_star_schema,
        )

        collected_star: list[StarSchemaTables] = []
        fallback_builder = StarSchemaBuilder()
        for ep in episodes:
            ep_metrics_dir = pathlib.Path(ep.episode_dir) / "metrics"
            ep_star = read_episode_star_tables(ep_metrics_dir)
            if ep_star is None:
                ep_star = fallback_builder.build_status_episode_tables(
                    run_id=ep.benchmark_id or benchmark_id,
                    episode_id=ep.episode_id,
                    descriptor=ep,
                )
            collected_star.append(ep_star)

        benchmark_star = StarSchemaBuilder.concat_tables(collected_star)
        write_star_schema(benchmark_star, combined_path.parent)

        series_parts: dict[str, list[pl.DataFrame]] = {}
        for ep in episodes:
            for path in sorted((pathlib.Path(ep.episode_dir) / "metrics").glob("series_*.parquet")):
                series_parts.setdefault(path.stem.removeprefix("series_"), []).append(pl.read_parquet(path))
        if series_parts:
            series_dir = combined_path.parent / "metrics"
            series_dir.mkdir(parents=True, exist_ok=True)
            for stem, frames in series_parts.items():
                pl.concat(frames, how="vertical_relaxed").write_parquet(series_dir / f"series_{stem}.parquet")

        print(f"Done. {len(all_metrics)} episode rows -> {combined_path}")
        return combined_path
