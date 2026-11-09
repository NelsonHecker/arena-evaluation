from __future__ import annotations

import json

import polars as pl
import yaml

from arena_evaluation.processing.mcap_reader import anchor_at
from arena_evaluation.processing.native_bundle import NativeBundle
from arena_evaluation.processing.pedestrian_frames import ALIGN_TOLERANCE_NS, peak_pedestrian_count
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, TopicBundle


def _parse_conditions(raw: str | None) -> list[dict] | None:
    """Parse the recorded episode `conditions_yaml` JSON string into a list of dicts."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def _env_offset(tf_static: pl.DataFrame | None, at_ns: int | None = None) -> tuple[float, float] | None:
    """Env packing translation from the `map -> env_<n>/map` transform in effect at `at_ns` (the latest when None), None if multi-env ambiguous."""
    if tf_static is None or len(tf_static) == 0:
        return (0.0, 0.0)
    rows = tf_static.filter((pl.col("parent_frame") == "map") & pl.col("child_frame").str.contains(r"^env_\d+/map$"))
    if len(rows) == 0:
        return (0.0, 0.0)
    if rows["child_frame"].n_unique() > 1:
        return None
    times = rows["t_ns"].to_list()
    anchors = list(zip(times, rows["tx"].cast(pl.Float64).to_list(), rows["ty"].cast(pl.Float64).to_list(), strict=True))
    return anchor_at(anchors, at_ns) if at_ns is not None else anchors[-1][1:]


def _episode_snapshot(snapshot: pl.DataFrame | None, start_time: int, end_time: int) -> pl.DataFrame | None:
    """Episode-scoped snapshot rows: in-window rows plus the seed, all rows of the latest
    snapshot stamp at-or-before `start_time`, or the first stamp inside the window if none."""
    if snapshot is None or len(snapshot) == 0:
        return None

    window = snapshot.filter((pl.col("t_ns") >= start_time) & (pl.col("t_ns") <= end_time))

    seed_candidates = snapshot.filter(pl.col("t_ns") <= start_time)
    if len(seed_candidates) > 0:
        seed_time = seed_candidates["t_ns"].max()
    elif len(window) > 0:
        seed_time = window["t_ns"].min()
    else:
        return None

    seed = snapshot.filter(pl.col("t_ns") == seed_time)
    return pl.concat([seed, window]).unique().sort("t_ns")


def _params_start_goal(raw: str | None) -> tuple[list, list]:
    """Start and goal of the first robot keyed in the episode params YAML, empty when it carries none."""
    if not raw:
        return [], []
    try:
        params = yaml.safe_load(raw) if raw else {}
    except yaml.YAMLError:
        return [], []
    if not isinstance(params, dict):
        return [], []
    robot = next((name.rpartition(".")[0] for name in params if name.rpartition(".")[2] in ("start", "goal")), None)
    if robot is None:
        return [], []
    return params.get(f"{robot}.start", []), params.get(f"{robot}.goal", [])


def _plan_yaw_at(plan: pl.DataFrame, start_time: int) -> float | None:
    """Heading of the first waypoint of the first plan published at or after start_time."""
    later = plan.filter(pl.col("t_ns") >= start_time)
    if len(later) == 0:
        return None
    first_stamp = later["t_ns"].min()
    return float(later.filter(pl.col("t_ns") == first_stamp).sort("waypoint_idx")["yaw"][0])


def _plan_goal_within(plan: pl.DataFrame, start_time: int, end_time: int) -> list[float]:
    """(x, y, yaw) of the last waypoint of the first plan published inside the episode window, empty when none."""
    inside = plan.filter((pl.col("t_ns") >= start_time) & (pl.col("t_ns") <= end_time))
    if len(inside) == 0:
        return []
    first_stamp = inside["t_ns"].min()
    last = inside.filter(pl.col("t_ns") == first_stamp).sort("waypoint_idx").row(-1, named=True)
    return [last["x"], last["y"], last["yaw"]]


def _ground_truth_pose(aligned_df: pl.DataFrame, index: int) -> list[float]:
    """Ground-truth (x, y, yaw) of an aligned sample, empty without one: odometry never stands in for the trajectory."""
    row = aligned_df.row(index, named=True)
    if row.get("pos_x_gt") is None:
        return []
    return [row["pos_x_gt"], row["pos_y_gt"], row.get("yaw_gt", 0.0)]


class EpisodeSplitter:
    """
    Splits continuous topic data into discrete episodes using EpisodeRecord messages.
    """

    def __init__(self, aligner: NativeBundle, min_episode_frames: int = 5):
        self.aligner = aligner
        self.min_episode_frames = min_episode_frames

    def split(self, bundle: TopicBundle, robot_name: str | None = None) -> list[AlignedEpisodeBundle]:
        """
        Split the raw bundle into aligned episodes.
        """
        if bundle.robot_odom is None:
            return []

        def _to_df(lf_or_df: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
            if isinstance(lf_or_df, pl.LazyFrame):
                return lf_or_df.collect()
            return lf_or_df

        odometry = _to_df(bundle.robot_odom)
        if odometry is None or len(odometry) == 0:
            return []
        last_odom_val = odometry["t_ns"].max()
        if not isinstance(last_odom_val, (int, float)):
            return []
        last_odom_ns = int(last_odom_val)

        record_df = _to_df(bundle.episode_lifecycle)
        initialpose_df = _to_df(bundle.robot_initialpose)
        plan_df = _to_df(bundle.robot_plans)
        semantic_snapshot_df = _to_df(bundle.world_semantics)
        tf_static_df = _to_df(bundle.tf_static)
        peds_df = _to_df(bundle.pedestrian_tracks)

        episodes = []

        if record_df is None or len(record_df) == 0:
            aligned_df = _to_df(self.aligner.align(bundle))
            if aligned_df is not None and len(aligned_df) >= self.min_episode_frames:
                episodes.append(
                    AlignedEpisodeBundle(
                        episode_id=0,
                        data=aligned_df,
                        start_pos=_ground_truth_pose(aligned_df, 0),
                        goal_pos=_ground_truth_pose(aligned_df, -1),
                        num_pedestrians=peak_pedestrian_count(peds_df, *_window_bounds(aligned_df)),
                        robot_name=robot_name,
                        semantic_snapshot=semantic_snapshot_df,
                        env_offset=_env_offset(tf_static_df),
                    )
                )
            return episodes

        rows = list(record_df.iter_rows(named=True))

        i = 0
        while i < len(rows):
            row = rows[i]
            start_time = row["t_ns"]

            if i + 1 < len(rows) and rows[i + 1]["episode_id"] == row["episode_id"]:
                end_time = rows[i + 1]["t_ns"]
                i += 2
            else:
                end_time = rows[i + 1]["t_ns"] - 1 if i + 1 < len(rows) else last_odom_ns
                i += 1

            aligned_df = _to_df(self.aligner.align(bundle, start_time, end_time))

            if aligned_df is None or len(aligned_df) < self.min_episode_frames:
                continue

            start_pos, goal_pos = _params_start_goal(row["params_yaml"])

            if not start_pos and initialpose_df is not None and len(initialpose_df) > 0:
                df_init = initialpose_df.filter(pl.col("t_ns") >= start_time)
                if len(df_init) > 0:
                    row_init = df_init.row(0, named=True)
                else:
                    row_init = initialpose_df.row(-1, named=True)
                start_pos = [row_init["pos_x"], row_init["pos_y"], row_init["yaw"]]

            if start_pos and len(start_pos) == 3 and plan_df is not None and len(plan_df) > 0:
                plan_yaw = _plan_yaw_at(plan_df, start_time)
                if plan_yaw is not None and plan_yaw != 0.0 and abs(start_pos[2] - plan_yaw) > 1.0:
                    start_pos[2] = plan_yaw

            if not start_pos and aligned_df is not None and len(aligned_df) > 0:
                start_pos = _ground_truth_pose(aligned_df, 0)

            if not goal_pos and plan_df is not None and len(plan_df) > 0:
                goal_pos = _plan_goal_within(plan_df, start_time, end_time)

            if not goal_pos and aligned_df is not None and len(aligned_df) > 0:
                goal_pos = _ground_truth_pose(aligned_df, -1)

            episodes.append(
                AlignedEpisodeBundle(
                    episode_id=row["episode_id"],
                    data=aligned_df,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    num_pedestrians=peak_pedestrian_count(peds_df, start_time - ALIGN_TOLERANCE_NS, end_time),
                    robot_name=robot_name,
                    semantic_snapshot=_episode_snapshot(semantic_snapshot_df, start_time, end_time),
                    conditions=_parse_conditions(row.get("conditions_yaml")),
                    env_offset=_env_offset(tf_static_df, start_time),
                )
            )

        return episodes


def _window_bounds(aligned_df: pl.DataFrame) -> tuple[int, int]:
    """Start and end stamps of an aligned frame on the odometry clock."""
    return int(aligned_df["time_ns"].min()), int(aligned_df["time_ns"].max())  # pyright: ignore[reportArgumentType]
