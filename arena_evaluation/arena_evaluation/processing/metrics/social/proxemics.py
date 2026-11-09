"""Personal-space occupancy of the robot, and per-pedestrian zone-intrusion counts.

The robot sample axis carries the count of pedestrians inside the personal-space radius, from
the vectorized robot-time to pedestrian-frame alignment. The count series is a 1NF table
(:meth:`ProxemicsCalculator.series_table`), never a list column inside the metrics row.
"""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.social.interaction_common import native_robot_frame
from arena_evaluation.processing.pedestrian_frames import distances_to_pedestrians
from arena_evaluation.processing.windows import DEFAULT_DETECTOR_PARAMS, pedestrian_pairs_lf
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

INTIMATE_CLEARANCE_M: Final[float] = 0.45
PERSONAL_CLEARANCE_M: Final[float] = 1.20

PERSONAL_SPACE_SERIES_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "time_s": pl.Float64,
        "peds_in_personal_space": pl.Int64,
    }
)


def personal_space_counts(
    robot_frame: pl.DataFrame, peds: pl.DataFrame, radius_m: float
) -> pl.DataFrame:
    """One row per robot ground-truth sample with the number of pedestrians inside ``radius_m`` of it."""
    time_ns = robot_frame.get_column("time_ns").to_numpy().astype(np.int64)
    pos_x = robot_frame.get_column("pos_x_gt").to_numpy().astype(np.float64)
    pos_y = robot_frame.get_column("pos_y_gt").to_numpy().astype(np.float64)
    within = (
        distances_to_pedestrians(time_ns, pos_x, pos_y, peds)
        .filter(pl.col("distance") < radius_m)
        .group_by("row")
        .len()
    )
    counts = np.zeros(len(time_ns), dtype=np.int64)
    counts[within["row"].to_numpy()] = within["len"].to_numpy()
    return pl.DataFrame(
        {"t_ns": time_ns, "count": counts},
        schema={"t_ns": pl.Int64, "count": pl.Int64},
    )


class ProxemicsCalculator(BaseMetricCalculator):
    """Calculates proxemics and personal space intrusion metrics."""

    NAME = "proxemics"
    CATEGORY = "social"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_tracks"]

    UNITS = {
        "num_pedestrians": "",
        "total_time_in_personal_space": "s",
        "avg_velocity_in_personal_space": "m/s",
    }

    PRIMARY_OUTPUTS = ["total_time_in_personal_space"]

    OUTPUT_DIRECTIONS = {"total_time_in_personal_space": "lower"}

    PERSONAL_SPACE_RADIUS = 1.2

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "num_pedestrians",
            "total_time_in_personal_space",
            "avg_velocity_in_personal_space",
        ]

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Dwell of the robot inside its personal space, and the mean velocity while inside."""
        robot_frame = self.resolve_robot_frame(episode)
        peds_df = self.native_ped_frame(episode)
        if robot_frame is None or robot_frame.height == 0 or peds_df is None or peds_df.height == 0:
            return dict.fromkeys(self.output_keys())
        counts = personal_space_counts(robot_frame, peds_df, self.PERSONAL_SPACE_RADIUS)
        time_ns = counts.get_column("t_ns").to_numpy()
        occupancy = counts.get_column("count").to_numpy()
        dt = np.append(np.diff(time_ns) / 1e9, 0.0)
        inside = np.flatnonzero(occupancy > 0)
        total_time_s = float(dt[inside].sum()) if len(inside) > 0 else 0.0
        velocity = prior_results.get("velocity", [])
        velocities = [velocity[i] for i in inside if velocity and i < len(velocity)]
        avg_velocity = float(np.mean(velocities)) if velocities else 0.0
        return {
            "num_pedestrians": episode.num_pedestrians,
            "total_time_in_personal_space": total_time_s,
            "avg_velocity_in_personal_space": avg_velocity,
        }

    def _window_metrics(self, window: AlignedEpisodeBundle) -> dict[str, typing.Any]:
        """Personal-space dwell over the window's own robot samples and pedestrian set."""
        robot = native_robot_frame(window)
        peds = self.native_ped_frame(window)
        if robot is None or robot.height == 0 or peds is None or peds.height == 0:
            return dict.fromkeys(self.output_keys())
        time_ns = robot.get_column("t_ns").to_numpy().astype(np.int64)
        counts = personal_space_counts(
            robot.select(
                time_ns=pl.col("t_ns"),
                pos_x_gt=pl.col("x"),
                pos_y_gt=pl.col("y"),
            ),
            peds,
            self.PERSONAL_SPACE_RADIUS,
        )
        occupancy = counts.get_column("count").to_numpy()
        step_s = np.append(np.diff(time_ns) / 1e9, 0.0)
        inside = occupancy > 0
        speed_mps = np.hypot(
            robot.get_column("vx").to_numpy(), robot.get_column("vy").to_numpy()
        )
        return {
            "num_pedestrians": int(peds.get_column("ped_id").n_unique()),
            "total_time_in_personal_space": float(step_s[inside].sum()),
            "avg_velocity_in_personal_space": float(speed_mps[inside].mean())
            if bool(inside.any())
            else 0.0,
        }

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE keeps the aligned population; an ENCOUNTER slice measures its own window telemetry."""
        scope = MetricScope.EPISODE if scope_slice is None else scope_slice.scope
        match scope:
            case MetricScope.EPISODE:
                return self.calculate(episode, prior_results)
            case MetricScope.ENCOUNTER:
                return self._window_metrics(scope_slice.window_bundle)
            case _:
                return dict.fromkeys(self.output_keys())

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF per-sample personal-space occupancy, one row per robot sample."""
        robot_frame = self.resolve_robot_frame(episode)
        peds_df = self.native_ped_frame(episode)
        if (
            robot_frame is None
            or robot_frame.height == 0
            or peds_df is None
            or peds_df.height == 0
        ):
            return pl.DataFrame(schema=PERSONAL_SPACE_SERIES_SCHEMA)
        counts = personal_space_counts(robot_frame, peds_df, self.PERSONAL_SPACE_RADIUS)
        first_t_ns = int(counts.select(pl.col("t_ns").min()).item())
        resolved_robot = robot_id or (episode.robot_name or "robot")
        return counts.select(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
            robot_id=pl.lit(resolved_robot, dtype=pl.String),
            t_ns=pl.col("t_ns"),
            time_s=(pl.col("t_ns") - first_t_ns).cast(pl.Float64) / 1e9,
            peds_in_personal_space=pl.col("count"),
        )


def pedestrian_intrusion_counts_frame(episode: AlignedEpisodeBundle, robot_radius_m: float) -> pl.DataFrame:
    """Zone-entry counts per pedestrian from the pair stream (edge clearance thresholds)."""
    topics = episode.topics if episode.topics is not None else {}
    pedestrian_tracks = topics.get("pedestrian_tracks")
    robot_tracks = topics.get("robot_tracks")
    if pedestrian_tracks is not None and isinstance(pedestrian_tracks, pl.LazyFrame):
        pedestrian_tracks = pedestrian_tracks.collect()
    if robot_tracks is not None and isinstance(robot_tracks, pl.LazyFrame):
        robot_tracks = robot_tracks.collect()
    if pedestrian_tracks is None or len(pedestrian_tracks) == 0 or robot_tracks is None or len(robot_tracks) == 0:
        return pl.DataFrame(schema={"ped_id": pl.String, "total_intrusions_intimate": pl.Int64, "total_intrusions_personal": pl.Int64})
    pairs = pedestrian_pairs_lf(robot_tracks, pedestrian_tracks, robot_radius_m, DEFAULT_DETECTOR_PARAMS).collect().sort(["actor_id", "t_ns"])
    entries = pairs.with_columns(
        intimate=pl.col("edge_clearance_m") < INTIMATE_CLEARANCE_M,
        personal=pl.col("edge_clearance_m") < PERSONAL_CLEARANCE_M,
    ).with_columns(
        intimate_entry=pl.col("intimate") & ~pl.col("intimate").shift(1, fill_value=False).over("actor_id"),
        personal_entry=pl.col("personal") & ~pl.col("personal").shift(1, fill_value=False).over("actor_id"),
    )
    return entries.group_by(pl.col("actor_id").alias("ped_id")).agg(
        total_intrusions_intimate=pl.col("intimate_entry").sum().cast(pl.Int64),
        total_intrusions_personal=pl.col("personal_entry").sum().cast(pl.Int64),
    )


class PedestrianIntrusionCalculator(BaseMetricCalculator):
    """Pedestrian-level zone intrusion counts (intimate: <0.45m, personal: <1.20m)."""

    NAME = "pedestrian_intrusions"
    CATEGORY = "social"
    SCOPE = MetricScope.PEDESTRIAN
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.PEDESTRIAN})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_tracks"]

    UNITS = {
        "total_intrusions_intimate": "",
        "total_intrusions_personal": "",
    }
    PRIMARY_OUTPUTS = ["total_intrusions_intimate", "total_intrusions_personal"]
    OUTPUT_DIRECTIONS = {
        "total_intrusions_intimate": "lower",
        "total_intrusions_personal": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["total_intrusions_intimate", "total_intrusions_personal"]

    def _calculate_for_target(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        target_id: str | None = None,
    ) -> dict[str, typing.Any]:
        prior_intimate = prior_results.get("total_intrusions_intimate")
        prior_personal = prior_results.get("total_intrusions_personal")
        if prior_intimate is not None and prior_personal is not None:
            return {
                "total_intrusions_intimate": int(prior_intimate),
                "total_intrusions_personal": int(prior_personal),
            }

        topics = self.native_topics(episode)
        robot_tracks = topics.get("robot_tracks")
        ped_tracks = topics.get("pedestrian_tracks")

        if isinstance(robot_tracks, pl.LazyFrame):
            robot_tracks = robot_tracks.collect()
        if isinstance(ped_tracks, pl.LazyFrame):
            ped_tracks = ped_tracks.collect()

        if robot_tracks is None or len(robot_tracks) == 0 or ped_tracks is None or len(ped_tracks) == 0:
            return {"total_intrusions_intimate": 0, "total_intrusions_personal": 0}

        resolved_target = target_id if target_id is not None else self.metrics_config.get("target_agent_id")
        if resolved_target is not None:
            ped_tracks = ped_tracks.filter(pl.col("ped_id") == str(resolved_target))

        if len(ped_tracks) == 0:
            return {"total_intrusions_intimate": 0, "total_intrusions_personal": 0}

        robot_radius_m = float(self.robot_params.robot_radius)
        pairs = pedestrian_pairs_lf(robot_tracks, ped_tracks, robot_radius_m, DEFAULT_DETECTOR_PARAMS).collect().sort(["actor_id", "t_ns"])
        if len(pairs) == 0:
            return {"total_intrusions_intimate": 0, "total_intrusions_personal": 0}

        entries = pairs.with_columns(
            intimate=pl.col("edge_clearance_m") < INTIMATE_CLEARANCE_M,
            personal=pl.col("edge_clearance_m") < PERSONAL_CLEARANCE_M,
        ).with_columns(
            intimate_entry=pl.col("intimate") & ~pl.col("intimate").shift(1, fill_value=False).over("actor_id"),
            personal_entry=pl.col("personal") & ~pl.col("personal").shift(1, fill_value=False).over("actor_id"),
        )
        return {
            "total_intrusions_intimate": int(entries.get_column("intimate_entry").sum()),
            "total_intrusions_personal": int(entries.get_column("personal_entry").sum()),
        }

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        return self._calculate_for_target(episode, prior_results)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        target_id = scope_slice.slice_id if scope_slice is not None and scope_slice.scope == MetricScope.PEDESTRIAN else None
        return self._calculate_for_target(episode, prior_results, target_id=target_id)
