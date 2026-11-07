from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.pedestrian_frames import distances_to_pedestrians
from arena_evaluation.processing.windows import DEFAULT_DETECTOR_PARAMS, pedestrian_pairs_lf
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

INTIMATE_CLEARANCE_M: typing.Final[float] = 0.45
PERSONAL_CLEARANCE_M: typing.Final[float] = 1.20


class ProxemicsCalculator(BaseMetricCalculator):
    """Calculates proxemics and personal space intrusion metrics."""

    NAME = "proxemics"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = ["robot_odom", "pedestrian_tracks"]

    UNITS = {
        "num_pedestrians": "",
        "time_in_personal_space": "",
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
            "time_in_personal_space",
            "total_time_in_personal_space",
            "avg_velocity_in_personal_space",
        ]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:

        peds_df = self.native_ped_frame(episode)
        if episode.data is None or "pos_x" not in episode.data.columns or peds_df is None:
            return {k: None for k in self.output_keys()}

        pos_x, pos_y, _, _, _, _ = self.resolve_robot_pose(episode)
        time_ns = episode.data["time_ns"].to_numpy()

        velocity = prior_results.get("velocity", [])

        N = len(pos_x)
        if N == 0:
            return {
                "num_pedestrians": 0,
                "time_in_personal_space": [],
                "total_time_in_personal_space": 0.0,
                "avg_velocity_in_personal_space": 0.0,
            }

        within = distances_to_pedestrians(time_ns, pos_x, pos_y, peds_df).filter(pl.col("distance") < self.PERSONAL_SPACE_RADIUS).group_by("row").len()
        counts = np.zeros(N, dtype=int)
        counts[within["row"].to_numpy()] = within["len"].to_numpy()
        in_personal_space_steps = counts.tolist()

        dt = np.append(np.diff(time_ns) / 1e9, 0.0)
        inside = np.flatnonzero(counts > 0)
        in_personal_space_time_s = float(dt[inside].sum())
        vel_in_space = [velocity[i] for i in inside if velocity and i < len(velocity)]

        if vel_in_space:
            avg_vel = float(np.nanmean(vel_in_space))
            if np.isnan(avg_vel) or np.isinf(avg_vel):
                avg_vel = 0.0
        else:
            avg_vel = 0.0
        return {
            "num_pedestrians": episode.num_pedestrians,
            "time_in_personal_space": in_personal_space_steps,
            "total_time_in_personal_space": in_personal_space_time_s,
            "avg_velocity_in_personal_space": avg_vel,
        }


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

        if robot_tracks is not None and isinstance(robot_tracks, pl.LazyFrame):
            robot_tracks = robot_tracks.collect()
        if ped_tracks is not None and isinstance(ped_tracks, pl.LazyFrame):
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
