from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.pedestrian_frames import distances_to_pedestrians
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


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
