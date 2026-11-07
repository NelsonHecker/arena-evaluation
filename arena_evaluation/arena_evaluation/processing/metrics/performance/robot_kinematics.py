from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class RobotKinematicsCalculator(BaseMetricCalculator):
    """Robot-level cumulative kinematics: path length, mean velocity, max velocity."""

    NAME = "robot_kinematics"
    CATEGORY = "performance"
    SCOPE = MetricScope.ROBOT
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = []

    UNITS = {
        "path_length_m": "m",
        "mean_speed_mps": "m/s",
        "max_speed_mps": "m/s",
    }
    PRIMARY_OUTPUTS = ["path_length_m", "mean_speed_mps"]
    OUTPUT_DIRECTIONS = {
        "path_length_m": "lower",
        "mean_speed_mps": "higher",
        "max_speed_mps": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["path_length_m", "mean_speed_mps", "max_speed_mps"]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        path_length_m = prior_results.get("path_length")
        mean_speed_mps = prior_results.get("velocity_mean")
        max_speed_mps = prior_results.get("velocity_max")

        # Fallback to direct calculation from native robot_tracks topic
        tracks = self.native_topics(episode).get("robot_tracks")
        if tracks is not None and isinstance(tracks, pl.LazyFrame):
            tracks = tracks.collect()

        if (path_length_m is None or mean_speed_mps is None or max_speed_mps is None) and tracks is not None and len(tracks) > 1:
            sorted_tracks = tracks.sort("t_ns")
            dx = sorted_tracks.get_column("x").diff().drop_nulls().to_numpy()
            dy = sorted_tracks.get_column("y").diff().drop_nulls().to_numpy()
            dists = np.hypot(dx, dy)
            if path_length_m is None:
                path_length_m = float(np.sum(dists))

            vx = sorted_tracks.get_column("vx").to_numpy().astype(np.float64)
            vy = sorted_tracks.get_column("vy").to_numpy().astype(np.float64)
            speeds = np.hypot(vx, vy)
            if mean_speed_mps is None and len(speeds) > 0:
                mean_speed_mps = float(np.mean(speeds))
            if max_speed_mps is None and len(speeds) > 0:
                max_speed_mps = float(np.max(speeds))

        return {
            "path_length_m": float(path_length_m) if path_length_m is not None else None,
            "mean_speed_mps": float(mean_speed_mps) if mean_speed_mps is not None else None,
            "max_speed_mps": float(max_speed_mps) if max_speed_mps is not None else None,
        }
