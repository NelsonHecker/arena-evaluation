from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
MIN_DT_S: typing.Final[float] = 1e-6


class MotionMetricsCalculator(BaseMetricCalculator):
    """Robot velocity, acceleration and jerk over the scope window's ground truth."""

    NAME = "motion_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks"]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})

    UNITS = {
        "velocity": "m/s",
        "velocity_mean": "m/s",
        "velocity_max": "m/s",
        "acceleration": "m/s²",
        "acceleration_mean": "m/s²",
        "acceleration_max": "m/s²",
        "jerk": "m/s³",
        "jerk_mean": "m/s³",
        "jerk_max": "m/s³",
    }

    PRIMARY_OUTPUTS = ["velocity_mean", "velocity_max", "jerk_mean", "jerk_max"]

    OUTPUT_DIRECTIONS = {"jerk_mean": "lower", "jerk_max": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "velocity",
            "velocity_mean",
            "velocity_max",
            "acceleration",
            "acceleration_mean",
            "acceleration_max",
            "jerk",
            "jerk_mean",
            "jerk_max",
        ]

    def _episode_kinematics(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray]:
        """Ground-truth speed (m/s) and sample times (ns) of the episode, the physical samples only."""
        pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        frame = episode.data
        if frame is not None and "vel_linear" in frame.columns:
            valid = pl.col("vel_linear").is_not_null() & ~pl.col("vel_linear").is_nan()
            max_linear_velocity = self.robot_params.max_linear_velocity
            if max_linear_velocity > 0.0:
                valid = valid & (pl.col("vel_linear").abs() <= max_linear_velocity)
            frame = frame.filter(valid)
            return np.abs(frame["vel_linear"].to_numpy().astype(np.float64)), frame["time_ns"].to_numpy().astype(np.int64)
        if len(pos_x) > 1:
            time_ns = frame["time_ns"].to_numpy().astype(np.int64) if frame is not None and "time_ns" in frame.columns else np.arange(len(pos_x), dtype=np.int64) * 100_000_000
            return self.speed_from_pose(pos_x, pos_y, time_ns), time_ns
        return np.array([]), np.array([], dtype=np.int64)

    def _native_kinematics(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray]:
        """Ground-truth speed (m/s) and sample times (ns) of the scope window from the native /tf track."""
        _pos_x, _pos_y, _yaw, time_ns = self.resolve_native_pose(episode)
        if len(time_ns) == 0:
            return np.array([]), np.array([], dtype=np.int64)
        tracks = self.native_topics(episode).get("robot_tracks")
        assert tracks is not None
        tracks = tracks.sort("t_ns")
        if "speed_mps" in tracks.columns:
            speed = tracks["speed_mps"].cast(pl.Float64).abs().to_numpy().astype(np.float64)
        else:
            speed = self.speed_from_pose(
                tracks["x"].cast(pl.Float64).to_numpy().astype(np.float64),
                tracks["y"].cast(pl.Float64).to_numpy().astype(np.float64),
                tracks["t_ns"].to_numpy().astype(np.int64),
            )
        return speed, tracks["t_ns"].to_numpy().astype(np.int64)

    @staticmethod
    def _summarize(speed_mps: np.ndarray, time_ns: np.ndarray) -> dict[str, typing.Any]:
        """Velocity, acceleration and jerk aggregates of one speed track; explicit nulls without enough samples."""
        if len(speed_mps) < 2:
            return {
                "velocity": speed_mps.tolist(),
                "velocity_mean": float(np.mean(speed_mps)) if len(speed_mps) else None,
                "velocity_max": float(np.max(speed_mps)) if len(speed_mps) else None,
                "acceleration": [],
                "acceleration_mean": None,
                "acceleration_max": None,
                "jerk": [],
                "jerk_mean": None,
                "jerk_max": None,
            }

        dt = np.diff(time_ns) / NANOSECONDS_PER_SECOND
        dt = np.where(dt <= 0.0, MIN_DT_S, dt)
        acceleration = np.diff(speed_mps) / dt
        result: dict[str, typing.Any] = {
            "velocity": speed_mps.tolist(),
            "velocity_mean": float(np.mean(speed_mps)),
            "velocity_max": float(np.max(speed_mps)),
            "acceleration": acceleration.tolist(),
            "acceleration_mean": float(np.mean(np.abs(acceleration))),
            "acceleration_max": float(np.max(np.abs(acceleration))),
            "jerk": [],
            "jerk_mean": None,
            "jerk_max": None,
        }
        if len(speed_mps) < 3:
            return result

        jerk = np.diff(acceleration) / dt[:-1]
        result["jerk"] = jerk.tolist()
        result["jerk_mean"] = float(np.mean(np.abs(jerk)))
        result["jerk_max"] = float(np.max(np.abs(jerk)))
        return result

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        speed_mps, time_ns = self._episode_kinematics(episode)
        return self._summarize(speed_mps, time_ns)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            speed_mps, time_ns = self._native_kinematics(episode)
            return self._summarize(speed_mps, time_ns)
        raise ValueError(f"motion_metrics does not serve the {scope_slice.scope.value} scope")