"""Robot speed, acceleration and jerk over the scope window's ground truth.

``calculate`` keeps only the scalar aggregates in the metric row. The per-sample
kinematics are published as one flat 1NF table by
:meth:`MotionMetricsCalculator.series_table`, never as list columns in Parquet.
"""

from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
MIN_DT_S: typing.Final[float] = 1e-6

SERIES_TABLE_STEM = "motion"

# One row per (episode, robot, sample) of the ground-truth kinematics; no list or struct columns.
SERIES_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
    "run_id": pl.String,
    "episode_id": pl.Int64,
    "robot_id": pl.String,
    "t_ns": pl.Int64,
    "time_s": pl.Float64,
    "speed_mps": pl.Float64,
    "acceleration_mps2": pl.Float64,
    "jerk_mps3": pl.Float64,
}

_SUMMARY_KEYS: typing.Final[tuple[str, ...]] = (
    "velocity_mean",
    "velocity_max",
    "acceleration_mean",
    "acceleration_max",
    "jerk_mean",
    "jerk_max",
)


def episode_speed_mps(calculator: BaseMetricCalculator, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth speed (m/s) and sample times (ns) of an episode, the physical samples only.

    The single derivation behind both the aggregate metric row and the 1NF motion
    series, so the published series and its summaries cannot disagree.
    """
    pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = calculator.resolve_robot_pose(episode)
    frame = episode.data
    if frame is not None and "vel_linear" in frame.columns:
        valid = pl.col("vel_linear").is_not_null() & ~pl.col("vel_linear").is_nan()
        max_linear_velocity = calculator.robot_params.max_linear_velocity
        if max_linear_velocity > 0.0:
            valid = valid & (pl.col("vel_linear").abs() <= max_linear_velocity)
        frame = frame.filter(valid)
        return np.abs(frame["vel_linear"].to_numpy().astype(np.float64)), frame["time_ns"].to_numpy().astype(np.int64)
    if len(pos_x) > 1:
        time_ns = frame["time_ns"].to_numpy().astype(np.int64) if frame is not None and "time_ns" in frame.columns else np.arange(len(pos_x), dtype=np.int64) * 100_000_000
        return calculator.speed_from_pose(pos_x, pos_y, time_ns), time_ns
    return np.array([]), np.array([], dtype=np.int64)


def _dt_s(time_ns: np.ndarray) -> np.ndarray:
    """Non-negative sample intervals in seconds, a degenerate interval clipped to ``MIN_DT_S``."""
    dt = np.diff(time_ns) / NANOSECONDS_PER_SECOND
    return np.where(dt <= 0.0, MIN_DT_S, dt)


def _acceleration_mps2(speed_mps: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    return np.diff(speed_mps) / dt_s


def _jerk_mps3(acceleration_mps2: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    return np.diff(acceleration_mps2) / dt_s[:-1]


def _summarize(speed_mps: np.ndarray, time_ns: np.ndarray) -> dict[str, float | None]:
    """Velocity, acceleration and jerk aggregates of one speed track; explicit nulls without enough samples."""
    result: dict[str, float | None] = dict.fromkeys(_SUMMARY_KEYS)
    if len(speed_mps) == 0:
        return result
    result["velocity_mean"] = float(np.mean(speed_mps))
    result["velocity_max"] = float(np.max(speed_mps))
    if len(speed_mps) < 2:
        return result

    dt_s = _dt_s(time_ns)
    acceleration_mps2 = _acceleration_mps2(speed_mps, dt_s)
    result["acceleration_mean"] = float(np.mean(np.abs(acceleration_mps2)))
    result["acceleration_max"] = float(np.max(np.abs(acceleration_mps2)))
    if len(speed_mps) < 3:
        return result

    jerk_mps3 = _jerk_mps3(acceleration_mps2, dt_s)
    result["jerk_mean"] = float(np.mean(np.abs(jerk_mps3)))
    result["jerk_max"] = float(np.max(np.abs(jerk_mps3)))
    return result


def _null_float_series(values: np.ndarray, name: str) -> pl.Series:
    """Float64 series with every NaN replaced by an explicit null."""
    return pl.Series(name, values, dtype=pl.Float64).fill_nan(None)


def _motion_series_frame(
    speed_mps: np.ndarray,
    time_ns: np.ndarray,
    *,
    run_id: str,
    episode_id: int,
    robot_id: str,
) -> pl.DataFrame:
    """1NF motion series of one speed track, schema-typed when empty.

    Acceleration is the backward difference of the speed at the sample it lands on, so
    the first sample has no acceleration; jerk likewise leaves its first two samples null.
    """
    sample_count = len(speed_mps)
    if sample_count == 0:
        return pl.DataFrame(schema=SERIES_SCHEMA)

    acceleration_mps2 = np.full(sample_count, np.nan, dtype=np.float64)
    jerk_mps3 = np.full(sample_count, np.nan, dtype=np.float64)
    if sample_count >= 2:
        dt_s = _dt_s(time_ns)
        acceleration = _acceleration_mps2(speed_mps, dt_s)
        acceleration_mps2[1:] = acceleration
        if sample_count >= 3:
            jerk_mps3[2:] = _jerk_mps3(acceleration, dt_s)

    time_s = (time_ns.astype(np.float64) - float(time_ns[0])) / NANOSECONDS_PER_SECOND
    return pl.DataFrame(
        {
            "run_id": pl.Series("run_id", [run_id] * sample_count, dtype=pl.String),
            "episode_id": pl.Series("episode_id", [int(episode_id)] * sample_count, dtype=pl.Int64),
            "robot_id": pl.Series("robot_id", [robot_id] * sample_count, dtype=pl.String),
            "t_ns": pl.Series("t_ns", time_ns, dtype=pl.Int64),
            "time_s": pl.Series("time_s", time_s, dtype=pl.Float64),
            "speed_mps": _null_float_series(speed_mps, "speed_mps"),
            "acceleration_mps2": _null_float_series(acceleration_mps2, "acceleration_mps2"),
            "jerk_mps3": _null_float_series(jerk_mps3, "jerk_mps3"),
        },
        schema=SERIES_SCHEMA,
    )


class MotionMetricsCalculator(BaseMetricCalculator):
    """Robot velocity, acceleration and jerk over the scope window's ground truth."""

    NAME = "motion_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks"]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})

    UNITS = {
        "velocity_mean": "m/s",
        "velocity_max": "m/s",
        "acceleration_mean": "m/sÂ²",
        "acceleration_max": "m/sÂ²",
        "jerk_mean": "m/sÂ³",
        "jerk_max": "m/sÂ³",
    }

    PRIMARY_OUTPUTS = ["velocity_mean", "velocity_max", "jerk_mean", "jerk_max"]

    OUTPUT_DIRECTIONS = {"jerk_mean": "lower", "jerk_max": "lower"}

    @classmethod
    def required_topics_for_scope(cls, scope: MetricScope) -> tuple[str | tuple[str, ...], ...]:
        # A window may legitimately lack an optional sensor stream; a missing stream is then per-metric
        # `no_samples`, not a row-level `topic_missing` that would block the whole encounter.
        return () if scope == MetricScope.ENCOUNTER else super().required_topics_for_scope(scope)

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(_SUMMARY_KEYS)

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

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        speed_mps, time_ns = episode_speed_mps(self, episode)
        return _summarize(speed_mps, time_ns)

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
            return _summarize(speed_mps, time_ns)
        raise ValueError(f"motion_metrics does not serve the {scope_slice.scope.value} scope")

    def series_table(self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = "") -> pl.DataFrame:
        """1NF flat motion series, one row per physical sample; schema-typed when empty."""
        speed_mps, time_ns = episode_speed_mps(self, episode)
        return _motion_series_frame(
            speed_mps,
            time_ns,
            run_id=run_id,
            episode_id=int(episode.episode_id),
            robot_id=robot_id or (episode.robot_name or "robot"),
        )
