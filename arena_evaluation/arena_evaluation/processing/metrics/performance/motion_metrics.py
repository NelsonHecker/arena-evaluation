"""Robot speed, acceleration and jerk over the scope window's ground truth.

Peak acceleration and jerk are estimated from the recorded ground-truth speed with a local
polynomial (Savitzky-Golay) fit over a declared **1.0 s** window and **polynomial order 2**
on a uniform grid of at most 20 Hz (a coarser recording keeps its own step). Differentiating the raw speed samples at the
recording rate amplifies sample noise into physically impossible peaks — a jackal with a
2.5 m/s² actuator limit showed median peak accelerations near 28 m/s² and median peak jerk
near 900 m/s³ — so the declared window is what the reported peaks measure. An episode with
fewer samples than the smoothing fit needs (or without a ground-truth position track) falls
back to plain finite differences, where smoothing is undefined.

``calculate`` keeps only the scalar aggregates in the metric row. The per-sample kinematics
are published as one flat 1NF table by :meth:`MotionMetricsCalculator.series_table`; both
come from the same derivation, so the series and its summaries cannot disagree.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Final

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
MIN_DT_S: typing.Final[float] = 1e-6

SERIES_TABLE_STEM = "motion"

# Declared smoothing of the derivative estimator: what the reported acceleration and jerk peaks measure.
SMOOTHING_WINDOW_S: Final[float] = 1.0
SMOOTHING_POLYORDER: Final[int] = 2
KINEMATICS_GRID_HZ: Final[float] = 20.0
MIN_SMOOTHING_SAMPLES: Final[int] = 5

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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Kinematics:
    """Speed, tangential acceleration and jerk on one time axis; undefined samples are NaN."""

    time_ns: np.ndarray
    speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    jerk_mps3: np.ndarray


def _empty_kinematics() -> Kinematics:
    empty = np.array([], dtype=np.float64)
    return Kinematics(
        time_ns=np.array([], dtype=np.int64),
        speed_mps=empty,
        acceleration_mps2=empty.copy(),
        jerk_mps3=empty.copy(),
    )


def _dt_s(time_ns: np.ndarray) -> np.ndarray:
    """Non-negative sample intervals in seconds, a degenerate interval clipped to ``MIN_DT_S``."""
    dt = np.diff(time_ns) / NANOSECONDS_PER_SECOND
    return np.where(dt <= 0.0, MIN_DT_S, dt)


def _pose_track(
    calculator: BaseMetricCalculator, episode: AlignedEpisodeBundle
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ground-truth position samples of the scope view: the aligned teleport-free frame, else the native track."""
    frame = calculator.resolve_robot_frame(episode)
    if frame is not None and frame.height >= 2:
        return (
            frame.get_column("time_ns").to_numpy().astype(np.int64),
            frame.get_column("pos_x_gt").to_numpy().astype(np.float64),
            frame.get_column("pos_y_gt").to_numpy().astype(np.float64),
        )
    tracks = calculator.native_topics(episode).get("robot_tracks")
    if tracks is None or tracks.height < 2:
        return None
    tracks = tracks.sort("t_ns")
    return (
        tracks.get_column("t_ns").to_numpy().astype(np.int64),
        tracks.get_column("x").cast(pl.Float64).to_numpy().astype(np.float64),
        tracks.get_column("y").cast(pl.Float64).to_numpy().astype(np.float64),
    )


def _unique_track(
    time_ns: np.ndarray, x_m: np.ndarray, y_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Samples in ascending time with one row per stamp and finite positions, None when fewer than two remain."""
    order = np.argsort(time_ns, kind="stable")
    time_sorted = time_ns[order]
    x_sorted = x_m[order]
    y_sorted = y_m[order]
    keep = np.concatenate(([True], np.diff(time_sorted) > 0))
    finite = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    keep &= finite
    time_sorted, x_sorted, y_sorted = time_sorted[keep], x_sorted[keep], y_sorted[keep]
    if len(time_sorted) < 2:
        return None
    return time_sorted, x_sorted, y_sorted


def _acceleration_from_speed(time_ns: np.ndarray, speed_mps: np.ndarray) -> np.ndarray:
    """Backward-difference acceleration of a scalar series, at the sample it lands on."""
    acceleration_mps2 = np.full(len(speed_mps), np.nan, dtype=np.float64)
    acceleration_mps2[1:] = np.diff(speed_mps) / _dt_s(time_ns)
    return acceleration_mps2


def _raw_kinematics(speed_mps: np.ndarray, time_ns: np.ndarray) -> Kinematics:
    """Plain finite differences on the recorded samples, the fallback where smoothing is undefined."""
    acceleration_mps2 = _acceleration_from_speed(time_ns, speed_mps)
    jerk_mps3 = np.full(len(speed_mps), np.nan, dtype=np.float64)
    if len(speed_mps) >= 3:
        jerk_mps3[2:] = np.diff(acceleration_mps2[1:]) / _dt_s(time_ns)[:-1]
    return Kinematics(
        time_ns=time_ns,
        speed_mps=speed_mps,
        acceleration_mps2=acceleration_mps2,
        jerk_mps3=jerk_mps3,
    )


def _speed_from_pose(x_m: np.ndarray, y_m: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
    """Ground-truth speed from one-sided position differences, zero on the first sample."""
    speed = np.zeros(len(x_m), dtype=np.float64)
    if len(x_m) >= 2:
        speed[1:] = np.hypot(np.diff(x_m), np.diff(y_m)) / _dt_s(time_ns)
    return speed


def _smooth_position_kinematics(
    time_ns: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> Kinematics | None:
    """Savitzky-Golay kinematics of a position track on a uniform grid, None when the window does not fit.

    Used where the recording has no speed stream: fitting the positions first keeps sample noise
    out of the derived speed instead of amplifying it through finite differences.
    """
    unique = _unique_track(time_ns, x_m, y_m)
    if unique is None:
        return None
    grid_time_ns, grid_x_m, grid_y_m = unique
    step_s = max(
        1.0 / KINEMATICS_GRID_HZ,
        float(np.median(np.diff(grid_time_ns))) / NANOSECONDS_PER_SECOND,
    )
    step_ns = max(1, int(round(step_s * NANOSECONDS_PER_SECOND)))
    grid = np.arange(grid_time_ns[0], grid_time_ns[-1] + 1, step_ns, dtype=np.int64)
    if len(grid) < MIN_SMOOTHING_SAMPLES:
        return None
    sample_step_s = step_ns / NANOSECONDS_PER_SECOND
    window = int(round(SMOOTHING_WINDOW_S / sample_step_s)) | 1
    window = min(window, (len(grid) // 2) * 2 - 1)
    if window < MIN_SMOOTHING_SAMPLES:
        return None
    grid_x = np.interp(grid, grid_time_ns, grid_x_m)
    grid_y = np.interp(grid, grid_time_ns, grid_y_m)
    velocity_x = savgol_filter(
        grid_x, window, SMOOTHING_POLYORDER, deriv=1, delta=sample_step_s
    )
    velocity_y = savgol_filter(
        grid_y, window, SMOOTHING_POLYORDER, deriv=1, delta=sample_step_s
    )
    speed_mps = np.hypot(velocity_x, velocity_y)
    acceleration_mps2 = savgol_filter(
        speed_mps, window, SMOOTHING_POLYORDER, deriv=1, delta=sample_step_s
    )
    jerk_mps3 = savgol_filter(
        speed_mps, window, SMOOTHING_POLYORDER, deriv=2, delta=sample_step_s
    )
    return Kinematics(
        time_ns=grid,
        speed_mps=speed_mps,
        acceleration_mps2=acceleration_mps2,
        jerk_mps3=jerk_mps3,
    )


def _smooth_speed_kinematics(
    time_ns: np.ndarray,
    speed_mps: np.ndarray,
) -> Kinematics | None:
    """Savitzky-Golay derivatives of the recorded speed on a uniform grid, None when the window does not fit."""
    order = np.argsort(time_ns, kind="stable")
    time_sorted = time_ns[order]
    speed_sorted = speed_mps[order]
    keep = np.concatenate(([True], np.diff(time_sorted) > 0)) & np.isfinite(speed_sorted)
    time_sorted, speed_sorted = time_sorted[keep], speed_sorted[keep]
    if len(time_sorted) < 2:
        return None
    # Never interpolate finer than the recorded samples: a coarser recording keeps its own step,
    # a finer one is thinned to the declared grid rate.
    step_s = max(
        1.0 / KINEMATICS_GRID_HZ,
        float(np.median(np.diff(time_sorted))) / NANOSECONDS_PER_SECOND,
    )
    step_ns = max(1, int(round(step_s * NANOSECONDS_PER_SECOND)))
    grid = np.arange(time_sorted[0], time_sorted[-1] + 1, step_ns, dtype=np.int64)
    if len(grid) < MIN_SMOOTHING_SAMPLES:
        return None
    sample_step_s = step_ns / NANOSECONDS_PER_SECOND
    window = int(round(SMOOTHING_WINDOW_S / sample_step_s)) | 1
    window = min(window, (len(grid) // 2) * 2 - 1)
    if window < MIN_SMOOTHING_SAMPLES:
        return None
    grid_speed_mps = np.interp(grid, time_sorted, speed_sorted)
    acceleration_mps2 = savgol_filter(
        grid_speed_mps, window, SMOOTHING_POLYORDER, deriv=1, delta=sample_step_s
    )
    jerk_mps3 = savgol_filter(
        grid_speed_mps, window, SMOOTHING_POLYORDER, deriv=2, delta=sample_step_s
    )
    return Kinematics(
        time_ns=grid,
        speed_mps=grid_speed_mps,
        acceleration_mps2=acceleration_mps2,
        jerk_mps3=jerk_mps3,
    )


def episode_speed_mps(
    calculator: BaseMetricCalculator, episode: AlignedEpisodeBundle
) -> tuple[np.ndarray, np.ndarray]:
    """Recorded speed (m/s) and sample times (ns) of an episode, the speed-only fallback stream.

    Used where the recording carries a speed stream but no ground-truth position track; the
    physical velocity limit of the robot drops impossible sample spikes.
    """
    frame = episode.data
    if frame is None or frame.height == 0:
        return np.array([]), np.array([], dtype=np.int64)
    try:
        frame = frame.select("time_ns", "vel_linear")
    except pl.exceptions.ColumnNotFoundError:
        return np.array([]), np.array([], dtype=np.int64)
    valid = pl.col("vel_linear").is_not_null() & ~pl.col("vel_linear").is_nan()
    max_linear_velocity = calculator.robot_params.max_linear_velocity
    if max_linear_velocity > 0.0:
        valid = valid & (pl.col("vel_linear").abs() <= max_linear_velocity)
    frame = frame.filter(valid)
    if frame.height == 0:
        return np.array([]), np.array([], dtype=np.int64)
    return (
        np.abs(frame["vel_linear"].to_numpy().astype(np.float64)),
        frame["time_ns"].to_numpy().astype(np.int64),
    )


def episode_kinematics(
    calculator: BaseMetricCalculator, episode: AlignedEpisodeBundle
) -> Kinematics:
    """Speed, acceleration and jerk of the scope view, smoothed over the declared window when it fits.

    The recorded ground-truth speed stream is the signal (the same stream the metric always
    reported); an episode without it derives the speed from the ground-truth pose track.
    """
    speed_mps, time_ns = episode_speed_mps(calculator, episode)
    if len(speed_mps) > 0:
        if len(speed_mps) > 1:
            smoothed = _smooth_speed_kinematics(time_ns, speed_mps)
            if smoothed is not None:
                return smoothed
        return _raw_kinematics(speed_mps, time_ns)

    track = _pose_track(calculator, episode)
    if track is None:
        return _empty_kinematics()
    unique = _unique_track(*track)
    if unique is None:
        return _empty_kinematics()
    unique_time_ns, unique_x_m, unique_y_m = unique
    smoothed = _smooth_position_kinematics(unique_time_ns, unique_x_m, unique_y_m)
    if smoothed is not None:
        return smoothed
    derived_speed_mps = _speed_from_pose(unique_x_m, unique_y_m, unique_time_ns)
    return _raw_kinematics(derived_speed_mps, unique_time_ns)


def _summarize(kinematics: Kinematics) -> dict[str, float | None]:
    """Velocity, acceleration and jerk aggregates of one kinematics track; explicit nulls without samples."""
    result: dict[str, float | None] = dict.fromkeys(_SUMMARY_KEYS)
    speed_mps = kinematics.speed_mps[np.isfinite(kinematics.speed_mps)]
    if len(speed_mps) == 0:
        return result
    result["velocity_mean"] = float(np.mean(speed_mps))
    result["velocity_max"] = float(np.max(speed_mps))
    acceleration_mps2 = kinematics.acceleration_mps2[np.isfinite(kinematics.acceleration_mps2)]
    if len(acceleration_mps2) > 0:
        result["acceleration_mean"] = float(np.mean(np.abs(acceleration_mps2)))
        result["acceleration_max"] = float(np.max(np.abs(acceleration_mps2)))
    jerk_mps3 = kinematics.jerk_mps3[np.isfinite(kinematics.jerk_mps3)]
    if len(jerk_mps3) > 0:
        result["jerk_mean"] = float(np.mean(np.abs(jerk_mps3)))
        result["jerk_max"] = float(np.max(np.abs(jerk_mps3)))
    return result


def _null_float_series(values: np.ndarray, name: str) -> pl.Series:
    """Float64 series with every NaN replaced by an explicit null."""
    return pl.Series(name, values, dtype=pl.Float64).fill_nan(None)


def _motion_series_frame(
    kinematics: Kinematics,
    *,
    run_id: str,
    episode_id: int,
    robot_id: str,
) -> pl.DataFrame:
    """1NF motion series of one kinematics track, schema-typed when empty."""
    sample_count = len(kinematics.speed_mps)
    if sample_count == 0:
        return pl.DataFrame(schema=SERIES_SCHEMA)
    time_s = (
        kinematics.time_ns.astype(np.float64) - float(kinematics.time_ns[0])
    ) / NANOSECONDS_PER_SECOND
    return pl.DataFrame(
        {
            "run_id": pl.Series("run_id", [run_id] * sample_count, dtype=pl.String),
            "episode_id": pl.Series("episode_id", [int(episode_id)] * sample_count, dtype=pl.Int64),
            "robot_id": pl.Series("robot_id", [robot_id] * sample_count, dtype=pl.String),
            "t_ns": pl.Series("t_ns", kinematics.time_ns, dtype=pl.Int64),
            "time_s": pl.Series("time_s", time_s, dtype=pl.Float64),
            "speed_mps": _null_float_series(kinematics.speed_mps, "speed_mps"),
            "acceleration_mps2": _null_float_series(
                kinematics.acceleration_mps2, "acceleration_mps2"
            ),
            "jerk_mps3": _null_float_series(kinematics.jerk_mps3, "jerk_mps3"),
        },
        schema=SERIES_SCHEMA,
    )


class MotionMetricsCalculator(BaseMetricCalculator):
    """Robot velocity, acceleration and jerk over the scope window's ground truth.

    Acceleration and jerk are local-polynomial derivatives over a declared
    ``SMOOTHING_WINDOW_S`` (1.0 s) window at ``SMOOTHING_POLYORDER`` 2, so the peaks
    measure the smoothed trajectory rather than sample-rate noise.
    """

    NAME = "motion_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks"]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset(
        {MetricScope.EPISODE, MetricScope.ENCOUNTER}
    )

    UNITS = {
        "velocity_mean": "m/s",
        "velocity_max": "m/s",
        "acceleration_mean": "m/s²",
        "acceleration_max": "m/s²",
        "jerk_mean": "m/s³",
        "jerk_max": "m/s³",
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

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        return _summarize(episode_kinematics(self, episode))

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            del prior_results
            return _summarize(episode_kinematics(self, episode))
        raise ValueError(f"motion_metrics does not serve the {scope_slice.scope.value} scope")

    def series_table(
        self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = ""
    ) -> pl.DataFrame:
        """1NF flat motion series, one row per kinematics sample; schema-typed when empty."""
        return _motion_series_frame(
            episode_kinematics(self, episode),
            run_id=run_id,
            episode_id=int(episode.episode_id),
            robot_id=robot_id or (episode.robot_name or "robot"),
        )
