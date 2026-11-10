from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..pose_segments import consistent_segment, ride_arrival_times

if typing.TYPE_CHECKING:
    from ...storage.schemas import AlignedEpisodeBundle, RobotParams

from ...storage.declarations import MetricScope, _coerce_scope

# A calculator may return `<output>_null_reason` next to a null output; the registry turns that
# companion string into the output's provenance code instead of the generic no_samples.
NULL_REASON_SUFFIX: typing.Final[str] = "_null_reason"


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeSlice:
    """Immutable spatiotemporal slice for one evaluation scope entity or window."""

    scope: MetricScope
    slice_id: str
    parent_id: str | None = None
    t_start_ns: int
    t_end_ns: int
    identity: Mapping[str, typing.Any]
    window_bundle: AlignedEpisodeBundle
    full_episode: AlignedEpisodeBundle


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeExecutionOutcome(Mapping[str, typing.Any]):
    """Output dictionary and per-key missingness provenance of a single scope execution."""

    scope: MetricScope
    results: dict[str, typing.Any]
    provenance: dict[str, str]

    def __getitem__(self, key: str) -> object:
        return self.results[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


def _anchored_odometry(
    xs: np.ndarray,
    ys: np.ndarray,
    yaws: np.ndarray,
    origin: tuple[float, float, float],
    start_pos: list[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Odometry turned and moved so that its first sample sits at the episode's start pose; unchanged without one."""
    if not start_pos or len(start_pos) < 2 or len(xs) == 0:
        return xs, ys, yaws
    start_yaw = start_pos[2] if len(start_pos) >= 3 else 0.0
    origin_x, origin_y, origin_yaw = origin
    theta = start_yaw - origin_yaw
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = xs - origin_x
    dy = ys - origin_y
    x = start_pos[0] + dx * cos_t - dy * sin_t
    y = start_pos[1] + dx * sin_t + dy * cos_t
    yaw = (yaws + theta + np.pi) % (2 * np.pi) - np.pi
    return x, y, yaw


class BaseMetricCalculator(ABC):
    """Abstract Base Class for all metric calculators."""

    NAME: str = ""
    SERIES_TABLE_STEM: typing.ClassVar[str] = ""
    CATEGORY: str = "general"
    SCOPE: MetricScope = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope] | set[MetricScope | str]] = frozenset()
    REQUIRES_PEDSIM: bool = False
    DEPENDS_ON: list[str] = []
    REQUIRED_TOPICS: list[str | list[str] | tuple[str, ...] | set[str]] = []
    UNITS: dict[str, str] = {}
    PRIMARY_OUTPUTS: list[str] = []
    OUTPUT_DIRECTIONS: dict[str, str] = {}
    # 1NF series tables this calculator reads. The pipeline publishes every series before scope
    # evaluation; the registry skips a calculator whose declared table is absent or empty and
    # explains the nulls with `series_missing:<stem>` provenance.
    SERIES_DEPENDS_ON: typing.ClassVar[tuple[str, ...]] = ()
    # Natural key of the series table this calculator publishes: one row per key tuple.
    SERIES_KEY: typing.ClassVar[tuple[str, ...]] = ("t_ns",)
    # Topics a series table needs to be emitted; empty when it derives from the aligned bundle alone.
    SERIES_REQUIRED_TOPICS: typing.ClassVar[tuple[str, ...]] = ()
    world: str | None = None

    def __init__(self, robot_params: RobotParams):
        """Initializes the calculator with robot parameters."""
        self.robot_params = robot_params
        self.metrics_config: dict[str, typing.Any] = {}

    @classmethod
    def effective_scopes(cls) -> frozenset[MetricScope]:
        """All evaluation scopes in which this calculator is registered."""
        if cls.SCOPES:
            return frozenset(_coerce_scope(s) for s in cls.SCOPES)
        return frozenset({_coerce_scope(cls.SCOPE)})

    @classmethod
    def required_topics_for_scope(cls, scope: MetricScope) -> tuple[str | tuple[str, ...], ...]:
        """Required topic names (or topic alternatives) for executing in `scope`."""
        del scope
        normalized: list[str | tuple[str, ...]] = []
        for req in cls.REQUIRED_TOPICS:
            if isinstance(req, str):
                normalized.append(req)
            else:
                normalized.append(tuple(req))
        return tuple(normalized)

    @classmethod
    def output_keys_for_scope(cls, scope: MetricScope) -> tuple[str, ...]:
        """Output keys produced when executing in `scope`."""
        del scope
        return tuple(cls.output_keys())

    def _consistent_ground_truth(self, episode: AlignedEpisodeBundle) -> tuple[pl.DataFrame, int, int] | None:
        """Ground-truth rows and the bounds of the longest teleport-free segment, None when no sample is usable."""
        if episode.data is None or "pos_x_gt" not in episode.data.columns:
            return None
        frame = episode.data.filter(pl.col("pos_x_gt").is_not_null() & ~pl.col("pos_x_gt").is_nan() & pl.col("pos_y_gt").is_not_null() & ~pl.col("pos_y_gt").is_nan() & pl.col("yaw_gt").is_not_null() & ~pl.col("yaw_gt").is_nan())
        if len(frame) == 0:
            return None
        start, end = consistent_segment(
            frame["pos_x_gt"].to_numpy().copy(),
            frame["pos_y_gt"].to_numpy().copy(),
            frame["time_ns"].to_numpy(),
            self._ride_arrivals(episode),
        )
        return frame, start, end

    def resolve_robot_frame(self, episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """The teleport-free ground-truth piece of the episode, sliced consistently with resolve_robot_pose."""
        resolved = self._consistent_ground_truth(episode)
        if resolved is None:
            return None
        frame, start, end = resolved
        return frame.slice(start, end - start)

    def resolve_robot_pose(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map-frame ground-truth pose (pos_x, pos_y, yaw) of the longest teleport-free piece of the track.

        The second triple is the odometry of the same piece, anchored to the episode start: an estimate for drift only.
        Without ground truth both are empty, because odometry never stands in for the trajectory.
        """
        empty = np.array([], dtype=np.float64)
        resolved = self._consistent_ground_truth(episode)
        if resolved is None:
            return empty, empty, empty, empty, empty, empty
        df, start, end = resolved

        track_x = df["pos_x_gt"].to_numpy().copy()
        track_y = df["pos_y_gt"].to_numpy().copy()
        track_yaw = df["yaw_gt"].to_numpy().copy()
        has_odometry = "pos_x" in df.columns and "yaw" in df.columns
        odom_x = df["pos_x"].to_numpy().astype(np.float64) if has_odometry else empty
        odom_y = df["pos_y"].to_numpy().astype(np.float64) if has_odometry else empty
        odom_yaw = df["yaw"].to_numpy().astype(np.float64) if has_odometry else empty
        odometry_origin = (odom_x[0], odom_y[0], odom_yaw[0]) if has_odometry and len(odom_x) > 0 else (0.0, 0.0, 0.0)

        odom_x_trans, odom_y_trans, odom_yaw_trans = _anchored_odometry(
            odom_x[start:end],
            odom_y[start:end],
            odom_yaw[start:end],
            odometry_origin,
            episode.start_pos,
        )
        return (
            track_x[start:end],
            track_y[start:end],
            track_yaw[start:end],
            odom_x_trans,
            odom_y_trans,
            odom_yaw_trans,
        )

    def series_table(self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = "") -> pl.DataFrame | None:
        """1NF long-format series this calculator publishes, None when it publishes none."""
        return None

    @staticmethod
    def _ride_arrivals(episode: AlignedEpisodeBundle) -> np.ndarray:
        """Elevator arrival and stair crossing log times of the robot's env, the teleports that are rides."""
        from .ecological.mechanism_common import robot_env_snapshot

        return ride_arrival_times(robot_env_snapshot(episode.semantic_snapshot, episode.robot_name))

    def native_topics(self, episode: AlignedEpisodeBundle) -> dict:
        """Raw native-rate topic frames keyed by topic name; {} when absent."""
        return episode.topics if isinstance(episode.topics, dict) else {}

    def series_frame(self, episode: AlignedEpisodeBundle, stem: str) -> pl.DataFrame | None:
        """The episode's published 1NF series frame for `stem`, None when it is absent or empty."""
        frame = episode.series.get(stem)
        if frame is None or frame.height == 0:
            return None
        return frame

    def missing_input_provenance(self, episode: AlignedEpisodeBundle) -> dict[str, str]:
        """Output key -> typed null_reason for nulls caused by an absent optional input stream.

        The registry uses these codes instead of the generic `no_samples` when a calculator ran but
        an optional input (a world snapshot, markers, an interaction stream) was not recorded.
        """
        del episode
        return {}

    def resolve_native_pose(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ground-truth robot pose on the native clock: (pos_x, pos_y, yaw, time_ns) float64 arrays, empty without ground truth.

        The robot's /tf track is the only trajectory source; wheel odometry never stands in for it.
        """
        empty = np.array([])
        tracks = self.native_topics(episode).get("robot_tracks")
        if tracks is None or not {"x", "y", "yaw"} <= set(tracks.columns):
            return empty, empty, empty, empty
        t_df = tracks.drop_nulls(["x", "y", "yaw"]).sort("t_ns")
        return (
            t_df["x"].to_numpy().astype(np.float64),
            t_df["y"].to_numpy().astype(np.float64),
            t_df["yaw"].to_numpy().astype(np.float64),
            t_df["t_ns"].to_numpy().astype(np.int64),
        )

    def native_ped_frame(self, episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """The robot's pedestrians as flat rows, one per pedestrian per frame, sorted by t_ns; None when absent."""
        peds = self.native_topics(episode).get("pedestrian_tracks")
        if peds is None or "t_ns" not in peds.columns:
            return None
        return peds.sort("t_ns")

    def pose_at_times(
        self,
        times_ns: np.ndarray,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        yaw: np.ndarray,
        time_ns: np.ndarray,
        tolerance_ns: int | None = 100_000_000,
        continuous: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate pose arrays onto query timestamps ``times_ns``."""
        if len(time_ns) == 0 or len(times_ns) == 0:
            return (
                np.zeros(len(times_ns)),
                np.zeros(len(times_ns)),
                np.zeros(len(times_ns)),
            )

        order = np.argsort(time_ns)
        t_sorted = time_ns[order]
        x_sorted = np.asarray(pos_x, dtype=np.float64)[order]
        y_sorted = np.asarray(pos_y, dtype=np.float64)[order]
        yaw_sorted = np.asarray(yaw, dtype=np.float64)[order]

        if continuous or tolerance_ns is None:
            px = np.interp(times_ns, t_sorted, x_sorted)
            py = np.interp(times_ns, t_sorted, y_sorted)
            ya = np.interp(times_ns, t_sorted, yaw_sorted)
            return px, py, ya

        import polars as pl

        df_ref = pl.DataFrame(
            {
                "time_ns": t_sorted,
                "pos_x": x_sorted,
                "pos_y": y_sorted,
                "yaw": yaw_sorted,
            }
        )
        df_query = pl.DataFrame({"time_ns": times_ns})
        merged = df_query.join_asof(df_ref, on="time_ns", strategy="backward", tolerance=tolerance_ns)
        px = np.nan_to_num(merged["pos_x"].to_numpy(), nan=0.0)
        py = np.nan_to_num(merged["pos_y"].to_numpy(), nan=0.0)
        ya = np.nan_to_num(merged["yaw"].to_numpy(), nan=0.0)
        return px, py, ya

    def values_at_times(
        self,
        values: np.ndarray,
        values_time_ns: np.ndarray,
        query_times_ns: np.ndarray,
        tolerance_ns: int | None = 100_000_000,
        continuous: bool = False,
    ) -> np.ndarray:
        """Interpolate values onto query timestamps."""
        if len(values_time_ns) == 0 or len(query_times_ns) == 0:
            return np.zeros(len(query_times_ns))

        order = np.argsort(values_time_ns)
        t_sorted = values_time_ns[order]
        v_sorted = np.asarray(values, dtype=np.float64)[order]

        if continuous or tolerance_ns is None:
            return np.interp(query_times_ns, t_sorted, v_sorted)

        import polars as pl

        df_ref = pl.DataFrame(
            {
                "time_ns": t_sorted,
                "val": v_sorted,
            }
        )
        df_query = pl.DataFrame({"time_ns": query_times_ns})
        merged = df_query.join_asof(df_ref, on="time_ns", strategy="backward", tolerance=tolerance_ns)
        return merged["val"].to_numpy()

    @staticmethod
    def speed_from_pose(pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
        """GT speed (m/s) by finite-differencing GT positions."""
        n = len(pos_x)
        speed = np.zeros(n)
        if n < 2:
            return speed
        dt = np.diff(time_ns) / 1e9
        dt_safe = np.where(dt <= 0.0, 1e-6, dt)
        speed[1:] = np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2) / dt_safe
        return np.nan_to_num(speed, nan=0.0)

    @staticmethod
    def velocity_from_pose(pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GT velocity vector (m/s) by finite-differencing GT positions."""
        n = len(pos_x)
        vx = np.zeros(n)
        vy = np.zeros(n)
        if n < 2:
            return vx, vy
        dt = np.diff(time_ns) / 1e9
        dt_safe = np.where(dt <= 0.0, 1e-6, dt)
        vx[1:] = np.diff(pos_x) / dt_safe
        vy[1:] = np.diff(pos_y) / dt_safe
        return np.nan_to_num(vx, nan=0.0), np.nan_to_num(vy, nan=0.0)

    @classmethod
    @abstractmethod
    def output_keys(cls) -> list[str]:
        """Returns a list of keys that this calculator will output."""
        pass

    @abstractmethod
    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Calculate metrics for a single episode."""
        pass

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """Calculate metrics for a specific scope slice, defaulting to `calculate`."""
        del scope_slice
        return self.calculate(episode, prior_results)
