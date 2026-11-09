from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.pedestrian_frames import nearest_pedestrian_distance
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

TRACK_ALIGN_TOLERANCE_NS: typing.Final[int] = 100_000_000
_RANGE_EPSILON_M: typing.Final[float] = 1e-6

SERIES_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
    "run_id": pl.String,
    "episode_id": pl.Int64,
    "robot_id": pl.String,
    "t_ns": pl.Int64,
    "obstacle_clearance_m": pl.Float64,
    "pedestrian_clearance_m": pl.Float64,
    "combined_clearance_m": pl.Float64,
}


def _null_series(values: np.ndarray, name: str) -> pl.Series:
    """Float64 series with every NaN replaced by an explicit null."""
    return pl.Series(name, values, dtype=pl.Float64).fill_nan(None)


def _summaries(obstacle_clearance_m: np.ndarray, pedestrian_clearance_m: np.ndarray) -> dict[str, float | None]:
    """Minimum and mean of the valid obstacle and pedestrian clearances, None without a single valid sample."""
    valid_obstacle = obstacle_clearance_m[~np.isnan(obstacle_clearance_m)]
    valid_pedestrian = pedestrian_clearance_m[~np.isnan(pedestrian_clearance_m)]
    return {
        "min_obstacle_clearance": float(np.min(valid_obstacle)) if len(valid_obstacle) else None,
        "mean_obstacle_clearance": float(np.mean(valid_obstacle)) if len(valid_obstacle) else None,
        "min_pedestrian_clearance": float(np.min(valid_pedestrian)) if len(valid_pedestrian) else None,
        "mean_pedestrian_clearance": float(np.mean(valid_pedestrian)) if len(valid_pedestrian) else None,
    }


class ClearanceMetricsCalculator(BaseMetricCalculator):
    """Minimum clearance distances to obstacles and pedestrians over the scope window."""

    NAME = "clearance_metrics"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("robot_scan_summary", "pedestrian_tracks")]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})

    UNITS = {
        "min_obstacle_clearance": "m",
        "mean_obstacle_clearance": "m",
        "min_pedestrian_clearance": "m",
        "mean_pedestrian_clearance": "m",
    }

    PRIMARY_OUTPUTS = ["min_obstacle_clearance", "min_pedestrian_clearance"]
    OUTPUT_DIRECTIONS = {
        "min_obstacle_clearance": "higher",
        "mean_obstacle_clearance": "higher",
        "min_pedestrian_clearance": "higher",
        "mean_pedestrian_clearance": "higher",
    }

    _PED_RADIUS_M = 0.3

    @classmethod
    @classmethod
    def required_topics_for_scope(cls, scope: MetricScope) -> tuple[str | tuple[str, ...], ...]:
        # A window may legitimately lack an optional sensor stream; a missing stream is then per-metric
        # `no_samples`, not a row-level `topic_missing` that would block the whole encounter.
        return () if scope == MetricScope.ENCOUNTER else super().required_topics_for_scope(scope)

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "min_obstacle_clearance",
            "mean_obstacle_clearance",
            "min_pedestrian_clearance",
            "mean_pedestrian_clearance",
        ]

    @staticmethod
    def _scan_arrays(scan_df: pl.DataFrame | None, time_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scan minimum and maximum range interpolated onto ``time_ns``, NaN where no scan sample is in tolerance."""
        if scan_df is None or "scan_min_m" not in scan_df.columns:
            empty = np.full(len(time_ns), np.nan)
            return empty, empty.copy()
        columns = ["t_ns", "scan_min_m"]
        if "range_max_m" in scan_df.columns:
            columns.append("range_max_m")
        scan = scan_df.select([pl.col("t_ns").cast(pl.Int64), *[pl.col(name).cast(pl.Float64) for name in columns[1:]]]).drop_nulls("t_ns").sort("t_ns")
        joined = (
            pl.DataFrame({"t_ns": time_ns.astype(np.int64)})
            .join_asof(scan, on="t_ns", strategy="backward", tolerance=TRACK_ALIGN_TOLERANCE_NS)
            .sort("t_ns")
        )
        scan_min = joined["scan_min_m"].to_numpy().astype(np.float64)
        scan_max = joined["range_max_m"].to_numpy().astype(np.float64) if "range_max_m" in joined.columns else np.full(len(time_ns), np.nan)
        return scan_min, scan_max

    def _clearance_arrays(
        self,
        time_ns: np.ndarray,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        scan_df: pl.DataFrame | None,
        peds_df: pl.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-sample obstacle and pedestrian clearance in metres, NaN where that source has no valid sample."""
        robot_radius_m = self.robot_params.robot_radius
        scan_min, scan_max = self._scan_arrays(scan_df, time_ns)
        undetected = np.isnan(scan_max) | (scan_min < scan_max - _RANGE_EPSILON_M)
        obstacle_valid = np.isfinite(scan_min) & (scan_min > 0.0) & undetected
        obstacle_clearance_m = np.where(obstacle_valid, np.maximum(0.0, scan_min - robot_radius_m), np.nan)

        if peds_df is None or len(peds_df) == 0:
            pedestrian_clearance_m = np.full(len(time_ns), np.nan)
        else:
            distance_m = nearest_pedestrian_distance(time_ns, pos_x, pos_y, peds_df, TRACK_ALIGN_TOLERANCE_NS)
            pedestrian_clearance_m = np.where(
                np.isnan(distance_m),
                np.nan,
                np.maximum(0.0, distance_m - robot_radius_m - self._PED_RADIUS_M),
            )
        return obstacle_clearance_m, pedestrian_clearance_m

    @staticmethod
    def _combined(obstacle_clearance_m: np.ndarray, pedestrian_clearance_m: np.ndarray) -> np.ndarray:
        """Sample-wise minimum of both sources, whichever is valid, NaN without either."""
        both_valid = ~np.isnan(obstacle_clearance_m) & ~np.isnan(pedestrian_clearance_m)
        combined = np.full(len(obstacle_clearance_m), np.nan)
        combined[both_valid] = np.minimum(obstacle_clearance_m[both_valid], pedestrian_clearance_m[both_valid])
        only_obstacle = ~np.isnan(obstacle_clearance_m) & ~both_valid
        combined[only_obstacle] = obstacle_clearance_m[only_obstacle]
        only_pedestrian = ~np.isnan(pedestrian_clearance_m) & ~both_valid
        combined[only_pedestrian] = pedestrian_clearance_m[only_pedestrian]
        return combined

    def _episode_arrays(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aligned sample times, ground-truth robot position and the two clearance series of the episode."""
        if episode.data is None or "time_ns" not in episode.data.columns:
            empty_int = np.array([], dtype=np.int64)
            empty = np.array([], dtype=np.float64)
            return empty_int, empty, empty.copy(), empty.copy()

        pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        frame = self.resolve_robot_frame(episode)
        if frame is None or len(frame) == 0 or len(pos_x) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        aligned_times_ns = frame["time_ns"].to_numpy().astype(np.int64)
        topic_scan = self.native_topics(episode).get("robot_scan_summary")
        peds_df = self.native_ped_frame(episode)
        obstacle, pedestrian = self._clearance_arrays(aligned_times_ns, pos_x, pos_y, topic_scan, peds_df)
        return aligned_times_ns, self._combined(obstacle, pedestrian), obstacle, pedestrian

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        _times, _combined, obstacle, pedestrian = self._episode_arrays(episode)
        return _summaries(obstacle, pedestrian)

    def _native_arrays(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Native-clock sample times, ground-truth robot position and the two clearance series of a scope window."""
        pos_x, pos_y, _yaw, time_ns = self.resolve_native_pose(episode)
        if len(time_ns) == 0:
            empty = np.array([], dtype=np.float64)
            return np.array([], dtype=np.int64), empty, empty.copy(), empty.copy()
        topics = self.native_topics(episode)
        obstacle, pedestrian = self._clearance_arrays(
            time_ns,
            pos_x,
            pos_y,
            topics.get("robot_scan_summary"),
            self.native_ped_frame(episode),
        )
        return time_ns, self._combined(obstacle, pedestrian), obstacle, pedestrian

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            _times, _combined, obstacle, pedestrian = self._native_arrays(episode)
            return _summaries(obstacle, pedestrian)
        raise ValueError(f"clearance_metrics does not serve the {scope_slice.scope.value} scope")

    def series_table(self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = "") -> pl.DataFrame:
        """1NF flat clearance series, one row per sample, explicit nulls where a source has no valid sample."""
        times_ns, combined, obstacle, pedestrian = self._episode_arrays(episode)
        if len(times_ns) == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        return pl.DataFrame(
            {
                "run_id": pl.Series("run_id", [run_id] * len(times_ns), dtype=pl.String),
                "episode_id": pl.Series("episode_id", [int(episode.episode_id)] * len(times_ns), dtype=pl.Int64),
                "robot_id": pl.Series("robot_id", [robot_id or (episode.robot_name or "robot")] * len(times_ns), dtype=pl.String),
                "t_ns": pl.Series("t_ns", times_ns, dtype=pl.Int64),
                "obstacle_clearance_m": _null_series(obstacle, "obstacle_clearance_m"),
                "pedestrian_clearance_m": _null_series(pedestrian, "pedestrian_clearance_m"),
                "combined_clearance_m": _null_series(combined, "combined_clearance_m"),
            },
            schema=SERIES_SCHEMA,
        )