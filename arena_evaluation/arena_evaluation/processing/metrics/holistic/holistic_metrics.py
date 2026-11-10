"""Holistic cross-domain metrics: acoustic exposure inside personal space, and the effective cost of transport.

AEPS integrates the published acoustic-exposure series over the frames whose nearest pedestrian is
inside the personal-space band. Both inputs are declared 1NF series dependencies: `acoustic_exposure`
carries the per-pedestrian level and `proxemics_extended` the per-frame nearest-pedestrian clearance,
so the metric reads the same tables the report consumes instead of retired nested list outputs. The
registry publishes `series_missing:<stem>` and a null value when a declared table is absent or empty;
the integral is never substituted with a nominal 45 dBA floor.
"""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PERSONAL_SPACE_CLEARANCE_M: Final[float] = 1.2
COLLISION_PENALTY: Final[float] = 2.0
INTRUSION_WEIGHT: Final[float] = 0.5
COT_DECIMALS: Final[int] = 6
MIN_EPISODE_TIME_S: Final[float] = 0.01


def _aligned_values(time_ns: np.ndarray, frame: pl.DataFrame, column: str) -> np.ndarray:
    """`frame[column]` reindexed onto the episode's sample times, NaN where the sample is absent."""
    if frame.height == 0:
        return np.full(len(time_ns), np.nan, dtype=np.float64)
    values = frame.select(pl.col("t_ns").cast(pl.Int64), pl.col(column).cast(pl.Float64)).drop_nulls("t_ns")
    values = values.unique(subset=["t_ns"], keep="last")
    joined = pl.DataFrame({"t_ns": time_ns.astype(np.int64)}).join(values, on="t_ns", how="left")
    return joined.get_column(column).to_numpy().astype(np.float64, copy=False)


class HolisticMetricsCalculator(BaseMetricCalculator):
    """Acoustic exposure inside personal space and the effective cost of transport."""

    NAME = "holistic_metrics"
    CATEGORY = "holistic"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = [
        "energy_extended",
        "collision_metrics",
        "proxemics_extended",
        "time_metrics",
        "acoustic_exposure",
    ]
    SERIES_DEPENDS_ON = ("acoustic_exposure", "proxemics_extended")
    SERIES_KEY = ("t_ns", "ped_id")
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_tracks"]

    UNITS = {
        "aeps_linear_s": "s",
        "e_cot": "",
    }

    PRIMARY_OUTPUTS = ["e_cot"]
    OUTPUT_DIRECTIONS = {"e_cot": "lower", "aeps_linear_s": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["aeps_linear_s", "e_cot"]

    def _level_series(self, episode: AlignedEpisodeBundle, time_ns: np.ndarray) -> np.ndarray | None:
        """Per-frame dBA: the loudest pedestrian exposure at each sample time from the published series."""
        frame = self.series_frame(episode, "acoustic_exposure")
        if frame is None:
            return None
        per_frame = frame.group_by("t_ns").agg(pl.col("exposure_dba").max().alias("level_dba")).sort("t_ns")
        levels = _aligned_values(time_ns, per_frame, "level_dba")
        return levels if np.isfinite(levels).any() else None

    def _clearance_series(self, episode: AlignedEpisodeBundle, time_ns: np.ndarray) -> np.ndarray | None:
        """Per-frame nearest-pedestrian clearance from the published proxemics-extended series."""
        frame = self.series_frame(episode, "proxemics_extended")
        if frame is None:
            return None
        clearances = _aligned_values(time_ns, frame, "min_ped_clearance_m")
        return clearances if np.isfinite(clearances).any() else None

    @staticmethod
    def _time_ns(episode: AlignedEpisodeBundle) -> np.ndarray | None:
        """Aligned sample timestamps of the episode view, none without aligned rows."""
        if episode.data is None or episode.data.height == 0:
            return None
        return episode.data.get_column("time_ns").to_numpy().astype(np.int64)

    def _aeps(self, episode: AlignedEpisodeBundle) -> float | None:
        """Integral of the linear pressure while a pedestrian stayed inside the personal space."""
        time_ns = self._time_ns(episode)
        if time_ns is None or len(time_ns) == 0:
            return None
        clearance = self._clearance_series(episode, time_ns)
        levels = self._level_series(episode, time_ns)
        if clearance is None or levels is None:
            return None
        step_s = np.diff(time_ns, prepend=time_ns[0]) / 1e9
        step_s[step_s < 0.0] = 0.0
        intruded = (clearance < PERSONAL_SPACE_CLEARANCE_M) & np.isfinite(levels) & np.isfinite(clearance)
        if not np.any(intruded):
            return 0.0
        return float(np.sum(np.power(10.0, levels[intruded] / 10.0) * step_s[intruded]))

    @staticmethod
    def _effective_cost_of_transport(
        prior_results: typing.Mapping[str, typing.Any],
    ) -> float | None:
        """Cost of transport inflated by the collision count and the personal-space intrusion rate."""
        cost_of_transport = prior_results.get("specific_cost_of_transport")
        if cost_of_transport is None or float(cost_of_transport) <= 0.0:
            return None
        collisions = prior_results.get("collision_amount")
        collision_count = 0 if collisions is None else int(collisions)
        intrusion_integral = prior_results.get("personal_space_intrusion_integral")
        intrusion_m_s = 0.0 if intrusion_integral is None else float(intrusion_integral)
        time_to_goal = prior_results.get("time_to_goal")
        duration_s = max(float(time_to_goal), MIN_EPISODE_TIME_S) if time_to_goal else MIN_EPISODE_TIME_S
        value = float(cost_of_transport) * (1.0 + COLLISION_PENALTY * collision_count + INTRUSION_WEIGHT * intrusion_m_s / duration_s)
        return round(value, COT_DECIMALS)

    def _episode_metrics(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: typing.Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Both holistic outputs over the given episode view."""
        return {
            "aeps_linear_s": self._aeps(episode),
            "e_cot": self._effective_cost_of_transport(prior_results),
        }

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Compute the episode-scope holistic metrics over the bundle's own samples."""
        return self._episode_metrics(episode, prior_results)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE scope only: the slice's own window and child-scope rollups are the population."""
        if scope_slice is not None and scope_slice.scope != MetricScope.EPISODE:
            return dict.fromkeys(self.output_keys())
        return self._episode_metrics(episode, prior_results)
