"""Holistic cross-domain metrics: acoustic exposure inside personal space, and the effective cost of transport.

AEPS integrates the recorded level over the frames whose nearest pedestrian is inside the
personal-space band. Its clearance series comes from the proxemics calculator when available,
otherwise from the same vectorized pedestrian-frame geometry; a missing level source and a
missing time axis leave the output null instead of substituting the historical 45 dBA floor.
"""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.social.interaction_common import DEFAULT_PED_RADIUS_M
from arena_evaluation.processing.pedestrian_frames import (
    ALIGN_TOLERANCE_NS,
    frame_geometry,
    nearest_pedestrian_frames,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PERSONAL_SPACE_CLEARANCE_M: Final[float] = 1.2
COLLISION_PENALTY: Final[float] = 2.0
INTRUSION_WEIGHT: Final[float] = 0.5
COT_DECIMALS: Final[int] = 6
MIN_EPISODE_TIME_S: Final[float] = 0.01


def _level_values(series: object) -> np.ndarray | None:
    """Per-frame dBA values of a nested or flat exposure series, NaN for frames without a level."""
    if not isinstance(series, (list, tuple)) or len(series) == 0:
        return None
    values = np.full(len(series), np.nan, dtype=np.float64)
    for index, frame in enumerate(series):
        if isinstance(frame, (list, tuple, np.ndarray)):
            if len(frame) > 0:
                values[index] = float(np.max(np.asarray(frame, dtype=np.float64)))
        elif frame is not None:
            values[index] = float(frame)  # pyright: ignore[reportArgumentType]
    return values


def _clearance_values(clearance: object) -> np.ndarray | None:
    """Per-frame edge clearances of a recorded series, NaN for frames without a measurement."""
    if not isinstance(clearance, (list, tuple)):
        return None
    values = np.full(len(clearance), np.nan, dtype=np.float64)
    for index, value in enumerate(clearance):
        if value is not None:
            values[index] = float(value)  # pyright: ignore[reportArgumentType]
    return values


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

    @property
    def _combined_radius_m(self) -> float:
        """Robot radius plus the nominal pedestrian radius of the personal-space band."""
        return float(self.robot_params.robot_radius) + DEFAULT_PED_RADIUS_M

    def _clearance_series(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: typing.Mapping[str, typing.Any],
    ) -> np.ndarray | None:
        """Edge clearance of the nearest pedestrian per frame: the proxemics series, else the frame geometry."""
        recorded = _clearance_values(prior_results.get("timeseries_min_ped_clearance"))
        if recorded is not None:
            return recorded
        geometry = frame_geometry(
            self.native_ped_frame(episode),
            self.native_topics(episode).get("robot_tracks"),
            tolerance_ns=ALIGN_TOLERANCE_NS,
        )
        nearest = nearest_pedestrian_frames(geometry)
        if nearest.height == 0:
            return None
        return nearest.select((pl.col("distance_m") - self._combined_radius_m).alias("clearance_m")).get_column("clearance_m").to_numpy().astype(np.float64)

    @staticmethod
    def _level_series(
        prior_results: typing.Mapping[str, typing.Any],
    ) -> np.ndarray | None:
        """Per-frame dBA values: the pedestrian exposure series, then the total level series."""
        for key in ("timeseries_acoustic_exposure_dba", "timeseries_total_level_dba"):
            recorded = _level_values(prior_results.get(key))
            if recorded is not None:
                return recorded
        return None

    @staticmethod
    def _time_ns(episode: AlignedEpisodeBundle) -> np.ndarray | None:
        """Aligned sample timestamps of the episode view, none without aligned rows."""
        if episode.data is None or episode.data.height == 0:
            return None
        return episode.data.get_column("time_ns").to_numpy().astype(np.int64)

    def _aeps(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: typing.Mapping[str, typing.Any],
    ) -> float | None:
        """Integral of the linear pressure while a pedestrian stayed inside the personal space."""
        clearance = self._clearance_series(episode, prior_results)
        levels = self._level_series(prior_results)
        time_ns = self._time_ns(episode)
        if clearance is None or levels is None or time_ns is None:
            return None
        length = min(len(clearance), len(levels), len(time_ns))
        if length == 0:
            return None
        clearance = clearance[:length]
        levels = levels[:length]
        timestamps = time_ns[:length]
        step_s = np.diff(timestamps, prepend=timestamps[0]) / 1e9
        step_s[step_s < 0.0] = 0.0
        intruded = (clearance < PERSONAL_SPACE_CLEARANCE_M) & np.isfinite(levels)
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
            "aeps_linear_s": self._aeps(episode, prior_results),
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
