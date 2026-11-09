"""Holistic cross-domain metrics: acoustic exposure inside personal space, and the effective cost of transport.

Both are genuine integrals over the resolved episode view: the personal-space mask comes from
the vectorized robot-time to pedestrian-frame alignment, and every undefined input leaves its
output null instead of substituting a nominal 45 dBA floor or a zero integral.
"""

from __future__ import annotations

import math
import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.social.interaction_common import DEFAULT_PED_RADIUS_M
from arena_evaluation.processing.pedestrian_frames import nearest_pedestrian_distance
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PERSONAL_SPACE_CLEARANCE_M: Final[float] = 1.2
COLLISION_PENALTY: Final[float] = 2.0
INTRUSION_WEIGHT: Final[float] = 0.5
COT_DECIMALS: Final[int] = 6


def _decibel_levels(series: object) -> pl.DataFrame | None:
    """Per-frame dBA series from the recorded exposure series, null on frames that observed nothing."""
    if not isinstance(series, (list, tuple)) or len(series) == 0:
        return None
    levels: list[float | None] = []
    for frame in series:
        if isinstance(frame, (list, tuple, np.ndarray)):
            levels.append(float(max(frame)) if len(frame) > 0 else None)
        else:
            levels.append(float(frame))  # pyright: ignore[reportArgumentType]
    return pl.DataFrame(
        {"frame_ix": np.arange(len(levels), dtype=np.int64), "level_dba": levels},
        schema={"frame_ix": pl.Int64, "level_dba": pl.Float64},
    )


def acoustic_exposure_in_personal_space_pa2s(
    robot_times_ns: np.ndarray,
    robot_x_m: np.ndarray,
    robot_y_m: np.ndarray,
    peds: pl.DataFrame | None,
    combined_radius_m: float,
    exposure_dba: pl.DataFrame | None,
) -> float | None:
    """Integral of the linear acoustic pressure while a pedestrian stayed inside the personal space."""
    if peds is None or peds.height == 0 or exposure_dba is None or len(robot_times_ns) == 0:
        return None
    distances_m = nearest_pedestrian_distance(
        robot_times_ns, robot_x_m, robot_y_m, peds
    )
    clearance_m = distances_m - combined_radius_m
    frames = pl.DataFrame(
        {
            "frame_ix": np.arange(len(robot_times_ns), dtype=np.int64),
            "time_ns": robot_times_ns.astype(np.int64),
            "clearance_m": clearance_m,
        },
        schema={
            "frame_ix": pl.Int64,
            "time_ns": pl.Int64,
            "clearance_m": pl.Float64,
        },
    ).join(exposure_dba, on="frame_ix", how="left")
    step_s = (pl.col("time_ns") - pl.col("time_ns").shift(1)).cast(pl.Float64) / 1e9
    integrated = frames.select(
        (
            pl.when(
                (pl.col("clearance_m") < PERSONAL_SPACE_CLEARANCE_M)
                & pl.col("level_dba").is_not_null()
                & (step_s > 0.0)
            )
            .then(
                pl.col("level_dba").truediv(10.0).pow(10.0)
                * step_s
            )
            .otherwise(0.0)
        )
        .sum()
        .alias("exposure_pa2s"),
        pl.col("level_dba").is_not_null().sum().alias("scored_frames"),
    ).row(0, named=True)
    if int(integrated["scored_frames"]) == 0:
        return None
    return float(integrated["exposure_pa2s"])


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

    def _robot_samples(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Ground-truth robot samples of the episode view: times, x and y, none without a resolved pose."""
        frame = self.resolve_robot_frame(episode)
        if frame is None or frame.height == 0:
            return None
        return (
            frame.get_column("time_ns").to_numpy().astype(np.int64),
            frame.get_column("pos_x_gt").to_numpy().astype(np.float64),
            frame.get_column("pos_y_gt").to_numpy().astype(np.float64),
        )

    def _aeps(
        self, episode: AlignedEpisodeBundle, prior_results: typing.Mapping[str, typing.Any]
    ) -> float | None:
        """Acoustic exposure in personal space, null while the exposure series or the pose is missing."""
        samples = self._robot_samples(episode)
        if samples is None:
            return None
        exposure = prior_results.get("timeseries_acoustic_exposure_dba")
        if exposure is None:
            exposure = prior_results.get("timeseries_total_level_dba")
        levels = _decibel_levels(exposure)
        if levels is None:
            return None
        time_ns, x_m, y_m = samples
        return acoustic_exposure_in_personal_space_pa2s(
            time_ns,
            x_m,
            y_m,
            self.native_ped_frame(episode),
            self._combined_radius_m,
            levels,
        )

    @staticmethod
    def _effective_cost_of_transport(
        prior_results: typing.Mapping[str, typing.Any],
    ) -> float | None:
        """Cost of transport inflated by collision and personal-space-intrusion penalties."""
        cost_of_transport = prior_results.get("specific_cost_of_transport")
        intrusion_integral = prior_results.get("personal_space_intrusion_integral")
        time_to_goal_s = prior_results.get("time_to_goal")
        if (
            cost_of_transport is None
            or intrusion_integral is None
            or time_to_goal_s is None
        ):
            return None
        collisions = prior_results.get("collision_amount", 0)
        collision_count = 0 if collisions is None else int(collisions)
        duration_s = max(float(time_to_goal_s), 0.01)
        value = float(cost_of_transport) * (
            1.0
            + COLLISION_PENALTY * collision_count
            + INTRUSION_WEIGHT * float(intrusion_integral) / duration_s
        )
        return round(value, COT_DECIMALS) if math.isfinite(value) else None

    def _episode_metrics(
        self, episode: AlignedEpisodeBundle, prior_results: typing.Mapping[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Both holistic outputs over the given episode view."""
        return {
            "aeps_linear_s": self._aeps(episode, prior_results),
            "e_cot": self._effective_cost_of_transport(prior_results),
        }

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
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
