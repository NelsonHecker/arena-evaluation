"""Pedestrian deflection, velocity delay and patrol rounds, per pedestrian and across the episode.

The per-pedestrian geometry is one Polars group-by over the flat track (path length, straight
chord, lateral deviation, displacement extremes); the two scopes share that frame and differ
only in the population they reduce: the slice's own pedestrian for PEDESTRIAN, the pedestrian
mean for EPISODE (the historical episode-wide value).
"""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.social.pedestrian_activity import (
    pedestrian_round_trips_frame,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PEDESTRIAN_MIN_POINTS: Final[int] = 5
ROUND_TRIP_MIN_DISPLACEMENT_M: Final[float] = 0.5
DEFLECTION_MIN_CHORD_M: Final[float] = 0.5
NOMINAL_DESIRED_SPEED_MPS: Final[float] = 1.2
VALID_POSITION_LIMIT_M: Final[float] = 1.0e6
SUMMED_VALUE_DECIMALS: Final[int] = 3

PEDESTRIAN_GEOMETRY_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "ped_id": pl.String,
        "n_points": pl.UInt32,
        "first_t_ns": pl.Int64,
        "last_t_ns": pl.Int64,
        "path_length_m": pl.Float64,
        "chord_m": pl.Float64,
        "max_displacement_m": pl.Float64,
        "lateral_deviation_m": pl.Float64,
    }
)


def _track_frame(peds: pl.DataFrame) -> pl.DataFrame:
    """Finite, plausibly located pedestrian rows sorted by pedestrian and time."""
    return (
        peds.select(
            t_ns=pl.col("t_ns").cast(pl.Int64),
            ped_id=pl.col("ped_id").cast(pl.String),
            x_m=pl.col("x").cast(pl.Float64),
            y_m=pl.col("y").cast(pl.Float64),
        )
        .filter(pl.col("x_m").is_finite() & pl.col("y_m").is_finite() & (pl.col("x_m").abs() < VALID_POSITION_LIMIT_M) & (pl.col("y_m").abs() < VALID_POSITION_LIMIT_M))
        .sort("ped_id", "t_ns")
    )


def pedestrian_geometry_frame(peds: pl.DataFrame | None) -> pl.DataFrame:
    """Per-pedestrian path length, straight chord, farthest displacement and mean lateral deviation."""
    if peds is None or peds.height == 0:
        return pl.DataFrame(schema=PEDESTRIAN_GEOMETRY_SCHEMA)
    track = _track_frame(peds)
    if track.height == 0:
        return pl.DataFrame(schema=PEDESTRIAN_GEOMETRY_SCHEMA)

    first_x_m = pl.col("x_m").first()
    first_y_m = pl.col("y_m").first()
    chord_x_m = pl.col("x_m").last() - first_x_m
    chord_y_m = pl.col("y_m").last() - first_y_m
    chord_m = (chord_x_m.pow(2) + chord_y_m.pow(2)).sqrt()
    lateral_m = ((pl.col("x_m") - first_x_m) * (chord_y_m / chord_m) - (pl.col("y_m") - first_y_m) * (chord_x_m / chord_m)).abs()
    return (
        track.group_by("ped_id")
        .agg(
            n_points=pl.len(),
            first_t_ns=pl.col("t_ns").first(),
            last_t_ns=pl.col("t_ns").last(),
            path_length_m=(pl.col("x_m").diff().pow(2) + pl.col("y_m").diff().pow(2)).sqrt().sum(),
            chord_m=chord_m,
            max_displacement_m=((pl.col("x_m") - first_x_m).pow(2) + (pl.col("y_m") - first_y_m).pow(2)).sqrt().max(),
            lateral_deviation_m=pl.when(chord_m > DEFLECTION_MIN_CHORD_M).then(lateral_m.mean()).otherwise(None).alias("lateral_deviation_m"),
        )
        .select("ped_id", "n_points", "first_t_ns", "last_t_ns", "path_length_m", "chord_m", "max_displacement_m", "lateral_deviation_m")
    )


def lateral_deflection_sums(geometry: pl.DataFrame) -> tuple[float, int]:
    """Sum of the per-pedestrian mean lateral deviations and the number of pedestrians they average over."""
    if geometry.height == 0:
        return 0.0, 0
    qualifying = geometry.filter((pl.col("n_points") >= PEDESTRIAN_MIN_POINTS) & pl.col("lateral_deviation_m").is_not_null())
    total = 0.0 if qualifying.height == 0 else float(qualifying.select(pl.col("lateral_deviation_m").sum()).item())
    return total, geometry.height


def velocity_delay_value(geometry: pl.DataFrame, duration_s: float | None) -> float:
    """Relative slowdown of the mean traveled distance against the nominal desired speed, 0 when undefined."""
    if duration_s is None or duration_s <= 0.0:
        return 0.0
    traveled = geometry.filter(pl.col("n_points") > 1)
    if traveled.height == 0:
        return 0.0
    mean_distance_m = float(traveled.select(pl.col("path_length_m").mean()).item())
    mean_speed_mps = mean_distance_m / duration_s
    if mean_speed_mps <= 0.0:
        return 0.0
    return max(0.0, 1.0 - mean_speed_mps / NOMINAL_DESIRED_SPEED_MPS)


def round_trips_completed(geometry: pl.DataFrame, peds: pl.DataFrame) -> int:
    """Completed away-and-return rounds of the pedestrians with enough samples to define one."""
    if geometry.height == 0:
        return 0
    qualifying_ids = geometry.filter((pl.col("n_points") >= PEDESTRIAN_MIN_POINTS) & (pl.col("max_displacement_m") > ROUND_TRIP_MIN_DISPLACEMENT_M)).get_column("ped_id")
    if len(qualifying_ids) == 0:
        return 0
    displacement_m = ((pl.col("x_m") - pl.col("x_m").first()).pow(2) + (pl.col("y_m") - pl.col("y_m").first()).pow(2)).sqrt().over("ped_id")
    trajectory = _track_frame(peds).filter(pl.col("ped_id").is_in(qualifying_ids.implode())).with_columns(displacement_m=displacement_m)
    completed = pedestrian_round_trips_frame(trajectory)
    if completed.height == 0:
        return 0
    return int(completed.select(pl.col("round_trips_completed").sum()).item())


class PedestrianDisturbanceCalculator(BaseMetricCalculator):
    """Per-pedestrian deflection and patrol rounds, and their pedestrian mean at EPISODE scope."""

    NAME = "pedestrian_disturbance"
    CATEGORY = "social"
    SCOPE = MetricScope.PEDESTRIAN
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.PEDESTRIAN, MetricScope.EPISODE})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["proxemics"]
    REQUIRED_TOPICS = ["pedestrian_tracks"]

    UNITS = {
        "ped_path_deflection_m": "m",
        "ped_velocity_delay_ratio": "",
        "ped_round_trips_completed": "",
    }

    PRIMARY_OUTPUTS = ["ped_path_deflection_m"]
    OUTPUT_DIRECTIONS = {"ped_path_deflection_m": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ped_path_deflection_m",
            "ped_velocity_delay_ratio",
            "ped_round_trips_completed",
        ]

    @staticmethod
    def _compute_trajectory_deflection(actual_coords: np.ndarray, reference_coords: np.ndarray) -> float:
        """Mean distance of the recorded points to their nearest reference point, ignoring non-finite rows."""
        if len(actual_coords) == 0 or len(reference_coords) == 0:
            return 0.0
        valid_actual = actual_coords[~np.isnan(actual_coords).any(axis=1)]
        valid_reference = reference_coords[~np.isnan(reference_coords).any(axis=1)]
        if len(valid_actual) == 0 or len(valid_reference) == 0:
            return 0.0
        actual_m = np.ascontiguousarray(valid_actual[:, :2], dtype=np.float64)
        reference_m = np.ascontiguousarray(valid_reference[:, :2], dtype=np.float64)
        if np.isfinite(reference_m).all() and np.isfinite(actual_m).all():
            distances, _ = cKDTree(reference_m).query(actual_m, k=1)
            return float(np.mean(distances))
        dense = np.hypot(
            actual_m[:, None, 0] - reference_m[None, :, 0],
            actual_m[:, None, 1] - reference_m[None, :, 1],
        )
        return float(np.mean(np.min(dense, axis=1)))

    def _pedestrian_outputs(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: typing.Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Deflection, delay ratio and rounds of the single pedestrian of a PEDESTRIAN slice."""
        peds = self.native_ped_frame(episode)
        if peds is None or peds.height == 0:
            return dict.fromkeys(self.output_keys())
        geometry = pedestrian_geometry_frame(peds)
        if geometry.height == 0:
            return {
                "ped_path_deflection_m": 0.0,
                "ped_velocity_delay_ratio": 0.0,
                "ped_round_trips_completed": 0,
            }
        span_ns = int(geometry.select(pl.col("last_t_ns").max()).item()) - int(geometry.select(pl.col("first_t_ns").min()).item())
        prior_deflection = prior_results.get("ped_path_deflection_m")
        return {
            "ped_path_deflection_m": None if prior_deflection is None else float(prior_deflection),
            "ped_velocity_delay_ratio": round(velocity_delay_value(geometry, span_ns / 1e9), SUMMED_VALUE_DECIMALS),
            "ped_round_trips_completed": round_trips_completed(geometry, peds),
        }

    def _episode_outputs(self, episode: AlignedEpisodeBundle) -> dict[str, typing.Any]:
        """Pedestrian mean of the disturbance geometry, the historical episode-wide value."""
        peds = self.native_ped_frame(episode)
        if peds is None or peds.height == 0:
            return dict.fromkeys(self.output_keys())
        geometry = pedestrian_geometry_frame(peds)
        if geometry.height == 0:
            return {
                "ped_path_deflection_m": 0.0,
                "ped_velocity_delay_ratio": 0.0,
                "ped_round_trips_completed": 0,
            }
        total_m, num_agents = lateral_deflection_sums(geometry)
        return {
            "ped_path_deflection_m": round(total_m / num_agents, SUMMED_VALUE_DECIMALS),
            "ped_velocity_delay_ratio": round(
                velocity_delay_value(geometry, self._episode_duration_s(episode)),
                SUMMED_VALUE_DECIMALS,
            ),
            "ped_round_trips_completed": round_trips_completed(geometry, peds) / num_agents,
        }

    @staticmethod
    def _episode_duration_s(episode: AlignedEpisodeBundle) -> float | None:
        """Wall seconds the episode's aligned rows cover, the denominator the delay ratio used."""
        if episode.data is None or episode.data.height < 2:
            return None
        span_ns = episode.data.select(pl.col("t_ns").max() - pl.col("t_ns").min()).item()
        duration_s = float(span_ns) / 1e9
        return duration_s if duration_s > 0.0 else None

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
        """EPISODE scope: the pedestrian mean of the per-pedestrian geometry."""
        del prior_results
        return self._episode_outputs(episode)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: typing.Mapping[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """PEDESTRIAN computes the slice's own pedestrian; EPISODE reduces them to their mean."""
        scope = MetricScope.EPISODE if scope_slice is None else scope_slice.scope
        if scope == MetricScope.PEDESTRIAN:
            return self._pedestrian_outputs(episode, prior_results)
        if scope == MetricScope.EPISODE:
            return self._episode_outputs(episode)
        return dict.fromkeys(self.output_keys())
