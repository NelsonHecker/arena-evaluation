from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

ROUND_TRIP_REACH_SHARE: typing.Final[float] = 0.75
ROUND_TRIP_RETURN_SHARE: typing.Final[float] = 0.25
ROUND_TRIP_MIN_DISPLACEMENT_M: typing.Final[float] = 0.5


def pedestrian_round_trips_frame(trajectory: pl.DataFrame) -> pl.DataFrame:
    """Completed patrol rounds per pedestrian: an away reach followed by a return, from displacement extremes."""
    reach_m = ROUND_TRIP_REACH_SHARE * pl.col("max_displacement_m")
    returned_m = ROUND_TRIP_RETURN_SHARE * pl.col("max_displacement_m")
    events = (
        trajectory.join(trajectory.group_by("ped_id").agg(max_displacement_m=pl.col("displacement_m").max()), on="ped_id")
        .filter(pl.col("max_displacement_m") > ROUND_TRIP_MIN_DISPLACEMENT_M)
        .filter((pl.col("displacement_m") >= reach_m) | (pl.col("displacement_m") <= returned_m))
        .with_columns(away=pl.col("displacement_m") >= reach_m)
        .with_columns(next_away=pl.col("away").shift(-1).over("ped_id"))
    )
    completed = events.filter(pl.col("away") & ~pl.col("next_away").fill_null(True))
    return completed.group_by("ped_id").len().rename({"len": "round_trips_completed"})


class PedestrianActivityCalculator(BaseMetricCalculator):
    """Pedestrian lifetime activity metrics: distance traveled, mean speed and patrol rounds."""

    NAME = "pedestrian_activity"
    CATEGORY = "social"
    SCOPE = MetricScope.PEDESTRIAN
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.PEDESTRIAN})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["pedestrian_tracks"]

    UNITS = {
        "total_distance_m": "m",
        "mean_speed_mps": "m/s",
        "round_trips_completed": "",
    }
    PRIMARY_OUTPUTS = ["total_distance_m", "mean_speed_mps"]
    OUTPUT_DIRECTIONS = {
        "total_distance_m": "higher",
        "mean_speed_mps": "higher",
        "round_trips_completed": "higher",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "total_distance_m",
            "mean_speed_mps",
            "round_trips_completed",
        ]

    def _calculate_for_target(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        target_id: str | None = None,
    ) -> dict[str, typing.Any]:
        ped_tracks = self.native_topics(episode).get("pedestrian_tracks")
        if ped_tracks is not None and isinstance(ped_tracks, pl.LazyFrame):
            ped_tracks = ped_tracks.collect()

        if ped_tracks is None or len(ped_tracks) == 0:
            return {
                "total_distance_m": None,
                "mean_speed_mps": None,
                "round_trips_completed": 0,
            }

        resolved_target = target_id if target_id is not None else self.metrics_config.get("target_agent_id")
        if resolved_target is not None:
            ped_tracks = ped_tracks.filter(pl.col("ped_id") == str(resolved_target))

        if len(ped_tracks) == 0:
            return {
                "total_distance_m": None,
                "mean_speed_mps": None,
                "round_trips_completed": 0,
            }

        tracks = ped_tracks.sort("t_ns").with_columns(derived_speed_mps=pl.col("speed_mps").fill_null((pl.col("vx").pow(2) + pl.col("vy").pow(2)).sqrt()))
        dx = tracks.get_column("x").diff().drop_nulls()
        dy = tracks.get_column("y").diff().drop_nulls()
        step_dist = np.hypot(dx.to_numpy(), dy.to_numpy())
        total_distance_m = float(np.sum(step_dist)) if len(step_dist) > 0 else 0.0

        speeds = tracks.get_column("derived_speed_mps").drop_nulls()
        mean_speed_mps = float(speeds.mean()) if len(speeds) > 0 else None

        spawn_x = float(tracks.get_column("x")[0])
        spawn_y = float(tracks.get_column("y")[0])
        x_arr = tracks.get_column("x").to_numpy().astype(np.float64)
        y_arr = tracks.get_column("y").to_numpy().astype(np.float64)
        displacements = np.hypot(x_arr - spawn_x, y_arr - spawn_y)
        max_disp = float(np.max(displacements)) if len(displacements) > 0 else 0.0

        round_trips = 0
        if max_disp > ROUND_TRIP_MIN_DISPLACEMENT_M:
            reach_m = ROUND_TRIP_REACH_SHARE * max_disp
            return_m = ROUND_TRIP_RETURN_SHARE * max_disp
            disp_df = pl.DataFrame({"disp": displacements}).filter((pl.col("disp") >= reach_m) | (pl.col("disp") <= return_m)).with_columns(is_away=pl.col("disp") >= reach_m).with_columns(next_away=pl.col("is_away").shift(-1))
            completed = disp_df.filter(pl.col("is_away") & ~pl.col("next_away").fill_null(True))
            round_trips = len(completed)

        return {
            "total_distance_m": total_distance_m,
            "mean_speed_mps": mean_speed_mps,
            "round_trips_completed": round_trips,
        }

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        return self._calculate_for_target(episode, prior_results)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        target_id = scope_slice.slice_id if scope_slice is not None and scope_slice.scope == MetricScope.PEDESTRIAN else None
        return self._calculate_for_target(episode, prior_results, target_id=target_id)
