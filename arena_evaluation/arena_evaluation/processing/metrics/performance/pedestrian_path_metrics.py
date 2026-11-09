from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.pedestrian_frames import pedestrians_at
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class PedestrianPathMetricsCalculator(BaseMetricCalculator):
    """Calculates time-synchronized pedestrian trajectory paths."""

    NAME = "pedestrian_path_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["pedestrian_tracks"]
    REQUIRES_PEDSIM = True

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.PEDESTRIAN})

    UNITS = {"pedestrian_path": "m"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["pedestrian_path"]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:

        peds_df = self.native_ped_frame(episode)
        if episode.data is None or peds_df is None or len(peds_df) == 0:
            return {"pedestrian_path": []}

        time_ns = episode.data["time_ns"].to_numpy()
        T = len(time_ns)
        if T == 0:
            return {"pedestrian_path": []}

        paths = {ped_id: [[float("nan")] * 3 for _ in range(T)] for ped_id in peds_df["ped_id"].unique().sort().to_list()}
        row_of = {int(stamp): row for row, stamp in enumerate(time_ns)}
        present = pedestrians_at(pl.Series("time_ns", time_ns.astype(np.int64)), peds_df)
        for ped_id, stamp, x, y, z in present.select("ped_id", "time_ns", "x", "y", "z").iter_rows():
            paths[ped_id][row_of[int(stamp)]] = [x, y, z]

        for path in paths.values():
            for t in range(1, T):
                if np.isnan(path[t][0]):
                    path[t] = path[t - 1]

        valid_paths = [path for path in paths.values() if any(not np.isnan(pt[0]) for pt in path)]
        return {"pedestrian_path": valid_paths}

    def _native_paths(self, episode: AlignedEpisodeBundle) -> list[list[list[float]]]:
        """One path per pedestrian from the window's native pedestrian track, gaps forward-filled per pedestrian."""
        peds = self.native_ped_frame(episode)
        if peds is None or len(peds) == 0 or not {"t_ns", "ped_id", "x", "y"} <= set(peds.columns):
            return []
        z = pl.col("z").forward_fill().over("ped_id") if "z" in peds.columns else pl.lit(0.0, dtype=pl.Float64)
        rows = (
            peds.sort(["ped_id", "t_ns"])
            .with_columns(
                x=pl.col("x").cast(pl.Float64).forward_fill().over("ped_id"),
                y=pl.col("y").cast(pl.Float64).forward_fill().over("ped_id"),
                z=z,
            )
            .filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
        )
        if len(rows) == 0:
            return []
        grouped = (
            rows.group_by("ped_id", maintain_order=True)
            .agg(pl.concat_list([pl.col("x"), pl.col("y"), pl.col("z")]).alias("path"))
            .sort("ped_id")
        )
        return [[list(point) for point in path] for path in grouped["path"].to_list()]

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.PEDESTRIAN:
            return {"pedestrian_path": self._native_paths(episode)}
        raise ValueError(f"pedestrian_path_metrics does not serve the {scope_slice.scope.value} scope")
