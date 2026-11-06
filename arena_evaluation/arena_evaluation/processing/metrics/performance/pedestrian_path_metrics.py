from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.pedestrian_frames import pedestrians_at
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class PedestrianPathMetricsCalculator(BaseMetricCalculator):
    """Calculates time-synchronized pedestrian trajectory paths."""

    NAME = "pedestrian_path_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["pedestrian_tracks"]
    REQUIRES_PEDSIM = True

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
