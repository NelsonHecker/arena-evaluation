from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.pedestrian_frames import nearest_pedestrian_distance
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class ClearanceMetricsCalculator(BaseMetricCalculator):
    """Computes minimum clearance distances to obstacles and pedestrians."""

    NAME = "clearance_metrics"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("robot_scan_summary", "pedestrian_tracks")]  # at least one of the two

    UNITS = {
        "min_obstacle_clearance": "m",
        "mean_obstacle_clearance": "m",
        "min_pedestrian_clearance": "m",
        "mean_pedestrian_clearance": "m",
        "clearance_timeseries": "m",
    }

    PRIMARY_OUTPUTS = ["min_obstacle_clearance", "min_pedestrian_clearance"]
    OUTPUT_DIRECTIONS = {
        "min_obstacle_clearance": "higher",
        "mean_obstacle_clearance": "higher",
        "min_pedestrian_clearance": "higher",
        "mean_pedestrian_clearance": "higher",
    }

    _PED_RADIUS = 0.3  # m

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "min_obstacle_clearance",
            "mean_obstacle_clearance",
            "min_pedestrian_clearance",
            "mean_pedestrian_clearance",
            "clearance_timeseries",
        ]

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        if episode.data is None:
            return {k: None for k in self.output_keys()}

        topics = self.native_topics(episode)
        scan_df = topics.get("robot_scan_summary")
        peds_df = self.native_ped_frame(episode)
        has_scan = scan_df is not None and "scan_min_m" in scan_df.columns
        has_peds = peds_df is not None and len(peds_df) > 0

        if not has_scan and not has_peds:
            return {k: None for k in self.output_keys()}

        pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        robot_radius = self.robot_params.robot_radius
        N = len(pos_x)
        if N == 0:
            return {k: None for k in self.output_keys()}

        odom_times_ns = episode.data["time_ns"].to_numpy()
        tol = 100_000_000

        scan_min_arr = None
        scan_max_arr = None
        if has_scan and scan_df is not None:
            s_df = scan_df.sort("t_ns")
            joined = (
                pl.DataFrame({"time_ns": odom_times_ns})
                .join_asof(
                    s_df.select(pl.col("t_ns").alias("time_ns"), "scan_min_m", "range_max_m"),
                    on="time_ns",
                    strategy="backward",
                    tolerance=tol,
                )
                .sort("time_ns")
            )
            scan_min_arr = joined["scan_min_m"].to_numpy()
            scan_max_arr = joined["range_max_m"].to_numpy()

        ped_distance = nearest_pedestrian_distance(odom_times_ns, pos_x, pos_y, peds_df, tol) if has_peds and peds_df is not None else np.full(N, np.nan)

        obstacle_clearances: list[float | None] = []
        ped_clearances: list[float | None] = []
        combined_clearances: list[float | None] = []

        for i in range(N):
            obs_clear = None
            if scan_min_arr is not None and i < len(scan_min_arr):
                sm = scan_min_arr[i]
                if sm is not None and not np.isnan(sm) and not np.isinf(sm) and sm > 0:
                    r_max = scan_max_arr[i] if scan_max_arr is not None and i < len(scan_max_arr) else None
                    # At or beyond max range means nothing was detected
                    if r_max is None or np.isnan(r_max) or sm < float(r_max) - 1e-6:
                        obs_clear = max(0.0, float(sm) - robot_radius)
            obstacle_clearances.append(obs_clear)

            ped_clear = None
            if not np.isnan(ped_distance[i]):
                ped_clear = max(0.0, float(ped_distance[i]) - robot_radius - self._PED_RADIUS)
            ped_clearances.append(ped_clear)

            if obs_clear is not None and ped_clear is not None:
                combined_clearances.append(float(min(obs_clear, ped_clear)))
            elif obs_clear is not None:
                combined_clearances.append(float(obs_clear))
            elif ped_clear is not None:
                combined_clearances.append(float(ped_clear))
            else:
                combined_clearances.append(None)

        valid_obs = [c for c in obstacle_clearances if c is not None]
        valid_ped = [c for c in ped_clearances if c is not None]

        return {
            "min_obstacle_clearance": float(np.min(valid_obs)) if valid_obs else None,
            "mean_obstacle_clearance": float(np.mean(valid_obs)) if valid_obs else None,
            "min_pedestrian_clearance": float(np.min(valid_ped)) if valid_ped else None,
            "mean_pedestrian_clearance": float(np.mean(valid_ped)) if valid_ped else None,
            "clearance_timeseries": combined_clearances,
        }
