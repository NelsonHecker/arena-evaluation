from __future__ import annotations
import typing
import numpy as np
import polars as pl

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class ClearanceMetricsCalculator(BaseMetricCalculator):
    """
    Clearance metrics: minimum distance to obstacles and pedestrians.

    - Obstacle clearance from the NATIVE scan topic (full multi-rate
      standard): scan_min minus the robot radius. Frames where nothing is
      within the sensor's range (scan_min >= range_max) are treated as NO
      DETECTION, not as far-away obstacles (range sentinel fix, 2026-08-12).
    - Pedestrian clearance is the effective edge-to-edge distance
      d_eff = min_dist - (r_robot + r_ped) (Arena Evaluation 3.0 zone
      standard).
    - clearance_timeseries is the per-frame combined minimum on the odom
      frame axis (native scan/peds sampled by backward-asof, 100 ms).
    """

    NAME = "clearance_metrics"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("scan", "peds")]  # At least one of scan or peds required

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
        scan_df = topics.get("scan")
        peds_df = topics.get("peds")
        has_scan = scan_df is not None and "scan_min" in scan_df.columns
        has_peds = peds_df is not None and "peds_positions" in peds_df.columns

        if not has_scan and not has_peds:
            return {k: None for k in self.output_keys()}

        pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        robot_radius = self.robot_params.robot_radius
        N = len(pos_x)
        if N == 0:
            return {k: None for k in self.output_keys()}

        odom_times_ns = episode.data["time_ns"].to_numpy()
        tol = 100_000_000

        # ── Native scan sampled onto the odom axis ──
        scan_min_arr = None
        scan_max_arr = None
        if has_scan:
            s_df = scan_df.sort("time_ns")
            joined = (
                pl.DataFrame({"time_ns": odom_times_ns})
                .join_asof(
                    s_df.select(["time_ns", "scan_min", "scan_range_max"]),
                    on="time_ns",
                    strategy="backward",
                    tolerance=tol,
                )
                .sort("time_ns")
            )
            scan_min_arr = joined["scan_min"].to_numpy()
            scan_max_arr = (
                joined["scan_range_max"].to_numpy()
                if "scan_range_max" in joined.columns
                else None
            )

        # ── Native peds sampled onto the odom axis ──
        peds_positions_list = None
        num_peds_col = None
        if has_peds:
            p_df = peds_df.sort("time_ns")
            cols = ["time_ns", "peds_positions"]
            if "num_pedestrians" in p_df.columns:
                cols.append("num_pedestrians")
            joined = (
                pl.DataFrame({"time_ns": odom_times_ns})
                .join_asof(
                    p_df.select(cols),
                    on="time_ns",
                    strategy="backward",
                    tolerance=tol,
                )
                .sort("time_ns")
            )
            peds_positions_list = joined["peds_positions"].to_list()
            if "num_pedestrians" in joined.columns:
                num_peds_col = joined["num_pedestrians"].to_numpy()

        obstacle_clearances: list[float | None] = []
        ped_clearances: list[float | None] = []
        combined_clearances: list[float | None] = []

        for i in range(N):
            # ── Obstacle clearance (native scan; sentinel-aware) ──
            obs_clear = None
            if scan_min_arr is not None and i < len(scan_min_arr):
                sm = scan_min_arr[i]
                if (
                    sm is not None
                    and not np.isnan(sm)
                    and not np.isinf(sm)
                    and sm > 0
                ):
                    r_max = (
                        scan_max_arr[i]
                        if scan_max_arr is not None and i < len(scan_max_arr)
                        else None
                    )
                    # No detection: scan reads at/beyond the sensor max range
                    if r_max is None or np.isnan(r_max) or sm < float(r_max) - 1e-6:
                        obs_clear = max(0.0, float(sm) - robot_radius)
            obstacle_clearances.append(obs_clear)

            # ── Pedestrian clearance (edge-to-edge) ──
            ped_clear = None
            if peds_positions_list is not None and i < len(peds_positions_list):
                peds_raw = peds_positions_list[i]
                peds_arr = self._parse_peds(
                    peds_raw,
                    num_peds_col[i] if num_peds_col is not None else None,
                )
                if peds_arr.shape[0] > 0:
                    rx, ry = pos_x[i], pos_y[i]
                    dx = peds_arr[:, 0] - rx
                    dy = peds_arr[:, 1] - ry
                    dists = np.sqrt(dx**2 + dy**2)
                    min_dist = float(np.min(dists))
                    ped_clear = max(0.0, min_dist - robot_radius - self._PED_RADIUS)
            ped_clearances.append(ped_clear)

            # ── Combined ──
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

    @staticmethod
    def _parse_peds(peds_raw, num_peds_hint=None):
        """Parse flat ped position list into (N, 2) or (N, 3) array."""
        if not peds_raw or len(peds_raw) == 0:
            return np.empty((0, 2))
        if isinstance(peds_raw, str):
            import ast
            try:
                peds_raw = ast.literal_eval(peds_raw)
            except Exception:
                return np.empty((0, 2))
        arr = np.array(peds_raw, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 2))
        if arr.ndim == 1:
            if num_peds_hint and num_peds_hint > 0:
                if num_peds_hint * 3 == len(arr):
                    arr = arr.reshape(-1, 3)
                elif num_peds_hint * 2 == len(arr):
                    arr = arr.reshape(-1, 2)
            else:
                if len(arr) % 3 == 0:
                    arr = arr.reshape(-1, 3)
                elif len(arr) % 2 == 0:
                    arr = arr.reshape(-1, 2)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.empty((0, 2))
        return arr
