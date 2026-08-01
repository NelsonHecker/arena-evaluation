from __future__ import annotations

import typing
import numpy as np
import polars as pl

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class TrajectoryMetricsCalculator(BaseMetricCalculator):
    NAME = "trajectory_naturalness"
    CATEGORY = "naturalness"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("odom", "tf_gt")]

    UNITS = {
        "ade": "m",
        "fde": "m",
        "adtw": "m",
        "path_irregularity": "rad/m",
        "topological_complexity": ""
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["ade", "fde", "adtw", "path_irregularity", "topological_complexity"]

    def _load_reference_path(self, episode: "AlignedEpisodeBundle") -> tuple[np.ndarray, np.ndarray] | None:
        if episode.run is None or episode.folder_manager is None:
            return None
        
        # Determine reference run planner
        ref_planner = f"{episode.run.planner}_unobstructed_robot"
        ref_run_dir = episode.folder_manager.run_dir(
            benchmark_id=episode.run.benchmark_id,
            planner=ref_planner,
            stage=episode.run.stage
        )
        
        topics_dir = episode.folder_manager.extracted_topics_path(ref_run_dir)
        if not topics_dir.exists():
            return None
            
        try:
            from ...parquet_store import TopicParquetStore
            from ...topic_aligner import TopicAligner
            from ...episode_splitter import EpisodeSplitter
            
            bundles = TopicParquetStore.read(topics_dir)
            if not bundles:
                return None
            
            robot_bundle = (
                bundles.get(episode.robot_name)
                if episode.robot_name
                else next(iter(bundles.values()), None)
            )
            if not robot_bundle:
                return None

            aligner = TopicAligner()
            splitter = EpisodeSplitter(aligner)
            episodes = splitter.split(robot_bundle, robot_name=episode.robot_name)
            
            for ep in episodes:
                if ep.episode_id == episode.episode_id:
                    pos_x, pos_y, _, _, _, _ = self.resolve_robot_pose(ep)
                    if len(pos_x) > 0:
                        return pos_x, pos_y
            return None
        except Exception:
            return None

    def calculate(self, episode: "AlignedEpisodeBundle", prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        pos_x, pos_y, yaw, _, _, _ = self.resolve_robot_pose(episode)
        
        if len(pos_x) == 0:
            return {k: None for k in self.output_keys()}
            
        results = {k: None for k in self.output_keys()}
        
        # Calculate Path Irregularity
        if len(pos_x) > 1:
            dx = np.diff(pos_x)
            dy = np.diff(pos_y)
            distances = np.sqrt(dx**2 + dy**2)
            L = np.sum(distances)
            
            if L > 0.1:
                dyaw = np.diff(yaw)
                dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                sum_abs_dyaw = np.sum(np.abs(dyaw))
                
                start_x, start_y = pos_x[0], pos_y[0]
                goal_x, goal_y = episode.goal_pos[0], episode.goal_pos[1]
                target_angle = np.arctan2(goal_y - start_y, goal_x - start_x)
                
                rot_to_target = (target_angle - yaw[0] + np.pi) % (2 * np.pi) - np.pi
                delta_theta_min = np.abs(rot_to_target)
                
                pi = (sum_abs_dyaw - delta_theta_min) / L
                results["path_irregularity"] = float(pi)
                
        # Calculate Topological Complexity (Winding Number around pedestrians)
        ep_peds = getattr(episode, "peds", None)
        if ep_peds is not None and len(ep_peds) > 0 and len(pos_x) > 1:
            try:
                # peds dataframe has ts_iso, ped_id, pos_x, pos_y
                # episode.data has ts_iso, pos_x_gt, pos_y_gt (if available) or pos_x, pos_y
                ts_col = "time_ns" if ("time_ns" in ep_peds.columns and "time_ns" in episode.data.columns) else "ts_iso"
                df_peds = ep_peds.drop_nulls(subset=["pos_x", "pos_y", ts_col]).sort(ts_col)
                
                # prepare robot df with common names
                use_gt = "pos_x_gt" in episode.data.columns
                x_col = "pos_x_gt" if use_gt else "pos_x"
                y_col = "pos_y_gt" if use_gt else "pos_y"
                
                df_robot = episode.data.drop_nulls(subset=[x_col, y_col, ts_col]).sort(ts_col)
                
                # Join exact or asof
                joined = df_peds.join_asof(df_robot, on=ts_col, strategy="nearest")
                
                total_winding = 0.0
                for _, group in joined.group_by("ped_id"):
                    rx = group[x_col].to_numpy()
                    ry = group[y_col].to_numpy()
                    px = group["pos_x"].to_numpy()
                    py = group["pos_y"].to_numpy()
                    
                    rel_x = rx - px
                    rel_y = ry - py
                    
                    angles = np.arctan2(rel_y, rel_x)
                    if len(angles) > 1:
                        d_angles = np.diff(angles)
                        d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi
                        total_winding += np.sum(np.abs(d_angles)) / (2 * np.pi)
                        
                results["topological_complexity"] = float(total_winding)
            except Exception:
                pass
                
        # Calculate Reference-based metrics
        ref_path = self._load_reference_path(episode)
        if ref_path is not None:
            ref_x, ref_y = ref_path
            if len(ref_x) > 0 and len(pos_x) > 0:
                # FDE
                fde = np.sqrt((pos_x[-1] - ref_x[-1])**2 + (pos_y[-1] - ref_y[-1])**2)
                results["fde"] = float(fde)
                
                # ADE (Point-to-Point interpolated)
                curr_dist = np.insert(np.cumsum(np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2)), 0, 0)
                ref_dist = np.insert(np.cumsum(np.sqrt(np.diff(ref_x)**2 + np.diff(ref_y)**2)), 0, 0)
                
                if curr_dist[-1] > 0 and ref_dist[-1] > 0:
                    curr_frac = curr_dist / curr_dist[-1]
                    ref_frac = ref_dist / ref_dist[-1]
                    
                    interp_ref_x = np.interp(curr_frac, ref_frac, ref_x)
                    interp_ref_y = np.interp(curr_frac, ref_frac, ref_y)
                    
                    ade = np.mean(np.sqrt((pos_x - interp_ref_x)**2 + (pos_y - interp_ref_y)**2))
                    results["ade"] = float(ade)
                    
                # ADTW
                try:
                    from dtaidistance import dtw_ndim
                    series1 = np.column_stack((pos_x, pos_y))
                    series2 = np.column_stack((ref_x, ref_y))
                    adtw = dtw_ndim.distance(series1, series2)
                    results["adtw"] = float(adtw)
                except ImportError:
                    pass

        return results
