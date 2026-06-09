from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class GazeMetricsCalculator(BaseMetricCalculator):
    """
    Calculates gaze-related social metrics.
    
    Metrics:
    - time_looking_at_peds: Steps where robot looks at a pedestrian
    - time_looking_at_peds_total: Total seconds looking at pedestrians
    - time_looked_at_by_peds: Steps where pedestrian looks at robot
    - time_looked_at_by_peds_total: Total seconds looked at by pedestrians
    """
    
    NAME = "gaze_metrics"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    
    GAZE_CONE_DEG = 5.0
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "time_looking_at_pedestrians",
            "total_time_looking_at_pedestrians",
            "time_looked_at_by_pedestrians",
            "total_time_looked_at_by_pedestrians",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        if episode.data is None or "pos_x" not in episode.data.columns or "peds_positions" not in episode.data.columns:
            return {k: None for k in self.output_keys()}
            
        pos_x = episode.data["pos_x"].to_numpy()
        pos_y = episode.data["pos_y"].to_numpy()
        yaw = episode.data["yaw"].to_numpy()
        time_ns = episode.data["time_ns"].to_numpy()
        
        peds_positions = episode.data["peds_positions"].to_list()
        # Ensure we have pedestrian headings if available, else skip
        if "peds_headings" in episode.data.columns:
            peds_headings = episode.data["peds_headings"].to_list()
        else:
            peds_headings = [None] * len(pos_x)
            
        N = len(pos_x)
        
        dt = np.diff(time_ns) / 1e9
        dt = np.append(dt, 0.0)
        
        looking_at = []
        looking_at_time = 0.0
        
        looked_at = []
        looked_at_time = 0.0
        
        cone_rad = np.radians(self.GAZE_CONE_DEG)
        
        def angle_diff(a1, a2):
            return np.pi - np.abs(np.abs(a1 - a2) - np.pi)
            
        for i in range(N):
            peds = peds_positions[i]
            headings = peds_headings[i]
            
            if not peds or len(peds) == 0:
                looking_at.append(0)
                looked_at.append(0)
                continue
                
            rx, ry, ryaw = pos_x[i], pos_y[i], yaw[i]
            
            if isinstance(peds, str):
                import ast
                try:
                    peds_arr = np.array(ast.literal_eval(peds))
                    head_arr = np.array(ast.literal_eval(headings)) if headings else np.zeros(len(peds_arr)//3)
                except:
                    peds_arr = np.array([])
                    head_arr = np.array([])
            else:
                peds_arr = np.array(peds)
                head_arr = np.array(headings) if headings is not None else np.zeros(len(peds_arr)//3)
                
            if peds_arr.ndim == 1 and len(peds_arr) % 3 == 0:
                peds_arr = peds_arr.reshape(-1, 3)
                
            if peds_arr.ndim != 2 or peds_arr.shape[1] < 2:
                looking_at.append(0)
                looked_at.append(0)
                continue
                
            dx = peds_arr[:, 0] - rx
            dy = peds_arr[:, 1] - ry
            
            # Angle from robot to pedestrian
            angle_to_ped = np.arctan2(dy, dx)
            diff_robot_to_ped = angle_diff(ryaw, angle_to_ped)
            
            is_looking = np.sum(diff_robot_to_ped <= cone_rad)
            looking_at.append(int(is_looking))
            if is_looking > 0:
                looking_at_time += dt[i]
                
            # Angle from pedestrian to robot
            angle_to_robot = np.arctan2(-dy, -dx)
            if len(head_arr) == len(peds_arr):
                diff_ped_to_robot = angle_diff(head_arr, angle_to_robot)
                is_looked_at = np.sum(diff_ped_to_robot <= cone_rad)
                looked_at.append(int(is_looked_at))
                if is_looked_at > 0:
                    looked_at_time += dt[i]
            else:
                looked_at.append(0)
                
        return {
            "time_looking_at_pedestrians": looking_at,
            "total_time_looking_at_pedestrians": looking_at_time,
            "time_looked_at_by_pedestrians": looked_at,
            "total_time_looked_at_by_pedestrians": looked_at_time,
        }
