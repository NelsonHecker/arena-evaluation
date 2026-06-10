from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class ProxemicsCalculator(BaseMetricCalculator):
    """
    Calculates proxemics metrics (distance to pedestrians).
    
    Metrics:
    - num_pedestrians: Number of pedestrians in the episode
    - time_in_personal_space: Steps where a pedestrian is within personal space
    - time_in_personal_space_total: Total seconds in personal space
    - avg_velocity_in_personal_space: Average velocity when in personal space
    
    Uses Hall's proxemics:
    Intimate space: < 0.45m
    Personal space: 0.45m - 1.2m
    """
    
    NAME = "proxemics"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = ["odom", "peds"]
    
    PERSONAL_SPACE_RADIUS = 1.2
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "num_pedestrians",
            "time_in_personal_space",
            "total_time_in_personal_space",
            "avg_velocity_in_personal_space",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        # We need peds positions and robot positions
        if episode.data is None or "pos_x" not in episode.data.columns or "peds_positions" not in episode.data.columns:
            return {k: None for k in self.output_keys()}
            
        pos_x, pos_y, _, _, _, _ = self.resolve_robot_pose(episode)
        time_ns = episode.data["time_ns"].to_numpy()
        peds_positions = episode.data["peds_positions"].to_list()
        
        velocity = prior_results.get("velocity", [])
        
        N = len(pos_x)
        if N == 0:
            return {
                "num_pedestrians": 0,
                "time_in_personal_space": [],
                "total_time_in_personal_space": 0.0,
                "avg_velocity_in_personal_space": 0.0,
            }
            
        in_personal_space_steps = []
        in_personal_space_time_s = 0.0
        vel_in_space = []
        
        # Calculate time diffs
        dt = np.diff(time_ns) / 1e9
        dt = np.append(dt, 0.0) # pad last element
        
        # Vectorized check for each step
        for i in range(N):
            peds = peds_positions[i]
            if not peds or len(peds) == 0:
                in_personal_space_steps.append(0)
                continue
                
            rx, ry = pos_x[i], pos_y[i]
            
            # Assuming peds is a flattened list or list of lists: [[x1, y1, z1], [x2, y2, z2]]
            # Convert to numpy array safely
            if isinstance(peds, str):
                import ast
                try:
                    peds_arr = np.array(ast.literal_eval(peds))
                except:
                    peds_arr = np.array([])
            else:
                peds_arr = np.array(peds)
                
            if peds_arr.size == 0:
                in_personal_space_steps.append(0)
                continue
                
            # If flattened list [x1,y1,z1, x2,y2,z2], reshape it
            if peds_arr.ndim == 1 and len(peds_arr) % 3 == 0:
                peds_arr = peds_arr.reshape(-1, 3)
            elif peds_arr.ndim == 1 and len(peds_arr) % 2 == 0:
                peds_arr = peds_arr.reshape(-1, 2)
                
            if peds_arr.ndim != 2 or peds_arr.shape[1] < 2:
                in_personal_space_steps.append(0)
                continue
                
            # Compute Euclidean distances to all peds
            dx = peds_arr[:, 0] - rx
            dy = peds_arr[:, 1] - ry
            distances = np.sqrt(dx**2 + dy**2)
            
            # Count how many are in personal space
            # Exclude intimate space if desired, but typically we just count anything < 1.2m
            in_space_count = np.sum(distances < self.PERSONAL_SPACE_RADIUS)
            
            in_personal_space_steps.append(int(in_space_count))
            
            if in_space_count > 0:
                in_personal_space_time_s += dt[i]
                if velocity and i < len(velocity):
                    vel_in_space.append(velocity[i])
                    
        avg_vel = float(np.mean(vel_in_space)) if vel_in_space else 0.0
        
        return {
            "num_pedestrians": episode.num_pedestrians,
            "time_in_personal_space": in_personal_space_steps,
            "total_time_in_personal_space": in_personal_space_time_s,
            "avg_velocity_in_personal_space": avg_vel,
        }
