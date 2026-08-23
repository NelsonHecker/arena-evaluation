from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class CollisionMetricsCalculator(BaseMetricCalculator):
    """Calculates collision counts, determines episode success status, and computes SPL."""
    
    NAME = "collision_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["time_metrics", "path_metrics", "trajectory_naturalness"]
    REQUIRED_TOPICS = [("collision_monitor_state", "collision_events"), ("tf_gt", "odom")]
    
    UNITS = {
        "collision_amount": "",
        "collisions": "",
        "result": "",
        "success": "",
        "spl": "",
    }

    PRIMARY_OUTPUTS = ["success", "collision_amount", "spl"]
    OUTPUT_DIRECTIONS = {"success": "higher", "spl": "higher"}
    
    TIMEOUT_THRESHOLD_S = 180.0
    MAX_COLLISIONS = 3
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "collision_amount",
            "collisions",
            "result",
            "success",
            "spl",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        collision_amount = 0
        collisions = []
        result = "GOAL_REACHED"
        success = True
                    
        if episode.data is not None and "action_type" in episode.data.columns:
            action_types = episode.data["action_type"].to_numpy()
            is_stopped = (action_types == 1)
            
            nav2_collisions = 0
            if len(is_stopped) > 0 and is_stopped[0]:
                nav2_collisions += 1
                
            for i in range(1, len(is_stopped)):
                if is_stopped[i] and not is_stopped[i-1]:
                    nav2_collisions += 1
            
            if nav2_collisions > collision_amount:
                collision_amount = nav2_collisions

        if episode.data is not None and "collision_event" in episode.data.columns:
            events_count = episode.data["collision_event"].to_numpy()
            
            valid_counts = np.nan_to_num(events_count.astype(float), nan=0.0)
            is_collision = (valid_counts > 0)
            
            arena_collisions = 0
            if len(is_collision) > 0 and is_collision[0]:
                arena_collisions += 1
                
            for i in range(1, len(is_collision)):
                if is_collision[i] and not is_collision[i-1]:
                    arena_collisions += 1
            
            if arena_collisions > collision_amount:
                collision_amount = arena_collisions
                
        time_to_goal = prior_results.get("time_to_goal")
        
        if time_to_goal is not None and float(time_to_goal) >= self.TIMEOUT_THRESHOLD_S:
            result = "TIMEOUT"
            success = False
        elif collision_amount >= self.MAX_COLLISIONS or prior_results.get("collision_amount", 0) >= self.MAX_COLLISIONS:
            result = "COLLISION"
            success = False
        elif time_to_goal is None and "time_to_goal" in prior_results:
            result = "COLLISION" if collision_amount > 0 or prior_results.get("collision_amount", 0) > 0 else "FAILED"
            success = False
        else:
            result = "GOAL_REACHED"
            success = True

        # Calculate Success weighted by Path Length (SPL)
        # SPL = Success * (L_0 / max(L_actual, L_0))
        actual_length = prior_results.get("path_length", 0.0) or 0.0
        l0 = prior_results.get("theta_star_length")
        if l0 is None or l0 <= 0:
            if episode.start_pos and episode.goal_pos and len(episode.start_pos) >= 2 and len(episode.goal_pos) >= 2:
                l0 = float(np.linalg.norm(np.array(episode.goal_pos[:2]) - np.array(episode.start_pos[:2])))
            else:
                l0 = 0.0

        if success:
            denom = max(float(actual_length), float(l0))
            if denom > 0:
                spl = float(l0 / denom)
                spl = min(max(spl, 0.0), 1.0)
            else:
                spl = 1.0
        else:
            spl = 0.0
            
        return {
            "collision_amount": collision_amount,
            "collisions": collisions,
            "result": result,
            "success": success,
            "spl": float(spl),
        }

