from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class CollisionMetricsCalculator(BaseMetricCalculator):
    """
    Calculates collision-related metrics and determines final episode result.
    
    Metrics:
    - collision_amount: Number of distinct collision events
    - collisions: Indices of collision events
    - result: GOAL_REACHED, COLLISION, or TIMEOUT
    - success: Boolean true if GOAL_REACHED
    """
    
    NAME = "collision_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["time_metrics"]
    
    TIMEOUT_THRESHOLD_S = 180.0
    MAX_COLLISIONS = 3
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "collision_amount",
            "collisions",
            "result",
            "success",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        # Defaults
        collision_amount = 0
        collisions = []
        result = "GOAL_REACHED"
        success = True
        
        # Determine collisions from scan data
        if episode.data is not None and "scan_ranges" in episode.data.columns:
            scan_ranges = episode.data["scan_ranges"].to_list()
            lower_bound = self.robot_params.robot_radius
            
            collisions_marker = []
            for i, scan in enumerate(scan_ranges):
                if scan is None:
                    collisions_marker.append(False)
                    continue
                    
                # Parse string rep of list if needed, else numpy array
                if isinstance(scan, str):
                    try:
                        import ast
                        arr = np.array(ast.literal_eval(scan))
                    except:
                        arr = np.array([])
                else:
                    arr = np.array(scan)
                    
                is_collision = len(arr[arr <= lower_bound]) > 0
                collisions_marker.append(is_collision)
                
                if is_collision:
                    collisions.append(i)
                    
            # Count distinct collision events (edge-triggered)
            for i in range(1, len(collisions_marker)):
                if collisions_marker[i] and not collisions_marker[i-1]:
                    collision_amount += 1
                    
        # Also check collision_events topic if aligned
        if episode.data is not None and "collision_event" in episode.data.columns:
            # If there are explicit collision events in the dataset, we can use them
            events = episode.data["collision_event"].to_numpy()
            event_count = np.sum(events != None)
            # Use max of laser-inferred and explicit events
            # This handles edge cases where laser misses a collision
            if event_count > collision_amount:
                collision_amount = int(event_count)
                
        time_to_goal = prior_results.get("time_to_goal", 0.0)
        
        if time_to_goal >= self.TIMEOUT_THRESHOLD_S:
            result = "TIMEOUT"
            success = False
        elif collision_amount >= self.MAX_COLLISIONS:
            result = "COLLISION"
            success = False
        else:
            result = "GOAL_REACHED"
            success = True
            
        return {
            "collision_amount": collision_amount,
            "collisions": collisions,
            "result": result,
            "success": success,
        }
