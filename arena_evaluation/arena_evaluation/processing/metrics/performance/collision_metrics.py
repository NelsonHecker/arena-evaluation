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
    REQUIRED_TOPICS = []
    
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
                    
        # Check nav2 collision_monitor_state topic
        if episode.data is not None and "action_type" in episode.data.columns:
            # action_type == 1 means STOP (which is triggered when inside a stop polygon)
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

        # Also check collision_events topic if aligned
        if episode.data is not None and "collision_event" in episode.data.columns:
            # Count times it transitions from 0 events to > 0 events
            events_count = episode.data["collision_event"].to_numpy()
            
            # Note: mcap_reader.py now stores the length of the events array, but
            # because of join_asof, some values may be null/NaN before the first message.
            # We treat nulls as 0.
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
