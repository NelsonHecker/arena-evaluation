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
    - collision_amount_wall/static/pedestrian: distinct collisions per kind, None when the recording carries no kind
    - collision_obstacles: ids of every entity the robot hit
    - collisions: Indices of collision events
    - result: GOAL_REACHED, COLLISION, or TIMEOUT
    - success: Boolean true if GOAL_REACHED
    """
    
    NAME = "collision_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["time_metrics"]
    REQUIRED_TOPICS = [("collision_monitor_state", "collision_events")]
    
    UNITS = {
        "collision_amount": "",
        "collision_amount_wall": "",
        "collision_amount_static": "",
        "collision_amount_pedestrian": "",
        "collision_obstacles": "",
        "collisions": "",
        "result": "",
        "success": "",
    }

    PRIMARY_OUTPUTS = ["success", "collision_amount"]
    OUTPUT_DIRECTIONS = {"success": "higher"}
    
    TIMEOUT_THRESHOLD_S = 180.0
    MAX_COLLISIONS = 3
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "collision_amount",
            "collision_amount_wall",
            "collision_amount_static",
            "collision_amount_pedestrian",
            "collision_obstacles",
            "collisions",
            "result",
            "success",
        ]

    @staticmethod
    def _count_rising_edges(is_active: np.ndarray) -> int:
        count = 0
        if len(is_active) > 0 and is_active[0]:
            count += 1
        for i in range(1, len(is_active)):
            if is_active[i] and not is_active[i - 1]:
                count += 1
        return count

    @classmethod
    def _column_rising_edges(cls, episode: AlignedEpisodeBundle, column: str) -> int | None:
        if episode.data is None or column not in episode.data.columns:
            return None
        counts = episode.data[column].to_numpy().astype(float)
        if np.isnan(counts).all():
            return None
        return cls._count_rising_edges(np.nan_to_num(counts, nan=0.0) > 0)
        
    @staticmethod
    def _hit_obstacles(episode: AlignedEpisodeBundle) -> list[str]:
        if episode.data is None or "collision_obstacle_ids" not in episode.data.columns:
            return []
        hit: set[str] = set()
        for ids in episode.data["collision_obstacle_ids"].to_list():
            if ids is not None:
                hit.update(i for i in ids if i)
        return sorted(hit)

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
            nav2_collisions = self._count_rising_edges(is_stopped)
            
            if nav2_collisions > collision_amount:
                collision_amount = nav2_collisions

        if episode.data is not None and "collision_event" in episode.data.columns:
            events_count = episode.data["collision_event"].to_numpy()
            valid_counts = np.nan_to_num(events_count.astype(float), nan=0.0)
            arena_collisions = self._count_rising_edges(valid_counts > 0)
            
            if arena_collisions > collision_amount:
                collision_amount = arena_collisions

        collision_amount_wall = self._column_rising_edges(episode, "collision_wall")
        collision_amount_static = self._column_rising_edges(episode, "collision_static")
        collision_amount_pedestrian = self._column_rising_edges(episode, "collision_pedestrian")
        collision_obstacles = self._hit_obstacles(episode)
                
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
            "collision_amount_wall": collision_amount_wall,
            "collision_amount_static": collision_amount_static,
            "collision_amount_pedestrian": collision_amount_pedestrian,
            "collision_obstacles": collision_obstacles,
            "collisions": collisions,
            "result": result,
            "success": success,
        }
