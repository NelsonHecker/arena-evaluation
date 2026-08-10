from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class PathEfficiencyCalculator(BaseMetricCalculator):
    """
    Calculates path efficiency.
    
    Metrics:
    - path_efficiency: Ratio of Euclidean distance (start to goal) over total path length.
                       Close to 1.0 means highly efficient.
    """
    
    NAME = "path_efficiency"
    CATEGORY = "performance"
    DEPENDS_ON = ["path_metrics"]
    REQUIRED_TOPICS = ["odom"]
    UNITS = {"path_efficiency": ""}

    PRIMARY_OUTPUTS = ["path_efficiency"]
    OUTPUT_DIRECTIONS = {"path_efficiency": "higher"}
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "path_efficiency",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        path_length = prior_results.get("path_length", 0.0)
        
        if path_length <= 1e-9 or not episode.start_pos or not episode.goal_pos:
            return {"path_efficiency": 0.0}
            
        start = np.array(episode.start_pos[:2])
        goal = np.array(episode.goal_pos[:2])
        
        euclidean_dist = np.linalg.norm(goal - start)
        
        if np.isnan(euclidean_dist) or np.isinf(euclidean_dist) or np.isnan(path_length):
            return {"path_efficiency": 0.0}
            
        try:
            efficiency = float(euclidean_dist / path_length)
            if np.isnan(efficiency):
                efficiency = 0.0
            else:
                efficiency = min(max(efficiency, 0.0), 1.0)
        except ZeroDivisionError:
            efficiency = 0.0

        
        return {
            "path_efficiency": efficiency,
        }
