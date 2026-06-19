from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class PedestrianPathMetricsCalculator(BaseMetricCalculator):
    """
    Calculates pedestrian trajectory paths over the episode.
    
    Metrics:
    - pedestrian_path: A list of independent 3D paths for each pedestrian.
      Each path is padded with NaNs to be exactly length T (the episode length)
      to guarantee perfect time synchronization in the trajectory plot slider.
    """
    
    NAME = "pedestrian_path_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["peds"]
    REQUIRES_PEDSIM = True
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return ["pedestrian_path"]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        if episode.data is None or "peds_positions" not in episode.data.columns:
            return {"pedestrian_path": []}
            
        peds_positions = episode.data["peds_positions"].to_list()
        
        T = len(peds_positions)
        if T == 0:
            return {"pedestrian_path": []}
            
        # Find maximum number of pedestrians across all timesteps
        max_peds = 0
        for step_peds in peds_positions:
            if isinstance(step_peds, list):
                max_peds = max(max_peds, len(step_peds) // 3)
                
        if max_peds == 0:
            return {"pedestrian_path": []}
            
        # Initialize paths with NaNs so they are strictly length T
        paths = [[[float('nan'), float('nan'), float('nan')] for _ in range(T)] for _ in range(max_peds)]
        
        for t, step_peds in enumerate(peds_positions):
            if not isinstance(step_peds, list):
                if t > 0:
                    for k in range(max_peds):
                        paths[k][t] = paths[k][t-1]
                continue
                
            num_peds = len(step_peds) // 3
            for k in range(num_peds):
                paths[k][t] = [step_peds[3*k], step_peds[3*k + 1], step_peds[3*k + 2]]
                
            for k in range(num_peds, max_peds):
                if t > 0:
                    paths[k][t] = paths[k][t-1]
                
        # Filter out paths that are completely entirely NaNs
        valid_paths = []
        for path in paths:
            if any(not np.isnan(pt[0]) for pt in path):
                valid_paths.append(path)
                
        return {"pedestrian_path": valid_paths}
