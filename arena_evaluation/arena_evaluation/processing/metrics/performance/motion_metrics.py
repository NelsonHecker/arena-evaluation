from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class MotionMetricsCalculator(BaseMetricCalculator):
    """
    Calculates velocity, acceleration, and jerk.
    
    Metrics:
    - velocity: Real velocity of the robot over time
    - velocity_mean: Average velocity
    - velocity_max: Maximum velocity
    - acceleration: Difference of velocities (dv/dt approximation)
    - acceleration_mean: Average acceleration
    - jerk: Rate at which acceleration changes (da/dt approximation)
    - jerk_mean: Average jerk
    """
    
    NAME = "motion_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "velocity",
            "velocity_mean",
            "velocity_max",
            "acceleration",
            "acceleration_mean",
            "jerk",
            "jerk_mean",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        if episode.data is None or "vel_linear" not in episode.data.columns:
            return {k: None for k in self.output_keys()}
            
        vel_linear = episode.data["vel_linear"].to_numpy()
        vel_angular = episode.data["vel_angular"].to_numpy()
        
        vel_abs = np.abs(vel_linear)
        
        N = len(vel_abs)
        if N < 2:
            return {
                "velocity": vel_abs.tolist(),
                "velocity_mean": float(np.mean(vel_abs)) if N > 0 else 0.0,
                "velocity_max": float(np.max(vel_abs)) if N > 0 else 0.0,
                "acceleration": [],
                "acceleration_mean": 0.0,
                "jerk": [],
                "jerk_mean": 0.0,
            }
            
        acceleration = np.diff(vel_abs)
        
        if N < 3:
            return {
                "velocity": vel_abs.tolist(),
                "velocity_mean": float(np.mean(vel_abs)),
                "velocity_max": float(np.max(vel_abs)),
                "acceleration": acceleration.tolist(),
                "acceleration_mean": float(np.mean(np.abs(acceleration))),
                "jerk": [],
                "jerk_mean": 0.0,
            }
            
        jerk = np.diff(acceleration)
        
        return {
            "velocity": vel_abs.tolist(),
            "velocity_mean": float(np.mean(vel_abs)),
            "velocity_max": float(np.max(vel_abs)),
            "acceleration": acceleration.tolist(),
            "acceleration_mean": float(np.mean(np.abs(acceleration))),
            "jerk": jerk.tolist(),
            "jerk_mean": float(np.mean(np.abs(jerk))),
        }
