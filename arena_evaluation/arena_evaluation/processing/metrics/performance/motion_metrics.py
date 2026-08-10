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
    REQUIRED_TOPICS = ["odom"]
    
    UNITS = {
        "velocity": "m/s",
        "velocity_mean": "m/s",
        "velocity_max": "m/s",
        "acceleration": "m/s²",
        "acceleration_mean": "m/s²",
        "jerk": "m/s³",
        "jerk_mean": "m/s³",
    }

    PRIMARY_OUTPUTS = ["velocity_mean", "velocity_max", "jerk_mean"]
    
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
            
        import polars as pl
        if len(episode.data) > 0:
            episode.data = episode.data.filter(
                pl.col("vel_linear").is_not_null() & ~pl.col("vel_linear").is_nan() &
                pl.col("vel_angular").is_not_null() & ~pl.col("vel_angular").is_nan()
            )
            
        if len(episode.data) > 0:
            vel_linear = episode.data["vel_linear"].to_numpy()
            vel_angular = episode.data["vel_angular"].to_numpy()
        else:
            vel_linear = np.array([], dtype=np.float64)
            vel_angular = np.array([], dtype=np.float64)
        
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
            
        time_ns = episode.data["time_ns"].to_numpy()
        dt = np.diff(time_ns) / 1e9
        dt = np.where(dt == 0.0, 1e-6, dt)

        acceleration = np.diff(vel_abs) / dt
        
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
            
        jerk = np.diff(acceleration) / dt[:-1]
        
        return {
            "velocity": vel_abs.tolist(),
            "velocity_mean": float(np.mean(vel_abs)),
            "velocity_max": float(np.max(vel_abs)),
            "acceleration": acceleration.tolist(),
            "acceleration_mean": float(np.mean(np.abs(acceleration))),
            "jerk": jerk.tolist(),
            "jerk_mean": float(np.mean(np.abs(jerk))),
        }
