import numpy as np
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.performance.path_metrics import PathMetricsCalculator

def test_path_length_straight():
    # Straight line along X axis
    odom_df = pl.DataFrame({
        "time_ns": [1, 2, 3, 4],
        "pos_x": [0.0, 1.0, 2.0, 3.0],
        "pos_y": [0.0, 0.0, 0.0, 0.0],
        "yaw": [0.0, 0.0, 0.0, 0.0]
    })
    
    episode = AlignedEpisodeBundle(episode_id=1, data=odom_df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = PathMetricsCalculator(params)
    
    results = calc.calculate(episode, {})
    
    assert np.isclose(results["path_length"], 3.0)
    assert np.isclose(results["curvature_mean"], 0.0)
    assert np.isclose(results["roughness_mean"], 0.0)
    assert np.isclose(results["angle_over_length"], 0.0)

def test_path_length_stationary():
    # Stationary robot
    odom_df = pl.DataFrame({
        "time_ns": [1, 2, 3, 4],
        "pos_x": [0.0, 0.0, 0.0, 0.0],
        "pos_y": [0.0, 0.0, 0.0, 0.0],
        "yaw": [0.0, 0.0, 0.0, 0.0]
    })
    
    episode = AlignedEpisodeBundle(episode_id=1, data=odom_df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = PathMetricsCalculator(params)
    
    results = calc.calculate(episode, {})
    
    assert np.isclose(results["path_length"], 0.0)
    assert np.isclose(results["angle_over_length"], 0.0)
