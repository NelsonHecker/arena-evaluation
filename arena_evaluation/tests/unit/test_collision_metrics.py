import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator

def test_collision_amount():
    df = pl.DataFrame({
        "scan_ranges": [
            [1.0, 2.0], # no collision
            [0.1, 2.0], # collision 1
            [0.15, 2.0], # still collision 1
            [1.0, 2.0], # no collision
            [0.1, 2.0]  # collision 2
        ]
    })
    
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0) # radius 0.2
    calc = CollisionMetricsCalculator(params)
    
    prior = {"time_to_goal": 10.0}
    results = calc.calculate(episode, prior)
    
    assert results["collision_amount"] == 2
    assert results["result"] == "GOAL_REACHED"
    assert results["success"] == True

def test_collision_timeout_and_fail():
    df = pl.DataFrame({
        "scan_ranges": [[0.1, 2.0]] * 5 # lots of collisions
    })
    
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)
    
    # Check timeout takes precedence or collision
    results_timeout = calc.calculate(episode, {"time_to_goal": 200.0})
    assert results_timeout["result"] == "TIMEOUT"
    assert results_timeout["success"] == False
