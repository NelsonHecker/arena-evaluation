import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator

def test_collision_amount():
    df = pl.DataFrame({
        "collision_event": [
            0, # no collision
            2, # collision 1
            2, # still collision 1
            0, # no collision
            1  # collision 2
        ]
    })

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    prior = {"time_to_goal": 10.0}
    results = calc.calculate(episode, prior)

    assert results["collision_amount"] == 2
    assert results["result"] == "GOAL_REACHED"
    assert results["success"] == True

def test_collision_timeout_and_fail():
    df = pl.DataFrame({
        "collision_event": [0, 1, 0, 1, 0, 1, 0, 1] # 4 collisions, > MAX_COLLISIONS
    })

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results_timeout = calc.calculate(episode, {"time_to_goal": 200.0})
    assert results_timeout["result"] == "TIMEOUT"
    assert results_timeout["success"] == False

    results_fail = calc.calculate(episode, {"time_to_goal": 50.0})
    assert results_fail["result"] == "COLLISION"
    assert results_fail["success"] == False
