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

def test_collision_amount_split_by_kind():
    df = pl.DataFrame({
        "collision_event": [0, 2, 2, 0, 1, 1],
        "collision_wall": [0, 1, 1, 0, 0, 0],
        "collision_static": [0, 1, 1, 0, 0, 0],
        "collision_pedestrian": [0, 0, 0, 0, 1, 1],
    })

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_amount"] == 2
    assert results["collision_amount_wall"] == 1
    assert results["collision_amount_static"] == 1
    assert results["collision_amount_pedestrian"] == 1

def test_collision_amount_split_by_kind_unknown():
    df = pl.DataFrame({
        "collision_event": [0, 2, 0],
        "collision_wall": [None, None, None],
        "collision_static": [None, None, None],
        "collision_pedestrian": [None, None, None],
    })

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_amount"] == 1
    assert results["collision_amount_wall"] is None
    assert results["collision_amount_static"] is None
    assert results["collision_amount_pedestrian"] is None

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


def test_collision_obstacles_recorded():
    df = pl.DataFrame({
        "collision_event": [0, 2, 1],
        "collision_obstacle_ids": [[], ["<wall>", "chair_1"], ["ped_3"]],
    })

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_obstacles"] == ["<wall>", "chair_1", "ped_3"]


def test_collision_obstacles_absent():
    df = pl.DataFrame({"collision_event": [0, 1]})

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_obstacles"] == []
