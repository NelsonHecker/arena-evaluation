import pytest
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.registry import MetricRegistry


def test_registry_skipping_with_available_topics():
    params = RobotParams(0.25, 0.0, 30.0)
    registry = MetricRegistry(params)

    df_odom_only = pl.DataFrame(
        {
            "t_ns": [1000, 2000, 3000],
            "pos_x": [0.0, 1.0, 2.0],
            "pos_y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "pos_x_gt": [0.0, 1.0, 2.0],
            "pos_y_gt": [0.0, 0.0, 0.0],
            "yaw_gt": [0.0, 0.0, 0.0],
            "vel_linear": [1.0, 1.0, 1.0],
            "vel_angular": [0.0, 0.0, 0.0],
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=df_odom_only, start_pos=[0.0, 0.0, 0.0], goal_pos=[2.0, 0.0, 0.0])

    available_topics = {"robot_tracks"}
    results = registry.run(episode, pedsim_available=True, available_topics=available_topics)

    assert results["path_length"] > 0
    assert results["time_to_goal"] == 2e-6  # 2000 ns
    assert results["velocity_mean"] == 1.0

    assert results["collision_amount"] is None
    assert results["collisions"] is None
    assert results["result"] is None
    assert results["success"] is None
    assert results["num_pedestrians"] is None
    assert results["total_time_in_personal_space"] is None
    assert results["total_time_looking_at_pedestrians"] is None


def test_registry_skipping_with_all_topics():
    params = RobotParams(0.25, 0.0, 30.0)
    registry = MetricRegistry(params)

    df_all = pl.DataFrame(
        {
            "t_ns": [1000, 2000, 3000],
            "pos_x": [0.0, 1.0, 2.0],
            "pos_y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "pos_x_gt": [0.0, 1.0, 2.0],
            "pos_y_gt": [0.0, 0.0, 0.0],
            "yaw_gt": [0.0, 0.0, 0.0],
            "vel_linear": [1.0, 1.0, 1.0],
            "vel_angular": [0.0, 0.0, 0.0],
            "collision_event": [0, 0, 0],
            "collision_monitor_state_action": ["none", "none", "none"],
        }
    )
    peds = pl.DataFrame(
        {
            "t_ns": [1000, 2000, 3000],
            "env_id": ["env_0"] * 3,
            "ped_id": ["ped_0"] * 3,
            "ped_name": ["ped_0"] * 3,
            "x": [10.0, 10.0, 10.0],
            "y": [10.0, 10.0, 10.0],
            "z": [0.0] * 3,
            "yaw": [0.0] * 3,
            "vx": [0.0] * 3,
            "vy": [0.0] * 3,
            "vz": [0.0] * 3,
            "speed_mps": [0.0] * 3,
            "radius_m": [0.35] * 3,
            "desired_speed": [1.0] * 3,
            "policy": [""] * 3,
            "agent_type": [""] * 3,
            "animation_state": [0] * 3,
        }
    )

    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=df_all,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[2.0, 0.0, 0.0],
        topics={"pedestrian_tracks": peds},
    )

    available_topics = {"robot_odom", "robot_tracks", "pedestrian_tracks", "collision_events", "collision_monitor_state"}
    results = registry.run(episode, pedsim_available=True, available_topics=available_topics)

    assert results["path_length"] > 0
    assert results["collision_amount"] == 0
    assert results["result"] == "GOAL_REACHED"
    assert results["success"] is True
    assert results["num_pedestrians"] == 0
    assert results["total_time_looking_at_pedestrians"] == 0.0
