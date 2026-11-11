"""Unit tests for full-identity fleet keys, self exclusion and per-actor radii."""

from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from arena_evaluation.processing.pipeline import _fleet_poses
from arena_evaluation.storage.schemas import TopicBundle

_POSE_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "pos_x": pl.Float64,
        "pos_y": pl.Float64,
        "yaw": pl.Float64,
    }
)


def _poses() -> pl.DataFrame:
    """Two judge-pose samples of one robot."""
    return pl.DataFrame(
        {
            "t_ns": np.array([0, 1_000_000_000], dtype=np.int64),
            "pos_x": [1.0, 2.0],
            "pos_y": [3.0, 4.0],
            "yaw": [0.0, 0.1],
        },
        schema=_POSE_SCHEMA,
    )


def test_same_named_robots_in_two_envs_stay_two_fleet_actors() -> None:
    robots = {
        "env_0_robot_1": TopicBundle(robot_task_pose=_poses()),
        "env_1_robot_1": TopicBundle(robot_task_pose=_poses()),
        "env_0_robot_2": TopicBundle(robot_task_pose=_poses()),
    }

    fleet = _fleet_poses(robots, (None, None), ego_robot_id="env_0_robot_1", actor_radius_m=0.35)

    assert set(fleet) == {"env_1_robot_1", "env_0_robot_2"}
    assert_series_equal(fleet["env_1_robot_1"]["radius_m"], pl.Series("radius_m", [0.35, 0.35], dtype=pl.Float64))


def test_fleet_poses_exclude_the_evaluated_robot_by_full_identity() -> None:
    robots = {
        "env_0_robot_1": TopicBundle(robot_task_pose=_poses()),
        "env_1_robot_1": TopicBundle(robot_task_pose=_poses()),
    }

    fleet = _fleet_poses(robots, (None, None), ego_robot_id="env_0_robot_1", actor_radius_m=0.25)

    assert set(fleet) == {"env_1_robot_1"}
