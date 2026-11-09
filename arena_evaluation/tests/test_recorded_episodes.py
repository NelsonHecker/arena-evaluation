from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arena_evaluation.processing.recorded_episodes import RecordedEpisodes
from musohu_bag import BEAMS, COUNT, cmd_vel, pose, scan_ranges, velocity

TOPICS = "episodes/episode_000/topics"
ODOM_COLUMNS = ["time_ns", "pos_x", "pos_y", "yaw", "vel_linear", "vel_angular"]
SCAN_COLUMNS = ["scan_min", "scan_range_max", "n_rays"]
CMD_COLUMNS = ["linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"]


def test_metadata(converted: Path):
    episodes = RecordedEpisodes(converted)
    assert len(episodes) == 1
    assert episodes.episode(0).planner == "human"
    assert episodes.metadata(0).robot_model == ["human"]
    assert list(episodes.robots(0)) == ["env_0_human"]
    frames = episodes.frames(0)
    assert frames.height == COUNT
    assert frames.columns == ODOM_COLUMNS + SCAN_COLUMNS + CMD_COLUMNS


def test_frames_carry_scan_pose_velocity_and_command_per_step(converted: Path):
    frames = RecordedEpisodes(converted).frames(0)
    assert frames["scan_min"][5] == pytest.approx(scan_ranges(5)[1])
    assert (frames["pos_x"][5], frames["pos_y"][5], frames["yaw"][5]) == pytest.approx(pose(5))
    assert (frames["vel_linear"][5], frames["vel_angular"][5]) == pytest.approx((velocity(5)[0], velocity(5)[2]))
    assert (frames["linear_x"][5], frames["angular_z"][5]) == pytest.approx(cmd_vel(4))
    assert frames["linear_x"][0] is None


def test_frames_match_extracted_topics(converted: Path):
    import polars as pl
    from polars.testing import assert_series_equal

    frames = RecordedEpisodes(converted).frames(0)
    odom = pl.read_parquet(converted / TOPICS / "robot_odom.parquet").sort("t_ns")
    scan = pl.read_parquet(converted / TOPICS / "robot_scan_summary.parquet").sort("t_ns")
    assert_series_equal(frames["pos_x"], odom["odom_x"], check_names=False)
    assert_series_equal(frames["pos_y"], odom["odom_y"], check_names=False)
    assert_series_equal(frames["yaw"], odom["odom_yaw"], check_names=False)
    assert_series_equal(frames["scan_min"], scan["scan_min_m"], check_names=False)
    assert_series_equal(frames["scan_range_max"], scan["range_max_m"], check_names=False)


def test_missing_episode_raises(converted: Path):
    with pytest.raises(IndexError, match="no episode 3"):
        RecordedEpisodes(converted).frames(3)


def test_unknown_robot_lists_the_recorded_ones(converted: Path):
    with pytest.raises(KeyError, match="env_0_human"):
        RecordedEpisodes(converted).frames(0, robot="env_0_jackal")
