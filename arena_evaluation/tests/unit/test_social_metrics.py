"""Episode-scope gaze and proxemics calculators: typed nulls, 1NF series tables, no list columns."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.social.gaze import (
    SERIES_SCHEMA as GAZE_SERIES_SCHEMA,
    GazeMetricsCalculator,
)
from arena_evaluation.processing.metrics.social.proxemics import (
    PERSONAL_SPACE_SERIES_SCHEMA,
    ProxemicsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

EMPTY_EPISODE = AlignedEpisodeBundle(
    episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0
)


def _calc(cls):
    return cls(RobotParams(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0))


def test_proxemics_no_peds():
    results = _calc(ProxemicsCalculator).calculate(EMPTY_EPISODE, {})
    assert results["total_time_in_personal_space"] is None
    assert results["avg_velocity_in_personal_space"] is None
    assert results["num_pedestrians"] is None


def test_proxemics_series_table_is_flat():
    table = _calc(ProxemicsCalculator).series_table(EMPTY_EPISODE)
    assert_frame_equal(table, pl.DataFrame(schema=PERSONAL_SPACE_SERIES_SCHEMA))


def test_proxemics_has_no_nested_series_key():
    assert "time_in_personal_space" not in ProxemicsCalculator.output_keys()


def test_gaze_no_peds():
    results = _calc(GazeMetricsCalculator).calculate(EMPTY_EPISODE, {})
    assert results["total_time_looking_at_pedestrians"] is None
    assert results["total_time_looked_at_by_pedestrians"] is None


def test_gaze_series_table_is_flat():
    table = _calc(GazeMetricsCalculator).series_table(EMPTY_EPISODE)
    assert_frame_equal(table, pl.DataFrame(schema=GAZE_SERIES_SCHEMA))


def test_gaze_has_no_nested_series_key():
    assert not any(
        key.startswith("time_looking_") or key.startswith("time_looked_")
        for key in GazeMetricsCalculator.output_keys()
    )
