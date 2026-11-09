"""Time aggregates: the retired ``time`` sample clock is the ``t_ns`` series column every 1NF table carries."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.metrics.performance.motion_metrics import SERIES_SCHEMA as MOTION_SERIES_SCHEMA
from arena_evaluation.processing.metrics.performance.path_metrics import SERIES_SCHEMA as PATH_SERIES_SCHEMA
from arena_evaluation.processing.metrics.performance.time_metrics import TimeMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _calculator() -> TimeMetricsCalculator:
    return TimeMetricsCalculator(RobotParams(robot_radius=0.2))


def _episode(time_ns: list[int], vel_linear: list[float | None] | None = None) -> AlignedEpisodeBundle:
    columns: dict[str, pl.Series] = {"time_ns": pl.Series("time_ns", time_ns, dtype=pl.Int64)}
    if vel_linear is not None:
        columns["vel_linear"] = pl.Series("vel_linear", vel_linear, dtype=pl.Float64)
    return AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(columns), start_pos=[], goal_pos=[])


def test_output_keys_do_not_carry_the_sample_clock():
    expected = ["time_diff", "time_to_goal", "idling_time"]

    assert TimeMetricsCalculator.output_keys() == expected
    assert "time" not in TimeMetricsCalculator.output_keys()
    assert set(TimeMetricsCalculator.UNITS) <= set(expected)


def test_duration_and_idling_from_the_motion_speed_track():
    calculator = _calculator()
    episode = _episode([1 * SEC, 2 * SEC, 3 * SEC, 4 * SEC], vel_linear=[0.5, 0.0, 0.0, 0.5])

    results = calculator.calculate(episode, {})

    assert results["time_to_goal"] == pytest.approx(3.0)
    assert results["time_diff"] == 3 * SEC
    assert results["idling_time"] == pytest.approx(2.0)


def test_a_single_sample_episode_has_no_duration_and_no_idling():
    results = _calculator().calculate(_episode([SEC]), {})

    assert results == {"time_diff": 0, "time_to_goal": 0.0, "idling_time": 0.0}


def test_without_a_speed_track_idling_is_zero_and_the_duration_survives():
    calculator = _calculator()
    episode = _episode([0, SEC, 2 * SEC])

    results = calculator.calculate(episode, {})

    assert results["time_to_goal"] == pytest.approx(2.0)
    assert results["idling_time"] == 0.0


def test_without_an_aligned_frame_every_time_output_is_absent():
    calculator = _calculator()
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])

    assert calculator.calculate(episode, {}) == dict.fromkeys(TimeMetricsCalculator.output_keys())


def test_the_retired_clock_is_the_t_ns_column_of_the_series_tables():
    """Every 1NF series table already carries the sample clock as ``t_ns``, so ``time`` needs no table of its own."""
    assert "t_ns" in MOTION_SERIES_SCHEMA
    assert "t_ns" in PATH_SERIES_SCHEMA
