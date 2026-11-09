"""Motion aggregates in the metric row per sample kinematics in the 1NF motion series table."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.metrics.performance.motion_metrics import SERIES_SCHEMA, MotionMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _episode(*, vel_linear: list[float | None] | None = None, pos_x_gt: list[float] | None = None, sample_count: int | None = None) -> AlignedEpisodeBundle:
    """Episode frame at 1 s spacing, with a twist stream or a ground-truth pose track."""
    if pos_x_gt is not None:
        count = len(pos_x_gt)
        data = pl.DataFrame(
            {
                "time_ns": pl.Series("time_ns", [index * SEC for index in range(count)], dtype=pl.Int64),
                "pos_x_gt": pl.Series("pos_x_gt", pos_x_gt, dtype=pl.Float64),
                "pos_y_gt": pl.Series("pos_y_gt", [0.0] * count, dtype=pl.Float64),
                "yaw_gt": pl.Series("yaw_gt", [0.0] * count, dtype=pl.Float64),
            }
        )
    else:
        count = len(vel_linear) if vel_linear is not None else (sample_count or 0)
        data = pl.DataFrame(
            {
                "time_ns": pl.Series("time_ns", [index * SEC for index in range(count)], dtype=pl.Int64),
                "vel_linear": pl.Series("vel_linear", vel_linear if vel_linear is not None else [None] * count, dtype=pl.Float64),
            }
        )
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[])


def _calculator() -> MotionMetricsCalculator:
    return MotionMetricsCalculator(RobotParams(robot_radius=0.2))


def test_output_keys_keep_only_the_aggregates():
    expected = ["velocity_mean", "velocity_max", "acceleration_mean", "acceleration_max", "jerk_mean", "jerk_max"]

    assert MotionMetricsCalculator.output_keys() == expected
    assert set(MotionMetricsCalculator.UNITS) <= set(expected)


def test_constant_velocity_series_has_zero_acceleration_and_jerk():
    table = _calculator().series_table(_episode(vel_linear=[1.0, 1.0, 1.0, 1.0]), run_id="run", robot_id="jackal_0")

    expected = pl.DataFrame(
        {
            "run_id": ["run"] * 4,
            "episode_id": [1] * 4,
            "robot_id": ["jackal_0"] * 4,
            "t_ns": [0, SEC, 2 * SEC, 3 * SEC],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "speed_mps": [1.0, 1.0, 1.0, 1.0],
            "acceleration_mps2": [None, 0.0, 0.0, 0.0],
            "jerk_mps3": [None, None, 0.0, 0.0],
        },
        schema=SERIES_SCHEMA,
    )

    assert_frame_equal(table, expected, check_dtypes=True, rel_tol=1e-9, abs_tol=1e-12)


def test_constant_acceleration_series_has_zero_jerk():
    calculator = _calculator()
    episode = _episode(vel_linear=[0.0, 1.0, 2.0, 3.0])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert_series_equal(table["acceleration_mps2"], pl.Series("acceleration_mps2", [None, 1.0, 1.0, 1.0], dtype=pl.Float64), check_names=False, rel_tol=1e-9)
    assert_series_equal(table["jerk_mps3"], pl.Series("jerk_mps3", [None, None, 0.0, 0.0], dtype=pl.Float64), check_names=False, rel_tol=1e-9)
    assert results["velocity_mean"] == pytest.approx(1.5)
    assert results["velocity_max"] == pytest.approx(3.0)
    assert results["acceleration_mean"] == pytest.approx(1.0)
    assert results["jerk_mean"] == pytest.approx(0.0)


def test_velocity_series_is_derived_from_ground_truth_without_a_twist_stream():
    table = _calculator().series_table(_episode(pos_x_gt=[0.0, 1.0, 2.0, 3.0]))

    assert_series_equal(table["speed_mps"], pl.Series("speed_mps", [0.0, 1.0, 1.0, 1.0], dtype=pl.Float64), check_names=False, rel_tol=1e-9)
    assert_series_equal(table["acceleration_mps2"], pl.Series("acceleration_mps2", [None, 1.0, 0.0, 0.0], dtype=pl.Float64), check_names=False, rel_tol=1e-9)


def test_series_summaries_match_the_metric_row():
    calculator = _calculator()
    episode = _episode(vel_linear=[0.0, 2.0, 1.0, 4.0])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert results["velocity_mean"] == pytest.approx(float(table["speed_mps"].mean()))
    assert results["velocity_max"] == pytest.approx(float(table["speed_mps"].max()))
    assert results["acceleration_max"] == pytest.approx(float(table["acceleration_mps2"].abs().max()))
    assert results["jerk_max"] == pytest.approx(float(table["jerk_mps3"].abs().max()))


def test_a_single_sample_has_no_acceleration_series():
    calculator = _calculator()
    episode = _episode(vel_linear=[1.0])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert table.height == 1
    assert table["acceleration_mps2"].is_null().all()
    assert table["jerk_mps3"].is_null().all()
    assert results["velocity_mean"] == pytest.approx(1.0)
    assert results["acceleration_mean"] is None
    assert results["jerk_mean"] is None


def test_two_samples_have_one_acceleration_and_no_jerk():
    calculator = _calculator()
    episode = _episode(vel_linear=[0.0, 2.0])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert_series_equal(table["acceleration_mps2"], pl.Series("acceleration_mps2", [None, 2.0], dtype=pl.Float64), check_names=False)
    assert table["jerk_mps3"].is_null().all()
    assert results["acceleration_mean"] == pytest.approx(2.0)
    assert results["jerk_mean"] is None


def test_without_an_aligned_frame_aggregates_and_series_are_empty():
    calculator = _calculator()
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])

    assert calculator.calculate(episode, {}) == dict.fromkeys(MotionMetricsCalculator.output_keys())
    assert_frame_equal(calculator.series_table(episode), pl.DataFrame(schema=SERIES_SCHEMA), check_dtypes=True)
