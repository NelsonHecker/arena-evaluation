import math

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.naturalness.trajectory import TrajectoryMetricsCalculator
from arena_evaluation.processing.metrics.performance.efficiency_metrics import PathEfficiencyCalculator
from arena_evaluation.processing.path import theta_star
from arena_evaluation.processing.path.theta_star import GeometricThetaStar
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000
OPEN_ROOM = "efficiency_test_open_room"


@pytest.fixture(autouse=True, scope="module")
def open_room_solver():
    theta_star._solver_instances[f"{OPEN_ROOM}_0.25"] = GeometricThetaStar(np.zeros((200, 200), dtype=bool), resolution=0.1, robot_radius=0.25)


def _params() -> RobotParams:
    return RobotParams(robot_radius=0.25)


def _gt(xs: list[float], ys: list[float]) -> pl.DataFrame:
    times = [i * SEC for i in range(len(xs))]
    return pl.DataFrame(
        {
            "time_ns": times,
            "t_ns": times,
            "pos_x_gt": [float(x) for x in xs],
            "pos_y_gt": [float(y) for y in ys],
            "yaw_gt": [0.0] * len(xs),
            "x": [float(x) for x in xs],
            "y": [float(y) for y in ys],
            "yaw": [0.0] * len(xs),
        }
    )


def _episode(gt: pl.DataFrame, goal: list[float], *, map_name: str | None = None, plans: pl.DataFrame | None = None) -> AlignedEpisodeBundle:
    topics = {"robot_tracks": gt, "tf_gt": gt}
    if plans is not None:
        topics["robot_plans"] = plans
    return AlignedEpisodeBundle(1, gt, [], goal, map=map_name, topics=topics)


def _straight_episode() -> AlignedEpisodeBundle:
    xs = [float(value) for value in np.linspace(0.0, 10.0, 21)]
    return _episode(_gt(xs, [0.0] * len(xs)), [10.0, 0.0], map_name=OPEN_ROOM)


def test_path_efficiency_uses_the_geodesic_reference_prior():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[3.0, 4.0])

    results = PathEfficiencyCalculator(_params()).calculate(episode, {"path_length": 10.0, "theta_star_length": 5.0})

    assert results["path_efficiency"] == pytest.approx(0.5)


def test_missing_geodesic_reference_is_null_not_a_straight_line_fallback():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[3.0, 4.0])

    results = PathEfficiencyCalculator(_params()).calculate(episode, {"path_length": 10.0})

    assert results["path_efficiency"] is None


def test_path_efficiency_reports_over_unity_without_clamping():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[3.0, 4.0])

    results = PathEfficiencyCalculator(_params()).calculate(episode, {"path_length": 4.0, "theta_star_length": 5.0})

    assert results["path_efficiency"] == pytest.approx(1.25)


def test_path_efficiency_is_null_for_a_non_positive_path_length():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[3.0, 4.0])

    results = PathEfficiencyCalculator(_params()).calculate(episode, {"path_length": 0.0, "theta_star_length": 5.0})

    assert results["path_efficiency"] is None


def test_path_efficiency_is_null_for_a_non_positive_geodesic():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[3.0, 4.0])

    results = PathEfficiencyCalculator(_params()).calculate(episode, {"path_length": 10.0, "theta_star_length": 0.0})

    assert results["path_efficiency"] is None


def test_straight_ten_metre_path_in_an_empty_map_has_efficiency_one():
    episode = _straight_episode()

    trajectory = TrajectoryMetricsCalculator(_params()).calculate(episode, {})
    efficiency = PathEfficiencyCalculator(_params()).calculate(episode, {"path_length": 10.0, "theta_star_length": trajectory["theta_star_length"]})

    assert trajectory["theta_star_length"] == pytest.approx(10.0)
    assert efficiency["path_efficiency"] == pytest.approx(1.0)


def test_ten_metre_reference_with_a_detour_has_efficiency_below_one():
    episode = _straight_episode()

    trajectory = TrajectoryMetricsCalculator(_params()).calculate(episode, {})
    efficiency = PathEfficiencyCalculator(_params()).calculate(episode, {"path_length": 12.0, "theta_star_length": trajectory["theta_star_length"]})

    assert trajectory["theta_star_length"] == pytest.approx(10.0)
    assert efficiency["path_efficiency"] == pytest.approx(10.0 / 12.0)


def test_closed_loop_without_a_recorded_route_has_no_geodesic_reference():
    gt = _gt([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    episode = _episode(gt, [0.05, 0.0], map_name=OPEN_ROOM)

    trajectory = TrajectoryMetricsCalculator(_params()).calculate(episode, {})
    efficiency = PathEfficiencyCalculator(_params()).calculate(episode, {"path_length": 2.0, "theta_star_length": trajectory["theta_star_length"]})

    assert trajectory["theta_star_length"] is None
    assert efficiency["path_efficiency"] is None
