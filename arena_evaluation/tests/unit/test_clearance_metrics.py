import polars as pl
import pytest
from polars.testing import assert_series_equal

from arena_evaluation.processing.metrics.performance.clearance_metrics import (
    ClearanceMetricsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
from flat_peds import flat_peds

SEC = 1_000_000_000
RADIUS = 0.25
PED_RADIUS = ClearanceMetricsCalculator._PED_RADIUS_M


def _episode(times_ns, scan=None, peds=None):
    """Robot at the origin for every sample; scan is the robot_scan_summary columns, peds one list of (x, y) per sample."""
    n = len(times_ns)
    data = pl.DataFrame(
        {
            "time_ns": times_ns,
            "pos_x": [0.0] * n,
            "pos_y": [0.0] * n,
            "yaw": [0.0] * n,
            "pos_x_gt": [0.0] * n,
            "pos_y_gt": [0.0] * n,
            "yaw_gt": [0.0] * n,
        }
    )
    topics = {"robot_tracks": pl.DataFrame({"t_ns": times_ns, "x": [0.0] * n, "y": [0.0] * n, "yaw": [0.0] * n})}
    if scan is not None:
        topics["robot_scan_summary"] = pl.DataFrame({"t_ns": times_ns, **scan})
    frames = None
    if peds is not None:
        frames = flat_peds([(t, [{"x": x, "y": y} for x, y in sample]) for t, sample in zip(times_ns, peds, strict=True)])
        topics["pedestrian_tracks"] = frames
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        num_pedestrians=1,
        topics=topics,
        peds=frames,
    )


@pytest.fixture
def calc():
    return ClearanceMetricsCalculator(RobotParams(robot_radius=RADIUS))


def _series(calc, episode, column):
    return calc.series_table(episode, run_id="run", robot_id="robot").get_column(column)


def test_no_topics_yields_all_none(calc):
    results = calc.calculate(_episode([0, SEC]), {})
    assert set(results) == set(calc.output_keys())
    assert all(v is None for v in results.values())


def test_series_table_is_flat_with_one_row_per_sample(calc):
    table = calc.series_table(_episode([0, SEC], scan={"scan_min_m": [2.0, 3.0], "range_max_m": [10.0, 10.0]}), run_id="run", robot_id="robot")
    assert table.schema == pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "obstacle_clearance_m": pl.Float64,
            "pedestrian_clearance_m": pl.Float64,
            "combined_clearance_m": pl.Float64,
        }
    )
    assert len(table) == 2
    assert table["run_id"].unique().to_list() == ["run"]


def test_obstacle_clearance_subtracts_the_robot_radius(calc):
    scan = {"scan_min_m": [2.0, 3.0], "range_max_m": [10.0, 10.0]}
    results = calc.calculate(_episode([0, SEC], scan=scan), {})
    assert results["min_obstacle_clearance"] == pytest.approx(2.0 - RADIUS)
    assert results["mean_obstacle_clearance"] == pytest.approx(2.5 - RADIUS)


def test_scan_at_max_range_is_no_detection_not_a_far_obstacle(calc):
    """The sentinel frame must not be averaged in as a huge clearance."""
    episode = _episode([0, SEC], scan={"scan_min_m": [2.0, 10.0], "range_max_m": [10.0, 10.0]})
    results = calc.calculate(episode, {})
    assert results["min_obstacle_clearance"] == pytest.approx(2.0 - RADIUS)
    assert results["mean_obstacle_clearance"] == pytest.approx(2.0 - RADIUS)
    assert _series(calc, episode, "obstacle_clearance_m")[1] is None


def test_every_frame_at_max_range_leaves_no_obstacle_clearance(calc):
    scan = {"scan_min_m": [10.0, 10.0], "range_max_m": [10.0, 10.0]}
    results = calc.calculate(_episode([0, SEC], scan=scan), {})
    assert results["min_obstacle_clearance"] is None
    assert results["mean_obstacle_clearance"] is None


def test_pedestrian_clearance_is_edge_to_edge(calc):
    results = calc.calculate(_episode([0, SEC], peds=[[(1.5, 0.0), (3.0, 0.0)], [(1.5, 0.0), (3.0, 0.0)]]), {})
    assert results["min_pedestrian_clearance"] == pytest.approx(1.5 - RADIUS - PED_RADIUS)


def test_pedestrian_clearance_is_clamped_at_zero(calc):
    results = calc.calculate(_episode([0, SEC], peds=[[(0.1, 0.0)], [(0.1, 0.0)]]), {})
    assert results["min_pedestrian_clearance"] == 0.0


def test_nearest_pedestrian_wins(calc):
    peds = [[(3.0, 0.0), (1.0, 0.0)], [(3.0, 0.0), (1.0, 0.0)]]
    results = calc.calculate(_episode([0, SEC], peds=peds), {})
    assert results["min_pedestrian_clearance"] == pytest.approx(1.0 - RADIUS - PED_RADIUS)


def test_timeseries_takes_the_minimum_of_both_sources(calc):
    episode = _episode([0, SEC], scan={"scan_min_m": [2.0, 0.5], "range_max_m": [10.0, 10.0]}, peds=[[(1.0, 0.0)], [(5.0, 0.0)]])
    calc.calculate(episode, {})
    expected = pl.Series("combined_clearance_m", [1.0 - RADIUS - PED_RADIUS, 0.5 - RADIUS], dtype=pl.Float64, nan_to_null=True)
    assert_series_equal(_series(calc, episode, "combined_clearance_m"), expected, check_names=False, rel_tol=1e-9)


def test_frames_without_pedestrians_fall_back_to_the_scan(calc):
    episode = _episode([0, SEC], scan={"scan_min_m": [2.0, 2.0], "range_max_m": [10.0, 10.0]}, peds=[[], []])
    results = calc.calculate(episode, {})
    assert results["min_pedestrian_clearance"] is None
    expected = pl.Series("combined_clearance_m", [2.0 - RADIUS, 2.0 - RADIUS], dtype=pl.Float64)
    assert_series_equal(_series(calc, episode, "combined_clearance_m"), expected, check_names=False, rel_tol=1e-9)