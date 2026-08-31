import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _episode_endpoints, _episode_window


def _aligned(with_gt: bool) -> pl.DataFrame:
    cols = {
        "time_ns": [0, 1_000_000_000],
        "pos_x": [0.05, 3.0],
        "pos_y": [0.02, 1.0],
        "yaw": [0.5, 0.6],
    }
    if with_gt:
        cols |= {"pos_x_gt": [28.45, 31.0], "pos_y_gt": [9.8, 10.5], "yaw_gt": [1.4, 1.5]}
    return pl.DataFrame(cols)


def test_start_prefers_ground_truth_over_odom():
    # Odom starts wherever the robot's odometry happened to be, ground truth is map frame.
    start, goal = _episode_endpoints(_aligned(with_gt=True), None)
    assert start == [28.45, 9.8, 1.4]
    assert goal == [31.0, 10.5, 1.5]


def test_start_falls_back_to_odom_without_ground_truth():
    start, goal = _episode_endpoints(_aligned(with_gt=False), None)
    assert start == [0.05, 0.02, 0.5]
    assert goal == [3.0, 1.0, 0.6]


def test_goal_is_the_last_planned_pose_when_a_plan_exists():
    plan = pl.DataFrame({"time_ns": [0], "poses_x": [[1.0, 18.9]], "poses_y": [[1.0, 17.95]], "poses_yaw": [[0.0, -1.2]]})
    start, goal = _episode_endpoints(_aligned(with_gt=True), plan.lazy())
    assert start == [28.45, 9.8, 1.4]
    assert goal == [18.9, 17.95, -1.2]


def test_plans_after_the_aligned_window_do_not_define_the_goal():
    # The next episode's first plan can land in the tail of this recording.
    plan = pl.DataFrame(
        {
            "time_ns": [500_000_000, 5_000_000_000],
            "poses_x": [[1.0, 18.9], [0.0, 11.2]],
            "poses_y": [[1.0, 17.95], [0.0, 2.05]],
            "poses_yaw": [[0.0, -1.2], [0.0, -2.4]],
        }
    )
    _, goal = _episode_endpoints(_aligned(with_gt=True), plan)
    assert goal == [18.9, 17.95, -1.2]


def test_episode_window_spans_running_to_terminal_record():
    # A latched terminal record from the previous episode can precede this episode's RUNNING row.
    record = pl.DataFrame({"time_ns": [50, 100, 900], "outcome_state": [2, 1, 4]})
    assert _episode_window(record.lazy()) == (100, 900)
    assert _episode_window(None) == (None, None)
    assert _episode_window(pl.DataFrame({"time_ns": []})) == (None, None)
