import polars as pl

from arena_evaluation.processing.topic_aligner import TopicAligner
from arena_evaluation.storage.schemas import TopicBundle


def _odometry(stamps: list[int], xs: list[float]) -> pl.DataFrame:
    n = len(stamps)
    return pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": ["env_0"] * n,
            "robot_id": ["env_0_jackal"] * n,
            "odom_x": xs,
            "odom_y": [0.0] * n,
            "odom_yaw": [0.0] * n,
            "odom_vx": [0.0] * n,
            "odom_wz": [0.0] * n,
            "drift_error_m": [0.0] * n,
        }
    )


def _scan_summary(stamps: list[int], scan_min: list[float]) -> pl.DataFrame:
    n = len(stamps)
    return pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": ["env_0"] * n,
            "robot_id": ["env_0_jackal"] * n,
            "scan_min_m": scan_min,
            "scan_max_m": scan_min,
            "range_min_m": [0.1] * n,
            "range_max_m": [10.0] * n,
            "n_valid_rays": [4] * n,
        }
    )


def test_a_reading_joins_the_latest_one_at_or_before_each_odometry_sample():
    bundle = TopicBundle(robot_odom=_odometry([1000, 2000, 3000, 4000], [1.0, 2.0, 3.0, 4.0]), robot_scan_summary=_scan_summary([1100, 3100], [0.5, 0.6]))

    aligned = TopicAligner(tolerance_ns=1000).align(bundle)

    assert len(aligned) == 4
    assert aligned["scan_min"].to_list() == [None, 0.5, None, 0.6]


def test_readings_outside_the_tolerance_do_not_join():
    bundle = TopicBundle(robot_odom=_odometry([1000, 2000], [1.0, 2.0]), robot_scan_summary=_scan_summary([1100, 2100], [0.5, 0.6]))

    aligned = TopicAligner(tolerance_ns=0).align(bundle)

    assert aligned["scan_min"].null_count() == 2


def test_unsorted_odometry_is_aligned_in_time_order():
    bundle = TopicBundle(robot_odom=_odometry([5000, 4000, 3000, 2000, 1000], [5.0, 4.0, 3.0, 2.0, 1.0]), robot_scan_summary=_scan_summary([5000, 1000], [0.5, 0.1]))

    aligned = TopicAligner(tolerance_ns=500).align(bundle)

    assert aligned["time_ns"].to_list() == [1000, 2000, 3000, 4000, 5000]
    assert aligned["scan_min"].to_list() == [0.1, None, None, None, 0.5]


def test_a_window_that_starts_after_it_ends_aligns_nothing():
    bundle = TopicBundle(robot_odom=_odometry([1000, 2000], [1.0, 2.0]))

    assert TopicAligner(tolerance_ns=500).align(bundle, start_time_ns=3000, end_time_ns=1000) is None
