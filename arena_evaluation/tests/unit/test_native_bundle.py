from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.native_bundle import NativeBundle
from arena_evaluation.storage.schemas import TopicBundle

SEC = 1_000_000_000


def _track(rows: list[tuple[int, float, float, float, float, float]]) -> pl.DataFrame:
    """Ground-truth track rows: (t_ns, x, y, yaw, speed_mps, omega_z)."""
    return pl.DataFrame(
        {
            "t_ns": [r[0] for r in rows],
            "x": [r[1] for r in rows],
            "y": [r[2] for r in rows],
            "yaw": [r[3] for r in rows],
            "speed_mps": [r[4] for r in rows],
            "omega_z": [r[5] for r in rows],
        }
    )


def test_ground_truth_track_is_the_clock_and_the_pose() -> None:
    bundle = TopicBundle(robot_tracks=_track([(0, 1.0, 2.0, 0.5, 0.3, 0.1), (SEC, 3.0, 4.0, 0.6, 0.4, 0.2)]))

    aligned = NativeBundle().align(bundle)

    assert aligned is not None
    frame = aligned.collect() if isinstance(aligned, pl.LazyFrame) else aligned
    assert frame["time_ns"].to_list() == [0, SEC]
    assert frame["pos_x"].to_list() == [1.0, 3.0]
    assert frame["pos_y"].to_list() == [2.0, 4.0]
    assert frame["yaw"].to_list() == [0.5, 0.6]
    assert frame["vel_linear"].to_list() == [0.3, 0.4]
    assert frame["vel_angular"].to_list() == [0.1, 0.2]


def test_ground_truth_aliases_mirror_the_pose_columns() -> None:
    bundle = TopicBundle(robot_tracks=_track([(0, 1.0, 2.0, 0.5, 0.3, 0.1)]))

    aligned = NativeBundle().align(bundle)

    frame = aligned if isinstance(aligned, pl.DataFrame) else aligned.collect()
    assert frame["pos_x_gt"].to_list() == frame["pos_x"].to_list()
    assert frame["pos_y_gt"].to_list() == frame["pos_y"].to_list()
    assert frame["yaw_gt"].to_list() == frame["yaw"].to_list()


def test_a_recording_without_odometry_still_evaluates() -> None:
    """Wheel odometry is not a clock: ground truth alone is enough to build the bundle."""
    bundle = TopicBundle(robot_tracks=_track([(0, 1.0, 2.0, 0.5, 0.3, 0.1)]), robot_odom=None)

    aligned = NativeBundle().align(bundle)

    assert aligned is not None
    frame = aligned if isinstance(aligned, pl.DataFrame) else aligned.collect()
    assert frame.height == 1


def test_secondary_tables_join_backwards_within_the_tolerance() -> None:
    scan = pl.DataFrame({"t_ns": [0, 200_000_000], "scan_min_m": [1.5, 2.5], "range_max_m": [9.0, 8.0], "n_valid_rays": [10, 11]})
    bundle = TopicBundle(robot_tracks=_track([(0, 0.0, 0.0, 0.0, 0.0, 0.0), (300_000_000, 0.0, 0.0, 0.0, 0.0, 0.0)]), robot_scan_summary=scan)

    aligned = NativeBundle(tolerance_ns=100_000_000).align(bundle)

    frame = aligned if isinstance(aligned, pl.DataFrame) else aligned.collect()
    assert frame["scan_min"].to_list() == [1.5, 2.5]


def test_window_bounds_filter_the_ground_truth_clock() -> None:
    bundle = TopicBundle(robot_tracks=_track([(0, 0.0, 0.0, 0.0, 0.0, 0.0), (SEC, 1.0, 1.0, 0.0, 0.0, 0.0), (2 * SEC, 2.0, 2.0, 0.0, 0.0, 0.0)]))

    aligned = NativeBundle().align(bundle, start_time_ns=SEC, end_time_ns=2 * SEC)

    frame = aligned if isinstance(aligned, pl.DataFrame) else aligned.collect()
    assert frame["time_ns"].to_list() == [SEC, 2 * SEC]


def test_without_ground_truth_the_bundle_is_absent() -> None:
    assert NativeBundle().align(TopicBundle()) is None
    assert NativeBundle().align(TopicBundle(robot_tracks=_track([]))) is None
