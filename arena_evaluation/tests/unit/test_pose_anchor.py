import polars as pl

from arena_evaluation.processing.pose_anchor import DENSE_GAP_NS, PoseSource, classify, resolve_pose_source
from arena_evaluation.storage.schemas import TopicBundle

SEC = 1_000_000_000
DT_NS = 33_333_333


def _tracks(stamps: list[int]) -> pl.DataFrame:
    return pl.DataFrame({"t_ns": stamps, "x": [float(index) for index in range(len(stamps))], "y": [0.0] * len(stamps), "pose_source": ["tf_gt"] * len(stamps)})


def _dense(n: int = 100) -> pl.DataFrame:
    return _tracks([index * DT_NS for index in range(n)])


def _odometry(n: int = 100) -> pl.DataFrame:
    return pl.DataFrame({"t_ns": [index * DT_NS for index in range(n)], "odom_x": [float(index) for index in range(n)], "odom_y": [0.0] * n})


def test_dense_ground_truth_is_the_tf_source_passed_through_unchanged():
    tracks = _dense()

    frame, source = resolve_pose_source(TopicBundle(robot_tracks=tracks), (None, None))

    assert source == PoseSource("tf", 100)
    assert frame.equals(tracks)


def test_sparse_ground_truth_is_kept_raw():
    tracks = _tracks([0, 5 * SEC])

    frame, source = resolve_pose_source(TopicBundle(robot_tracks=tracks), (None, None))

    assert source == PoseSource("tf_sparse", 2)
    assert frame.equals(tracks)


def test_a_robot_without_ground_truth_has_no_trajectory():
    frame, source = resolve_pose_source(TopicBundle(), (None, None))

    assert frame is None
    assert source == PoseSource("absent", 0)


def test_odometry_never_stands_in_for_missing_ground_truth():
    frame, source = resolve_pose_source(TopicBundle(robot_odom=_odometry()), (None, None))

    assert frame is None
    assert source == PoseSource("absent", 0)


def test_the_episode_window_bounds_the_track_on_log_time():
    tracks = _dense(100)
    window = (int(tracks["t_ns"][10]), int(tracks["t_ns"][20]))

    frame, source = resolve_pose_source(TopicBundle(robot_tracks=tracks), window)

    assert len(frame) == 11
    assert source == PoseSource("tf", 11)


def test_classify_reads_the_median_gap_of_the_log_time():
    assert classify(_dense()) == "dense"
    assert classify(_dense().head(1)) == "sparse"
    assert classify(None) == "absent"
    assert classify(_dense().head(0)) == "absent"


def test_a_stream_a_second_apart_is_sparse():
    assert classify(_tracks([index * SEC for index in range(10)])) == "sparse"


def test_the_dense_threshold_is_inclusive_at_the_median_gap():
    at_threshold = _tracks([index * DENSE_GAP_NS for index in range(5)])
    just_over = _tracks([index * (DENSE_GAP_NS + 1) for index in range(5)])

    assert classify(at_threshold) == "dense"
    assert classify(just_over) == "sparse"
