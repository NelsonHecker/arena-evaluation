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


def test_an_empty_odometry_table_aligns_nothing():
    empty = pl.DataFrame(schema={"t_ns": pl.Int64, "env_id": pl.String, "robot_id": pl.String, "odom_x": pl.Float64})

    assert TopicAligner().align(TopicBundle(robot_odom=empty)) is None


def test_a_bundle_without_odometry_aligns_nothing():
    assert TopicAligner().align(TopicBundle()) is None


def test_an_aligned_frame_carries_the_odometry_under_the_metric_column_names():
    aligned = TopicAligner().align(TopicBundle(robot_odom=_odometry([1000], [1.0])))

    assert {"time_ns", "pos_x", "pos_y", "yaw", "vel_linear", "vel_angular"} <= set(aligned.columns)
    assert "odom_x" not in aligned.columns
