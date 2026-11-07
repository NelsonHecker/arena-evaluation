"""Unit tests for TopicBundle.available() and the pipeline's native-topic collection."""

import pytest

pytest.importorskip("polars")

import polars as pl
from arena_evaluation.processing.pipeline import _collect_native_topics
from arena_evaluation.storage.schemas import TopicBundle


def _odometry() -> pl.DataFrame:
    return pl.DataFrame({"t_ns": [0, 1], "env_id": ["env_0"] * 2, "robot_id": ["env_0_jackal"] * 2, "odom_x": [0.0, 1.0]})


def test_topic_bundle_available_reports_only_populated_fields():
    schedule = pl.DataFrame({"t_ns": [0], "phase_label": ["a"], "phase_kind": ["linear"]})
    bundle = TopicBundle(robot_odom=_odometry(), robot_characterization_schedule=schedule)

    assert bundle.available() == {"robot_odom", "robot_characterization_schedule"}


def test_collect_native_topics_excludes_shared_frames_but_keeps_the_schedule():
    schedule = pl.DataFrame({"t_ns": [0], "phase_label": ["a"], "phase_kind": ["linear"], "vx_target": [1.0]})
    bundle = TopicBundle(
        robot_odom=_odometry(),
        robot_characterization_schedule=schedule,
        tf=pl.DataFrame({"t_ns": [0], "parent_frame": ["map"], "child_frame": ["base"]}),
        tf_static=pl.DataFrame({"t_ns": [0], "parent_frame": ["map"], "child_frame": ["base"]}),
        world_semantics=pl.DataFrame({"t_ns": [0], "entity_name": ["x"]}),
        episode_lifecycle=pl.DataFrame({"t_ns": [0], "episode_id": [0]}),
    )

    topics = _collect_native_topics(bundle)

    assert set(topics) == {"robot_odom", "robot_characterization_schedule"}


def test_collect_native_topics_keeps_an_empty_gesture_table():
    gestures = pl.DataFrame(schema={"t_ns": pl.Int64, "ped_id": pl.String, "clip": pl.String})
    interactions = pl.DataFrame(schema={"t_ns": pl.Int64, "interaction_id": pl.Int64})

    topics = _collect_native_topics(TopicBundle(robot_odom=_odometry(), pedestrian_gestures=gestures, pedestrian_interactions=interactions))

    assert "pedestrian_gestures" in topics
    assert len(topics["pedestrian_gestures"]) == 0
    assert "pedestrian_interactions" not in topics
