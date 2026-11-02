from __future__ import annotations

import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_people_msgs.msg")

import pyarrow.parquet as pq
from arena_people_msgs.msg import Pedestrian, Pedestrians
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import MCAPReader

_TOPIC = "/arena/env_0/arena_peds"


def test_rendered_peds_keep_agent_type_and_gait_and_leave_engine_fields_null(tmp_path: pathlib.Path) -> None:
    msg = Pedestrians()
    msg.header.frame_id = "map"
    msg.pedestrians = [Pedestrian(name="env_0/Pedestrian_0", id=1, agent_type="child", gait_phase=1.5, gait_cadence=0.5)]
    bag = tmp_path / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, _TOPIC, "arena_people_msgs/msg/Pedestrians", "cdr"))
    writer.write(_TOPIC, serialize_message(msg), 1_000_000_000)
    writer.close()
    MCAPReader(bag).read(tmp_path / "out")

    (row,) = pq.read_table(tmp_path / "out" / "env_0" / "peds.parquet").to_pylist()
    assert row["peds_names"] == ["env_0/Pedestrian_0"]
    assert row["peds_agent_types"] == ["child"]
    assert (row["peds_gait_phases"], row["peds_gait_cadences"]) == ([1.5], [0.5])
    assert (row["peds_radii"], row["peds_desired_speeds"], row["peds_policies"]) == ([None], [None], [None])
