from __future__ import annotations

import json
import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("task_generator_msgs.msg")

import pyarrow.parquet as pq
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.serialization import serialize_message
from task_generator_msgs.msg import EpisodeRecord

from arena_evaluation.processing.mcap_reader import MCAPReader

_TOPIC = "/arena/env_0/task_generator_node/state/episode"


def _record_rows(tmp_path: pathlib.Path, msg: EpisodeRecord) -> list[dict]:
    bag = tmp_path / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, _TOPIC, "task_generator_msgs/msg/EpisodeRecord", "cdr"))
    writer.write(_TOPIC, serialize_message(msg), 1_000_000_000)
    writer.close()
    MCAPReader(bag).read(tmp_path / "out")
    return pq.read_table(tmp_path / "out" / "env_0" / "episode_record.parquet").to_pylist()


def test_episode_record_keeps_world_seed_and_accepted_human_params(tmp_path: pathlib.Path) -> None:
    msg = EpisodeRecord(episode_id=4, world="hospital_1", seed=107, outcome_state=EpisodeRecord.RUNNING)
    msg.human_params = [
        Parameter(name="humansim.robots_visible", value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=False)),
        Parameter(name="humansim.global_planner.resolution", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=0.1)),
    ]
    (row,) = _record_rows(tmp_path, msg)
    assert (row["world"], row["seed"]) == ("hospital_1", 107)
    assert json.loads(row["human_params"]) == {"humansim.robots_visible": False, "humansim.global_planner.resolution": 0.1}
