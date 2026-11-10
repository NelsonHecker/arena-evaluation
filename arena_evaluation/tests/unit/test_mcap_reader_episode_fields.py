from __future__ import annotations

import json
import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("task_generator_msgs.msg")

import pyarrow as pa
import pyarrow.parquet as pq
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.serialization import serialize_message
from task_generator_msgs.msg import EpisodeRecord

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import PARAM, PARAM_VALUE, _msgdef, _write_bag

_TOPIC = "/arena/env_0/task_generator_node/state/episode"
TRIMMED_EPISODE_RECORD: tuple[str, str] = (
    "task_generator_msgs/msg/EpisodeRecord",
    _msgdef("uint32 episode_id\nuint8 outcome_state\nstring outcome_info\nstring goal_uuid\nrcl_interfaces/Parameter[] robots_params\nstring phases\nstring conditions", PARAM, PARAM_VALUE),
)


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
    assert pq.read_schema(tmp_path / "out" / "env_0" / "episode_record.parquet").field("seed").type == pa.int64()
    assert (row["world"], row["seed"]) == ("hospital_1", 107)
    assert json.loads(row["human_params"]) == {"humansim.robots_visible": False, "humansim.global_planner.resolution": 0.1}


def test_unresolved_episode_record_reads_world_and_seed_as_null(tmp_path: pathlib.Path) -> None:
    msg = EpisodeRecord(episode_id=4, world="", seed=-1, outcome_state=EpisodeRecord.QUEUED)
    (row,) = _record_rows(tmp_path, msg)
    assert (row["world"], row["seed"]) == (None, None)


def test_episode_record_without_world_seed_and_human_params_reads_them_as_null(tmp_path: pathlib.Path) -> None:
    record = {"episode_id": 4, "outcome_state": 1, "outcome_info": "", "goal_uuid": "", "robots_params": [], "phases": "", "conditions": ""}
    bag = _write_bag(tmp_path / "episode.mcap", [(_TOPIC, TRIMMED_EPISODE_RECORD, record, 1_000_000_000)])
    MCAPReader(bag).read(tmp_path / "out")
    (row,) = pq.read_table(tmp_path / "out" / "env_0" / "episode_record.parquet").to_pylist()
    assert (row["world"], row["seed"], row["human_params"]) == (None, None, None)
