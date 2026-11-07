import json
import pathlib

import polars as pl
import pytest

pytest.importorskip("mcap_ros2.writer")

from mcap_ros2.writer import Writer

from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.pipeline import _robot_phases

_PARAMETER_DEFS = """
================================================================================
MSG: diagnostic_msgs/KeyValue
string key
string value
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: rcl_interfaces/Parameter
string name
rcl_interfaces/ParameterValue value
================================================================================
MSG: rcl_interfaces/ParameterValue
uint8 type
bool bool_value
int64 integer_value
float64 double_value
string string_value
byte[] byte_array_value
bool[] bool_array_value
int64[] integer_array_value
float64[] double_array_value
string[] string_array_value
"""

_RECORD_FIELDS = """uint32 episode_id
string world
int64 seed
string task_layer
diagnostic_msgs/KeyValue[] modes
string[] robots
uint8 outcome_state
string outcome_info
string goal_uuid
float32 goal_dist_start
float32 goal_dist_min
float32 path_length
builtin_interfaces/Time start_time
string[] integrity
string conditions
{extra}rcl_interfaces/Parameter[] params
rcl_interfaces/Parameter[] human_params
"""

_PHASES = {
    "jackal": {
        "phases": [{"goto": [10.0, 12.0, 0.0], "tolerance_radius": 0.5}, {"goto": [10.0, 17.0, 1.57], "tolerance_radius": 0.5}],
        "conditions": [],
        "map_poses": [[15.0, 17.0, 0.0], [15.0, 22.0, 1.57]],
    }
}


def _records(tmp_path: pathlib.Path, payloads: list[str | None]) -> pl.DataFrame:
    with_phases = payloads[0] is not None
    bag = tmp_path / "run.mcap"
    with bag.open("wb") as f:
        writer = Writer(f)
        fields = _RECORD_FIELDS.format(extra="string phases\n" if with_phases else "")
        schema = writer.register_msgdef("arena_runtime_msgs/msg/EpisodeRecord", fields + _PARAMETER_DEFS)
        for i, payload in enumerate(payloads):
            msg = {
                "episode_id": 7,
                "world": "test_world",
                "seed": 42,
                "task_layer": "standard",
                "modes": [],
                "robots": ["jackal"],
                "outcome_state": 1 + i,
                "outcome_info": "",
                "goal_uuid": "",
                "goal_dist_start": 5.0,
                "goal_dist_min": 1.0,
                "path_length": 6.2,
                "start_time": {"sec": 1, "nanosec": 0},
                "integrity": [],
                "conditions": "",
                "params": [],
                "human_params": [],
            }
            if with_phases:
                msg["phases"] = payload
            writer.write_message("/env_0/arena_env_node/state/episode", schema, msg, log_time=i + 1, publish_time=i + 1)
        writer.finish()
    out = tmp_path / "topics"
    out.mkdir()
    MCAPReader(bag).read(out)
    return pl.read_parquet(out / "episode_lifecycle.parquet")


def test_phases_of_the_episode_record_reach_the_robot(tmp_path: pathlib.Path) -> None:
    records = _records(tmp_path, ["", json.dumps(_PHASES)])
    assert records["phases_yaml"].to_list() == ["", json.dumps(_PHASES)]
    assert _robot_phases(records.sort("t_ns"), "env_0_jackal") == _PHASES["jackal"]
    assert _robot_phases(records.sort("t_ns"), "env_0_burger") is None


def test_recording_from_before_phases_reads_none_for_the_robot(tmp_path: pathlib.Path) -> None:
    records = _records(tmp_path, [None, None])
    assert records["phases_yaml"].to_list() == ["", ""]
    assert records["outcome_state"].to_list() == [1, 2]
    assert _robot_phases(records.sort("t_ns"), "env_0_jackal") is None
