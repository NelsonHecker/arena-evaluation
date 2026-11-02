from __future__ import annotations

import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")
pytest.importorskip("mcap_ros2.writer")

import pyarrow.parquet as pq
from arena_humansim_msgs.msg import AgentFrame, AgentMeta, AgentState, AgentStates
from mcap_ros2.writer import Writer
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import _PEDS_SCHEMA, MCAPReader

_TOPIC = "/arena/env_0/task_generator_node/agent_states"
_META_TOPIC = "/arena/env_0/task_generator_node/agent_meta"
_FRAME = "arena_humansim_msgs/msg/AgentFrame"

# (agent_id, kind, x, y, theta, vx, vy, radius, desired_velocity, policy_idx, policy, agent_type)
_AGENTS = [
    (3, AgentState.KIND_HUMAN, 1.5, -2.0, 0.75, 0.3, 0.4, 0.35, 1.25, 0, "sfm", "adult"),
    (99, AgentState.KIND_ROBOT, 4.0, 4.0, 0.0, 1.0, 0.0, 0.3, 0.0, -1, "", ""),
    (7, AgentState.KIND_HUMAN, -0.5, 3.25, -1.5, -0.2, 0.1, 0.4, 0.75, 1, "orca", "child"),
]

_ROW = {
    "time_ns": 2_000_000_000,
    "peds_frame_id": "map",
    "num_pedestrians": 2,
    "peds_names": ["3", "7"],
    "peds_positions": [1.5, -2.0, 0.0, -0.5, 3.25, 0.0],
    "peds_headings": [0.75, -1.5],
    "peds_twists": [0.3, 0.4, 0.0, -0.2, 0.1, 0.0],
    "peds_ids": [3, 7],
    "peds_animation_states": [0, 0],
    "peds_interaction_ids": [-1, -1],
    "peds_interaction_types": [0, 0],
    "peds_radii": [0.35, 0.4],
    "peds_desired_speeds": [1.25, 0.75],
    "peds_policies": ["sfm", "orca"],
    "peds_agent_types": ["adult", "child"],
    "peds_gait_phases": [None, None],
    "peds_gait_cadences": [None, None],
}


def _nested() -> AgentStates:
    msg = AgentStates()
    msg.header.frame_id = "map"
    msg.header.stamp.sec = 2
    for agent_id, kind, x, y, theta, vx, vy, radius, desired, _, policy, agent_type in _AGENTS:
        agent = AgentState(agent_id=agent_id, kind=kind, radius=radius, desired_velocity=desired, policy=policy, agent_type=agent_type)
        agent.pose.x, agent.pose.y, agent.pose.theta = x, y, theta
        agent.velocity.x, agent.velocity.y = vx, vy
        msg.agents.append(agent)
    return msg


def _flat() -> AgentFrame:
    msg = AgentFrame()
    msg.header.frame_id = "map"
    msg.header.stamp.sec = 2
    columns = [list(column) for column in zip(*_AGENTS, strict=True)]
    msg.agent_id, msg.kind, msg.x, msg.y, msg.theta, msg.vx, msg.vy, msg.radius, msg.desired_velocity, msg.policy_idx = columns[:10]
    return msg


def _meta() -> AgentMeta:
    msg = AgentMeta()
    msg.policies = ["sfm", "orca"]
    msg.agent_id = [agent[0] for agent in _AGENTS]
    msg.name = ["" for _ in _AGENTS]
    msg.handedness = ["" for _ in _AGENTS]
    msg.agent_type = [agent[11] for agent in _AGENTS]
    return msg


def _write(tmp_path: pathlib.Path, name: str, messages: list[tuple[str, str, object, int]]) -> pathlib.Path:
    bag = tmp_path / name
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    for i, (topic, type_name) in enumerate(dict.fromkeys((topic, type_name) for topic, type_name, _, _ in messages)):
        writer.create_topic(rosbag2_py.TopicMetadata(i, topic, type_name, "cdr"))
    for topic, _, msg, t in messages:
        writer.write(topic, serialize_message(msg), t)
    writer.close()
    out_dir = tmp_path / f"{name}_out"
    MCAPReader(bag).read(out_dir)
    return out_dir


def test_agent_states_nested_and_flat_layouts_read_into_identical_peds(tmp_path: pathlib.Path) -> None:
    nested = _write(tmp_path, "nested", [(_TOPIC, "arena_humansim_msgs/msg/AgentStates", _nested(), 2_000_000_000)])
    flat = _write(tmp_path, "flat", [(_META_TOPIC, "arena_humansim_msgs/msg/AgentMeta", _meta(), 1_000_000_000), (_TOPIC, _FRAME, _flat(), 2_000_000_000)])
    nested_rows = pq.read_table(nested / "env_0" / "peds.parquet").to_pylist()
    flat_rows = pq.read_table(flat / "env_0" / "peds.parquet").to_pylist()
    assert nested_rows == flat_rows == [_ROW]


def _robot_only() -> AgentFrame:
    msg = AgentFrame()
    msg.header.frame_id = "map"
    msg.header.stamp.sec = 1
    msg.agent_id, msg.kind, msg.x, msg.y, msg.theta, msg.vx, msg.vy = [99], [AgentState.KIND_ROBOT], [4.0], [4.0], [0.0], [1.0], [0.0]
    return msg


def test_agent_frames_without_humans_before_the_first_flush_keep_the_peds_schema(tmp_path: pathlib.Path) -> None:
    messages: list[tuple[str, str, object, int]] = [(_TOPIC, _FRAME, _robot_only(), 1_000_000_000 + i) for i in range(10_000)]
    messages.append((_TOPIC, _FRAME, _flat(), 2_000_000_000))
    out_dir = _write(tmp_path, "late_humans", messages)

    table = pq.read_table(out_dir / "env_0" / "peds_physics.parquet")
    assert table.schema.equals(_PEDS_SCHEMA)
    assert table.num_rows == 10_001
    assert table.slice(0, 1).to_pylist()[0]["peds_positions"] == []
    # without a latched agent_meta in the bag, policies and agent types stay unknown
    assert table.slice(10_000).to_pylist() == [{**_ROW, "peds_policies": [None, None], "peds_agent_types": [None, None]}]


def _flat_with_interactions() -> AgentFrame:
    msg = _flat()
    msg.animation_state = [1, 0, 2]
    msg.interaction_id = [4, -1, -1]
    msg.interaction_type = [10, 0, 0]
    return msg


def test_agent_frames_keep_the_animation_and_interaction_arrays_of_the_humans(tmp_path: pathlib.Path) -> None:
    out_dir = _write(tmp_path, "flat_interactions", [(_TOPIC, _FRAME, _flat_with_interactions(), 2_000_000_000)])
    (row,) = pq.read_table(out_dir / "env_0" / "peds.parquet").to_pylist()
    assert row["peds_ids"] == [3, 7]
    assert row["peds_animation_states"] == [1, 2]
    assert row["peds_interaction_ids"] == [4, -1]
    assert row["peds_interaction_types"] == [10, 0]


def test_agent_meta_reads_an_empty_type_as_adult(tmp_path: pathlib.Path) -> None:
    meta = _meta()
    meta.agent_type = ["", "", "child"]
    out_dir = _write(tmp_path, "empty_type", [(_META_TOPIC, "arena_humansim_msgs/msg/AgentMeta", meta, 1_000_000_000), (_TOPIC, _FRAME, _flat(), 2_000_000_000)])
    (row,) = pq.read_table(out_dir / "env_0" / "peds.parquet").to_pylist()
    assert row["peds_agent_types"] == ["adult", "child"]


def test_teleport_only_humans_read_a_null_policy_in_both_layouts(tmp_path: pathlib.Path) -> None:
    nested_msg = _nested()
    nested_msg.agents[0].policy = ""
    flat_msg = _flat()
    flat_msg.policy_idx = [-1, -1, 1]
    nested = _write(tmp_path, "teleport_nested", [(_TOPIC, "arena_humansim_msgs/msg/AgentStates", nested_msg, 2_000_000_000)])
    flat = _write(tmp_path, "teleport_flat", [(_META_TOPIC, "arena_humansim_msgs/msg/AgentMeta", _meta(), 1_000_000_000), (_TOPIC, _FRAME, flat_msg, 2_000_000_000)])
    (nested_row,) = pq.read_table(nested / "env_0" / "peds.parquet").to_pylist()
    (flat_row,) = pq.read_table(flat / "env_0" / "peds.parquet").to_pylist()
    assert nested_row["peds_policies"] == flat_row["peds_policies"] == [None, "orca"]


def test_nested_agent_states_read_an_unset_type_as_null(tmp_path: pathlib.Path) -> None:
    msg = _nested()
    msg.agents[0].agent_type = ""
    out_dir = _write(tmp_path, "unset_type", [(_TOPIC, "arena_humansim_msgs/msg/AgentStates", msg, 2_000_000_000)])
    (row,) = pq.read_table(out_dir / "env_0" / "peds.parquet").to_pylist()
    assert row["peds_agent_types"] == [None, "child"]


def test_agent_frame_gait_reads_zero_as_null(tmp_path: pathlib.Path) -> None:
    msg = _flat()
    msg.gait_phase = [0.0, 0.0, 2.0]
    msg.gait_cadence = [0.0, 0.0, 1.25]
    out_dir = _write(tmp_path, "gait_zero", [(_TOPIC, _FRAME, msg, 2_000_000_000)])
    (row,) = pq.read_table(out_dir / "env_0" / "peds.parquet").to_pylist()
    assert row["peds_gait_phases"] == [None, 2.0]
    assert row["peds_gait_cadences"] == [None, 1.25]


def test_agent_meta_with_mismatched_type_lengths_leaves_types_null(tmp_path: pathlib.Path) -> None:
    meta = _meta()
    meta.agent_type = ["adult", "child"]
    out_dir = _write(tmp_path, "mismatched_types", [(_META_TOPIC, "arena_humansim_msgs/msg/AgentMeta", meta, 1_000_000_000), (_TOPIC, _FRAME, _flat(), 2_000_000_000)])
    (row,) = pq.read_table(out_dir / "env_0" / "peds.parquet").to_pylist()
    assert row["peds_policies"] == ["sfm", "orca"]
    assert row["peds_agent_types"] == [None, None]


_SEP = "\n" + "=" * 80 + "\n"
_TIME = "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec"
_HEADER = "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id"
# AgentMeta as recorded before agent_type existed, and an AgentFrame with the fields the reader needs for policies
_OLD_META = "string[] policies\nint32[] agent_id"
_OLD_FRAME = _SEP.join(["std_msgs/Header header\nint32[] agent_id\nfloat64[] x\nfloat64[] y\nfloat64[] theta\nfloat64[] vx\nfloat64[] vy\nuint8[] kind\nint16[] policy_idx", _HEADER, _TIME])


def test_agent_meta_without_agent_types_keeps_the_policies(tmp_path: pathlib.Path) -> None:
    bag = tmp_path / "older_meta.mcap"
    with bag.open("wb") as fh:
        writer = Writer(fh)
        meta_schema = writer.register_msgdef("arena_humansim_msgs/msg/AgentMeta", _OLD_META)
        frame_schema = writer.register_msgdef(_FRAME, _OLD_FRAME)
        writer.write_message(_META_TOPIC, meta_schema, {"policies": ["sfm", "orca"], "agent_id": [3, 99, 7]}, log_time=1_000_000_000)
        writer.write_message(
            _TOPIC,
            frame_schema,
            {
                "header": {"stamp": {"sec": 2, "nanosec": 0}, "frame_id": "map"},
                "agent_id": [3, 99, 7],
                "x": [1.5, 4.0, -0.5],
                "y": [-2.0, 4.0, 3.25],
                "theta": [0.75, 0.0, -1.5],
                "vx": [0.3, 1.0, -0.2],
                "vy": [0.4, 0.0, 0.1],
                "kind": [AgentState.KIND_HUMAN, AgentState.KIND_ROBOT, AgentState.KIND_HUMAN],
                "policy_idx": [0, -1, 1],
            },
            log_time=2_000_000_000,
        )
        writer.finish()
    out_dir = tmp_path / "older_meta_out"
    MCAPReader(bag).read(out_dir)
    (row,) = pq.read_table(out_dir / "env_0" / "peds.parquet").to_pylist()
    assert row["peds_ids"] == [3, 7]
    assert row["peds_policies"] == ["sfm", "orca"]
    assert row["peds_agent_types"] == [None, None]
