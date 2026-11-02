from __future__ import annotations

import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")

import pyarrow.parquet as pq
from arena_humansim_msgs.msg import AgentFrame, AgentMeta, AgentState, AgentStates
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import _PEDS_SCHEMA, MCAPReader

_TOPIC = "/arena/env_0/task_generator_node/agent_states"
_META_TOPIC = "/arena/env_0/task_generator_node/agent_meta"
_FRAME = "arena_humansim_msgs/msg/AgentFrame"

# (agent_id, kind, x, y, theta, vx, vy, radius, desired_velocity, policy_idx, policy, agent_type)
_AGENTS = [
    (3, AgentState.KIND_HUMAN, 1.5, -2.0, 0.75, 0.3, 0.4, 0.35, 1.25, 0, "sfm", ""),
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
    "peds_agent_types": ["", "child"],
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
