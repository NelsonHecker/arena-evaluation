"""Pedestrian telemetry: each recorded pedestrian becomes one flat row per timestamp, never a stride-packed list."""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from arena_evaluation.processing.telemetry.messages import Envelope, env_row, pedestrian_id, yaw_from_orientation
from arena_evaluation.processing.telemetry.writer import TableSink

if TYPE_CHECKING:
    from arena_humansim_msgs.msg import AgentFrame, AgentStates
    from arena_humansim_msgs.msg import AgentMeta as RosAgentMeta
    from arena_people_msgs.msg import Pedestrians

ADULT_AGENT_TYPE = "adult"
HUMAN_KIND = 0
# Values of a human whose message carries no interaction: the crowd engine's own "none".
NO_INTERACTION_ID = -1
NO_INTERACTION_TYPE = 0
ARENA_TABLE = "arena_peds"
ENGINE_TABLE = "pedestrians"
GESTURE_TABLE = "pedestrian_gestures"
PEDESTRIANS_SCHEMA = "arena_people_msgs/msg/Pedestrians"
AGENT_FRAME_SCHEMA = "arena_humansim_msgs/msg/AgentFrame"
AGENT_STATES_SCHEMA = "arena_humansim_msgs/msg/AgentStates"


@dataclasses.dataclass(frozen=True, slots=True)
class AgentMeta:
    """Crowd-engine metadata: policy names by index and agent types by agent id."""

    policies: tuple[str, ...] = ()
    types_by_id: Mapping[int, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class _Agent:
    """One pedestrian of a frame, before it becomes a row of the pedestrian table."""

    number: int
    name: str
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vz: float
    radius_m: float | None
    desired_speed: float | None
    policy: str | None
    agent_type: str | None
    animation_state: int | None
    interaction_id: int | None
    interaction_type: int | None
    gait_phase: float | None
    gait_cadence: float | None

    def table_columns(self) -> dict[str, Any]:
        return {
            "ped_id": pedestrian_id(self.number),
            "ped_name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "speed_mps": math.hypot(self.vx, self.vy),
            "radius_m": self.radius_m,
            "desired_speed": self.desired_speed,
            "policy": self.policy,
            "agent_type": self.agent_type,
            "animation_state": self.animation_state,
            "interaction_id": self.interaction_id,
            "interaction_type": self.interaction_type,
            "gait_phase": self.gait_phase,
            "gait_cadence": self.gait_cadence,
        }


class PedestrianRecorder:
    """Writes pedestrian frames as flat rows and remembers the frames each environment was stamped in."""

    def __init__(self, sink: TableSink) -> None:
        self._sink = sink
        self._meta: dict[str, AgentMeta] = {}
        self._frames: dict[tuple[str, str], set[str]] = defaultdict(set)

    def record_meta(self, envelope: Envelope[RosAgentMeta]) -> None:
        self._meta[envelope.ref.env_id] = _agent_meta(envelope)

    def record_frame(self, table: str, envelope: Envelope[Pedestrians | AgentFrame | AgentStates]) -> None:
        env_id = envelope.ref.env_id
        self._frames[(env_id, table)].add(envelope.body.header.frame_id.strip("/"))
        for agent in _agents(envelope, self._meta.get(env_id, AgentMeta())):
            self._sink.append(table, env_row(envelope, **agent.table_columns()))
        for row in _gesture_rows(envelope):
            self._sink.append(GESTURE_TABLE, row)

    def frames(self) -> dict[tuple[str, str], frozenset[str]]:
        return {key: frozenset(frames) for key, frames in self._frames.items()}


def _agent_meta(envelope: Envelope[RosAgentMeta]) -> AgentMeta:
    meta = envelope.body
    policies = tuple(str(policy) for policy in meta.policies)
    types = {int(agent_id): _agent_type(agent_type) for agent_id, agent_type in zip(meta.agent_id, meta.agent_type, strict=True)}
    return AgentMeta(policies=policies, types_by_id=types)


def _agents(envelope: Envelope[Pedestrians | AgentFrame | AgentStates], meta: AgentMeta) -> list[_Agent]:
    schema = envelope.schema.name
    if schema == PEDESTRIANS_SCHEMA:
        columns = _pedestrians_columns(cast("Envelope[Pedestrians]", envelope))
    elif schema == AGENT_FRAME_SCHEMA:
        columns = _agent_frame_columns(cast("Envelope[AgentFrame]", envelope), meta)
    elif schema == AGENT_STATES_SCHEMA:
        columns = _agent_states_columns(cast("Envelope[AgentStates]", envelope))
    else:
        raise ValueError(f"unsupported pedestrian schema {schema!r}")
    return [_Agent(**dict(zip(columns, values, strict=True))) for values in zip(*columns.values(), strict=True)]


def _pedestrians_columns(envelope: Envelope[Pedestrians]) -> dict[str, list[Any]]:
    peds = envelope.body.pedestrians
    nulls = [None] * len(peds)
    return {
        "number": [int(ped.id) for ped in peds],
        "name": [ped.name or str(ped.id) for ped in peds],
        "x": [float(ped.pose.position.x) for ped in peds],
        "y": [float(ped.pose.position.y) for ped in peds],
        "z": [float(ped.pose.position.z) for ped in peds],
        "yaw": [yaw_from_orientation(ped.pose.orientation) for ped in peds],
        "vx": [float(ped.twist.linear.x) for ped in peds],
        "vy": [float(ped.twist.linear.y) for ped in peds],
        "vz": [float(ped.twist.linear.z) for ped in peds],
        "radius_m": nulls,
        "desired_speed": nulls,
        "policy": nulls,
        "agent_type": [_agent_type(ped.agent_type) for ped in peds],
        "animation_state": [int(ped.animation_state) for ped in peds],
        "interaction_id": [int(ped.interaction_id) for ped in peds],
        "interaction_type": [int(ped.interaction_type) for ped in peds],
        "gait_phase": [_nonzero(float(ped.gait_phase)) for ped in peds],
        "gait_cadence": [_nonzero(float(ped.gait_cadence)) for ped in peds],
    }


def _agent_frame_columns(envelope: Envelope[AgentFrame], meta: AgentMeta) -> dict[str, list[Any]]:
    frame = envelope.body
    humans = _uint8(frame.kind) == HUMAN_KIND
    agent_ids = np.asarray(frame.agent_id, dtype=np.int64)[humans].tolist()
    policy_indices = np.asarray(frame.policy_idx, dtype=np.int64)[humans].tolist()
    return {
        "number": agent_ids,
        "name": [str(agent_id) for agent_id in agent_ids],
        "x": _floats(frame.x, humans),
        "y": _floats(frame.y, humans),
        "z": [0.0] * len(agent_ids),
        "yaw": _floats(frame.theta, humans),
        "vx": _floats(frame.vx, humans),
        "vy": _floats(frame.vy, humans),
        "vz": [0.0] * len(agent_ids),
        "radius_m": [_nonzero(radius) for radius in _floats(frame.radius, humans)],
        "desired_speed": [_nonzero(speed) for speed in _floats(frame.desired_velocity, humans)],
        "policy": [_policy_name(meta.policies, index) for index in policy_indices],
        "agent_type": [meta.types_by_id.get(agent_id) for agent_id in agent_ids],
        "animation_state": _bytes(frame.animation_state, humans),
        "interaction_id": np.asarray(frame.interaction_id, dtype=np.int64)[humans].tolist(),
        "interaction_type": _bytes(frame.interaction_type, humans),
        "gait_phase": [_nonzero(phase) for phase in _floats(frame.gait_phase, humans)],
        "gait_cadence": [_nonzero(cadence) for cadence in _floats(frame.gait_cadence, humans)],
    }


def _agent_states_columns(envelope: Envelope[AgentStates]) -> dict[str, list[Any]]:
    agents = [agent for agent in envelope.body.agents if agent.kind == HUMAN_KIND]
    nulls = [None] * len(agents)
    return {
        "number": [int(agent.agent_id) for agent in agents],
        "name": [agent.name or str(agent.agent_id) for agent in agents],
        "x": [float(agent.pose.x) for agent in agents],
        "y": [float(agent.pose.y) for agent in agents],
        "z": [0.0] * len(agents),
        "yaw": [float(agent.pose.theta) for agent in agents],
        "vx": [float(agent.velocity.x) for agent in agents],
        "vy": [float(agent.velocity.y) for agent in agents],
        "vz": [float(agent.velocity.z) for agent in agents],
        "radius_m": [_nonzero(float(agent.radius)) for agent in agents],
        "desired_speed": [_nonzero(float(agent.desired_velocity)) for agent in agents],
        "policy": [_text_or_none(agent.policy) for agent in agents],
        "agent_type": [_text_or_none(agent.agent_type) for agent in agents],
        "animation_state": [int(agent.animation_state) for agent in agents],
        "interaction_id": [NO_INTERACTION_ID] * len(agents),
        "interaction_type": [NO_INTERACTION_TYPE] * len(agents),
        "gait_phase": nulls,
        "gait_cadence": nulls,
    }


def _gesture_rows(envelope: Envelope) -> list[dict[str, Any]]:
    if envelope.schema.name != PEDESTRIANS_SCHEMA:
        return []
    peds = cast("Envelope[Pedestrians]", envelope).body.pedestrians
    return [
        env_row(
            envelope,
            ped_id=pedestrian_id(int(ped.id)),
            slot=str(gesture.slot),
            clip=str(gesture.clip),
            hand=str(gesture.hand),
            at_x=float(gesture.at.x),
            at_y=float(gesture.at.y),
            at_z=float(gesture.at.z),
            render_pose_override=bool(gesture.render_pose_override),
        )
        for ped in peds
        for gesture in ped.gestures
    ]


def _floats(values: object, humans: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float64)[humans].tolist()


def _uint8(values: object) -> np.ndarray:
    return np.frombuffer(values, dtype=np.uint8) if isinstance(values, (bytes, bytearray)) else np.asarray(values, dtype=np.uint8)


def _bytes(values: object, humans: np.ndarray) -> list[int]:
    return _uint8(values)[humans].tolist()


def _policy_name(policies: tuple[str, ...], index: int | None) -> str | None:
    return policies[index] if index is not None and 0 <= index < len(policies) else None


def _agent_type(raw: object) -> str:
    return str(raw) or ADULT_AGENT_TYPE


def _text_or_none(raw: object) -> str | None:
    return str(raw) or None


def _nonzero(value: float | None) -> float | None:
    """Zero is the producer's 'not provided' value for gait, radius and speed."""
    return value or None
