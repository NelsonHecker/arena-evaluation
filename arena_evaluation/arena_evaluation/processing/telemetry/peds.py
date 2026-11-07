"""Pedestrian telemetry: each recorded pedestrian becomes one flat row per timestamp, never a stride-packed list."""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from arena_evaluation.processing.telemetry.messages import Envelope, env_row, pedestrian_id, yaw_from_orientation
from arena_evaluation.processing.telemetry.writer import TableSink

if TYPE_CHECKING:
    from arena_humansim_msgs.msg import AgentFrame, AgentStates
    from arena_humansim_msgs.msg import AgentMeta as RosAgentMeta
    from arena_people_msgs.msg import Pedestrians

ADULT_AGENT_TYPE = "adult"
HUMAN_KIND = 0
# Values of a human whose message carries no interaction or animation state: the crowd engine's own "none".
NO_INTERACTION_ID = -1
NO_INTERACTION_TYPE = 0
NO_ANIMATION = 0
ARENA_TABLE = "pedestrian_tracks"
ENGINE_TABLE = "pedestrian_tracks_physics"
GESTURE_TABLE = "pedestrian_gestures"
PEDESTRIANS_SCHEMA = "arena_people_msgs/msg/Pedestrians"
AGENT_FRAME_SCHEMA = "arena_humansim_msgs/msg/AgentFrame"


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
    # agent-contracts: ignore MCAP schema evolution
    policies = tuple(str(policy) for policy in getattr(meta, "policies", ()))
    # agent-contracts: ignore MCAP schema evolution
    agent_type = getattr(meta, "agent_type", ())
    typed = len(meta.agent_id) == len(agent_type)
    types = {int(agent_id): _agent_type(t) for agent_id, t in zip(meta.agent_id, agent_type, strict=True)} if typed else {}
    return AgentMeta(policies=policies, types_by_id=types)


def _agents(envelope: Envelope[Pedestrians | AgentFrame | AgentStates], meta: AgentMeta) -> list[_Agent]:
    schema_name = envelope.schema.name
    if schema_name == PEDESTRIANS_SCHEMA:
        columns = _pedestrians_columns(envelope)  # type: ignore[arg-type]
    elif schema_name == AGENT_FRAME_SCHEMA:
        columns = _agent_frame_columns(envelope, meta)  # type: ignore[arg-type]
    else:
        columns = _agent_states_columns(envelope)  # type: ignore[arg-type]
    return [_Agent(**dict(zip(columns, values, strict=True))) for values in zip(*columns.values(), strict=True)]


def _pedestrians_columns(envelope: Envelope[Pedestrians]) -> dict[str, list[Any]]:
    peds = envelope.body.pedestrians
    nulls = [None] * len(peds)
    # agent-contracts: ignore MCAP schema evolution
    agent_type = [_agent_type(getattr(ped, "agent_type", "")) for ped in peds]
    # agent-contracts: ignore MCAP schema evolution
    animation_state = [int(getattr(ped, "animation_state", NO_ANIMATION)) for ped in peds]
    # agent-contracts: ignore MCAP schema evolution
    interaction_id = [int(getattr(ped, "interaction_id", NO_INTERACTION_ID)) for ped in peds]
    # agent-contracts: ignore MCAP schema evolution
    interaction_type = [int(getattr(ped, "interaction_type", NO_INTERACTION_TYPE)) for ped in peds]
    # agent-contracts: ignore MCAP schema evolution
    gait_phase = [_nonzero(float(getattr(ped, "gait_phase", 0.0))) for ped in peds]
    # agent-contracts: ignore MCAP schema evolution
    gait_cadence = [_nonzero(float(getattr(ped, "gait_cadence", 0.0))) for ped in peds]
    return {
        "number": [int(ped.id) for ped in peds],
        "name": [str(ped.name) for ped in peds],
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
        "agent_type": agent_type,
        "animation_state": animation_state,
        "interaction_id": interaction_id,
        "interaction_type": interaction_type,
        "gait_phase": gait_phase,
        "gait_cadence": gait_cadence,
    }


def _agent_frame_columns(envelope: Envelope[AgentFrame], meta: AgentMeta) -> dict[str, list[Any]]:
    frame = envelope.body
    humans = np.frombuffer(bytes(frame.kind), dtype=np.uint8) == HUMAN_KIND
    agent_ids = np.asarray(frame.agent_id, dtype=np.int64)[humans].tolist()
    count = len(agent_ids)
    # agent-contracts: ignore MCAP schema evolution
    policy_indices = _per_human(getattr(frame, "policy_idx", None), humans, int, missing=None)
    # agent-contracts: ignore MCAP schema evolution
    animation = _per_human(_byte_values(getattr(frame, "animation_state", ())), humans, int, missing=NO_ANIMATION)
    # agent-contracts: ignore MCAP schema evolution
    interaction_ids = _per_human(getattr(frame, "interaction_id", None), humans, int, missing=NO_INTERACTION_ID)
    # agent-contracts: ignore MCAP schema evolution
    interaction_types = _per_human(_byte_values(getattr(frame, "interaction_type", ())), humans, int, missing=NO_INTERACTION_TYPE)
    # agent-contracts: ignore MCAP schema evolution
    radii = _per_human(getattr(frame, "radius", None), humans, float, missing=None)
    # agent-contracts: ignore MCAP schema evolution
    desired = _per_human(getattr(frame, "desired_velocity", None), humans, float, missing=None)
    # agent-contracts: ignore MCAP schema evolution
    phases = _per_human(getattr(frame, "gait_phase", None), humans, float, missing=None)
    # agent-contracts: ignore MCAP schema evolution
    cadences = _per_human(getattr(frame, "gait_cadence", None), humans, float, missing=None)
    return {
        "number": agent_ids,
        "name": [str(agent_id) for agent_id in agent_ids],
        "x": _floats(frame.x, humans),
        "y": _floats(frame.y, humans),
        "z": [0.0] * count,
        "yaw": _floats(frame.theta, humans),
        "vx": _floats(frame.vx, humans),
        "vy": _floats(frame.vy, humans),
        "vz": [0.0] * count,
        "radius_m": [_nonzero(radius) for radius in radii],
        "desired_speed": [_nonzero(speed) for speed in desired],
        "policy": [_policy_name(meta.policies, index) for index in policy_indices],
        "agent_type": [meta.types_by_id.get(agent_id) for agent_id in agent_ids],
        "animation_state": animation,
        "interaction_id": interaction_ids,
        "interaction_type": interaction_types,
        "gait_phase": [_nonzero(phase) for phase in phases],
        "gait_cadence": [_nonzero(cadence) for cadence in cadences],
    }


def _agent_states_columns(envelope: Envelope[AgentStates]) -> dict[str, list[Any]]:
    state = envelope.body
    agents = [agent for agent in state.agents if agent.kind == HUMAN_KIND]
    nulls = [None] * len(agents)
    return {
        "number": [int(agent.agent_id) for agent in agents],
        "name": [str(agent.agent_id) for agent in agents],
        "x": [float(agent.pose.x) for agent in agents],
        "y": [float(agent.pose.y) for agent in agents],
        "z": [0.0] * len(agents),
        "yaw": [float(agent.pose.theta) for agent in agents],
        "vx": [float(agent.velocity.x) for agent in agents],
        "vy": [float(agent.velocity.y) for agent in agents],
        "vz": [float(agent.velocity.z) for agent in agents],
        # agent-contracts: ignore MCAP schema evolution
        "radius_m": [_nonzero(float(getattr(agent, "radius", 0.0))) for agent in agents],
        # agent-contracts: ignore MCAP schema evolution
        "desired_speed": [_nonzero(float(getattr(agent, "desired_velocity", 0.0))) for agent in agents],
        # agent-contracts: ignore MCAP schema evolution
        "policy": [_text_or_none(getattr(agent, "policy", "")) for agent in agents],
        # agent-contracts: ignore MCAP schema evolution
        "agent_type": [_text_or_none(getattr(agent, "agent_type", "")) for agent in agents],
        # agent-contracts: ignore MCAP schema evolution
        "animation_state": [int(getattr(agent, "animation_state", NO_ANIMATION)) for agent in agents],
        "interaction_id": [NO_INTERACTION_ID] * len(agents),
        "interaction_type": [NO_INTERACTION_TYPE] * len(agents),
        "gait_phase": nulls,
        "gait_cadence": nulls,
    }


def _gesture_rows(envelope: Envelope) -> list[dict[str, Any]]:
    if envelope.schema.name != PEDESTRIANS_SCHEMA:
        return []
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
        for ped in envelope.body.pedestrians
        # agent-contracts: ignore MCAP schema evolution
        for gesture in getattr(ped, "gestures", ())
    ]


def _per_human(values: Sequence[Any] | np.ndarray | None, humans: np.ndarray, cast: Callable[[Any], Any], missing: object) -> list[Any]:
    """Values of the human agents of a parallel array, `missing` for each human when the array is absent or misaligned."""
    if values is None or len(values) != len(humans):
        return [missing] * int(humans.sum())
    return [cast(value) for value in np.asarray(values)[humans].tolist()]


def _floats(values: Sequence[float], humans: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float64)[humans].tolist()


def _byte_values(raw: object) -> np.ndarray:
    return np.frombuffer(bytes(raw), dtype=np.uint8)  # pyright: ignore[reportArgumentType]


def _policy_name(policies: Sequence[str], index: int | None) -> str | None:
    return policies[index] if index is not None and 0 <= index < len(policies) else None


def _agent_type(raw: object) -> str:
    return str(raw) or ADULT_AGENT_TYPE


def _text_or_none(raw: object) -> str | None:
    return str(raw) or None


def _nonzero(value: float | None) -> float | None:
    """Zero is the producer's 'not provided' value for gait, radius and speed."""
    return value or None
