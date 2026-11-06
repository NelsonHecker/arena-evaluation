"""Episode lifecycle, world semantics, plans, characterization and pose markers, and transforms as flat rows."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import yaml

from arena_evaluation.processing.telemetry.messages import Envelope, RobotRef, declares, env_row, robot_row, stamp_ns, yaw_from_orientation
from arena_evaluation.processing.telemetry.writer import TableSink
from arena_evaluation.storage.legacy import legacy_modes, legacy_params

_log = logging.getLogger(__name__)
_BOOL, _INTEGER, _DOUBLE, _STRING = 1, 2, 3, 4
_BYTE_ARRAY, _BOOL_ARRAY, _INTEGER_ARRAY, _DOUBLE_ARRAY, _STRING_ARRAY = 5, 6, 7, 8, 9


def record_episode(sink: TableSink, envelope: Envelope) -> None:
    record, schema = envelope.body, envelope.schema
    sink.append(
        "episode_lifecycle",
        env_row(
            envelope,
            episode_id=int(record.episode_id),
            outcome_state=int(record.outcome_state),
            outcome_info=str(record.outcome_info),
            goal_uuid=str(record.goal_uuid),
            phases_yaml=str(record.phases) if declares(schema, record, "phases") else "",
            params_yaml=_params_yaml(envelope),
            conditions_yaml=str(record.conditions or "") if declares(schema, record, "conditions") else "",
            world=str(record.world) if declares(schema, record, "world") and record.world else None,
            seed=int(record.seed) if declares(schema, record, "seed") and record.seed >= 0 else None,
            human_params=_human_params_json(record) if declares(schema, record, "human_params") else None,
        ),
    )
    flags = record.integrity if declares(schema, record, "integrity") else []
    if isinstance(flags, list):
        for flag_idx, flag in enumerate(flags):
            sink.append("episode_integrity", env_row(envelope, flag_idx=flag_idx, flag=str(flag)))


def record_semantics(sink: TableSink, envelope: Envelope) -> None:
    snapshot = envelope.body
    env_id = f"env_{int(snapshot.env_id)}"
    for entity in snapshot.entities:
        base = {"t_ns": envelope.t_ns, "env_id": env_id, "world": str(snapshot.world), "entity_name": str(entity.entity), "kind": str(entity.kind)}
        for name, value in zip(entity.discrete_names, entity.discrete_values, strict=True):
            sink.append("world_semantics", {**base, "field_name": str(name), "field_kind": "discrete", "value_str": str(value), "value_num": None, "value_bool": None})
        for name, value in zip(entity.continuous_names, entity.continuous_values, strict=True):
            sink.append("world_semantics", {**base, "field_name": str(name), "field_kind": "continuous", "value_str": None, "value_num": float(value), "value_bool": None})
        for name, value in zip(entity.predicate_names, entity.predicate_values, strict=True):
            sink.append("world_semantics", {**base, "field_name": str(name), "field_kind": "predicate", "value_str": None, "value_num": None, "value_bool": bool(value)})
        members = ",".join(sorted(str(member) for member in entity.members))
        sink.append("world_semantics", {**base, "field_name": "members", "field_kind": "members", "value_str": members, "value_num": None, "value_bool": None})


class PlanRecorder:
    """Writes each global plan as numbered waypoint rows, one plan per message and robot."""

    def __init__(self, sink: TableSink) -> None:
        self._sink = sink
        self._plan_counts: defaultdict[RobotRef, int] = defaultdict(int)

    def record(self, envelope: Envelope) -> None:
        plan_id = self._plan_counts[envelope.ref]
        self._plan_counts[envelope.ref] += 1
        for waypoint_idx, stamped in enumerate(envelope.body.poses):
            pose = stamped.pose
            self._sink.append(
                "robot_plans",
                robot_row(
                    envelope,
                    plan_id=plan_id,
                    waypoint_idx=waypoint_idx,
                    x=float(pose.position.x),
                    y=float(pose.position.y),
                    yaw=yaw_from_orientation(pose.orientation),
                ),
            )


def record_characterization_phase(sink: TableSink, envelope: Envelope) -> None:
    sink.append("robot_characterization_phase", robot_row(envelope, label=str(envelope.body.data)))


class ScheduleRecorder:
    """Keeps the characterization schedule the sweep actually ran: the latest valid payload per robot wins."""

    def __init__(self) -> None:
        self._latest: dict[RobotRef, tuple[int, list[dict[str, Any]]]] = {}

    def record(self, envelope: Envelope) -> None:
        phases = _schedule_phases(str(envelope.body.data))
        if phases is not None:
            self._latest[envelope.ref] = (envelope.t_ns, phases)

    def write(self, sink: TableSink) -> None:
        for ref, (t_ns, phases) in self._latest.items():
            for phase_idx, phase in enumerate(phases):
                sink.append(
                    "robot_characterization_schedule",
                    {"t_ns": t_ns, "env_id": ref.env_id, "robot_id": ref.robot_id, "phase_idx": phase_idx, **phase},
                )


def record_initial_pose(sink: TableSink, envelope: Envelope) -> None:
    pose = envelope.body.pose.pose
    sink.append(
        "robot_initialpose",
        robot_row(envelope, pos_x=float(pose.position.x), pos_y=float(pose.position.y), yaw=yaw_from_orientation(pose.orientation)),
    )


def record_task_pose(sink: TableSink, envelope: Envelope) -> None:
    message = envelope.body
    sink.append(
        "robot_task_pose",
        robot_row(
            envelope,
            stamp_ns=stamp_ns(message.header),
            pos_x=float(message.pose.position.x),
            pos_y=float(message.pose.position.y),
            yaw=yaw_from_orientation(message.pose.orientation),
        ),
    )


def record_transforms(sink: TableSink, table: str, envelope: Envelope) -> None:
    for transform in envelope.body.transforms:
        translation, rotation = transform.transform.translation, transform.transform.rotation
        sink.append(
            table,
            {
                "t_ns": envelope.t_ns,
                "parent_frame": str(transform.header.frame_id),
                "child_frame": str(transform.child_frame_id),
                "tx": float(translation.x),
                "ty": float(translation.y),
                "tz": float(translation.z),
                "qx": float(rotation.x),
                "qy": float(rotation.y),
                "qz": float(rotation.z),
                "qw": float(rotation.w),
            },
        )


def _params_yaml(envelope: Envelope) -> str:
    record, schema = envelope.body, envelope.schema
    if not (declares(schema, record, "params") or declares(schema, record, "robots_params")):
        return ""
    try:
        return yaml.dump(_record_params(envelope))
    except (ValueError, TypeError, KeyError) as error:
        _log.warning(f"Failed to dump episode params: {error}")
        return ""


def _record_params(envelope: Envelope) -> dict[str, Any]:
    record, schema = envelope.body, envelope.schema
    if declares(schema, record, "params"):
        return {param.name: _param_value(param.value) for param in record.params}
    modes = legacy_modes(robots=record.tm_robots, obstacles=record.tm_obstacles, modules=record.tm_modules)
    robots = {param.name: _param_value(param.value) for param in record.robots_params}
    obstacles = {param.name: _param_value(param.value) for param in record.obstacles_params}
    return legacy_params(modes, robots, obstacles)


def _human_params_json(record: Any) -> str:
    return json.dumps({param.name: _param_value(param.value) for param in record.human_params})


def _param_value(value: Any) -> Any:
    kind = value.type
    if kind == _BOOL:
        return value.bool_value
    if kind == _INTEGER:
        return value.integer_value
    if kind == _DOUBLE:
        return value.double_value
    if kind == _STRING:
        return value.string_value
    if kind == _BYTE_ARRAY:
        return [byte[0] if isinstance(byte, bytes) else int(byte) for byte in value.byte_array_value]
    if kind == _BOOL_ARRAY:
        return list(value.bool_array_value)
    if kind == _INTEGER_ARRAY:
        return list(value.integer_array_value)
    if kind == _DOUBLE_ARRAY:
        return list(value.double_array_value)
    if kind == _STRING_ARRAY:
        return list(value.string_array_value)
    return ""


def _schedule_phases(payload: str) -> list[dict[str, Any]] | None:
    try:
        return [
            {
                "phase_label": str(phase["name"]),
                "phase_kind": str(phase["kind"]),
                "vx_target": float(phase["vx_target"]),
                "vy_target": float(phase["vy_target"]),
                "wz_target": float(phase["wz_target"]),
                "duration_s": float(phase["duration_s"]),
                "ramp_s": float(phase["ramp_s"]),
                "turn_radius_m": float(phase["radius_m"]),
            }
            for phase in json.loads(payload)["phases"]
        ]
    except (ValueError, KeyError, TypeError) as error:
        _log.warning(f"characterization_schedule payload rejected: {error!r}")
        return None
