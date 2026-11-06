"""Collision and pedestrian-interaction telemetry, normalized so that every participant list is its own table."""

from __future__ import annotations

from typing import Any

from arena_evaluation.processing.telemetry.messages import Envelope, declares, env_row, pedestrian_id, robot_row
from arena_evaluation.processing.telemetry.writer import TableSink

WALL_KIND = "wall"
STATIC_KIND = "static"
PEDESTRIAN_KIND = "pedestrian"
MEMBER_ROLE_NAMES = {0: "participant", 1: "queue"}
INTERACTION_EVENT_NAMES = {0: "ACTIVATED", 1: "HOLD_ONSET", 2: "RELEASED", 3: "INTERRUPTED", 4: "CANCELED"}


def record_collision_events(sink: TableSink, envelope: Envelope) -> None:
    message, schema = envelope.body, envelope.schema
    events = message.events
    has_kind = declares(schema, message, "kind")
    has_position = declares(schema, message, "obstacle_position")
    has_distance = declares(schema, message, "distance")
    sink.append(
        "collision_events",
        robot_row(
            envelope,
            event_count=len(events),
            wall_count=_count_kind(events, WALL_KIND) if has_kind else None,
            static_count=_count_kind(events, STATIC_KIND) if has_kind else None,
            ped_count=_count_kind(events, PEDESTRIAN_KIND) if has_kind else None,
        ),
    )
    for event in events:
        x, y, z = _position(event) if has_position else (None, None, None)
        sink.append(
            "collision_contacts",
            robot_row(
                envelope,
                obstacle_id=str(event.obstacle_id),
                kind=str(event.kind) if has_kind else None,
                obstacle_x=x,
                obstacle_y=y,
                obstacle_z=z,
                distance=float(event.distance) if has_distance else None,
            ),
        )


def record_collision_monitor(sink: TableSink, envelope: Envelope) -> None:
    monitor = envelope.body
    sink.append(
        "robot_collision_monitor",
        robot_row(envelope, action_type=int(monitor.action_type), polygon_name=str(monitor.polygon_name)),
    )


def record_interactions(sink: TableSink, envelope: Envelope) -> None:
    frame = envelope.body
    offsets = [int(offset) for offset in frame.member_offset]
    member_ids = [int(member) for member in frame.member_id]
    member_roles = bytes(frame.member_role)
    interaction_types = bytes(frame.interaction_type)
    outcomes = bytes(frame.outcome)
    for index, interaction_id in enumerate(frame.interaction_id):
        interaction = int(interaction_id)
        sink.append(
            "pedestrian_interactions",
            env_row(
                envelope,
                interaction_id=interaction,
                interaction_type=int(interaction_types[index]),
                outcome=int(outcomes[index]),
                object_id=str(frame.object_id[index]),
                duration=float(frame.duration[index]),
                arrived=bool(frame.arrived[index]),
                holding=bool(frame.holding[index]),
            ),
        )
        start, end = offsets[index], offsets[index + 1]
        for member, role in zip(member_ids[start:end], member_roles[start:end], strict=True):
            role_name = MEMBER_ROLE_NAMES.get(role)
            if role_name is not None:
                sink.append(
                    "pedestrian_interaction_members",
                    env_row(envelope, interaction_id=interaction, ped_id=pedestrian_id(member), member_role=role_name),
                )


def record_interaction_events(sink: TableSink, envelope: Envelope) -> None:
    for event in envelope.body.events:
        event_code = int(event.event)
        participants: list[int | None] = [int(participant) for participant in event.participants] or [None]
        for participant in participants:
            sink.append(
                "pedestrian_interaction_events",
                env_row(
                    envelope,
                    interaction_id=int(event.interaction_id),
                    interaction_type=int(event.interaction_type),
                    event=event_code,
                    event_name=INTERACTION_EVENT_NAMES.get(event_code, str(event_code)),
                    ped_id=None if participant is None else pedestrian_id(participant),
                ),
            )


def record_animations(sink: TableSink, envelope: Envelope) -> None:
    for ped in envelope.body.peds:
        ped_id = pedestrian_id(int(ped.id))
        for slot in list(ped.slots) or [None]:
            sink.append("pedestrian_animations", _animation_row(envelope, ped_id, ped, slot))


def _count_kind(events: Any, kind: str) -> int:
    return sum(1 for event in events if event.kind == kind)


def _position(event: Any) -> tuple[float, float, float]:
    position = event.obstacle_position
    return float(position.x), float(position.y), float(position.z)


def _animation_row(envelope: Envelope, ped_id: str, ped: Any, slot: Any) -> dict[str, Any]:
    base = {"ped_id": ped_id, "ped_name": str(ped.name), "base": str(ped.base), "base_phase": float(ped.base_phase)}
    if slot is None:
        return env_row(envelope, **base, slot="", clip="", playhead=None, duration=None, weight=None, loop=None)
    return env_row(
        envelope,
        **base,
        slot=str(slot.slot),
        clip=str(slot.clip),
        playhead=float(slot.playhead),
        duration=float(slot.duration),
        weight=float(slot.weight),
        loop=bool(slot.loop),
    )
