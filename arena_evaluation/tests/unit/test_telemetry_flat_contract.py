"""Contract tests for the flat telemetry tables that replaced the pre-overhaul wide layouts.

The old per-topic tables (`peds.parquet` with per-pedestrian list columns, `episode_record.parquet`,
`collision_contacts` under a per-robot directory) no longer exist. These assert that the flat schemas
carry the same information, one row per entity per stamp.
"""

from __future__ import annotations

from arena_evaluation.processing.telemetry import writer as telemetry_writer


def _names(table: str) -> set[str]:
    return set(telemetry_writer.TABLE_SCHEMAS[table].names)


def test_collision_contacts_carry_the_header_stamp_and_obstacle_geometry() -> None:
    names = _names("collision_contacts")
    assert {"t_ns", "env_id", "robot_id", "kind", "obstacle_id", "obstacle_x", "obstacle_y", "obstacle_z", "distance"} <= names


def test_collision_events_keep_the_per_message_counts() -> None:
    assert {"t_ns", "env_id", "robot_id", "event_count"} <= _names("collision_events")


def test_pedestrian_rows_carry_the_crowd_engine_attributes() -> None:
    assert {"radius_m", "desired_speed", "policy", "agent_type", "gait_phase", "gait_cadence"} <= _names("arena_peds")


def test_pedestrian_tables_share_one_flat_schema() -> None:
    schemas = {
        name: telemetry_writer.TABLE_SCHEMAS[name].names
        for name in ("arena_peds", "pedestrians", "pedestrian_tracks", "pedestrian_tracks_physics")
    }
    assert len({tuple(names) for names in schemas.values()}) == 1


def test_episode_lifecycle_carries_world_seed_and_accepted_human_params() -> None:
    assert {"world", "seed", "human_params", "episode_id", "outcome_state"} <= _names("episode_lifecycle")


def test_lifecycle_flags_have_their_own_flat_table() -> None:
    assert {"t_ns", "env_id", "flag_idx", "flag"} <= _names("episode_integrity")


def test_every_robot_table_uses_the_canonical_column_names() -> None:
    robot_tables = [name for name in telemetry_writer.TABLE_SCHEMAS if name.startswith("robot_")]
    assert robot_tables
    for name in robot_tables:
        names = _names(name)
        assert {"t_ns", "env_id"} <= names, name
