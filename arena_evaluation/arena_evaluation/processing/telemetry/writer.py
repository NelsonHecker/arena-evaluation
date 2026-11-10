"""Flat, typed Parquet tables for ingested telemetry and the streaming sink that writes them."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION = "zstd"
FLUSH_ROW_LIMIT = 50_000

_INT = pa.int64()
_FLOAT = pa.float64()
_TEXT = pa.string()
_FLAG = pa.bool_()

ROBOT_KEYS = (("env_id", _TEXT), ("robot_id", _TEXT))
ENV_KEYS = (("env_id", _TEXT),)
NO_KEYS: tuple[()] = ()

_ODOMETRY = (
    ("odom_x", _FLOAT),
    ("odom_y", _FLOAT),
    ("odom_yaw", _FLOAT),
    ("odom_vx", _FLOAT),
    ("odom_wz", _FLOAT),
    ("drift_error_m", _FLOAT),
)
_ROBOT_TRACK = (
    ("x", _FLOAT),
    ("y", _FLOAT),
    ("z", _FLOAT),
    ("yaw", _FLOAT),
    ("vx", _FLOAT),
    ("vy", _FLOAT),
    ("vz", _FLOAT),
    ("omega_z", _FLOAT),
    ("speed_mps", _FLOAT),
    ("pose_source", _TEXT),
)
_TWIST = (
    ("linear_x", _FLOAT),
    ("linear_y", _FLOAT),
    ("linear_z", _FLOAT),
    ("angular_x", _FLOAT),
    ("angular_y", _FLOAT),
    ("angular_z", _FLOAT),
)
_SCAN_SUMMARY = (
    ("scan_min_m", _FLOAT),
    ("scan_max_m", _FLOAT),
    ("range_min_m", _FLOAT),
    ("range_max_m", _FLOAT),
    ("n_valid_rays", _INT),
)
_SCAN_RAYS = (("ray_idx", _INT), ("range_m", _FLOAT), ("angle_rad", _FLOAT))
_JOINT_STATES = (("joint_name", _TEXT), ("position", _FLOAT), ("velocity", _FLOAT), ("effort", _FLOAT))
_POWER = (
    ("total_power_w", _FLOAT),
    ("static_power_w", _FLOAT),
    ("total_mech_power_w", _FLOAT),
    ("total_thermal_power_w", _FLOAT),
)
_JOINT_POWER = (
    ("joint_name", _TEXT),
    ("mechanical_power_w", _FLOAT),
    ("thermal_power_w", _FLOAT),
    ("total_power_w", _FLOAT),
)
_ENERGY = (("total_energy_consumed_wh", _FLOAT), ("battery_soc_percent", _FLOAT))
_ACOUSTICS = (
    ("total_level_af_dba", _FLOAT),
    ("total_level_zf_db", _FLOAT),
    ("baseline_level_dba", _FLOAT),
    ("drivetrain_level_dba", _FLOAT),
    ("operating_state", _TEXT),
    ("calibration_status", _TEXT),
    ("uncertainty_dba", _FLOAT),
    ("validity_flags", _INT),
)
_COLLISION_EVENTS = (
    ("event_count", _INT),
    ("wall_count", _INT),
    ("static_count", _INT),
    ("ped_count", _INT),
)
_COLLISION_CONTACTS = (
    ("obstacle_id", _TEXT),
    ("kind", _TEXT),
    ("obstacle_x", _FLOAT),
    ("obstacle_y", _FLOAT),
    ("obstacle_z", _FLOAT),
    ("distance", _FLOAT),
)
_COLLISION_MONITOR = (("action_type", _INT), ("polygon_name", _TEXT))
_PLAN_WAYPOINTS = (
    ("plan_id", _INT),
    ("waypoint_idx", _INT),
    ("x", _FLOAT),
    ("y", _FLOAT),
    ("yaw", _FLOAT),
)
_CHARACTERIZATION_PHASE = (("label", _TEXT),)
_CHARACTERIZATION_SCHEDULE = (
    ("phase_idx", _INT),
    ("phase_label", _TEXT),
    ("phase_kind", _TEXT),
    ("vx_target", _FLOAT),
    ("vy_target", _FLOAT),
    ("wz_target", _FLOAT),
    ("duration_s", _FLOAT),
    ("ramp_s", _FLOAT),
    ("turn_radius_m", _FLOAT),
)
_INITIAL_POSE = (("pos_x", _FLOAT), ("pos_y", _FLOAT), ("yaw", _FLOAT))
_TASK_POSE = (("stamp_ns", _INT), ("pos_x", _FLOAT), ("pos_y", _FLOAT), ("yaw", _FLOAT))
_PEDESTRIAN = (
    ("ped_id", _TEXT),
    ("ped_name", _TEXT),
    ("x", _FLOAT),
    ("y", _FLOAT),
    ("z", _FLOAT),
    ("yaw", _FLOAT),
    ("vx", _FLOAT),
    ("vy", _FLOAT),
    ("vz", _FLOAT),
    ("speed_mps", _FLOAT),
    ("radius_m", _FLOAT),
    ("desired_speed", _FLOAT),
    ("policy", _TEXT),
    ("agent_type", _TEXT),
    ("animation_state", _INT),
    ("interaction_id", _INT),
    ("interaction_type", _INT),
    ("gait_phase", _FLOAT),
    ("gait_cadence", _FLOAT),
)
_GESTURE = (
    ("ped_id", _TEXT),
    ("slot", _TEXT),
    ("clip", _TEXT),
    ("hand", _TEXT),
    ("at_x", _FLOAT),
    ("at_y", _FLOAT),
    ("at_z", _FLOAT),
    ("render_pose_override", _FLAG),
)
_ANIMATION = (
    ("ped_id", _TEXT),
    ("ped_name", _TEXT),
    ("base", _TEXT),
    ("base_phase", _FLOAT),
    ("slot", _TEXT),
    ("clip", _TEXT),
    ("playhead", _FLOAT),
    ("duration", _FLOAT),
    ("weight", _FLOAT),
    ("loop", _FLAG),
)
_INTERACTION = (
    ("interaction_id", _INT),
    ("interaction_type", _INT),
    ("outcome", _INT),
    ("object_id", _TEXT),
    ("duration", _FLOAT),
    ("arrived", _FLAG),
    ("holding", _FLAG),
)
_INTERACTION_MEMBER = (("interaction_id", _INT), ("ped_id", _TEXT), ("member_role", _TEXT))
_INTERACTION_EVENT = (
    ("interaction_id", _INT),
    ("interaction_type", _INT),
    ("event", _INT),
    ("event_name", _TEXT),
    ("ped_id", _TEXT),
)
_EPISODE_LIFECYCLE = (
    ("episode_id", _INT),
    ("outcome_state", _INT),
    ("outcome_info", _TEXT),
    ("goal_uuid", _TEXT),
    ("task_layer", _TEXT),
    ("modes_json", _TEXT),
    ("robots_json", _TEXT),
    ("goal_dist_start_m", _FLOAT),
    ("goal_dist_min_m", _FLOAT),
    ("path_length_m", _FLOAT),
    ("start_t_ns", _INT),
    ("phases_yaml", _TEXT),
    ("params_yaml", _TEXT),
    ("conditions_yaml", _TEXT),
    ("world", _TEXT),
    ("seed", _INT),
    ("human_params", _TEXT),
)
_EPISODE_INTEGRITY = (("flag_idx", _INT), ("flag", _TEXT))
_WORLD_SEMANTICS = (
    ("env_id", _TEXT),
    ("world", _TEXT),
    ("entity_name", _TEXT),
    ("kind", _TEXT),
    ("field_name", _TEXT),
    ("field_kind", _TEXT),
    ("value_str", _TEXT),
    ("value_num", _FLOAT),
    ("value_bool", _FLAG),
)
_TRANSFORM = (
    ("parent_frame", _TEXT),
    ("child_frame", _TEXT),
    ("tx", _FLOAT),
    ("ty", _FLOAT),
    ("tz", _FLOAT),
    ("qx", _FLOAT),
    ("qy", _FLOAT),
    ("qz", _FLOAT),
    ("qw", _FLOAT),
)

_ROBOT_TABLES = {
    "robot_tracks": _ROBOT_TRACK,
    "robot_odom": _ODOMETRY,
    "robot_odom_controller": _ODOMETRY,
    "robot_cmd_vel": _TWIST,
    "robot_cmd_vel_controller": _TWIST,
    "robot_scan_summary": _SCAN_SUMMARY,
    "robot_scan_rays": _SCAN_RAYS,
    "robot_joint_states": _JOINT_STATES,
    "robot_power": _POWER,
    "robot_joint_power": _JOINT_POWER,
    "robot_energy": _ENERGY,
    "robot_acoustics": _ACOUSTICS,
    "collision_events": _COLLISION_EVENTS,
    "collision_contacts": _COLLISION_CONTACTS,
    "robot_collision_monitor": _COLLISION_MONITOR,
    "robot_plans": _PLAN_WAYPOINTS,
    "robot_characterization_phase": _CHARACTERIZATION_PHASE,
    "robot_characterization_schedule": _CHARACTERIZATION_SCHEDULE,
    "robot_initialpose": _INITIAL_POSE,
    "robot_task_pose": _TASK_POSE,
}
_ENV_TABLES = {
    "arena_peds": _PEDESTRIAN,
    "pedestrians": _PEDESTRIAN,
    "pedestrian_tracks": _PEDESTRIAN,
    "pedestrian_tracks_physics": _PEDESTRIAN,
    "pedestrian_gestures": _GESTURE,
    "pedestrian_animations": _ANIMATION,
    "pedestrian_interactions": _INTERACTION,
    "pedestrian_interaction_members": _INTERACTION_MEMBER,
    "pedestrian_interaction_events": _INTERACTION_EVENT,
    "episode_lifecycle": _EPISODE_LIFECYCLE,
    "episode_integrity": _EPISODE_INTEGRITY,
}
_GLOBAL_TABLES = {
    "tf": _TRANSFORM,
    "tf_static": _TRANSFORM,
    "world_semantics": _WORLD_SEMANTICS,
}


def _schemas(keys: tuple, tables: Mapping[str, tuple]) -> dict[str, pa.Schema]:
    return {name: pa.schema([pa.field("t_ns", _INT, nullable=False), *(pa.field(column, dtype) for column, dtype in keys), *(pa.field(column, dtype) for column, dtype in columns)]) for name, columns in tables.items()}


ROBOT_TABLES: tuple[str, ...] = tuple(_ROBOT_TABLES)
ENV_TABLES: tuple[str, ...] = tuple(_ENV_TABLES)
GLOBAL_TABLES: tuple[str, ...] = tuple(_GLOBAL_TABLES)
TABLE_SCHEMAS: dict[str, pa.Schema] = {
    **_schemas(ROBOT_KEYS, _ROBOT_TABLES),
    **_schemas(ENV_KEYS, _ENV_TABLES),
    **_schemas(NO_KEYS, _GLOBAL_TABLES),
}


class TableSink:
    """Buffers flat rows per table and streams every table into one ZSTD Parquet file, sorted by t_ns."""

    def __init__(self, out_dir: pathlib.Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._columns: dict[str, dict[str, list[Any]]] = {}
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._last_stamp: dict[str, int] = {}
        self._unsorted: set[str] = set()
        self._pending_rows = 0

    def append(self, table: str, row: Mapping[str, Any]) -> None:
        columns = self._columns.setdefault(table, {name: [] for name in TABLE_SCHEMAS[table].names})
        stamp = row["t_ns"]
        previous = self._last_stamp.get(table)
        if previous is not None and stamp < previous:
            self._unsorted.add(table)
        self._last_stamp[table] = stamp
        for name, values in columns.items():
            values.append(row[name])
        self._pending_rows += 1
        if self._pending_rows >= FLUSH_ROW_LIMIT:
            self.flush()

    def flush(self) -> None:
        for table, columns in self._columns.items():
            if columns["t_ns"]:
                batch = pa.RecordBatch.from_pydict(columns, schema=TABLE_SCHEMAS[table])
                self._writer(table).write_batch(batch)
                for values in columns.values():
                    values.clear()
        self._pending_rows = 0

    def close(self) -> None:
        self.flush()
        for writer in self._writers.values():
            writer.close()
        for table in self._unsorted:
            _sort_by_stamp(self._out_dir / f"{table}.parquet")

    def _writer(self, table: str) -> pq.ParquetWriter:
        if table not in self._writers:
            self._writers[table] = pq.ParquetWriter(self._out_dir / f"{table}.parquet", TABLE_SCHEMAS[table], compression=COMPRESSION)
        return self._writers[table]


def _sort_by_stamp(path: pathlib.Path) -> None:
    pl.read_parquet(path).sort("t_ns", maintain_order=True).write_parquet(path, compression=COMPRESSION)
