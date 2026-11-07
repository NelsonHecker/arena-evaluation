"""Aligns the flat telemetry tables onto the odometry time axis under the aligned column names the metrics read."""

from __future__ import annotations

import typing

import polars as pl

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import TopicBundle

TIME_COLUMN = "time_ns"
_ODOMETRY = {"odom_x": "pos_x", "odom_y": "pos_y", "odom_yaw": "yaw", "odom_vx": "vel_linear", "odom_wz": "vel_angular"}
_GROUND_TRUTH = {"x": "pos_x_gt", "y": "pos_y_gt", "yaw": "yaw_gt"}
_SCAN = {"scan_min_m": "scan_min", "range_max_m": "scan_range_max", "n_valid_rays": "n_rays"}
_CMD_VEL = {"linear_x": "linear_x", "linear_y": "linear_y", "linear_z": "linear_z", "angular_x": "angular_x", "angular_y": "angular_y", "angular_z": "angular_z"}
_COLLISIONS = {"event_count": "collision_event", "wall_count": "collision_wall", "static_count": "collision_static", "ped_count": "collision_pedestrian"}
_MONITOR = {"action_type": "action_type", "polygon_name": "polygon_name"}
_POWER = {"total_power_w": "total_power_w", "static_power_w": "static_power_w", "total_mech_power_w": "total_mechanical_power_w", "total_thermal_power_w": "total_thermal_power_w"}
_ENERGY = {"total_energy_consumed_wh": "total_energy_consumed_wh", "battery_soc_percent": "battery_soc_percent"}
_ACOUSTICS = {
    "total_level_af_dba": "total_level_af_dba",
    "total_level_zf_db": "total_level_zf_db",
    "baseline_level_dba": "baseline_level_dba",
    "drivetrain_level_dba": "drivetrain_level_dba",
    "uncertainty_dba": "uncertainty_1sigma_dba",
    "validity_flags": "validity_flags",
}
_MARKERS = {"label": "label"}


class TopicAligner:
    """Aligns the flat telemetry tables onto the odometry time axis with backward asof joins."""

    def __init__(self, tolerance_ns: int = 100_000_000):
        self.tolerance_ns = tolerance_ns

    def align(
        self,
        bundle: TopicBundle,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Align the robot's tables onto its odometry time axis, optionally filtered to [start_time_ns, end_time_ns]."""
        if bundle.robot_odom is None or _is_empty(bundle.robot_odom):
            return None
        primary = _window(_lazy(bundle.robot_odom), "t_ns", start_time_ns, end_time_ns)
        if _is_empty(primary):
            return None
        df = _select_aligned(primary, _ODOMETRY).sort(TIME_COLUMN)

        for table, renames in _secondary_tables(bundle):
            df = self._join(df, table, renames, start_time_ns, end_time_ns)

        if bundle.robot_characterization_phase is not None and not _is_empty(bundle.robot_characterization_phase):
            markers = _select_aligned(_lazy(bundle.robot_characterization_phase), _MARKERS).sort(TIME_COLUMN)
            df = df.join_asof(markers, on=TIME_COLUMN, strategy="backward")

        if isinstance(bundle.robot_odom, pl.LazyFrame):
            return df
        return df.collect()

    def _join(
        self,
        primary: pl.LazyFrame,
        table: pl.DataFrame | pl.LazyFrame | None,
        renames: dict[str, str],
        start_time_ns: int | None,
        end_time_ns: int | None,
    ) -> pl.LazyFrame:
        if table is None or _is_empty(table):
            return primary
        tolerance = self.tolerance_ns
        secondary = _window(_lazy(table), "t_ns", _widen(start_time_ns, -tolerance), _widen(end_time_ns, tolerance))
        if _is_empty(secondary):
            return primary
        secondary = _select_aligned(secondary, renames).sort(TIME_COLUMN)
        return primary.join_asof(secondary, on=TIME_COLUMN, strategy="backward", tolerance=tolerance)


def _secondary_tables(bundle: TopicBundle) -> tuple[tuple[pl.DataFrame | pl.LazyFrame | None, dict[str, str]], ...]:
    return (
        (bundle.robot_tracks, _GROUND_TRUTH),
        (bundle.robot_scan_summary, _SCAN),
        (bundle.robot_cmd_vel, _CMD_VEL),
        (bundle.collision_events, _COLLISIONS),
        (bundle.robot_collision_monitor, _MONITOR),
        (bundle.robot_power, _POWER),
        (bundle.robot_energy, _ENERGY),
        (bundle.robot_acoustics, _ACOUSTICS),
    )


def _select_aligned(frame: pl.LazyFrame, renames: dict[str, str]) -> pl.LazyFrame:
    return frame.select(pl.col("t_ns").alias(TIME_COLUMN), *(pl.col(source).alias(target) for source, target in renames.items()))


def _window(frame: pl.LazyFrame, time_column: str, start_ns: int | None, end_ns: int | None) -> pl.LazyFrame:
    if start_ns is not None:
        frame = frame.filter(pl.col(time_column) >= start_ns)
    if end_ns is not None:
        frame = frame.filter(pl.col(time_column) <= end_ns)
    return frame


def _widen(bound: int | None, margin: int) -> int | None:
    return None if bound is None else bound + margin


def _lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def _is_empty(frame: pl.DataFrame | pl.LazyFrame) -> bool:
    if isinstance(frame, pl.LazyFrame):
        return frame.limit(1).collect().height == 0
    return len(frame) == 0
