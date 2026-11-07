"""Multi-resolution scoped evaluator: one episode in, five typed fact tables out.

The evaluator drives the scope-aware calculators over the natural spatiotemporal windows
of an episode (episode, encounter, contact, pedestrian, robot) and pivots their outputs
into the star-schema fact contracts. Detector windows come from :mod:`processing.windows`
and are fully vectorized; calculators run once per window over a window-restricted episode
view, and the fact tables are materialized only at the serialization boundary.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import typing
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.metrics.social.contact_attribution import ContactAttributionCalculator
from arena_evaluation.processing.metrics.social.encounter_costs import EncounterCostsCalculator
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_PED_RADIUS_M,
    PEDESTRIAN_TRACKS_TOPIC,
    ROBOT_TRACKS_TOPIC,
    resolve_crowd_condition,
)
from arena_evaluation.processing.metrics.social.probst_responsibility import ProbstResponsibilityCalculator
from arena_evaluation.processing.windows import (
    ACTOR_PEDESTRIAN,
    DEFAULT_DETECTOR_PARAMS,
    DetectorParams,
    detect_contact_windows_lf,
    detect_encounter_windows_lf,
    empty_contact_windows,
    empty_encounter_windows,
    fleet_pairs_lf,
    pedestrian_pairs_lf,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

EVALUATOR_VERSION: Final[str] = "wp2-scoped-evaluator-v1"

NO_SAMPLES: Final[str] = "no_samples"
PEDSIM_UNAVAILABLE: Final[str] = "pedsim_unavailable"
REFERENCE_MISSING: Final[str] = "reference_missing"
ZERO_DENOMINATOR: Final[str] = "zero_denominator"
TOPIC_MISSING: Final[str] = "topic_missing"

COLLISION_CONTACTS_TOPIC: Final[str] = "collision_contacts"
ROBOT_POWER_TOPIC: Final[str] = "robot_power"
ROBOT_ACOUSTICS_TOPIC: Final[str] = "robot_acoustics"

ACOUSTIC_REFERENCE_PRESSURE_PA: Final[float] = 20e-6
ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2: Final[float] = ACOUSTIC_REFERENCE_PRESSURE_PA * ACOUSTIC_REFERENCE_PRESSURE_PA
DEFAULT_ROBOT_RADIUS_M: Final[float] = 0.25
DEFAULT_ROBOT_ACTOR_RADIUS_M: Final[float] = 0.3
CONTACT_PRE_WINDOW_S: Final[float] = 0.5
CONTACT_POST_WINDOW_S: Final[float] = 0.5
INTIMATE_CLEARANCE_M: Final[float] = 0.45
PERSONAL_CLEARANCE_M: Final[float] = 1.20
ROUND_TRIP_REACH_SHARE: Final[float] = 0.75
ROUND_TRIP_RETURN_SHARE: Final[float] = 0.25
ROUND_TRIP_MIN_DISPLACEMENT_M: Final[float] = 0.5
METRES_PER_WH_J: Final[float] = 3600.0
_ENV_PREFIX: Final[re.Pattern[str]] = re.compile(r"^env_[0-9]+_")

FACT_TABLE_NAMES: Final[tuple[str, ...]] = ("episodes", "encounters", "contacts", "pedestrians", "robots")

EPISODE_FACT_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "result": pl.String,
        "success": pl.Boolean,
        "time_to_goal_s": pl.Float64,
        "path_length_m": pl.Float64,
        "path_efficiency": pl.Float64,
        "total_energy_wh": pl.Float64,
        "energy_per_meter_j_m": pl.Float64,
        "mean_velocity_mps": pl.Float64,
        "max_acceleration_mps2": pl.Float64,
        "jerk_max_mps3": pl.Float64,
        "min_obstacle_clearance_m": pl.Float64,
        "min_ped_clearance_m": pl.Float64,
        "total_collisions_ped": pl.Int64,
        "total_collisions_static": pl.Int64,
        "success_no_robot_fault": pl.Boolean,
        "mar": pl.Float64,
        "pfi": pl.Float64,
        "robot_path_dev_m": pl.Float64,
        "total_acoustic_dose_pa2s": pl.Float64,
        "null_reason": pl.String,
    }
)

ENCOUNTER_FACT_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "encounter_id": pl.String,
        "actor_1": pl.String,
        "actor_2": pl.String,
        "actor_2_kind": pl.String,
        "t_start_ns": pl.Int64,
        "t_end_ns": pl.Int64,
        "duration_s": pl.Float64,
        "approach_type": pl.String,
        "min_clearance_m": pl.Float64,
        "detour_time_s": pl.Float64,
        "path_deviation_m": pl.Float64,
        "encounter_energy_j": pl.Float64,
        "energy_surge_ratio": pl.Float64,
        "sound_exposure_dose_pa2s": pl.Float64,
        "probst_R_robot": pl.Float64,
        "probst_R_actor2": pl.Float64,
        "probst_R_time": pl.Float64,
        "probst_E_robot": pl.Float64,
        "probst_E_actor2": pl.Float64,
        "had_contact": pl.Boolean,
        "contact_id": pl.String,
        "null_reason": pl.String,
    }
)

CONTACT_FACT_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "contact_id": pl.String,
        "encounter_id": pl.String,
        "collider_kind": pl.String,
        "obstacle_id": pl.String,
        "t_onset_ns": pl.Int64,
        "t_end_ns": pl.Int64,
        "duration_s": pl.Float64,
        "impact_speed_mps": pl.Float64,
        "robot_speed_at_impact_mps": pl.Float64,
        "obstacle_speed_at_impact_mps": pl.Float64,
        "closer_label": pl.String,
        "robot_closing_share": pl.Float64,
        "counterfactual_freeze_persists": pl.Boolean,
        "fault_attributed_to": pl.String,
        "null_reason": pl.String,
    }
)

PEDESTRIAN_FACT_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "env_id": pl.String,
        "ped_id": pl.String,
        "ped_name": pl.String,
        "ped_policy": pl.String,
        "is_blind": pl.Boolean,
        "desired_speed": pl.Float64,
        "radius_m": pl.Float64,
        "spawn_x": pl.Float64,
        "spawn_y": pl.Float64,
        "spawn_yaw": pl.Float64,
        "total_distance_m": pl.Float64,
        "mean_speed_mps": pl.Float64,
        "cumulative_sound_dose_pa2s": pl.Float64,
        "total_encounters": pl.Int64,
        "total_intrusions_intimate": pl.Int64,
        "total_intrusions_personal": pl.Int64,
        "ped_path_deflection_m": pl.Float64,
        "round_trips_completed": pl.Int64,
        "null_reason": pl.String,
    }
)

ROBOT_FACT_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "env_id": pl.String,
        "robot_id": pl.String,
        "robot_name": pl.String,
        "path_length_m": pl.Float64,
        "mean_speed_mps": pl.Float64,
        "max_speed_mps": pl.Float64,
        "mean_power_w": pl.Float64,
        "peak_power_w": pl.Float64,
        "mechanical_work_j": pl.Float64,
        "thermal_energy_j": pl.Float64,
        "electrical_energy_wh": pl.Float64,
        "acoustic_emission_pa2s": pl.Float64,
        "drivetrain_scrub_j": pl.Float64,
        "null_reason": pl.String,
    }
)

FACT_SCHEMAS: Final[dict[str, pl.Schema]] = {
    "episodes": EPISODE_FACT_SCHEMA,
    "encounters": ENCOUNTER_FACT_SCHEMA,
    "contacts": CONTACT_FACT_SCHEMA,
    "pedestrians": PEDESTRIAN_FACT_SCHEMA,
    "robots": ROBOT_FACT_SCHEMA,
}

_EPISODE_NULLABLE: Final[tuple[str, ...]] = (
    "result",
    "success",
    "time_to_goal_s",
    "path_length_m",
    "path_efficiency",
    "total_energy_wh",
    "energy_per_meter_j_m",
    "mean_velocity_mps",
    "max_acceleration_mps2",
    "jerk_max_mps3",
    "min_obstacle_clearance_m",
    "min_ped_clearance_m",
    "total_collisions_ped",
    "total_collisions_static",
    "success_no_robot_fault",
    "mar",
    "pfi",
    "robot_path_dev_m",
    "total_acoustic_dose_pa2s",
)
_ENCOUNTER_NULLABLE: Final[tuple[str, ...]] = (
    "detour_time_s",
    "path_deviation_m",
    "encounter_energy_j",
    "energy_surge_ratio",
    "sound_exposure_dose_pa2s",
    "probst_R_robot",
    "probst_R_actor2",
    "probst_R_time",
    "probst_E_robot",
    "probst_E_actor2",
)
_CONTACT_NULLABLE: Final[tuple[str, ...]] = (
    "impact_speed_mps",
    "robot_speed_at_impact_mps",
    "obstacle_speed_at_impact_mps",
    "closer_label",
    "robot_closing_share",
)
_PEDESTRIAN_NULLABLE: Final[tuple[str, ...]] = (
    "ped_policy",
    "desired_speed",
    "radius_m",
    "spawn_x",
    "spawn_y",
    "spawn_yaw",
    "mean_speed_mps",
    "cumulative_sound_dose_pa2s",
    "ped_path_deflection_m",
)
_ROBOT_NULLABLE: Final[tuple[str, ...]] = (
    "path_length_m",
    "mean_speed_mps",
    "max_speed_mps",
    "mean_power_w",
    "peak_power_w",
    "mechanical_work_j",
    "thermal_energy_j",
    "electrical_energy_wh",
    "acoustic_emission_pa2s",
    "drivetrain_scrub_j",
)


@dataclasses.dataclass(frozen=True, slots=True)
class ScopedFactTables:
    """The five typed fact tables of one evaluated episode."""

    episodes: pl.DataFrame
    encounters: pl.DataFrame
    contacts: pl.DataFrame
    pedestrians: pl.DataFrame
    robots: pl.DataFrame

    def by_name(self) -> dict[str, pl.DataFrame]:
        """Fact tables keyed by their star-schema table name."""
        return {"episodes": self.episodes, "encounters": self.encounters, "contacts": self.contacts, "pedestrians": self.pedestrians, "robots": self.robots}


def write_fact_tables(tables: ScopedFactTables, directory: pathlib.Path) -> None:
    """Write the five fact tables as ``fact_<name>.parquet`` in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.by_name().items():
        frame.write_parquet(directory / f"fact_{name}.parquet")


def _table(episode: AlignedEpisodeBundle, topic: str) -> pl.DataFrame | None:
    """Collected flat table of an episode topic, None when absent."""
    topics = episode.topics
    if topics is None:
        return None
    table = topics.get(topic)
    if table is None:
        return None
    return table.collect() if isinstance(table, pl.LazyFrame) else table


def _has_rows(table: pl.DataFrame | None) -> bool:
    return table is not None and len(table) > 0


def _float(value: object) -> float | None:
    return None if value is None else float(typing.cast(float, value))


def _int(value: object) -> int | None:
    return None if value is None else int(typing.cast(int, value))


def _row_null_reason(row: Mapping[str, object], nullable: tuple[str, ...]) -> str:
    """``field:code`` provenance for every mapped field that stayed null."""
    return ";".join(f"{field}:{NO_SAMPLES}" for field in nullable if row.get(field) is None)


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    """Typed fact frame of the given rows, empty with the canonical schema when there are none."""
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


class ScopedEvaluator:
    """Evaluates one episode at episode, encounter, contact, pedestrian and robot resolution."""

    def __init__(self, robot_params: RobotParams, metrics_config: Mapping[str, Any] | None = None, detector_params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> None:
        self.robot_params = robot_params
        self.metrics_config: dict[str, Any] = dict(metrics_config or {})
        self.detector_params = detector_params

    def evaluate(self, episode: AlignedEpisodeBundle, *, run_id: str, episode_id: int, episode_metrics: Mapping[str, Any] | None = None) -> ScopedFactTables:
        """Evaluate one episode and return its five fact tables."""
        topics = episode.topics or {}
        robot_tracks = _table(episode, ROBOT_TRACKS_TOPIC)
        pedestrian_tracks = _table(episode, PEDESTRIAN_TRACKS_TOPIC)
        contact_windows = empty_contact_windows()
        encounter_windows = empty_encounter_windows()

        if _has_rows(robot_tracks):
            encounter_windows = self._detect_encounters(episode, typing.cast(pl.DataFrame, robot_tracks), pedestrian_tracks)
            contact_windows = self._detect_contacts(episode, encounter_windows)

        metrics = dict(episode_metrics) if episode_metrics is not None else self._run_registry(episode)
        contact_rows = self._contact_fact_rows(episode, run_id, episode_id, contact_windows, encounter_windows)
        encounter_rows = self._encounter_fact_rows(episode, run_id, episode_id, encounter_windows, contact_rows)
        episode_row = self._episode_fact_row(episode, run_id, episode_id, metrics, encounter_rows, contact_rows)
        pedestrians = self._pedestrian_fact_frame(episode, run_id, episode_id, encounter_windows, encounter_rows)
        robots = self._robot_fact_frame(episode, run_id, episode_id, metrics)
        return ScopedFactTables(
            episodes=_frame([episode_row], EPISODE_FACT_SCHEMA),
            encounters=_frame(encounter_rows, ENCOUNTER_FACT_SCHEMA),
            contacts=_frame(contact_rows, CONTACT_FACT_SCHEMA),
            pedestrians=pedestrians,
            robots=robots,
        )

    def _detect_encounters(self, episode: AlignedEpisodeBundle, robot_tracks: pl.DataFrame, pedestrian_tracks: pl.DataFrame | None) -> pl.DataFrame:
        """Encounter windows of this robot with its pedestrians and, when known, the other robots."""
        pairs = []
        if _has_rows(pedestrian_tracks):
            pairs.append(pedestrian_pairs_lf(robot_tracks, typing.cast(pl.DataFrame, pedestrian_tracks), float(self.robot_params.robot_radius), self.detector_params))
        if episode.fleet:
            pairs.append(fleet_pairs_lf(robot_tracks, episode.fleet, float(self.robot_params.robot_radius), params=self.detector_params))
        if not pairs:
            return empty_encounter_windows()
        pairs_lf = pairs[0] if len(pairs) == 1 else pl.concat(pairs)
        return detect_encounter_windows_lf(pairs_lf, self.detector_params).collect()

    def _detect_contacts(self, episode: AlignedEpisodeBundle, encounters: pl.DataFrame) -> pl.DataFrame:
        """Contact windows of the recorded collision stream, linked to their enclosing encounters."""
        collision_contacts = _table(episode, COLLISION_CONTACTS_TOPIC)
        if not _has_rows(collision_contacts):
            return empty_contact_windows()
        return detect_contact_windows_lf(typing.cast(pl.DataFrame, collision_contacts), encounters, self.detector_params).collect()

    def _run_registry(self, episode: AlignedEpisodeBundle) -> dict[str, Any]:
        """Episode-scope registry output, used when the caller does not pass the already-computed metrics."""
        topics = episode.topics or {}
        pedestrian_tracks = _table(episode, PEDESTRIAN_TRACKS_TOPIC)
        registry = MetricRegistry(self.robot_params, world=episode.map, metrics_config=dict(self.metrics_config))
        return registry.run(episode, available_topics=set(topics), pedsim_available=_has_rows(pedestrian_tracks))

    def _window_episode(self, episode: AlignedEpisodeBundle, t_start_ns: int, t_end_ns: int, actor_id: str, actor_kind: str) -> AlignedEpisodeBundle:
        """Episode view cut to ``[t_start_ns, t_end_ns]`` with the pedestrian stream restricted to the window's actor."""
        topics: dict[str, pl.DataFrame] = {}
        for name, table in (episode.topics or {}).items():
            frame = table.collect() if isinstance(table, pl.LazyFrame) else table
            if name == PEDESTRIAN_TRACKS_TOPIC:
                frame = self._actor_track_frame(episode, frame, actor_id, actor_kind)
            if len(frame) > 0:
                frame = frame.filter((pl.col("t_ns") >= t_start_ns) & (pl.col("t_ns") <= t_end_ns))
            topics[name] = frame
        return dataclasses.replace(episode, data=pl.DataFrame(), topics=topics, fleet={})

    def _actor_track_frame(self, episode: AlignedEpisodeBundle, pedestrian_tracks: pl.DataFrame, actor_id: str, actor_kind: str) -> pl.DataFrame:
        """Pedestrian-shaped track of the window's actor: the pedestrian rows or another robot's task poses."""
        if actor_kind == ACTOR_PEDESTRIAN or len(pedestrian_tracks) == 0:
            if len(pedestrian_tracks) == 0:
                return pedestrian_tracks
            return pedestrian_tracks.filter(pl.col("ped_id") == actor_id)
        task_poses = episode.fleet.get(actor_id)
        if task_poses is None or len(task_poses) < 2:
            return pedestrian_tracks.head(0)
        previous_time_s = pl.col("t_ns").diff().cast(pl.Float64) / 1e9
        dt_s = pl.when(previous_time_s > 0.0).then(previous_time_s).otherwise(None)
        return (
            task_poses.sort("t_ns")
            .with_columns(
                ped_id=pl.lit(actor_id),
                ped_name=pl.lit(actor_id),
                x=pl.col("pos_x").cast(pl.Float64),
                y=pl.col("pos_y").cast(pl.Float64),
                vx=(pl.col("pos_x").diff() / dt_s).fill_null(0.0),
                vy=(pl.col("pos_y").diff() / dt_s).fill_null(0.0),
                radius_m=pl.lit(DEFAULT_ROBOT_ACTOR_RADIUS_M, dtype=pl.Float64),
                desired_speed=pl.lit(None, dtype=pl.Float64),
            )
            .select("t_ns", "ped_id", "ped_name", "x", "y", "vx", "vy", "radius_m", "desired_speed")
        )

    def _window_config(self, actor_id: str, t_start_ns: int, t_end_ns: int, crowd_condition: str) -> dict[str, Any]:
        """Calculator config addressing one encounter window."""
        config = dict(self.metrics_config)
        config.update(
            target_agent_id=actor_id,
            window_start_ns=int(t_start_ns),
            window_end_ns=int(t_end_ns),
            crowd_condition=crowd_condition,
            default_ped_radius_m=float(config.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M)),
        )
        return config

    @staticmethod
    def _run_calculator(calculator_cls: type[BaseMetricCalculator], robot_params: RobotParams, config: Mapping[str, Any], episode: AlignedEpisodeBundle) -> dict[str, Any]:
        calculator = calculator_cls(robot_params)
        calculator.metrics_config = dict(config)
        return calculator.calculate(episode, {})

    def _window_mean_power_w(self, episode: AlignedEpisodeBundle, t_start_ns: int, t_end_ns: int) -> float | None:
        """Mean recorded total electrical power inside the window, None without a power table."""
        power = _table(episode, ROBOT_POWER_TOPIC)
        if not _has_rows(power):
            return None
        window = typing.cast(pl.DataFrame, power).filter((pl.col("t_ns") >= t_start_ns) & (pl.col("t_ns") <= t_end_ns)).get_column("total_power_w").drop_nulls()
        return float(window.mean()) if len(window) > 0 else None

    def _episode_mean_power_w(self, episode: AlignedEpisodeBundle) -> float | None:
        power = _table(episode, ROBOT_POWER_TOPIC)
        if not _has_rows(power):
            return None
        values = typing.cast(pl.DataFrame, power).get_column("total_power_w").drop_nulls()
        return float(values.mean()) if len(values) > 0 else None

    def _acoustic_dose_pa2s(self, episode: AlignedEpisodeBundle, t_start_ns: int | None = None, t_end_ns: int | None = None) -> float | None:
        """Integral of the linearized emitted sound pressure squared over the window, Pa^2*s."""
        acoustics = _table(episode, ROBOT_ACOUSTICS_TOPIC)
        if not _has_rows(acoustics):
            return None
        frame = typing.cast(pl.DataFrame, acoustics).select("t_ns", "total_level_af_dba").drop_nulls().sort("t_ns")
        if t_start_ns is not None:
            frame = frame.filter(pl.col("t_ns") >= t_start_ns)
        if t_end_ns is not None:
            frame = frame.filter(pl.col("t_ns") <= t_end_ns)
        if len(frame) < 2:
            return None
        time_ns = frame.get_column("t_ns").to_numpy().astype(np.int64)
        level_dba = frame.get_column("total_level_af_dba").to_numpy().astype(np.float64)
        dt_s = np.diff(time_ns).astype(np.float64) / 1e9
        linear_pressure_sq = ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2 * np.power(10.0, level_dba / 10.0)
        return float(np.sum(0.5 * (linear_pressure_sq[:-1] + linear_pressure_sq[1:]) * dt_s))

    @staticmethod
    def _short_robot_token(robot_id: str) -> str:
        """Compact robot token used in fact ids: ``env_0_jackal_0`` -> ``jackal_0``."""
        return _ENV_PREFIX.sub("", robot_id, count=1)

    @staticmethod
    def _robot_actor_id(episode: AlignedEpisodeBundle, robot_tracks: pl.DataFrame | None) -> str:
        """Robot identity of the bundle: its robot column, else its bundle name."""
        if _has_rows(robot_tracks):
            robot_ids = typing.cast(pl.DataFrame, robot_tracks).get_column("robot_id").drop_nulls()
            if len(robot_ids) > 0:
                return str(robot_ids[0])
        return str(episode.robot_name or "robot")

    def _episode_fact_row(self, episode: AlignedEpisodeBundle, run_id: str, episode_id: int, metrics: Mapping[str, Any], encounter_rows: list[dict[str, object]], contact_rows: list[dict[str, object]]) -> dict[str, object]:
        """Episode-level fact row from the registry output and the child fact tables."""
        robot_tracks = _table(episode, ROBOT_TRACKS_TOPIC)
        robot_id = self._robot_actor_id(episode, robot_tracks)
        robot_fault_contacts = sum(1 for contact in contact_rows if contact.get("fault_attributed_to") == "robot")
        pedestrian_contacts = sum(1 for contact in contact_rows if contact.get("collider_kind") == ACTOR_PEDESTRIAN)
        static_contacts = sum(1 for contact in contact_rows if contact.get("collider_kind") in ("wall", "static"))
        success = metrics.get("success")
        energy_per_meter = _float(metrics.get("energy_per_meter"))
        collisions_ped = _int(metrics.get("collision_amount_pedestrian"))
        collisions_wall = _int(metrics.get("collision_amount_wall"))
        collisions_static = _int(metrics.get("collision_amount_static"))
        row: dict[str, object] = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "robot_id": robot_id,
            "result": metrics.get("result"),
            "success": None if success is None else bool(success),
            "time_to_goal_s": _float(metrics.get("time_to_goal")),
            "path_length_m": _float(metrics.get("path_length")),
            "path_efficiency": _float(metrics.get("path_efficiency")),
            "total_energy_wh": _float(metrics.get("energy_total_wh")),
            "energy_per_meter_j_m": None if energy_per_meter is None else energy_per_meter * METRES_PER_WH_J,
            "mean_velocity_mps": _float(metrics.get("velocity_mean")),
            "max_acceleration_mps2": _float(metrics.get("acceleration_max")),
            "jerk_max_mps3": _float(metrics.get("jerk_max")),
            "min_obstacle_clearance_m": _float(metrics.get("min_obstacle_clearance")),
            "min_ped_clearance_m": _float(metrics.get("min_pedestrian_clearance")),
            "total_collisions_ped": collisions_ped if collisions_ped is not None else pedestrian_contacts,
            "total_collisions_static": (collisions_wall or 0) + (collisions_static or 0) if collisions_wall is not None or collisions_static is not None else static_contacts,
            "success_no_robot_fault": None if success is None else bool(success) and robot_fault_contacts == 0,
            "mar": _float(metrics.get("mar")),
            "pfi": _float(metrics.get("pfi")),
            "robot_path_dev_m": _float(metrics.get("robot_path_deviation_m")),
            "total_acoustic_dose_pa2s": self._acoustic_dose_pa2s(episode),
        }
        row["null_reason"] = self._episode_null_reason(episode, row)
        return row

    def _episode_null_reason(self, episode: AlignedEpisodeBundle, row: Mapping[str, object]) -> str:
        """Episode provenance: missing-field codes, missing trajectory/pedestrian stream topics."""
        reasons = [_row_null_reason(row, _EPISODE_NULLABLE)]
        if not _has_rows(_table(episode, ROBOT_TRACKS_TOPIC)):
            reasons.append(f"{TOPIC_MISSING}:{ROBOT_TRACKS_TOPIC}")
        if not _has_rows(_table(episode, PEDESTRIAN_TRACKS_TOPIC)):
            reasons.append(PEDSIM_UNAVAILABLE)
        return ";".join(reason for reason in reasons if reason)

    def _contact_fact_rows(self, episode: AlignedEpisodeBundle, run_id: str, episode_id: int, contact_windows: pl.DataFrame, encounter_windows: pl.DataFrame) -> list[dict[str, object]]:
        """One fact row per detected contact, attributed by the contact-attribution calculator."""
        if len(contact_windows) == 0:
            return []
        robot_tracks = _table(episode, ROBOT_TRACKS_TOPIC)
        robot_id = self._robot_actor_id(episode, robot_tracks)
        crowd_condition = resolve_crowd_condition(episode, self.metrics_config)
        contact_token = self._short_robot_token(robot_id)
        encounter_ids = {int(index): self._encounter_id(episode_id, contact_token, int(index)) for index in encounter_windows.get_column("encounter_idx").to_numpy()}
        rows: list[dict[str, object]] = []
        columns = contact_windows
        for contact_idx, collider_kind, obstacle_id, t_onset_ns, t_end_ns, obstacle_x, obstacle_y, encounter_idx in zip(
            columns.get_column("contact_idx").to_numpy(),
            columns.get_column("collider_kind").to_numpy(),
            columns.get_column("obstacle_id").to_numpy(),
            columns.get_column("t_onset_ns").to_numpy(),
            columns.get_column("t_end_ns").to_numpy(),
            columns.get_column("obstacle_x").to_numpy(),
            columns.get_column("obstacle_y").to_numpy(),
            columns.get_column("encounter_idx").fill_null(-1).to_numpy(),
            strict=True,
        ):
            internal_kind = ACTOR_PEDESTRIAN if collider_kind in (ACTOR_PEDESTRIAN, "robot") else str(collider_kind)
            actor_id = str(obstacle_id)
            pre_window_s = float(self.metrics_config.get("pre_contact_window_s", CONTACT_PRE_WINDOW_S))
            post_window_s = float(self.metrics_config.get("post_contact_window_s", CONTACT_POST_WINDOW_S))
            window = self._window_episode(episode, int(t_onset_ns - round(pre_window_s * 1e9)), int(t_end_ns + round(post_window_s * 1e9)), actor_id, internal_kind)
            static_point = None if collider_kind in (ACTOR_PEDESTRIAN, "robot") else (float(obstacle_x), float(obstacle_y))
            config = dict(self.metrics_config)
            config.update(
                contact_t0_ns=int(t_onset_ns),
                contact_t_end_ns=int(t_end_ns),
                obstacle_kind=internal_kind,
                obstacle_id=actor_id,
                target_agent_id=actor_id if internal_kind == ACTOR_PEDESTRIAN else None,
                static_point=static_point,
                crowd_condition=crowd_condition,
                pre_contact_window_s=pre_window_s,
                post_contact_window_s=post_window_s,
            )
            attribution = self._run_calculator(ContactAttributionCalculator, self.robot_params, config, window).get("contact_records", [])
            record = attribution[0] if attribution else {}
            encounter_index = None if int(encounter_idx) < 0 else int(encounter_idx)
            row: dict[str, object] = {
                "run_id": run_id,
                "episode_id": int(episode_id),
                "robot_id": robot_id,
                "contact_id": self._contact_id(episode_id, contact_token, int(contact_idx)),
                "encounter_id": None if encounter_index is None else encounter_ids.get(encounter_index),
                "collider_kind": str(collider_kind),
                "obstacle_id": actor_id,
                "t_onset_ns": int(t_onset_ns),
                "t_end_ns": int(t_end_ns),
                "duration_s": float(max(0, int(t_end_ns) - int(t_onset_ns))) / 1e9,
                "impact_speed_mps": _float(record.get("relative_impact_speed_mps")),
                "robot_speed_at_impact_mps": _float(record.get("v_robot_impact_mps")),
                "obstacle_speed_at_impact_mps": _float(record.get("v_target_impact_mps")),
                "closer_label": record.get("closer_label"),
                "robot_closing_share": _float(record.get("s_robot")),
                "counterfactual_freeze_persists": record.get("counterfactual_freeze_persists"),
                "fault_attributed_to": self._fault_attribution(collider_kind, record),
            }
            row["null_reason"] = _row_null_reason(row, _CONTACT_NULLABLE)
            rows.append(row)
        return rows

    @staticmethod
    def _fault_attribution(collider_kind: object, record: Mapping[str, Any]) -> str:
        """Final blame of a contact: a static-world impact is the robot's, otherwise the counterfactual record decides."""
        if collider_kind in ("wall", "static"):
            return "robot"
        if record.get("is_robot_fault") is True or record.get("is_robot_fault_lower_bound") is True:
            return "robot"
        if record.get("is_robot_fault") is False:
            return "pedestrian"
        return "unavoidable"

    def _encounter_fact_rows(self, episode: AlignedEpisodeBundle, run_id: str, episode_id: int, encounter_windows: pl.DataFrame, contact_rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """One fact row per encounter window, from the encounter-cost and Probst calculators."""
        if len(encounter_windows) == 0:
            return []
        robot_tracks = _table(episode, ROBOT_TRACKS_TOPIC)
        robot_id = self._robot_actor_id(episode, robot_tracks)
        crowd_condition = resolve_crowd_condition(episode, self.metrics_config)
        robot_token = self._short_robot_token(robot_id)
        episode_mean_power_w = self._episode_mean_power_w(episode)
        contacts_by_encounter: dict[str, dict[str, object]] = {}
        for contact in contact_rows:
            encounter_id = contact.get("encounter_id")
            if isinstance(encounter_id, str) and encounter_id not in contacts_by_encounter:
                contacts_by_encounter[encounter_id] = contact
        rows: list[dict[str, object]] = []
        columns = encounter_windows
        for encounter_idx, actor_id, actor_kind, t_start_ns, t_end_ns, min_clearance_m, approach_type in zip(
            columns.get_column("encounter_idx").fill_null(-1).to_numpy(),
            columns.get_column("actor_id").to_numpy(),
            columns.get_column("actor_kind").to_numpy(),
            columns.get_column("t_start_ns").to_numpy(),
            columns.get_column("t_end_ns").to_numpy(),
            columns.get_column("min_clearance_m").to_numpy(),
            columns.get_column("approach_type").to_numpy(),
            strict=True,
        ):
            actor = str(actor_id)
            kind = str(actor_kind)
            window = self._window_episode(episode, int(t_start_ns), int(t_end_ns), actor, kind)
            config = self._window_config(actor, int(t_start_ns), int(t_end_ns), crowd_condition)
            costs = self._run_calculator(EncounterCostsCalculator, self.robot_params, config, window)
            probst = self._run_calculator(ProbstResponsibilityCalculator, self.robot_params, config, window)
            encounter_id = self._encounter_id(episode_id, robot_token, int(encounter_idx))
            window_mean_power_w = self._window_mean_power_w(episode, int(t_start_ns), int(t_end_ns))
            energy_surge_ratio = None
            if window_mean_power_w is not None and episode_mean_power_w is not None and episode_mean_power_w > 0.0:
                energy_surge_ratio = window_mean_power_w / episode_mean_power_w
            linked_contact = contacts_by_encounter.get(encounter_id)
            row: dict[str, object] = {
                "run_id": run_id,
                "episode_id": int(episode_id),
                "robot_id": robot_id,
                "encounter_id": encounter_id,
                "actor_1": robot_id,
                "actor_2": actor,
                "actor_2_kind": kind,
                "t_start_ns": int(t_start_ns),
                "t_end_ns": int(t_end_ns),
                "duration_s": float(max(0, int(t_end_ns) - int(t_start_ns))) / 1e9,
                "approach_type": str(approach_type),
                "min_clearance_m": _float(min_clearance_m),
                "detour_time_s": _float(costs.get("cost_ped_delay_s")),
                "path_deviation_m": _float(costs.get("cost_ped_deflection_m")),
                "encounter_energy_j": _float(costs.get("cost_energy_j")),
                "energy_surge_ratio": energy_surge_ratio,
                "sound_exposure_dose_pa2s": _float(costs.get("cost_sound_pa2s")),
                "probst_R_robot": _float(probst.get("probst_r_robot")),
                "probst_R_actor2": _float(probst.get("probst_r_target")),
                "probst_R_time": _float(probst.get("probst_r_time")),
                "probst_E_robot": _float(probst.get("probst_e_robot")),
                "probst_E_actor2": _float(probst.get("probst_e_target")),
                "had_contact": linked_contact is not None,
                "contact_id": None if linked_contact is None else linked_contact.get("contact_id"),
            }
            row["null_reason"] = self._encounter_null_reason(row, has_reference=bool(self.metrics_config.get("reference_peds") or self.metrics_config.get("reference_episode")), has_power=_has_rows(_table(episode, ROBOT_POWER_TOPIC)))
            rows.append(row)
        return rows

    @staticmethod
    def _encounter_null_reason(row: Mapping[str, object], has_reference: bool, has_power: bool) -> str:
        """Encounter provenance: twin-dependent values are reference_missing without a twin, the power ratio is zero_denominator without power."""
        reasons: list[str] = []
        for field in _ENCOUNTER_NULLABLE:
            if row.get(field) is None:
                if field in ("detour_time_s", "path_deviation_m") and not has_reference:
                    reasons.append(f"{field}:{REFERENCE_MISSING}")
                elif field == "energy_surge_ratio" and has_power:
                    reasons.append(f"{field}:{ZERO_DENOMINATOR}")
                else:
                    reasons.append(f"{field}:{NO_SAMPLES}")
        return ";".join(reasons)

    def _pedestrian_fact_frame(self, episode: AlignedEpisodeBundle, run_id: str, episode_id: int, encounter_windows: pl.DataFrame, encounter_rows: list[dict[str, object]]) -> pl.DataFrame:
        """Per-pedestrian lifetime summaries, joined from the tracks, the pair stream and the encounter facts."""
        pedestrian_tracks = _table(episode, PEDESTRIAN_TRACKS_TOPIC)
        if not _has_rows(pedestrian_tracks):
            return pl.DataFrame(schema=PEDESTRIAN_FACT_SCHEMA)
        tracks = typing.cast(pl.DataFrame, pedestrian_tracks).sort(["ped_id", "t_ns"])
        crowd_condition = resolve_crowd_condition(episode, self.metrics_config)
        trajectory = tracks.with_columns(
            step_distance_m=(pl.col("x").diff().over("ped_id").pow(2) + pl.col("y").diff().over("ped_id").pow(2)).sqrt(),
            speed_mps=pl.col("speed_mps").fill_null((pl.col("vx").pow(2) + pl.col("vy").pow(2)).sqrt()),
        ).with_columns(
            spawn_x=pl.col("x").first().over("ped_id"),
            spawn_y=pl.col("y").first().over("ped_id"),
            displacement_m=((pl.col("x") - pl.col("x").first().over("ped_id")).pow(2) + (pl.col("y") - pl.col("y").first().over("ped_id")).pow(2)).sqrt(),
        )
        stats = trajectory.group_by("ped_id").agg(
            env_id=pl.col("env_id").first(),
            ped_name=pl.col("ped_name").drop_nulls().first(),
            ped_policy=pl.col("policy").drop_nulls().first(),
            desired_speed=pl.col("desired_speed").drop_nulls().first(),
            radius_m=pl.col("radius_m").drop_nulls().first(),
            spawn_x=pl.col("spawn_x").first(),
            spawn_y=pl.col("spawn_y").first(),
            spawn_yaw=pl.col("yaw").drop_nulls().first(),
            total_distance_m=pl.col("step_distance_m").sum(),
            mean_speed_mps=pl.col("speed_mps").mean(),
            max_displacement_m=pl.col("displacement_m").max(),
        )
        round_trips = self._round_trips_frame(trajectory)
        intrusions = self._intrusion_counts_frame(episode)
        encounter_aggregates = self._encounter_aggregates_frame(encounter_rows)
        facts = (
            stats.join(round_trips, on="ped_id", how="left")
            .join(intrusions, on="ped_id", how="left")
            .join(encounter_aggregates, on="ped_id", how="left")
            .with_columns(
                run_id=pl.lit(run_id),
                episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
                is_blind=pl.lit(crowd_condition == "blind"),
                total_encounters=pl.col("total_encounters").fill_null(0).cast(pl.Int64),
                total_intrusions_intimate=pl.col("total_intrusions_intimate").fill_null(0).cast(pl.Int64),
                total_intrusions_personal=pl.col("total_intrusions_personal").fill_null(0).cast(pl.Int64),
                round_trips_completed=pl.col("round_trips_completed").fill_null(0).cast(pl.Int64),
            )
        )
        reason_parts = [pl.when(pl.col(column).is_null()).then(pl.lit(f"{column}:{NO_SAMPLES}")) for column in _PEDESTRIAN_NULLABLE]
        facts = facts.with_columns(null_reason=pl.concat_str(reason_parts, separator=";", ignore_nulls=True).fill_null(""))
        return facts.select(PEDESTRIAN_FACT_SCHEMA.names()).cast(PEDESTRIAN_FACT_SCHEMA)

    @staticmethod
    def _round_trips_frame(trajectory: pl.DataFrame) -> pl.DataFrame:
        """Completed patrol rounds per pedestrian: an away reach followed by a return, from displacement extremes."""
        reach_m = ROUND_TRIP_REACH_SHARE * pl.col("max_displacement_m")
        returned_m = ROUND_TRIP_RETURN_SHARE * pl.col("max_displacement_m")
        events = (
            trajectory.join(trajectory.group_by("ped_id").agg(max_displacement_m=pl.col("displacement_m").max()), on="ped_id")
            .filter(pl.col("max_displacement_m") > ROUND_TRIP_MIN_DISPLACEMENT_M)
            .filter((pl.col("displacement_m") >= reach_m) | (pl.col("displacement_m") <= returned_m))
            .with_columns(away=pl.col("displacement_m") >= reach_m)
            .with_columns(next_away=pl.col("away").shift(-1).over("ped_id"))
        )
        completed = events.filter(pl.col("away") & ~pl.col("next_away").fill_null(True))
        return completed.group_by("ped_id").len().rename({"len": "round_trips_completed"})

    def _intrusion_counts_frame(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Zone-entry counts per pedestrian from the pair stream (edge clearance thresholds)."""
        pedestrian_tracks = _table(episode, PEDESTRIAN_TRACKS_TOPIC)
        robot_tracks = _table(episode, ROBOT_TRACKS_TOPIC)
        if not _has_rows(pedestrian_tracks) or not _has_rows(robot_tracks):
            return pl.DataFrame(schema={"ped_id": pl.String, "total_intrusions_intimate": pl.Int64, "total_intrusions_personal": pl.Int64})
        robot_radius_m = float(self.robot_params.robot_radius)
        pairs = pedestrian_pairs_lf(typing.cast(pl.DataFrame, robot_tracks), typing.cast(pl.DataFrame, pedestrian_tracks), robot_radius_m, DEFAULT_DETECTOR_PARAMS).collect().sort(["actor_id", "t_ns"])
        entries = pairs.with_columns(
            intimate=pl.col("edge_clearance_m") < INTIMATE_CLEARANCE_M,
            personal=pl.col("edge_clearance_m") < PERSONAL_CLEARANCE_M,
        ).with_columns(
            intimate_entry=pl.col("intimate") & ~pl.col("intimate").shift(1, fill_value=False).over("actor_id"),
            personal_entry=pl.col("personal") & ~pl.col("personal").shift(1, fill_value=False).over("actor_id"),
        )
        return entries.group_by(pl.col("actor_id").alias("ped_id")).agg(
            total_intrusions_intimate=pl.col("intimate_entry").sum().cast(pl.Int64),
            total_intrusions_personal=pl.col("personal_entry").sum().cast(pl.Int64),
        )

    @staticmethod
    def _encounter_aggregates_frame(encounter_rows: list[dict[str, object]]) -> pl.DataFrame:
        """Per-actor aggregates of the encounter facts: count, sound dose and reference-twin deflection."""
        if not encounter_rows:
            return pl.DataFrame(schema={"ped_id": pl.String, "total_encounters": pl.Int64, "cumulative_sound_dose_pa2s": pl.Float64, "ped_path_deflection_m": pl.Float64})
        frame = pl.DataFrame(encounter_rows)
        return frame.group_by(pl.col("actor_2").alias("ped_id")).agg(
            total_encounters=pl.len().cast(pl.Int64),
            cumulative_sound_dose_pa2s=pl.col("sound_exposure_dose_pa2s").sum(),
            ped_path_deflection_m=pl.col("path_deviation_m").sum(),
        )

    def _robot_fact_frame(self, episode: AlignedEpisodeBundle, run_id: str, episode_id: int, metrics: Mapping[str, Any]) -> pl.DataFrame:
        """Per-robot lifetime summary of work, emissions and drivetrain losses."""
        robot_tracks = _table(episode, ROBOT_TRACKS_TOPIC)
        robot_id = self._robot_actor_id(episode, robot_tracks)
        env_id = None
        if _has_rows(robot_tracks):
            environments = typing.cast(pl.DataFrame, robot_tracks).get_column("env_id").drop_nulls()
            env_id = str(environments[0]) if len(environments) > 0 else None
        duration_s = _float(metrics.get("time_to_goal"))
        electrical_energy_wh = _float(metrics.get("energy_total_wh"))
        mechanical_energy_wh = _float(metrics.get("energy_mechanical_wh"))
        thermal_energy_wh = _float(metrics.get("energy_thermal_wh"))
        mean_power_w = None
        if electrical_energy_wh is not None and duration_s is not None and duration_s > 0.0:
            mean_power_w = electrical_energy_wh * METRES_PER_WH_J / duration_s
        row: dict[str, object] = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "env_id": env_id,
            "robot_id": robot_id,
            "robot_name": episode.robot_name,
            "path_length_m": _float(metrics.get("path_length")),
            "mean_speed_mps": _float(metrics.get("velocity_mean")),
            "max_speed_mps": _float(metrics.get("velocity_max")),
            "mean_power_w": mean_power_w,
            "peak_power_w": _float(metrics.get("power_peak_w")),
            "mechanical_work_j": None if mechanical_energy_wh is None else mechanical_energy_wh * METRES_PER_WH_J,
            "thermal_energy_j": None if thermal_energy_wh is None else thermal_energy_wh * METRES_PER_WH_J,
            "electrical_energy_wh": electrical_energy_wh,
            "acoustic_emission_pa2s": self._acoustic_dose_pa2s(episode),
            "drivetrain_scrub_j": _float(metrics.get("friction_dissipation_j")),
        }
        row["null_reason"] = _row_null_reason(row, _ROBOT_NULLABLE)
        return _frame([row], ROBOT_FACT_SCHEMA)

    @staticmethod
    def _encounter_id(episode_id: int, robot_token: str, encounter_idx: int) -> str:
        return f"enc-{episode_id:03d}-{robot_token}-{encounter_idx:03d}"

    @staticmethod
    def _contact_id(episode_id: int, robot_token: str, contact_idx: int) -> str:
        return f"con-{episode_id:03d}-{robot_token}-{contact_idx:03d}"
