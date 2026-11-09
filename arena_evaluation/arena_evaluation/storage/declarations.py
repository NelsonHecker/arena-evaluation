from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from enum import StrEnum
import pathlib
from typing import Any, Final, Literal

import polars as pl

# Topic and parameter constants
ROBOT_TRACKS_TOPIC: Final[str] = "robot_tracks"
PEDESTRIAN_TRACKS_TOPIC: Final[str] = "pedestrian_tracks"
ROBOT_VISIBILITY_PARAM: Final[str] = "humansim.robots_visible"

CrowdCondition = Literal["visible", "blind"]


class MetricScope(StrEnum):
    """The five first-class fact table evaluation scopes."""

    EPISODE = "episodes"
    ENCOUNTER = "encounters"
    CONTACT = "contacts"
    PEDESTRIAN = "pedestrians"
    ROBOT = "robots"


def coerce_scope(value: MetricScope | str) -> MetricScope:
    """Normalize a MetricScope enum or scope token string into a MetricScope."""
    if isinstance(value, MetricScope):
        return value
    token = str(value).strip().lower()
    for member in MetricScope:
        if token in (member.value, member.name.lower()):
            return member
    raise ValueError(f"Unknown MetricScope: {value!r}")


_coerce_scope = coerce_scope


@dataclasses.dataclass(frozen=True, slots=True)
class MetricDeclaration:
    """Static schema and scope declaration for a benchmark metric."""

    name: str
    unit: str = ""
    scopes: tuple[MetricScope, ...] = (MetricScope.EPISODE,)
    dtype: pl.DataType = pl.Float64
    description: str = ""
    direction: str = ""  # "higher", "lower", or ""


def resolve_crowd_condition(
    episode: Any,
    metrics_config: Mapping[str, Any] | None = None,
) -> CrowdCondition:
    """Crowd visibility of the episode: explicit metrics config first, else the recorded humansim parameter, else visible."""
    config = metrics_config or {}
    configured = config.get("crowd_condition")
    if configured in ("blind", "visible"):
        return "blind" if configured == "blind" else "visible"
    for key in ("crowd_visible", "robots_visible"):
        visible = config.get(key)
        if visible is not None:
            return "visible" if bool(visible) else "blind"

    params = getattr(episode, "human_params", None)
    if isinstance(params, dict):
        val = params.get(ROBOT_VISIBILITY_PARAM)
        if val is not None:
            return "visible" if bool(val) else "blind"
        vis_leaf = ROBOT_VISIBILITY_PARAM.rsplit(".", maxsplit=1)[-1]
        for name, parameter in params.items():
            if name.rsplit(".", maxsplit=1)[-1] == vis_leaf:
                return "visible" if bool(parameter) else "blind"

    run = episode.run
    if run is not None and run.crowd_condition is not None:
        return run.crowd_condition
    return "visible"


FACT_TABLE_NAMES: Final[tuple[str, ...]] = (
    "episodes",
    "encounters",
    "contacts",
    "pedestrians",
    "robots",
)

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
        return {
            "episodes": self.episodes,
            "encounters": self.encounters,
            "contacts": self.contacts,
            "pedestrians": self.pedestrians,
            "robots": self.robots,
        }


def write_fact_tables(tables: ScopedFactTables, directory: pathlib.Path) -> None:
    """Write the five fact tables as ``fact_<name>.parquet`` in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.by_name().items():
        frame.write_parquet(directory / f"fact_{name}.parquet")


def read_fact_tables(directory: pathlib.Path) -> ScopedFactTables | None:
    """Read the five fact tables from ``fact_<name>.parquet`` in ``directory``."""
    frames: dict[str, pl.DataFrame] = {}
    for name in FACT_TABLE_NAMES:
        path = directory / f"fact_{name}.parquet"
        if not path.is_file():
            return None
        frames[name] = pl.read_parquet(path)
    return ScopedFactTables(
        episodes=frames["episodes"],
        encounters=frames["encounters"],
        contacts=frames["contacts"],
        pedestrians=frames["pedestrians"],
        robots=frames["robots"],
    )
