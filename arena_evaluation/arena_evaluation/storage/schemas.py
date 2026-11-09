from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

    import polars as pl

import typing
from dataclasses import dataclass, field, fields

from pydantic import BaseModel, Field, model_validator

from .legacy import read_metadata


class RunMetadata(BaseModel):
    """Metadata for a single episode, serialized to episode_XXX.yaml."""

    benchmark_id: str
    planner: str
    robot_model: list[str] = Field(default_factory=list)
    map: str
    stage: str
    episode_id: int | None = None
    episodes_requested: int = 0
    suite_name: str = ""
    contest_name: str = ""
    local_planner: str = ""
    inter_planner: str = ""
    agent_name: str = ""
    task_generator_episode_id: int | None = None
    # Terminal outcome written by the runner's stop_episode service call
    # (QUEUED=0, RUNNING=1, SUCCESS=2, FAILED=3, SKIPPED=4, FATAL=5).
    outcome_state: int | None = None
    outcome_info: str = ""
    integrity: list[str] = Field(default_factory=list)
    recording_started_at: str
    arena_git_sha: str | None = None
    arena_git_dirty: bool = False
    python_version: str
    ros_distro: str

    task_layer: str = ""
    modes: dict[str, list[str]] = Field(default_factory=dict)
    params: dict[str, typing.Any] = Field(default_factory=dict)

    recording_ended_at: str | None = None
    episodes_recorded: int | None = None
    pedsim_available: bool | None = None
    recorded_topics: list[str] | None = None

    processing_completed_at: str | None = None
    episodes_valid: int | None = None
    pipeline_version: str | None = None

    rosbag2_message_count: int | None = None
    rosbag2_topics: list[typing.Any] | None = None

    env_ns_root: str | None = None
    is_reference: bool = False
    reference_type: str | None = None
    parent_episode_id: int | None = None
    crowd_condition: typing.Literal["visible", "blind"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: object) -> object:
        return read_metadata(data) if isinstance(data, dict) else data


@dataclass(frozen=True)
class RobotParams:
    """Robot physical parameters loaded at runtime."""

    model: str = "unknown"
    robot_radius: float = 0.25
    laser_min_range: float = 0.0
    laser_max_range: float = 30.0
    base_mass: float = 0.0
    component_masses: dict[str, float] = field(default_factory=dict)
    max_linear_velocity: float = 0.0

    @property
    def mass(self) -> float:
        """Total mass in kg, or 0.0 if undeclared."""
        return self.base_mass + sum(self.component_masses.values())

    @classmethod
    def load(cls, model: str) -> RobotParams:
        """Load parameters from arena_robots share directory, returning defaults on failure."""
        import os

        import yaml
        from ament_index_python.packages import get_package_share_directory

        defaults = cls()
        radius = defaults.robot_radius
        base_mass = defaults.base_mass
        component_masses: dict[str, float] = {}

        try:
            robot_dir = os.path.join(get_package_share_directory("arena_robots"), "robots", model.partition("[")[0])
        except Exception:
            return cls(model=model)

        def _read(path: str) -> dict:
            try:
                with open(path) as file:
                    return yaml.safe_load(file) or {}
            except Exception:
                return {}

        radius = float(_read(os.path.join(robot_dir, "caps", "mobile.yaml")).get("radius", radius))
        base_mass = float(_read(os.path.join(robot_dir, "model_params.yaml")).get("mass", {}).get("base_kg", base_mass))
        max_linear_velocity = 0.0
        for block in _read(os.path.join(robot_dir, "control.yaml")).values():
            params = block.get("ros__parameters", {}) if isinstance(block, dict) else {}
            if "linear.x.max_velocity" in params:
                max_linear_velocity = float(params["linear.x.max_velocity"])

        components_root = os.path.join(os.path.dirname(os.path.dirname(robot_dir)), "components")
        for kind, entries in _read(os.path.join(robot_dir, "assembly.yaml")).get("defaults", {}).items():
            for entry in entries or []:
                variant = (entry or {}).get("variant")
                if not variant:
                    continue
                kg = _read(os.path.join(components_root, kind, variant, "component.yaml")).get("mass", {}).get("kg")
                if kg:
                    component_masses[f"{kind}/{variant}"] = float(kg)

        return cls(
            model=model,
            robot_radius=radius,
            laser_min_range=defaults.laser_min_range,
            laser_max_range=defaults.laser_max_range,
            base_mass=base_mass,
            component_masses=component_masses,
            max_linear_velocity=max_linear_velocity,
        )


@dataclass(frozen=True)
class RunDescriptor:
    """Descriptor for a single run directory."""

    run_dir: str
    benchmark_id: str
    planner: str
    stage: str


@dataclass(frozen=True)
class EpisodeDescriptor:
    """Descriptor for an episode directory in flat storage."""

    episode_dir: str
    benchmark_id: str
    episode_id: int
    planner: str
    stage: str
    map: str = "unknown"
    is_reference: bool = False
    reference_type: str | None = None
    crowd_condition: typing.Literal["visible", "blind"] | None = None


@dataclass
class TopicBundle:
    """Flat telemetry tables of one robot: the episode's ingested tables filtered to its robot and environment."""

    tf: pl.DataFrame | pl.LazyFrame | None = None
    tf_static: pl.DataFrame | pl.LazyFrame | None = None
    world_semantics: pl.DataFrame | pl.LazyFrame | None = None
    robot_tracks: pl.DataFrame | pl.LazyFrame | None = None
    robot_odom: pl.DataFrame | pl.LazyFrame | None = None
    robot_cmd_vel: pl.DataFrame | pl.LazyFrame | None = None
    robot_scan_summary: pl.DataFrame | pl.LazyFrame | None = None
    robot_scan_rays: pl.DataFrame | pl.LazyFrame | None = None
    robot_joint_states: pl.DataFrame | pl.LazyFrame | None = None
    robot_power: pl.DataFrame | pl.LazyFrame | None = None
    robot_joint_power: pl.DataFrame | pl.LazyFrame | None = None
    robot_energy: pl.DataFrame | pl.LazyFrame | None = None
    robot_acoustics: pl.DataFrame | pl.LazyFrame | None = None
    collision_events: pl.DataFrame | pl.LazyFrame | None = None
    collision_contacts: pl.DataFrame | pl.LazyFrame | None = None
    robot_collision_monitor: pl.DataFrame | pl.LazyFrame | None = None
    robot_plans: pl.DataFrame | pl.LazyFrame | None = None
    robot_characterization_phase: pl.DataFrame | pl.LazyFrame | None = None
    robot_characterization_schedule: pl.DataFrame | pl.LazyFrame | None = None
    robot_initialpose: pl.DataFrame | pl.LazyFrame | None = None
    robot_task_pose: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_tracks: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_tracks_physics: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_gestures: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_animations: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_interactions: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_interaction_members: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_interaction_events: pl.DataFrame | pl.LazyFrame | None = None
    episode_lifecycle: pl.DataFrame | pl.LazyFrame | None = None
    episode_integrity: pl.DataFrame | pl.LazyFrame | None = None

    def available(self) -> set[str]:
        """Names of the topics this bundle carries."""
        return {f.name for f in fields(self) if getattr(self, f.name) is not None}


@dataclass
class AlignedEpisodeBundle:
    """Aligned topic data for a single episode."""

    episode_id: int
    data: pl.DataFrame
    start_pos: list[float]
    goal_pos: list[float]
    num_pedestrians: int = 0
    robot_name: str | None = None
    semantic_snapshot: pl.DataFrame | None = None
    conditions: list[dict] | None = None
    # Recorded EpisodeRecord.human_params as parameter name -> value, None when no record carried any.
    human_params: dict[str, typing.Any] | None = None
    # This robot's entry of EpisodeRecord.phases ({phases, conditions, map_poses, goal_inputs, instructions}), None when the record has none.
    phases: dict | None = None
    # Judge poses of every robot in the env by bare name, this robot included.
    fleet: dict[str, pl.DataFrame] = field(default_factory=dict)
    # Final EpisodeRecord.outcome_state of the recording, None when no record was captured.
    outcome_state: int | None = None
    outcome_info: str | None = None
    run: typing.Any = None
    folder_manager: typing.Any = None
    peds: pl.DataFrame | None = None
    map: str | None = None
    topics: dict[str, pl.DataFrame] | None = None
    env_offset: tuple[float, float] | None = None
    run_dir: pathlib.Path | None = None


class PlotDataSpec(BaseModel):
    """Specification for data binding: catalog source, SQL query, stored query, or evidence artifact."""

    source: str | None = None
    query: str | None = None
    stored_query: str | None = None
    params: dict[str, typing.Any] | None = None
    evidence_id: str | None = None
    where: str | None = None


class PlotChannels(BaseModel):
    """Multi-dimensional visual channel mappings for grammar of graphics."""

    x: str | list[str] | None = None
    y: str | list[str] | None = None
    color: str | None = None
    facet_col: str | None = None
    facet_row: str | None = None
    size: str | None = None
    symbol: str | None = None
    split: bool = False
    hierarchy: str | list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_hue(cls, data: object) -> object:
        if isinstance(data, dict):
            d = dict(data)
            if "hue" in d and "color" not in d:
                d["color"] = d.pop("hue")
            return d
        return data


class PlotSliceSpec(BaseModel):
    """Spatiotemporal or hierarchical slice for targeted telemetry and encounter evaluation."""

    scope: str | None = None
    episode_id: int | None = None
    encounter_id: str | None = None
    contact_id: str | None = None


class PlotSpec(BaseModel):
    """Specification for a single plot in a report manifest or OLAP presentation."""

    id: str = ""
    type: str = ""
    title: str = ""
    uses: str | None = None
    data: PlotDataSpec | None = None
    channels: PlotChannels = Field(default_factory=PlotChannels)
    slice: PlotSliceSpec | None = None
    options: dict[str, typing.Any] = Field(default_factory=dict)
    checks: dict[str, typing.Any] = Field(default_factory=dict)

    # Legacy fields retained with sensible defaults for full backward compatibility
    data_key: str = ""
    group_by: list[str] | str | None = None
    differentiate: str | None = "planner"
    auto_differentiate: bool = True
    filter: dict[str, typing.Any] | None = None
    layout_group: str | None = None
    data_source: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_spec(cls, data: object) -> object:
        if isinstance(data, dict):
            d = dict(data)
            # Normalize recipe aliases
            if "uses" in d and not d.get("id"):
                d["id"] = d["uses"]
            if "encodings" in d and "channels" not in d:
                d["channels"] = d.pop("encodings")
            if "source" in d and "data" not in d:
                d["data"] = {"source": d.pop("source")}
            if "stored_query" in d:
                sq = d.pop("stored_query")
                sp = d.pop("params", None)
                if "data" not in d or d["data"] is None:
                    d["data"] = {"stored_query": sq, "params": sp}
                elif isinstance(d["data"], dict):
                    d["data"]["stored_query"] = sq
                    if sp is not None and "params" not in d["data"]:
                        d["data"]["params"] = sp
            if "params" in d and isinstance(d.get("data"), dict) and "params" not in d["data"]:
                d["data"]["params"] = d.pop("params")

            # 1. Normalize 'data'
            if "data" in d and isinstance(d["data"], str):
                s = d["data"].strip()
                if s.upper().startswith(("SELECT", "WITH")):
                    d["data"] = {"query": s}
                elif s.startswith("ev-"):
                    d["data"] = {"evidence_id": s}
                else:
                    d["data"] = {"source": s}
            elif "data_source" in d and "data" not in d and d["data_source"] is not None:
                d["data"] = {"source": d["data_source"]}
            elif "data" in d and isinstance(d["data"], dict) and "source" in d["data"] and "data_source" not in d:
                d["data_source"] = d["data"]["source"]

            # 2. Normalize 'channels'
            if "channels" in d and isinstance(d["channels"], dict):
                ch = dict(d["channels"])
                if "facet" in ch and "facet_col" not in ch:
                    ch["facet_col"] = ch.pop("facet")
                if "y" in ch and isinstance(ch["y"], str) and not d.get("data_key"):
                    d["data_key"] = ch["y"]
                elif "x" in ch and isinstance(ch["x"], str) and not d.get("data_key"):
                    d["data_key"] = ch["x"]
                if "color" in ch and "differentiate" not in d:
                    d["differentiate"] = ch["color"]
                elif "hue" in ch and "differentiate" not in d:
                    d["differentiate"] = ch["hue"]
                d["channels"] = ch

            return d
        return data
