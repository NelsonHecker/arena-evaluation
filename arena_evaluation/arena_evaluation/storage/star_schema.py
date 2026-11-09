"""Benchmark Star Schema Lakehouse (Layer 3).

Defines the four dimension schemas (``dim_episodes``, ``dim_robots``,
``dim_pedestrians``, ``dim_world_entities``), re-exports the eleven fact schemas
from :mod:`arena_evaluation.processing.scoped_evaluator`, and provides
:class:`StarSchemaBuilder` plus Parquet write/scan utilities.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import typing
from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from arena_evaluation.storage.declarations import (
    CHILD_FACT_TABLE_NAMES,
    COLLISION_FACT_SCHEMA,
    CONTACT_FACT_SCHEMA,
    EDGE_CASE_FACT_SCHEMA,
    ENCOUNTER_FACT_SCHEMA,
    EPISODE_FACT_SCHEMA,
    FACT_SCHEMAS,
    FACT_TABLE_NAMES,
    FORMATION_FACT_SCHEMA,
    JUDGE_DISAGREEMENT_FACT_SCHEMA,
    MECHANISM_STATE_FACT_SCHEMA,
    PEDESTRIAN_FACT_SCHEMA,
    PEDESTRIAN_TRACKS_TOPIC,
    PHASE_OUTCOME_FACT_SCHEMA,
    ROBOT_FACT_SCHEMA,
    ROBOT_TRACKS_TOPIC,
    ScopedFactTables,
    resolve_crowd_condition,
    write_fact_tables,
)
from arena_evaluation.storage.planner_names import split_planner_name
from arena_evaluation.storage.schemas import (
    AlignedEpisodeBundle,
    EpisodeDescriptor,
    RobotParams,
    RunMetadata,
)

__all__ = [
    "CHILD_FACT_TABLE_NAMES",
    "COLLISION_FACT_SCHEMA",
    "CONTACT_FACT_SCHEMA",
    "EDGE_CASE_FACT_SCHEMA",
    "DIMENSION_SCHEMAS",
    "DIMENSION_TABLE_NAMES",
    "DIM_EPISODES_SCHEMA",
    "DIM_PEDESTRIANS_SCHEMA",
    "DIM_ROBOTS_SCHEMA",
    "DIM_WORLD_ENTITIES_SCHEMA",
    "ENCOUNTER_FACT_SCHEMA",
    "EPISODE_FACT_SCHEMA",
    "FACT_CONTACTS_SCHEMA",
    "FACT_ENCOUNTERS_SCHEMA",
    "FACT_EPISODES_SCHEMA",
    "FACT_PEDESTRIANS_SCHEMA",
    "FACT_ROBOTS_SCHEMA",
    "FACT_SCHEMAS",
    "FORMATION_FACT_SCHEMA",
    "JUDGE_DISAGREEMENT_FACT_SCHEMA",
    "MECHANISM_STATE_FACT_SCHEMA",
    "PHASE_OUTCOME_FACT_SCHEMA",
    "FACT_TABLE_NAMES",
    "PEDESTRIAN_FACT_SCHEMA",
    "ROBOT_FACT_SCHEMA",
    "STAR_SCHEMA_SCHEMAS",
    "STAR_TABLE_NAMES",
    "StarSchemaBuilder",
    "StarSchemaTables",
    "read_episode_star_tables",
    "scan_star_schema",
    "write_episode_star_tables",
    "write_star_schema",
]

FACT_EPISODES_SCHEMA: Final[pl.Schema] = EPISODE_FACT_SCHEMA
FACT_ENCOUNTERS_SCHEMA: Final[pl.Schema] = ENCOUNTER_FACT_SCHEMA
FACT_CONTACTS_SCHEMA: Final[pl.Schema] = CONTACT_FACT_SCHEMA
FACT_PEDESTRIANS_SCHEMA: Final[pl.Schema] = PEDESTRIAN_FACT_SCHEMA
FACT_ROBOTS_SCHEMA: Final[pl.Schema] = ROBOT_FACT_SCHEMA
FACT_EDGE_CASES_SCHEMA: Final[pl.Schema] = EDGE_CASE_FACT_SCHEMA
FACT_FORMATIONS_SCHEMA: Final[pl.Schema] = FORMATION_FACT_SCHEMA
FACT_PHASE_OUTCOMES_SCHEMA: Final[pl.Schema] = PHASE_OUTCOME_FACT_SCHEMA
FACT_JUDGE_DISAGREEMENTS_SCHEMA: Final[pl.Schema] = JUDGE_DISAGREEMENT_FACT_SCHEMA
FACT_MECHANISM_STATES_SCHEMA: Final[pl.Schema] = MECHANISM_STATE_FACT_SCHEMA
FACT_COLLISIONS_SCHEMA: Final[pl.Schema] = COLLISION_FACT_SCHEMA

DIM_EPISODES_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "env_id": pl.String,
        "robot_id": pl.String,
        "benchmark_id": pl.String,
        "suite_name": pl.String,
        "contest_name": pl.String,
        "stage": pl.String,
        "world": pl.String,
        "scenario": pl.String,
        "planner": pl.String,
        "inter_planner": pl.String,
        "dispatcher": pl.String,
        "robot_model": pl.String,
        "crowd_condition": pl.String,
        "ped_policy_default": pl.String,
        "crowd_size": pl.Int64,
        "seed": pl.Int64,
        "is_reference": pl.Boolean,
        "reference_type": pl.String,
        "duration_s": pl.Float64,
        "outcome_state": pl.Int64,
        "outcome_info": pl.String,
    }
)

DIM_ROBOTS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "env_id": pl.String,
        "robot_id": pl.String,
        "robot_model": pl.String,
        "planner": pl.String,
        "inter_planner": pl.String,
        "dispatcher": pl.String,
        "max_linear_vel": pl.Float64,
        "max_angular_vel": pl.Float64,
        "linear_acc": pl.Float64,
        "linear_decel": pl.Float64,
        "robot_radius": pl.Float64,
        "robot_mass_kg": pl.Float64,
        "static_power_w": pl.Float64,
        "lidar_variant": pl.String,
        "start_x": pl.Float64,
        "start_y": pl.Float64,
        "goal_x": pl.Float64,
        "goal_y": pl.Float64,
    }
)

DIM_PEDESTRIANS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "env_id": pl.String,
        "ped_id": pl.String,
        "ped_name": pl.String,
        "agent_type": pl.String,
        "ped_policy": pl.String,
        "is_blind": pl.Boolean,
        "desired_speed": pl.Float64,
        "radius_m": pl.Float64,
        "spawn_x": pl.Float64,
        "spawn_y": pl.Float64,
        "spawn_yaw": pl.Float64,
        "waypoints_count": pl.Int64,
    }
)

DIM_WORLD_ENTITIES_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "env_id": pl.String,
        "entity_name": pl.String,
        "entity_kind": pl.String,
        "zone": pl.String,
        "activation_dist": pl.Float64,
        "acoustic_tl_db": pl.Float64,
        "initial_state": pl.String,
    }
)

DIMENSION_TABLE_NAMES: Final[tuple[str, ...]] = (
    "dim_episodes",
    "dim_robots",
    "dim_pedestrians",
    "dim_world_entities",
)

DIMENSION_SCHEMAS: Final[dict[str, pl.Schema]] = {
    "dim_episodes": DIM_EPISODES_SCHEMA,
    "dim_robots": DIM_ROBOTS_SCHEMA,
    "dim_pedestrians": DIM_PEDESTRIANS_SCHEMA,
    "dim_world_entities": DIM_WORLD_ENTITIES_SCHEMA,
}

STAR_TABLE_NAMES: Final[tuple[str, ...]] = (
    *DIMENSION_TABLE_NAMES,
    *(f"fact_{name}" for name in FACT_TABLE_NAMES),
)

STAR_SCHEMA_SCHEMAS: Final[dict[str, pl.Schema]] = {
    **DIMENSION_SCHEMAS,
    **{f"fact_{name}": schema for name, schema in FACT_SCHEMAS.items()},
}

_ENV_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^(env_\d+)")
_LIDAR_BRACKET_RE: Final[re.Pattern[str]] = re.compile(r"lidar=([a-zA-Z0-9_\-]+)")
_WORLD_ENTITIES_TOPIC: Final[str] = "world_entities"
_WORLD_SEMANTICS_TOPIC: Final[str] = "world_semantics"


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    val = float(typing.cast(float, value))
    return None if val != val else val


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    return int(typing.cast(int, value))


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _env_from_robot_id(robot_id: str, fallback: str = "env_0") -> str:
    match = _ENV_TOKEN_RE.match(robot_id)
    return match.group(1) if match is not None else fallback


def _parse_lidar_variant(
    raw_model: str, robot_params: RobotParams | None
) -> str | None:
    match = _LIDAR_BRACKET_RE.search(raw_model)
    if match is not None:
        return match.group(1)
    if robot_params is not None:
        for key in robot_params.component_masses:
            if key.startswith("lidar/"):
                return key.split("/", 1)[1]
    return None


def _scoped_frames_by_table(
    facts: ScopedFactTables,
) -> dict[str, tuple[pl.DataFrame, pl.Schema]]:
    """The scoped evaluator frames paired with their canonical schema, by short table name."""
    frames = facts.by_name()
    return {name: (frames[name], schema) for name, schema in FACT_SCHEMAS.items()}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StarSchemaTables:
    """Complete set of 4 dimension tables and 5 fact tables for a benchmark or episode."""

    dim_episodes: pl.DataFrame
    dim_robots: pl.DataFrame
    dim_pedestrians: pl.DataFrame
    dim_world_entities: pl.DataFrame
    fact_episodes: pl.DataFrame
    fact_encounters: pl.DataFrame
    fact_contacts: pl.DataFrame
    fact_pedestrians: pl.DataFrame
    fact_robots: pl.DataFrame
    fact_edge_cases: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=EDGE_CASE_FACT_SCHEMA)
    )
    fact_formations: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=FORMATION_FACT_SCHEMA)
    )
    fact_phase_outcomes: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=PHASE_OUTCOME_FACT_SCHEMA)
    )
    fact_judge_disagreements: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=JUDGE_DISAGREEMENT_FACT_SCHEMA)
    )
    fact_mechanism_states: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=MECHANISM_STATE_FACT_SCHEMA)
    )
    fact_collisions: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=COLLISION_FACT_SCHEMA)
    )

    @classmethod
    def empty(cls) -> StarSchemaTables:
        """Construct an empty StarSchemaTables instance with canonical schemas."""
        return cls(
            **{
                name: pl.DataFrame(schema=schema)
                for name, schema in STAR_SCHEMA_SCHEMAS.items()
            }
        )

    def dimension_tables(self) -> dict[str, pl.DataFrame]:
        """Return the four dimension DataFrames keyed by table name."""
        return {
            "dim_episodes": self.dim_episodes,
            "dim_robots": self.dim_robots,
            "dim_pedestrians": self.dim_pedestrians,
            "dim_world_entities": self.dim_world_entities,
        }

    def fact_tables(self) -> dict[str, pl.DataFrame]:
        """Return every fact DataFrame keyed by ``fact_<name>``."""
        return {
            "fact_episodes": self.fact_episodes,
            "fact_encounters": self.fact_encounters,
            "fact_contacts": self.fact_contacts,
            "fact_pedestrians": self.fact_pedestrians,
            "fact_robots": self.fact_robots,
            "fact_edge_cases": self.fact_edge_cases,
            "fact_formations": self.fact_formations,
            "fact_phase_outcomes": self.fact_phase_outcomes,
            "fact_judge_disagreements": self.fact_judge_disagreements,
            "fact_mechanism_states": self.fact_mechanism_states,
            "fact_collisions": self.fact_collisions,
        }

    def by_name(self) -> dict[str, pl.DataFrame]:
        """Return every Star Schema DataFrame keyed by canonical table name."""
        return {
            **self.dimension_tables(),
            **self.fact_tables(),
        }

    def lazy_by_name(self) -> dict[str, pl.LazyFrame]:
        """Return every Star Schema table as a LazyFrame."""
        return {name: frame.lazy() for name, frame in self.by_name().items()}


class StarSchemaBuilder:
    """Builds and aggregates Benchmark Star Schema dimension and fact tables."""

    def __init__(
        self,
        *,
        robot_params: RobotParams | None = None,
        metrics_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.robot_params = robot_params
        self.metrics_config: dict[str, Any] = dict(metrics_config or {})

    def build_episode_tables(
        self,
        episode: AlignedEpisodeBundle,
        facts: ScopedFactTables,
        *,
        run_id: str,
        episode_id: int,
        robot_params: RobotParams | None = None,
        descriptor: EpisodeDescriptor | None = None,
        metadata: RunMetadata | None = None,
        metrics_config: Mapping[str, Any] | None = None,
    ) -> StarSchemaTables:
        """Extract dimension tables from an evaluated episode bundle and combine with scoped facts."""
        resolved_params = robot_params or self.robot_params or RobotParams()
        resolved_cfg = dict(self.metrics_config)
        if metrics_config is not None:
            resolved_cfg.update(metrics_config)

        meta: RunMetadata | None = metadata
        if meta is None and isinstance(episode.run, RunMetadata):
            meta = episode.run

        robot_id = self._resolve_robot_id(episode, facts)
        env_id = self._resolve_env_id(episode, facts, robot_id, meta)
        crowd_condition = resolve_crowd_condition(episode, resolved_cfg)

        dim_ep_df = self._build_dim_episodes(
            episode,
            facts,
            run_id=run_id,
            episode_id=episode_id,
            env_id=env_id,
            robot_id=robot_id,
            crowd_condition=crowd_condition,
            robot_params=resolved_params,
            descriptor=descriptor,
            metadata=meta,
            metrics_config=resolved_cfg,
        )
        dim_rob_df = self._build_dim_robots(
            episode,
            run_id=run_id,
            episode_id=episode_id,
            env_id=env_id,
            robot_id=robot_id,
            robot_params=resolved_params,
            descriptor=descriptor,
            metadata=meta,
            metrics_config=resolved_cfg,
        )
        dim_ped_df = self._build_dim_pedestrians(
            episode,
            facts,
            metrics_config=resolved_cfg,
        )
        dim_world_df = self._build_dim_world_entities(
            episode,
            run_id=run_id,
            episode_id=episode_id,
            env_id=env_id,
        )

        return StarSchemaTables(
            dim_episodes=dim_ep_df,
            dim_robots=dim_rob_df,
            dim_pedestrians=dim_ped_df,
            dim_world_entities=dim_world_df,
            **{
                f"fact_{name}": frame.select(list(schema.names())).cast(schema)
                for name, (frame, schema) in _scoped_frames_by_table(facts).items()
            },
        )

    def build_status_episode_tables(
        self,
        *,
        run_id: str,
        episode_id: int,
        descriptor: EpisodeDescriptor,
        metadata: RunMetadata | None = None,
        robot_id: str = "",
    ) -> StarSchemaTables:
        """Build StarSchemaTables for an episode that had no trajectory or recording."""
        resolved_robot_id = robot_id or (
            metadata.robot_model[0].partition("[")[0]
            if metadata is not None and metadata.robot_model
            else "robot"
        )
        env_id = _env_from_robot_id(
            resolved_robot_id,
            fallback=metadata.env_ns_root or "env_0" if metadata else "env_0",
        )
        planner = (
            metadata.planner
            if metadata is not None and metadata.planner
            else descriptor.planner
        )
        inter_planner = (
            metadata.inter_planner
            if metadata is not None and metadata.inter_planner
            else split_planner_name(planner)[1]
        )
        dispatcher = (
            str(metadata.params.get("dispatcher", "none"))
            if metadata is not None
            else "none"
        )
        raw_model = (
            metadata.robot_model[0]
            if metadata is not None and metadata.robot_model
            else "unknown"
        )
        robot_model = raw_model.partition("[")[0] or "unknown"
        world = (
            metadata.map if metadata is not None and metadata.map else descriptor.map
        )
        scenario = (
            str(metadata.params.get("scenario", descriptor.stage))
            if metadata is not None
            else descriptor.stage
        )
        seed_val = (
            _opt_int(metadata.params.get("seed")) if metadata is not None else None
        )
        if seed_val is None:
            seed_val = int(episode_id)

        row: dict[str, object] = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "env_id": env_id,
            "robot_id": resolved_robot_id,
            "benchmark_id": descriptor.benchmark_id or run_id,
            "suite_name": metadata.suite_name if metadata is not None else "",
            "contest_name": metadata.contest_name if metadata is not None else "",
            "stage": descriptor.stage,
            "world": world,
            "scenario": scenario,
            "planner": planner,
            "inter_planner": inter_planner,
            "dispatcher": dispatcher,
            "robot_model": robot_model,
            "crowd_condition": "visible",
            "ped_policy_default": "sfm",
            "crowd_size": 0,
            "seed": seed_val,
            "is_reference": bool(descriptor.is_reference),
            "reference_type": descriptor.reference_type,
            "duration_s": None,
            "outcome_state": metadata.outcome_state if metadata is not None else None,
            "outcome_info": metadata.outcome_info if metadata is not None else None,
        }
        meta_params = metadata.params if metadata is not None else {}
        robot_row: dict[str, object] = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "env_id": env_id,
            "robot_id": resolved_robot_id,
            "robot_model": robot_model,
            "planner": planner,
            "inter_planner": inter_planner,
            "dispatcher": dispatcher,
            "lidar_variant": _opt_str(meta_params.get("lidar_variant")),
            "max_linear_vel": _opt_float(meta_params.get("robot.mobile.max_linear_vel"))
            or _opt_float(meta_params.get("max_linear_vel"))
            or (
                float(self.robot_params.max_linear_velocity)
                if self.robot_params and self.robot_params.max_linear_velocity > 0.0
                else None
            ),
            "max_angular_vel": _opt_float(meta_params.get("max_angular_vel")),
            "linear_acc": _opt_float(meta_params.get("linear_acc")),
            "linear_decel": _opt_float(meta_params.get("linear_decel")),
            "robot_mass_kg": float(self.robot_params.total_mass)
            if self.robot_params
            else None,
            "robot_radius_m": float(self.robot_params.robot_radius)
            if self.robot_params
            else None,
            "static_power_w": _opt_float(meta_params.get("static_power_w")),
            "start_x": None,
            "start_y": None,
            "goal_x": None,
            "goal_y": None,
        }
        empty = StarSchemaTables.empty()
        return dataclasses.replace(
            empty,
            dim_episodes=pl.DataFrame([row], schema=DIM_EPISODES_SCHEMA),
            dim_robots=pl.DataFrame([robot_row], schema=DIM_ROBOTS_SCHEMA),
        )

    @staticmethod
    def concat_tables(tables_seq: Sequence[StarSchemaTables]) -> StarSchemaTables:
        """Concatenate multiple StarSchemaTables in a single vertical pass per table (I2)."""
        if not tables_seq:
            return StarSchemaTables.empty()
        if len(tables_seq) == 1:
            return tables_seq[0]

        return StarSchemaTables(
            **{
                name: pl.concat([t.by_name()[name] for t in tables_seq], how="vertical")
                for name in STAR_TABLE_NAMES
            }
        )

    def build_benchmark(
        self,
        episodes: Sequence[tuple[AlignedEpisodeBundle, ScopedFactTables]],
        *,
        run_id: str,
        robot_params: RobotParams | None = None,
        metrics_config: Mapping[str, Any] | None = None,
    ) -> StarSchemaTables:
        """Build StarSchemaTables across a sequence of (AlignedEpisodeBundle, ScopedFactTables) pairs."""
        built = [
            self.build_episode_tables(
                ep_bundle,
                ep_facts,
                run_id=run_id,
                episode_id=int(ep_bundle.episode_id),
                robot_params=robot_params,
                metrics_config=metrics_config,
            )
            for ep_bundle, ep_facts in episodes
        ]
        return self.concat_tables(built)

    @staticmethod
    def _resolve_robot_id(
        episode: AlignedEpisodeBundle, facts: ScopedFactTables
    ) -> str:
        if not facts.episodes.is_empty():
            val = facts.episodes.item(0, "robot_id")
            if isinstance(val, str) and val:
                return val
        topics = episode.topics or {}
        rt_df = topics.get(ROBOT_TRACKS_TOPIC)
        if isinstance(rt_df, pl.LazyFrame):
            rt_df = rt_df.collect()
        if rt_df is not None and not rt_df.is_empty():
            non_null = rt_df.get_column("robot_id").drop_nulls()
            if len(non_null) > 0:
                return str(non_null[0])
        return str(episode.robot_name or "robot")

    @staticmethod
    def _resolve_env_id(
        episode: AlignedEpisodeBundle,
        facts: ScopedFactTables,
        robot_id: str,
        metadata: RunMetadata | None,
    ) -> str:
        if not facts.robots.is_empty():
            val = facts.robots.item(0, "env_id")
            if isinstance(val, str) and val:
                return val
        topics = episode.topics or {}
        rt_df = topics.get(ROBOT_TRACKS_TOPIC)
        if isinstance(rt_df, pl.LazyFrame):
            rt_df = rt_df.collect()
        if rt_df is not None and not rt_df.is_empty():
            non_null = rt_df.get_column("env_id").drop_nulls()
            if len(non_null) > 0:
                return str(non_null[0])
        fallback = (
            metadata.env_ns_root
            if metadata is not None and metadata.env_ns_root
            else "env_0"
        )
        return _env_from_robot_id(robot_id, fallback=fallback)

    def _build_dim_episodes(
        self,
        episode: AlignedEpisodeBundle,
        facts: ScopedFactTables,
        *,
        run_id: str,
        episode_id: int,
        env_id: str,
        robot_id: str,
        crowd_condition: str,
        robot_params: RobotParams,
        descriptor: EpisodeDescriptor | None,
        metadata: RunMetadata | None,
        metrics_config: Mapping[str, Any],
    ) -> pl.DataFrame:
        human_params = episode.human_params or {}
        meta_params = metadata.params if metadata is not None else {}

        benchmark_id = (
            (
                metadata.benchmark_id
                if metadata is not None and metadata.benchmark_id
                else None
            )
            or (
                descriptor.benchmark_id
                if descriptor is not None and descriptor.benchmark_id
                else None
            )
            or str(metrics_config.get("benchmark_id", run_id))
        )
        suite_name = (
            metadata.suite_name
            if metadata is not None and metadata.suite_name
            else str(metrics_config.get("suite_name", ""))
        )
        contest_name = (
            metadata.contest_name
            if metadata is not None and metadata.contest_name
            else str(metrics_config.get("contest_name", ""))
        )
        stage = (
            (metadata.stage if metadata is not None and metadata.stage else None)
            or (
                descriptor.stage
                if descriptor is not None and descriptor.stage
                else None
            )
            or str(metrics_config.get("stage", "default"))
        )
        world = (
            episode.map
            or (metadata.map if metadata is not None and metadata.map else None)
            or (descriptor.map if descriptor is not None and descriptor.map else None)
            or str(metrics_config.get("world", "unknown"))
        )
        scenario = str(
            meta_params.get("scenario")
            or human_params.get("scenario")
            or metrics_config.get("scenario")
            or stage
        )
        planner = (
            (metadata.planner if metadata is not None and metadata.planner else None)
            or (
                descriptor.planner
                if descriptor is not None and descriptor.planner
                else None
            )
            or str(metrics_config.get("planner", "unknown"))
        )
        inter_planner = (
            (
                metadata.inter_planner
                if metadata is not None and metadata.inter_planner
                else None
            )
            or _opt_str(metrics_config.get("inter_planner"))
            or split_planner_name(planner)[1]
        )
        dispatcher = str(
            meta_params.get("dispatcher") or metrics_config.get("dispatcher") or "none"
        )
        raw_robot_model = (
            (
                metadata.robot_model[0]
                if metadata is not None and metadata.robot_model
                else None
            )
            or _opt_str(metrics_config.get("robot_model"))
            or robot_params.model
        )
        robot_model = raw_robot_model.partition("[")[0] or "unknown"

        ped_policy_default = _opt_str(human_params.get("ped_policy")) or _opt_str(
            metrics_config.get("ped_policy_default")
        )
        if ped_policy_default is None and not facts.pedestrians.is_empty():
            policies = facts.pedestrians.get_column("ped_policy").drop_nulls()
            if len(policies) > 0:
                ped_policy_default = str(policies[0])
        if ped_policy_default is None:
            ped_policy_default = "sfm"

        crowd_size = (
            int(episode.num_pedestrians)
            if episode.num_pedestrians > 0
            else int(facts.pedestrians.height)
        )
        seed_raw = (
            human_params.get("seed")
            if human_params.get("seed") is not None
            else meta_params.get("seed", metrics_config.get("seed"))
        )
        seed = _opt_int(seed_raw)
        if seed is None:
            seed = int(episode_id)

        is_reference = bool(
            (metadata.is_reference if metadata is not None else False)
            or (descriptor.is_reference if descriptor is not None else False)
            or bool(metrics_config.get("is_reference", False))
        )
        reference_type = (
            (
                metadata.reference_type
                if metadata is not None and metadata.reference_type
                else None
            )
            or (
                descriptor.reference_type
                if descriptor is not None and descriptor.reference_type
                else None
            )
            or _opt_str(metrics_config.get("reference_type"))
        )

        duration_s: float | None = None
        if not facts.episodes.is_empty():
            duration_s = _opt_float(facts.episodes.item(0, "time_to_goal_s"))
        if duration_s is None:
            topics = episode.topics or {}
            rt_df = topics.get(ROBOT_TRACKS_TOPIC)
            if isinstance(rt_df, pl.LazyFrame):
                rt_df = rt_df.collect()
            if rt_df is not None and not rt_df.is_empty():
                t_col = rt_df.get_column("t_ns").drop_nulls()
                t_min = t_col.min()
                t_max = t_col.max()
                if isinstance(t_min, int) and isinstance(t_max, int) and t_max >= t_min:
                    duration_s = float(t_max - t_min) / 1e9

        outcome_state = (
            episode.outcome_state
            if episode.outcome_state is not None
            else (
                metadata.outcome_state
                if metadata is not None
                else _opt_int(metrics_config.get("outcome_state"))
            )
        )
        if outcome_state is None and not facts.episodes.is_empty():
            succ = facts.episodes.item(0, "success")
            if succ is True:
                outcome_state = 2
            elif succ is False:
                outcome_state = 3

        outcome_info = (
            episode.outcome_info
            if episode.outcome_info is not None
            else (
                metadata.outcome_info
                if metadata is not None and metadata.outcome_info
                else _opt_str(metrics_config.get("outcome_info"))
            )
        )
        if outcome_info is None and not facts.episodes.is_empty():
            outcome_info = _opt_str(facts.episodes.item(0, "result"))

        row: dict[str, object] = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "env_id": env_id,
            "robot_id": robot_id,
            "benchmark_id": benchmark_id,
            "suite_name": suite_name,
            "contest_name": contest_name,
            "stage": stage,
            "world": world,
            "scenario": scenario,
            "planner": planner,
            "inter_planner": inter_planner,
            "dispatcher": dispatcher,
            "robot_model": robot_model,
            "crowd_condition": crowd_condition,
            "ped_policy_default": ped_policy_default,
            "crowd_size": crowd_size,
            "seed": seed,
            "is_reference": is_reference,
            "reference_type": reference_type,
            "duration_s": duration_s,
            "outcome_state": outcome_state,
            "outcome_info": outcome_info,
        }
        return pl.DataFrame([row], schema=DIM_EPISODES_SCHEMA)

    def _build_dim_robots(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        env_id: str,
        robot_id: str,
        robot_params: RobotParams,
        descriptor: EpisodeDescriptor | None,
        metadata: RunMetadata | None,
        metrics_config: Mapping[str, Any],
    ) -> pl.DataFrame:
        meta_params = metadata.params if metadata is not None else {}
        planner = (
            (metadata.planner if metadata is not None and metadata.planner else None)
            or (
                descriptor.planner
                if descriptor is not None and descriptor.planner
                else None
            )
            or str(metrics_config.get("planner", "unknown"))
        )
        inter_planner = (
            (
                metadata.inter_planner
                if metadata is not None and metadata.inter_planner
                else None
            )
            or _opt_str(metrics_config.get("inter_planner"))
            or split_planner_name(planner)[1]
        )
        dispatcher = str(
            meta_params.get("dispatcher") or metrics_config.get("dispatcher") or "none"
        )
        raw_robot_model = (
            (
                metadata.robot_model[0]
                if metadata is not None and metadata.robot_model
                else None
            )
            or _opt_str(metrics_config.get("robot_model"))
            or robot_params.model
        )
        robot_model = raw_robot_model.partition("[")[0] or "unknown"
        lidar_variant = (
            _opt_str(meta_params.get("lidar_variant"))
            or _opt_str(metrics_config.get("lidar_variant"))
            or _parse_lidar_variant(raw_robot_model, robot_params)
        )

        max_linear_vel = (
            _opt_float(meta_params.get("robot.mobile.max_linear_vel"))
            or _opt_float(meta_params.get("max_linear_vel"))
            or _opt_float(metrics_config.get("max_linear_vel"))
            or (
                float(robot_params.max_linear_velocity)
                if robot_params.max_linear_velocity > 0.0
                else None
            )
        )
        max_angular_vel = _opt_float(
            meta_params.get("max_angular_vel", metrics_config.get("max_angular_vel"))
        )
        linear_acc = _opt_float(
            meta_params.get("linear_acc", metrics_config.get("linear_acc"))
        )
        linear_decel = _opt_float(
            meta_params.get("linear_decel", metrics_config.get("linear_decel"))
        )
        static_power_w = _opt_float(
            meta_params.get("static_power_w", metrics_config.get("static_power_w"))
        )

        start_x = float(episode.start_pos[0]) if len(episode.start_pos) >= 2 else None
        start_y = float(episode.start_pos[1]) if len(episode.start_pos) >= 2 else None
        goal_x = float(episode.goal_pos[0]) if len(episode.goal_pos) >= 2 else None
        goal_y = float(episode.goal_pos[1]) if len(episode.goal_pos) >= 2 else None

        row: dict[str, object] = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "env_id": env_id,
            "robot_id": robot_id,
            "robot_model": robot_model,
            "planner": planner,
            "inter_planner": inter_planner,
            "dispatcher": dispatcher,
            "max_linear_vel": max_linear_vel,
            "max_angular_vel": max_angular_vel,
            "linear_acc": linear_acc,
            "linear_decel": linear_decel,
            "robot_radius": float(robot_params.robot_radius),
            "robot_mass_kg": float(robot_params.mass),
            "static_power_w": static_power_w,
            "lidar_variant": lidar_variant,
            "start_x": start_x,
            "start_y": start_y,
            "goal_x": goal_x,
            "goal_y": goal_y,
        }
        return pl.DataFrame([row], schema=DIM_ROBOTS_SCHEMA)

    @staticmethod
    def _build_dim_pedestrians(
        episode: AlignedEpisodeBundle,
        facts: ScopedFactTables,
        *,
        metrics_config: Mapping[str, Any],
    ) -> pl.DataFrame:
        if facts.pedestrians.is_empty():
            return pl.DataFrame(schema=DIM_PEDESTRIANS_SCHEMA)

        human_params = episode.human_params or {}
        default_waypoints = (
            _opt_int(
                human_params.get(
                    "waypoints_count", metrics_config.get("waypoints_count")
                )
            )
            or 0
        )

        topics = episode.topics or {}
        ped_tracks = topics.get(PEDESTRIAN_TRACKS_TOPIC)
        if isinstance(ped_tracks, pl.LazyFrame):
            ped_tracks = ped_tracks.collect()

        base_lf = facts.pedestrians.lazy()
        if ped_tracks is not None and not ped_tracks.is_empty():
            agent_types_lf = (
                ped_tracks.lazy()
                .group_by("ped_id")
                .agg(agent_type=pl.col("agent_type").drop_nulls().first())
            )
            base_lf = base_lf.join(
                agent_types_lf, on="ped_id", how="left"
            ).with_columns(
                agent_type=pl.col("agent_type").fill_null("adult"),
                waypoints_count=pl.lit(default_waypoints, dtype=pl.Int64),
            )
        else:
            base_lf = base_lf.with_columns(
                agent_type=pl.lit("adult", dtype=pl.String),
                waypoints_count=pl.lit(default_waypoints, dtype=pl.Int64),
            )

        return (
            base_lf.select(list(DIM_PEDESTRIANS_SCHEMA.names()))
            .cast(DIM_PEDESTRIANS_SCHEMA)
            .collect()
        )

    @staticmethod
    def _build_dim_world_entities(
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        env_id: str,
    ) -> pl.DataFrame:
        topics = episode.topics or {}
        explicit_entities = topics.get(_WORLD_ENTITIES_TOPIC)
        if isinstance(explicit_entities, pl.LazyFrame):
            explicit_entities = explicit_entities.collect()
        if explicit_entities is not None and not explicit_entities.is_empty():
            return (
                explicit_entities.lazy()
                .with_columns(
                    run_id=pl.lit(run_id, dtype=pl.String),
                    episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
                    env_id=pl.col("env_id").fill_null(env_id),
                )
                .select(list(DIM_WORLD_ENTITIES_SCHEMA.names()))
                .cast(DIM_WORLD_ENTITIES_SCHEMA)
                .collect()
            )

        sem_df = episode.semantic_snapshot
        if sem_df is None:
            raw_sem = topics.get(_WORLD_SEMANTICS_TOPIC)
            if isinstance(raw_sem, pl.LazyFrame):
                sem_df = raw_sem.collect()
            elif isinstance(raw_sem, pl.DataFrame):
                sem_df = raw_sem

        if sem_df is None or sem_df.is_empty():
            return pl.DataFrame(schema=DIM_WORLD_ENTITIES_SCHEMA)

        return (
            sem_df.lazy()
            .sort("t_ns")
            .group_by("env_id", "entity_name", maintain_order=True)
            .agg(
                entity_kind=pl.col("kind").drop_nulls().first().fill_null("entity"),
                zone=pl.col("value_str")
                .filter(pl.col("field_name") == "zone")
                .drop_nulls()
                .first(),
                activation_dist=pl.col("value_num")
                .filter(
                    pl.col("field_name").is_in(
                        ["activation_dist", "activation_distance"]
                    )
                )
                .drop_nulls()
                .first(),
                acoustic_tl_db=pl.col("value_num")
                .filter(
                    pl.col("field_name").is_in(
                        ["acoustic_tl_db", "transmission_loss_db"]
                    )
                )
                .drop_nulls()
                .first(),
                initial_state=pl.col("value_str")
                .filter(
                    pl.col("field_name").is_in(
                        ["state", "state_label", "initial_state"]
                    )
                )
                .drop_nulls()
                .first(),
            )
            .with_columns(
                run_id=pl.lit(run_id, dtype=pl.String),
                episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
                env_id=pl.col("env_id").fill_null(env_id),
            )
            .select(list(DIM_WORLD_ENTITIES_SCHEMA.names()))
            .cast(DIM_WORLD_ENTITIES_SCHEMA)
            .collect()
        )


def write_episode_star_tables(
    tables: StarSchemaTables, directory: pathlib.Path
) -> dict[str, pathlib.Path]:
    """Write per-episode dimension and fact tables into ``directory`` (e.g. ``episode_dir / 'metrics'``)."""
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}
    for name, frame in tables.by_name().items():
        path = directory / f"{name}.parquet"
        frame.write_parquet(path)
        written[name] = path
    return written


def read_episode_star_tables(directory: pathlib.Path) -> StarSchemaTables | None:
    """Read per-episode dimension and fact tables from ``directory``, returning None when absent."""
    if not directory.is_dir():
        return None
    dim_ep_path = directory / "dim_episodes.parquet"
    if not dim_ep_path.is_file():
        return None

    loaded: dict[str, pl.DataFrame] = {}
    for name, schema in STAR_SCHEMA_SCHEMAS.items():
        path = directory / f"{name}.parquet"
        if path.is_file():
            loaded[name] = (
                pl.read_parquet(path).select(list(schema.names())).cast(schema)
            )
        else:
            loaded[name] = pl.DataFrame(schema=schema)

    return StarSchemaTables(**loaded)


def write_star_schema(
    tables: StarSchemaTables, benchmark_dir: pathlib.Path
) -> dict[str, pathlib.Path]:
    """Write ``dim_*.parquet`` at ``benchmark_dir`` and ``fact_*.parquet`` under ``benchmark_dir / 'metrics'``."""
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = benchmark_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, pathlib.Path] = {}
    for name, frame in tables.dimension_tables().items():
        path = benchmark_dir / f"{name}.parquet"
        frame.write_parquet(path)
        written[name] = path

    fact_frames = tables.fact_tables()
    write_fact_tables(
        ScopedFactTables(
            **{
                name.removeprefix("fact_"): frame
                for name, frame in fact_frames.items()
            }
        ),
        metrics_dir,
    )
    for name in fact_frames:
        written[name] = metrics_dir / f"{name}.parquet"

    return written


def scan_star_schema(benchmark_dir: pathlib.Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan all dimension and fact Parquet tables from ``benchmark_dir`` via ``pl.scan_parquet`` (L1)."""
    metrics_dir = benchmark_dir / "metrics"
    scanned: dict[str, pl.LazyFrame] = {}

    for name, schema in DIMENSION_SCHEMAS.items():
        path = benchmark_dir / f"{name}.parquet"
        if path.is_file():
            scanned[name] = pl.scan_parquet(path)
        else:
            scanned[name] = pl.LazyFrame(schema=schema)

    for short_name, schema in FACT_SCHEMAS.items():
        table_name = f"fact_{short_name}"
        path = metrics_dir / f"{table_name}.parquet"
        if not path.is_file():
            path = benchmark_dir / f"{table_name}.parquet"
        if path.is_file():
            scanned[table_name] = pl.scan_parquet(path)
        else:
            scanned[table_name] = pl.LazyFrame(schema=schema)

    return scanned
