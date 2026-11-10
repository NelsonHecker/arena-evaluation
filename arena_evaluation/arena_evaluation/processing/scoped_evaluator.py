"""Multi-resolution scoped evaluator: one episode in, five typed fact tables out.

The evaluator drives the scope-aware calculators through the MetricRegistry over
the natural spatiotemporal windows of an episode (episode, encounter, contact,
pedestrian, robot) and pivots their outputs into the star-schema fact contracts.
Detector windows come from :mod:`processing.windows` and are fully vectorized;
calculators run once per window over an isolated episode view, and fact tables are
materialized only at the serialization boundary.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    MetricScope,
    ScopeExecutionOutcome,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.metrics.safety.contact_attribution import (
    COLLISION_EVENTS_TOPIC,
    _contacts_from_bumper_events,
    _contacts_from_clearance,
    _resolve_contact_targets,
    _static_point,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    NS_PER_S,
    PEDESTRIAN_TRACKS_TOPIC,
    ROBOT_TRACKS_TOPIC,
    Trajectory2D,
    extract_ped_tracks,
    resolve_crowd_condition,
)
from arena_evaluation.processing.metrics.social.proxemics import (
    pedestrian_intrusion_counts_frame,
)
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
from arena_evaluation.storage.declarations import (
    _COLLISION_NULLABLE,
    _CONTACT_NULLABLE,
    _EDGE_CASE_NULLABLE,
    _ENCOUNTER_NULLABLE,
    _EPISODE_NULLABLE,
    _FORMATION_NULLABLE,
    _JUDGE_DISAGREEMENT_NULLABLE,
    _MECHANISM_STATE_NULLABLE,
    _PEDESTRIAN_NULLABLE,
    _PHASE_OUTCOME_NULLABLE,
    _ROBOT_NULLABLE,
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
    PHASE_OUTCOME_FACT_SCHEMA,
    ROBOT_FACT_SCHEMA,
    FactTable,
    MetricDeclaration,
    ScopedFactTables,
    declarations_for,
    write_fact_tables,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

EVALUATOR_VERSION: Final[str] = "wp2-scoped-evaluator-v1"

NO_SAMPLES: Final[str] = "no_samples"
PEDSIM_UNAVAILABLE: Final[str] = "pedsim_unavailable"
REFERENCE_MISSING: Final[str] = "reference_missing"
ZERO_DENOMINATOR: Final[str] = "zero_denominator"
TOPIC_MISSING: Final[str] = "topic_missing"
SERIES_MISSING: Final[str] = "series_missing"

# Registry provenance kinds that keep the legacy `field:no_samples` base reason; any other code
# (e.g. `reference_exceeds_path`) replaces it as the field's distinct reason.
_GENERIC_PROVENANCE_KINDS: Final[tuple[str, ...]] = (NO_SAMPLES, PEDSIM_UNAVAILABLE, TOPIC_MISSING, "exception")

__all__ = [
    "COLLISION_FACT_SCHEMA",
    "CONTACT_FACT_SCHEMA",
    "EDGE_CASE_FACT_SCHEMA",
    "ENCOUNTER_FACT_SCHEMA",
    "EPISODE_FACT_SCHEMA",
    "FORMATION_FACT_SCHEMA",
    "JUDGE_DISAGREEMENT_FACT_SCHEMA",
    "MECHANISM_STATE_FACT_SCHEMA",
    "PHASE_OUTCOME_FACT_SCHEMA",
    "FACT_SCHEMAS",
    "FACT_TABLE_NAMES",
    "PEDESTRIAN_FACT_SCHEMA",
    "ROBOT_FACT_SCHEMA",
    "ScopedFactTables",
    "write_fact_tables",
    "EpisodeScopeHierarchy",
    "ScopedEvaluator",
    "EVALUATOR_VERSION",
]

COLLISION_CONTACTS_TOPIC: Final[str] = "collision_contacts"
ROBOT_POWER_TOPIC: Final[str] = "robot_power"
ROBOT_ACOUSTICS_TOPIC: Final[str] = "robot_acoustics"

DEFAULT_ROBOT_ACTOR_RADIUS_M: Final[float] = 0.3
CONTACT_PRE_WINDOW_S: Final[float] = 0.5
CONTACT_POST_WINDOW_S: Final[float] = 0.5
METRES_PER_WH_J: Final[float] = 3600.0
_ENV_PREFIX: Final[re.Pattern[str]] = re.compile(r"^env_[0-9]+_")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeScopeHierarchy:
    """Immutable 5-scope slice hierarchy decomposed from a single episode bundle."""

    contacts: tuple[ScopeSlice, ...]
    encounters: tuple[ScopeSlice, ...]
    pedestrians: tuple[ScopeSlice, ...]
    robots: tuple[ScopeSlice, ...]
    episodes: tuple[ScopeSlice, ...]
    encounter_windows: pl.DataFrame
    contact_windows: pl.DataFrame


def _materialize_episode_bundle(episode: AlignedEpisodeBundle) -> AlignedEpisodeBundle:
    """Materialize all LazyFrame topic and fleet tables once via pl.collect_all before slicing."""
    lazy_keys: list[tuple[str, str]] = []
    lazy_frames: list[pl.LazyFrame] = []

    raw_topics = episode.topics if isinstance(episode.topics, Mapping) else {}
    topics: dict[str, pl.DataFrame] = {}
    for name, table in raw_topics.items():
        if isinstance(table, pl.LazyFrame):
            lazy_keys.append(("topic", name))
            lazy_frames.append(table)
        elif table is not None:
            topics[name] = table

    raw_fleet = episode.fleet if isinstance(episode.fleet, Mapping) else {}
    fleet: dict[str, pl.DataFrame] = {}
    for name, table in raw_fleet.items():
        if isinstance(table, pl.LazyFrame):
            lazy_keys.append(("fleet", name))
            lazy_frames.append(table)
        elif table is not None:
            fleet[name] = table

    if isinstance(episode.data, pl.LazyFrame):
        lazy_keys.append(("data", "data"))
        lazy_frames.append(episode.data)

    data_df = episode.data if isinstance(episode.data, pl.DataFrame) else pl.DataFrame()
    if lazy_frames:
        collected = pl.collect_all(lazy_frames)
        for (kind, name), df in zip(lazy_keys, collected, strict=True):
            if kind == "topic":
                topics[name] = df
            elif kind == "fleet":
                fleet[name] = df
            else:
                data_df = df

    return dataclasses.replace(episode, data=data_df, topics=topics, fleet=fleet)


def _table(episode: AlignedEpisodeBundle, topic: str) -> pl.DataFrame | None:
    """Materialized flat table of an episode topic, None when absent."""
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


def _row_null_reason(row: Mapping[str, object], nullable: tuple[str, ...], provenance: Mapping[str, str] | None = None) -> str:
    """``field:code`` provenance for every mapped field that stayed null; a calculator code wins over no_samples."""
    codes: list[str] = []
    for field in nullable:
        if row.get(field) is not None:
            continue
        override = (provenance or {}).get(field)
        if isinstance(override, str) and override:
            kind = override.partition(":")[2].partition(":")[0]
            if kind not in _GENERIC_PROVENANCE_KINDS:
                codes.append(override)
                continue
        codes.append(f"{field}:{NO_SAMPLES}")
    return ";".join(codes)


def _null_reason_expression(nullable: tuple[str, ...]) -> pl.Expr:
    """``field:code`` provenance expression for every nullable column of a child table row."""
    if not nullable:
        return pl.lit("", dtype=pl.String)
    parts = [pl.when(pl.col(name).is_null()).then(pl.lit(f"{name}:{NO_SAMPLES}", dtype=pl.String)).otherwise(pl.lit(None, dtype=pl.String)) for name in nullable]
    return pl.concat_str(parts, separator=";", ignore_nulls=True).fill_null("")


def _merge_provenance_reasons(base_reason: str, provenance: Mapping[str, str], row: Mapping[str, object]) -> str:
    """Combine field missingness codes with registry topic_missing and exception provenance."""
    parts = [part for part in base_reason.split(";") if part]
    seen = set(parts)
    for key, code in provenance.items():
        if ":exception:" in code or (":topic_missing:" in code and row.get(key) is None) or (":series_missing:" in code and row.get(key) is None):
            if code not in seen:
                parts.append(code)
                seen.add(code)
    return ";".join(parts)


def _declared_value(value: object, declaration: MetricDeclaration) -> object:
    """One calculator value coerced to its declared fact dtype; None stays null for provenance."""
    if value is None:
        return None
    match declaration.dtype:
        case pl.Float64:
            return float(typing.cast(float, value))
        case pl.Int64:
            return int(typing.cast(int, value))
        case pl.Boolean:
            if not isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                raise TypeError(f"{declaration.name}: expected a boolean, got {type(value).__name__}")
            return bool(value)
        case pl.String:
            if not isinstance(value, str):
                raise TypeError(f"{declaration.name}: expected a string, got {type(value).__name__}")
            return value
        case unsupported:
            raise TypeError(f"{declaration.name}: unsupported fact dtype {unsupported}")


def _declared_fact_values(results: Mapping[str, object], table: FactTable) -> dict[str, object]:
    """Every declared fact column of ``table``, copied from the calculator result of its name."""
    return {declaration.name: _declared_value(results.get(declaration.name), declaration) for declaration in declarations_for(table)}


_EDGE_CASE_LIST_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("step", "edge_case_steps"),
    ("agent", "edge_case_agents"),
    ("missing", "edge_case_missing"),
    ("unplayed", "edge_case_unplayed"),
)
_FORMATION_LIST_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("member", "edge_case_formation_members"),
    ("absent", "edge_case_formation_absent"),
)
_EDGE_CASE_ELEMENT_COLUMNS: Final[tuple[str, ...]] = tuple(key for _, key in _EDGE_CASE_LIST_FIELDS)
_FORMATION_ELEMENT_COLUMNS: Final[tuple[str, ...]] = tuple(key for _, key in _FORMATION_LIST_FIELDS)
_MECHANISM_FIELDS: Final[tuple[str, ...]] = ("locked", "progress")


def _sequence(value: object) -> list[object]:
    """The list a child-table builder reshapes; a non-list output has no rows."""
    if isinstance(value, list):
        return typing.cast(list[object], value)
    return []


def _with_element_columns(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Add the element columns a sparse list made absent, as null strings."""
    return frame.with_columns([pl.col(column) if column in frame.columns else pl.lit(None, dtype=pl.String).alias(column) for column in columns])


def _event_time_ns(episode: AlignedEpisodeBundle) -> np.ndarray:
    """Nanosecond clock the EPISODE-scope calculators sampled their per-sample lists on."""
    frame = episode.data
    if frame is None or "t_ns" not in frame.columns:
        return np.array([], dtype=np.int64)
    return frame.get_column("t_ns").to_numpy().astype(np.int64)


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    """Typed fact frame of the given rows, empty with the canonical schema when there are none."""
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def _robot_trajectory_from_frame(robot_tracks: pl.DataFrame | None, radius_m: float) -> Trajectory2D:
    """Construct Trajectory2D from materialized robot_tracks DataFrame."""
    import numpy as np

    empty = np.array([], dtype=np.float64)
    if not _has_rows(robot_tracks):
        return Trajectory2D(
            t_ns=np.array([], dtype=np.int64),
            x=empty,
            y=empty.copy(),
            vx=empty.copy(),
            vy=empty.copy(),
            radius_m=radius_m,
        )
    rows = typing.cast(pl.DataFrame, robot_tracks).select("t_ns", "x", "y", "vx", "vy").drop_nulls(["t_ns", "x", "y"]).with_columns(pl.col("vx").fill_null(0.0), pl.col("vy").fill_null(0.0)).sort("t_ns")
    if len(rows) == 0:
        return Trajectory2D(
            t_ns=np.array([], dtype=np.int64),
            x=empty,
            y=empty.copy(),
            vx=empty.copy(),
            vy=empty.copy(),
            radius_m=radius_m,
        )
    return Trajectory2D(
        t_ns=rows.get_column("t_ns").to_numpy().astype(np.int64),
        x=rows.get_column("x").to_numpy().astype(np.float64),
        y=rows.get_column("y").to_numpy().astype(np.float64),
        vx=np.nan_to_num(
            rows.get_column("vx").to_numpy().astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        vy=np.nan_to_num(
            rows.get_column("vy").to_numpy().astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        radius_m=radius_m,
    )


class ScopedEvaluator:
    """Multi-resolution scoped evaluator delegating metric calculations to the scope-native registry."""

    def __init__(
        self,
        robot_params: RobotParams,
        metrics_config: Mapping[str, Any] | None = None,
        detector_params: DetectorParams = DEFAULT_DETECTOR_PARAMS,
    ) -> None:
        self.robot_params = robot_params
        self.metrics_config: dict[str, Any] = dict(metrics_config or {})
        self.detector_params = detector_params

    def decompose_hierarchy(self, episode: AlignedEpisodeBundle, *, run_id: str, episode_id: int) -> EpisodeScopeHierarchy:
        """Stage 1: Materialize episode topics once and decompose into immutable ScopeSlice hierarchy."""
        materialized = _materialize_episode_bundle(episode)
        robot_tracks = _table(materialized, ROBOT_TRACKS_TOPIC)
        pedestrian_tracks = _table(materialized, PEDESTRIAN_TRACKS_TOPIC)
        contact_windows = empty_contact_windows()
        encounter_windows = empty_encounter_windows()

        if _has_rows(robot_tracks):
            encounter_windows = self._detect_encounters(materialized, typing.cast(pl.DataFrame, robot_tracks), pedestrian_tracks)
            contact_windows = self._detect_contacts(
                materialized,
                encounter_windows,
                typing.cast(pl.DataFrame, robot_tracks),
                pedestrian_tracks,
            )

        robot_id = self._robot_actor_id(materialized, robot_tracks)
        robot_token = self._short_robot_token(robot_id)
        episode_slice_id = f"{run_id}:{int(episode_id)}:{robot_id}"
        crowd_condition = resolve_crowd_condition(materialized, self.metrics_config)

        ep_start_ns = 0
        ep_end_ns = 0
        env_id: str | None = None
        if _has_rows(robot_tracks):
            rt_df = typing.cast(pl.DataFrame, robot_tracks)
            t_col = rt_df.get_column("t_ns").drop_nulls()
            min_t = t_col.min()
            max_t = t_col.max()
            if isinstance(min_t, int):
                ep_start_ns = min_t
            if isinstance(max_t, int):
                ep_end_ns = max_t
            envs = rt_df.get_column("env_id").drop_nulls()
            if len(envs) > 0:
                env_id = str(envs[0])

        encounter_ids = {int(index): self._encounter_id(episode_id, robot_token, int(index)) for index in encounter_windows.get_column("encounter_idx").to_numpy()}

        contact_slices: list[ScopeSlice] = []
        if len(contact_windows) > 0:
            pre_window_s = float(self.metrics_config.get("pre_contact_window_s", CONTACT_PRE_WINDOW_S))
            post_window_s = float(self.metrics_config.get("post_contact_window_s", CONTACT_POST_WINDOW_S))
            for (
                contact_idx,
                collider_kind,
                obstacle_id,
                t_onset_ns,
                t_end_ns,
                obstacle_x,
                obstacle_y,
                encounter_idx,
            ) in zip(
                contact_windows.get_column("contact_idx").to_numpy(),
                contact_windows.get_column("collider_kind").to_numpy(),
                contact_windows.get_column("obstacle_id").to_numpy(),
                contact_windows.get_column("t_onset_ns").to_numpy(),
                contact_windows.get_column("t_end_ns").to_numpy(),
                contact_windows.get_column("obstacle_x").to_numpy(),
                contact_windows.get_column("obstacle_y").to_numpy(),
                contact_windows.get_column("encounter_idx").fill_null(-1).to_numpy(),
                strict=True,
            ):
                internal_kind = ACTOR_PEDESTRIAN if collider_kind in (ACTOR_PEDESTRIAN, "robot") else str(collider_kind)
                actor_id = str(obstacle_id)
                win_start_ns = int(t_onset_ns - round(pre_window_s * NS_PER_S))
                win_end_ns = int(t_end_ns + round(post_window_s * NS_PER_S))
                window_bundle = self._window_episode(materialized, win_start_ns, win_end_ns, actor_id, internal_kind)
                static_pt = None if collider_kind in (ACTOR_PEDESTRIAN, "robot") else _static_point(float(obstacle_x), float(obstacle_y))
                enc_index = None if int(encounter_idx) < 0 else int(encounter_idx)
                parent_encounter_id = None if enc_index is None else encounter_ids.get(enc_index)
                contact_id = self._contact_id(episode_id, robot_token, int(contact_idx))
                contact_slices.append(
                    ScopeSlice(
                        scope=MetricScope.CONTACT,
                        slice_id=contact_id,
                        parent_id=parent_encounter_id,
                        t_start_ns=int(t_onset_ns),
                        t_end_ns=int(t_end_ns),
                        identity={
                            "run_id": run_id,
                            "episode_id": int(episode_id),
                            "robot_id": robot_id,
                            "contact_id": contact_id,
                            "encounter_id": parent_encounter_id,
                            "collider_kind": str(collider_kind),
                            "obstacle_id": actor_id,
                            "static_point": static_pt,
                            "crowd_condition": crowd_condition,
                            "pre_contact_window_s": pre_window_s,
                            "post_contact_window_s": post_window_s,
                        },
                        window_bundle=window_bundle,
                        full_episode=materialized,
                    )
                )

        encounter_slices: list[ScopeSlice] = []
        if len(encounter_windows) > 0:
            for (
                encounter_idx,
                actor_id,
                actor_kind,
                t_start_ns,
                t_end_ns,
                min_clearance_m,
                approach_type,
            ) in zip(
                encounter_windows.get_column("encounter_idx").fill_null(-1).to_numpy(),
                encounter_windows.get_column("actor_id").to_numpy(),
                encounter_windows.get_column("actor_kind").to_numpy(),
                encounter_windows.get_column("t_start_ns").to_numpy(),
                encounter_windows.get_column("t_end_ns").to_numpy(),
                encounter_windows.get_column("min_clearance_m").to_numpy(),
                encounter_windows.get_column("approach_type").to_numpy(),
                strict=True,
            ):
                actor = str(actor_id)
                kind = str(actor_kind)
                window_bundle = self._window_episode(materialized, int(t_start_ns), int(t_end_ns), actor, kind)
                encounter_id = self._encounter_id(episode_id, robot_token, int(encounter_idx))
                encounter_slices.append(
                    ScopeSlice(
                        scope=MetricScope.ENCOUNTER,
                        slice_id=encounter_id,
                        parent_id=episode_slice_id,
                        t_start_ns=int(t_start_ns),
                        t_end_ns=int(t_end_ns),
                        identity={
                            "run_id": run_id,
                            "episode_id": int(episode_id),
                            "robot_id": robot_id,
                            "encounter_id": encounter_id,
                            "actor_1": robot_id,
                            "actor_2": actor,
                            "actor_2_kind": kind,
                            "approach_type": str(approach_type),
                            "min_clearance_m": _float(min_clearance_m),
                            "crowd_condition": crowd_condition,
                        },
                        window_bundle=window_bundle,
                        full_episode=materialized,
                    )
                )

        pedestrian_slices: list[ScopeSlice] = []
        if _has_rows(pedestrian_tracks):
            tracks_sorted = typing.cast(pl.DataFrame, pedestrian_tracks).sort(["ped_id", "t_ns"])
            for ped_df in tracks_sorted.partition_by("ped_id", maintain_order=True):
                ped_id = str(ped_df.get_column("ped_id")[0])
                ped_t_col = ped_df.get_column("t_ns").drop_nulls()
                min_ped_t = ped_t_col.min()
                max_ped_t = ped_t_col.max()
                ped_start_ns = min_ped_t if isinstance(min_ped_t, int) else ep_start_ns
                ped_end_ns = max_ped_t if isinstance(max_ped_t, int) else ep_end_ns
                ped_topics = dict(materialized.topics or {})
                ped_topics[PEDESTRIAN_TRACKS_TOPIC] = ped_df
                ped_bundle = dataclasses.replace(materialized, data=pl.DataFrame(), topics=ped_topics, fleet={})

                ped_envs = ped_df.get_column("env_id").drop_nulls()
                ped_names = ped_df.get_column("ped_name").drop_nulls()
                ped_policies = ped_df.get_column("policy").drop_nulls()
                ped_speeds = ped_df.get_column("desired_speed").drop_nulls()
                ped_radii = ped_df.get_column("radius_m").drop_nulls()
                ped_xs = ped_df.get_column("x").drop_nulls()
                ped_ys = ped_df.get_column("y").drop_nulls()
                ped_yaws = ped_df.get_column("yaw").drop_nulls()

                pedestrian_slices.append(
                    ScopeSlice(
                        scope=MetricScope.PEDESTRIAN,
                        slice_id=ped_id,
                        parent_id=episode_slice_id,
                        t_start_ns=ped_start_ns,
                        t_end_ns=ped_end_ns,
                        identity={
                            "run_id": run_id,
                            "episode_id": int(episode_id),
                            "env_id": str(ped_envs[0]) if len(ped_envs) > 0 else env_id,
                            "ped_id": ped_id,
                            "ped_name": str(ped_names[0]) if len(ped_names) > 0 else ped_id,
                            "ped_policy": str(ped_policies[0]) if len(ped_policies) > 0 else None,
                            "is_blind": crowd_condition == "blind",
                            "desired_speed": _float(ped_speeds[0]) if len(ped_speeds) > 0 else None,
                            "radius_m": _float(ped_radii[0]) if len(ped_radii) > 0 else None,
                            "spawn_x": _float(ped_xs[0]) if len(ped_xs) > 0 else None,
                            "spawn_y": _float(ped_ys[0]) if len(ped_ys) > 0 else None,
                            "spawn_yaw": _float(ped_yaws[0]) if len(ped_yaws) > 0 else None,
                        },
                        window_bundle=ped_bundle,
                        full_episode=materialized,
                    )
                )

        robot_slice = ScopeSlice(
            scope=MetricScope.ROBOT,
            slice_id=robot_id,
            parent_id=episode_slice_id,
            t_start_ns=ep_start_ns,
            t_end_ns=ep_end_ns,
            identity={
                "run_id": run_id,
                "episode_id": int(episode_id),
                "env_id": env_id,
                "robot_id": robot_id,
                "robot_name": materialized.robot_name,
            },
            window_bundle=materialized,
            full_episode=materialized,
        )

        episode_slice = ScopeSlice(
            scope=MetricScope.EPISODE,
            slice_id=episode_slice_id,
            parent_id=None,
            t_start_ns=ep_start_ns,
            t_end_ns=ep_end_ns,
            identity={
                "run_id": run_id,
                "episode_id": int(episode_id),
                "robot_id": robot_id,
            },
            window_bundle=materialized,
            full_episode=materialized,
        )

        return EpisodeScopeHierarchy(
            contacts=tuple(contact_slices),
            encounters=tuple(encounter_slices),
            pedestrians=tuple(pedestrian_slices),
            robots=(robot_slice,),
            episodes=(episode_slice,),
            encounter_windows=encounter_windows,
            contact_windows=contact_windows,
        )

    def evaluate(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        episode_metrics: Mapping[str, Any] | None = None,
        episode_provenance: Mapping[str, str] | None = None,
    ) -> ScopedFactTables:
        """Evaluate one episode and return its five fact tables."""
        hierarchy = self.decompose_hierarchy(episode, run_id=run_id, episode_id=episode_id)
        episode_slice = hierarchy.episodes[0]
        materialized = episode_slice.full_episode
        pedestrian_tracks = _table(materialized, PEDESTRIAN_TRACKS_TOPIC)

        registry = MetricRegistry(
            self.robot_params,
            world=materialized.map,
            metrics_config=dict(self.metrics_config),
        )

        # Bottom-up Stage 2 execution: CONTACT -> ENCOUNTER -> PEDESTRIAN -> EPISODE & ROBOT
        contact_rows = self._contact_fact_rows(registry, hierarchy.contacts)
        encounter_rows = self._encounter_fact_rows(registry, materialized, hierarchy.encounters, contact_rows)
        pedestrians = self._pedestrian_fact_frame(registry, materialized, hierarchy.pedestrians, encounter_rows)

        provenance: dict[str, str] = dict(episode_provenance or {})
        if episode_metrics is not None:
            metrics = dict(episode_metrics)
            if "total_acoustic_dose_pa2s" not in metrics or metrics["total_acoustic_dose_pa2s"] is None:
                acous_calc = registry.calculators_by_scope[MetricScope.EPISODE].get("acoustic_dose")
                if acous_calc is not None and _has_rows(_table(materialized, ROBOT_ACOUSTICS_TOPIC)):
                    dose_res = acous_calc.calculate_for_scope(materialized, metrics, episode_slice)
                    if dose_res.get("total_acoustic_dose_pa2s") is not None:
                        metrics["total_acoustic_dose_pa2s"] = dose_res["total_acoustic_dose_pa2s"]
                elif not _has_rows(_table(materialized, ROBOT_ACOUSTICS_TOPIC)):
                    provenance["total_acoustic_dose_pa2s"] = f"total_acoustic_dose_pa2s:{TOPIC_MISSING}:{ROBOT_ACOUSTICS_TOPIC}"
        else:
            ep_outcome = registry.run_scope(
                MetricScope.EPISODE,
                materialized,
                available_topics=set(materialized.topics or {}),
                pedsim_available=_has_rows(pedestrian_tracks),
                scope_slice=episode_slice,
            )
            metrics = ep_outcome.results
            provenance = dict(ep_outcome.provenance)

        episode_row = self._episode_fact_row(
            materialized,
            run_id,
            episode_id,
            metrics,
            encounter_rows,
            contact_rows,
            provenance,
        )
        robots = self._robot_fact_frame(registry, hierarchy.robots[0], metrics, provenance)
        robot_id = self._robot_actor_id(materialized, _table(materialized, ROBOT_TRACKS_TOPIC))
        child_identity = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "robot_id": robot_id,
            "metrics": metrics,
        }

        return ScopedFactTables(
            episodes=_frame([episode_row], EPISODE_FACT_SCHEMA),
            encounters=_frame(encounter_rows, ENCOUNTER_FACT_SCHEMA),
            contacts=_frame(contact_rows, CONTACT_FACT_SCHEMA),
            pedestrians=pedestrians,
            robots=robots,
            edge_cases=self._edge_case_fact_frame(**child_identity),
            formations=self._formation_fact_frame(**child_identity),
            phase_outcomes=self._phase_outcome_fact_frame(**child_identity),
            judge_disagreements=self._judge_disagreement_fact_frame(**child_identity),
            mechanism_states=self._mechanism_state_fact_frame(materialized, **child_identity),
            collisions=self._collision_fact_frame(materialized, **child_identity),
        )

    def _detect_encounters(
        self,
        episode: AlignedEpisodeBundle,
        robot_tracks: pl.DataFrame,
        pedestrian_tracks: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Encounter windows of this robot with its pedestrians and other fleet robots."""
        pairs = []
        if _has_rows(pedestrian_tracks):
            pairs.append(
                pedestrian_pairs_lf(
                    robot_tracks,
                    typing.cast(pl.DataFrame, pedestrian_tracks),
                    float(self.robot_params.robot_radius),
                    self.detector_params,
                )
            )
        if episode.fleet:
            pairs.append(
                fleet_pairs_lf(
                    robot_tracks,
                    episode.fleet,
                    float(self.robot_params.robot_radius),
                    params=self.detector_params,
                )
            )
        if not pairs:
            return empty_encounter_windows()
        pairs_lf = pairs[0] if len(pairs) == 1 else pl.concat(pairs)
        return detect_encounter_windows_lf(pairs_lf, self.detector_params).collect()

    def _detect_contacts(
        self,
        episode: AlignedEpisodeBundle,
        encounters: pl.DataFrame,
        robot_tracks: pl.DataFrame | None = None,
        pedestrian_tracks: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Contact windows of the recorded collision stream or clearance/bumper fallback, linked to their enclosing encounters."""
        collision_contacts = _table(episode, COLLISION_CONTACTS_TOPIC)
        collision_events = _table(episode, COLLISION_EVENTS_TOPIC)
        if collision_contacts is None and collision_events is None:
            return empty_contact_windows()

        robot = _robot_trajectory_from_frame(
            robot_tracks if robot_tracks is not None else _table(episode, ROBOT_TRACKS_TOPIC),
            float(self.robot_params.robot_radius),
        )
        tracks = extract_ped_tracks(pedestrian_tracks if pedestrian_tracks is not None else _table(episode, PEDESTRIAN_TRACKS_TOPIC))

        if _has_rows(collision_contacts):
            cc_df = typing.cast(pl.DataFrame, collision_contacts)
            resolved = _resolve_contact_targets(cc_df, robot, tracks)
            prepared = cc_df.with_columns(
                obstacle_id=resolved.get_column("obstacle_id"),
                kind=resolved.get_column("obstacle_kind"),
            )
            return detect_contact_windows_lf(prepared, encounters, self.detector_params).collect()

        merge_gap_ns = int(round(self.detector_params.contact_merge_gap_s * NS_PER_S))
        fallback_events = _contacts_from_clearance(robot=robot, tracks=tracks, merge_gap_ns=merge_gap_ns)
        fallback_events.extend(_contacts_from_bumper_events(collision_events))
        if not fallback_events:
            return empty_contact_windows()

        step_ns = max(1, merge_gap_ns // 2)
        syn_t_ns: list[int] = []
        syn_obs_id: list[str] = []
        syn_kind: list[str] = []
        for ev in fallback_events:
            t_cur = int(ev.t0_ns)
            t_stop = int(ev.t_end_ns)
            while t_cur < t_stop:
                syn_t_ns.append(t_cur)
                syn_obs_id.append(ev.obstacle_id)
                syn_kind.append(ev.obstacle_kind)
                t_cur += step_ns
            syn_t_ns.append(t_stop)
            syn_obs_id.append(ev.obstacle_id)
            syn_kind.append(ev.obstacle_kind)

        n_syn = len(syn_t_ns)
        synthesized = pl.DataFrame(
            {
                "t_ns": syn_t_ns,
                "obstacle_id": syn_obs_id,
                "kind": syn_kind,
                "obstacle_x": [0.0] * n_syn,
                "obstacle_y": [0.0] * n_syn,
                "obstacle_z": [0.0] * n_syn,
            },
            schema={
                "t_ns": pl.Int64,
                "obstacle_id": pl.String,
                "kind": pl.String,
                "obstacle_x": pl.Float64,
                "obstacle_y": pl.Float64,
                "obstacle_z": pl.Float64,
            },
        )
        return detect_contact_windows_lf(synthesized, encounters, self.detector_params).collect()

    def _window_episode(
        self,
        episode: AlignedEpisodeBundle,
        t_start_ns: int,
        t_end_ns: int,
        actor_id: str,
        actor_kind: str,
    ) -> AlignedEpisodeBundle:
        """Episode view cut to ``[t_start_ns, t_end_ns]`` with the pedestrian stream restricted to the window's actor."""
        topics: dict[str, pl.DataFrame] = {}
        for name, frame in (episode.topics or {}).items():
            if not isinstance(frame, pl.DataFrame):
                continue
            if name == PEDESTRIAN_TRACKS_TOPIC:
                frame = self._actor_track_frame(episode, frame, actor_id, actor_kind)
            if len(frame) > 0:
                frame = frame.filter((pl.col("t_ns") >= t_start_ns) & (pl.col("t_ns") <= t_end_ns))
            topics[name] = frame
        if PEDESTRIAN_TRACKS_TOPIC not in topics and actor_kind != ACTOR_PEDESTRIAN:
            actor_frame = self._actor_track_frame(episode, pl.DataFrame(), actor_id, actor_kind)
            if len(actor_frame) > 0:
                actor_frame = actor_frame.filter((pl.col("t_ns") >= t_start_ns) & (pl.col("t_ns") <= t_end_ns))
            topics[PEDESTRIAN_TRACKS_TOPIC] = actor_frame
        return dataclasses.replace(episode, data=pl.DataFrame(), topics=topics, fleet={})

    def _actor_track_frame(
        self,
        episode: AlignedEpisodeBundle,
        pedestrian_tracks: pl.DataFrame,
        actor_id: str,
        actor_kind: str,
    ) -> pl.DataFrame:
        """Pedestrian-shaped track of the window's actor: pedestrian rows or another robot's task poses."""
        if actor_kind == ACTOR_PEDESTRIAN:
            if len(pedestrian_tracks) == 0:
                return pedestrian_tracks
            return pedestrian_tracks.filter(pl.col("ped_id") == actor_id)
        task_poses = episode.fleet.get(actor_id)
        if task_poses is None or len(task_poses) < 2:
            return (
                pedestrian_tracks.head(0)
                if len(pedestrian_tracks) > 0
                else pl.DataFrame(
                    schema={
                        "t_ns": pl.Int64,
                        "ped_id": pl.String,
                        "ped_name": pl.String,
                        "x": pl.Float64,
                        "y": pl.Float64,
                        "vx": pl.Float64,
                        "vy": pl.Float64,
                        "radius_m": pl.Float64,
                        "desired_speed": pl.Float64,
                    }
                )
            )
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
            .select(
                "t_ns",
                "ped_id",
                "ped_name",
                "x",
                "y",
                "vx",
                "vy",
                "radius_m",
                "desired_speed",
            )
        )

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

    def _episode_fact_row(
        self,
        episode: AlignedEpisodeBundle,
        run_id: str,
        episode_id: int,
        metrics: Mapping[str, Any],
        encounter_rows: list[dict[str, object]],
        contact_rows: list[dict[str, object]],
        provenance: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        """Episode-level fact row from the registry output and child fact tables."""
        del encounter_rows
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
        # Without the recorded collision stream a zero would claim "no collision happened"; the
        # honest value is unknown, carried as null and explained by `null_reason`.
        # Presence of the stream, not its row count: an empty stream means "no collision happened",
        # an absent one means "unknown". Only the second is null.
        collision_stream = _table(episode, COLLISION_EVENTS_TOPIC) is not None
        collision_known = collision_stream or bool(contact_rows)
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
            "total_collisions_ped": collisions_ped if collisions_ped is not None else (pedestrian_contacts if collision_known else None),
            "total_collisions_static": ((collisions_wall or 0) + (collisions_static or 0) if collisions_wall is not None or collisions_static is not None else (static_contacts if collision_known else None)),
            "success_no_robot_fault": None if success is None else bool(success) and robot_fault_contacts == 0,
            "mar": _float(metrics.get("mar")),
            "pfi": _float(metrics.get("pfi")),
            "robot_path_dev_m": _float(metrics.get("robot_path_deviation_m")),
            "total_acoustic_dose_pa2s": _float(metrics.get("total_acoustic_dose_pa2s")),
        }
        row.update(_declared_fact_values(metrics, FactTable.EPISODES))
        row["null_reason"] = self._episode_null_reason(episode, row, provenance or {})
        return row

    def _episode_null_reason(
        self,
        episode: AlignedEpisodeBundle,
        row: Mapping[str, object],
        provenance: Mapping[str, str],
    ) -> str:
        """Episode provenance: missing-field codes, missing trajectory/pedestrian stream topics, and calculator exceptions."""
        reasons = [_row_null_reason(row, _EPISODE_NULLABLE, provenance)]
        if not _has_rows(_table(episode, ROBOT_TRACKS_TOPIC)):
            reasons.append(f"{TOPIC_MISSING}:{ROBOT_TRACKS_TOPIC}")
        if not _has_rows(_table(episode, PEDESTRIAN_TRACKS_TOPIC)):
            reasons.append(PEDSIM_UNAVAILABLE)
        base = ";".join(reason for reason in reasons if reason)
        return _merge_provenance_reasons(base, provenance, row)

    def _contact_fact_rows(self, registry: MetricRegistry, contact_slices: tuple[ScopeSlice, ...]) -> list[dict[str, object]]:
        """One fact row per detected contact slice, attributed by the scope-aware contact calculators."""
        rows: list[dict[str, object]] = []
        for contact_slice in contact_slices:
            outcome: ScopeExecutionOutcome = registry.run_scope(
                MetricScope.CONTACT,
                contact_slice.window_bundle,
                available_topics=set(contact_slice.window_bundle.topics or {}),
                scope_slice=contact_slice,
            )
            contact_res = outcome.results
            ident = contact_slice.identity
            row: dict[str, object] = {
                "run_id": ident["run_id"],
                "episode_id": int(ident["episode_id"]),
                "robot_id": ident["robot_id"],
                "contact_id": contact_slice.slice_id,
                "encounter_id": contact_slice.parent_id,
                "collider_kind": str(ident["collider_kind"]),
                "obstacle_id": str(ident["obstacle_id"]),
                "t_onset_ns": int(contact_slice.t_start_ns),
                "t_end_ns": int(contact_slice.t_end_ns),
                "duration_s": float(max(0, int(contact_slice.t_end_ns) - int(contact_slice.t_start_ns))) / 1e9,
                "impact_speed_mps": _float(contact_res.get("impact_speed_mps")),
                "robot_speed_at_impact_mps": _float(contact_res.get("robot_speed_at_impact_mps")),
                "obstacle_speed_at_impact_mps": _float(contact_res.get("obstacle_speed_at_impact_mps")),
                "closer_label": contact_res.get("closer_label"),
                "robot_closing_share": _float(contact_res.get("robot_closing_share")),
                "counterfactual_freeze_persists": contact_res.get("counterfactual_freeze_persists"),
                "fault_attributed_to": contact_res.get("fault_attributed_to", "unavoidable"),
            }
            row.update(_declared_fact_values(contact_res, FactTable.CONTACTS))
            base_reason = _row_null_reason(row, _CONTACT_NULLABLE, outcome.provenance)
            row["null_reason"] = _merge_provenance_reasons(base_reason, outcome.provenance, row)
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

    def _encounter_fact_rows(
        self,
        registry: MetricRegistry,
        episode: AlignedEpisodeBundle,
        encounter_slices: tuple[ScopeSlice, ...],
        contact_rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """One fact row per encounter slice, from registered encounter calculators."""
        if not encounter_slices:
            return []

        power_tbl = _table(episode, ROBOT_POWER_TOPIC)
        has_power = _has_rows(power_tbl)
        episode_mean_power_w = (
            float(typing.cast(pl.DataFrame, power_tbl).get_column("total_power_w").drop_nulls().mean())  # type: ignore[arg-type]
            if has_power and len(typing.cast(pl.DataFrame, power_tbl).get_column("total_power_w").drop_nulls()) > 0
            else None
        )
        has_reference = self.metrics_config.get("reference_peds") is not None or self.metrics_config.get("reference_episode") is not None

        contacts_by_encounter: dict[str, dict[str, object]] = {}
        for contact in contact_rows:
            encounter_id = contact.get("encounter_id")
            if isinstance(encounter_id, str) and encounter_id not in contacts_by_encounter:
                contacts_by_encounter[encounter_id] = contact

        rows: list[dict[str, object]] = []
        for enc_slice in encounter_slices:
            priors = {"episode_mean_power_w": episode_mean_power_w}
            outcome: ScopeExecutionOutcome = registry.run_scope(
                MetricScope.ENCOUNTER,
                enc_slice.window_bundle,
                available_topics=set(enc_slice.window_bundle.topics or {}),
                prior_results=priors,
                scope_slice=enc_slice,
            )
            enc_res = outcome.results
            ident = enc_slice.identity
            encounter_id = enc_slice.slice_id
            linked_contact = contacts_by_encounter.get(encounter_id)
            row: dict[str, object] = {
                "run_id": ident["run_id"],
                "episode_id": int(ident["episode_id"]),
                "robot_id": ident["robot_id"],
                "encounter_id": encounter_id,
                "actor_1": ident["actor_1"],
                "actor_2": ident["actor_2"],
                "actor_2_kind": ident["actor_2_kind"],
                "t_start_ns": int(enc_slice.t_start_ns),
                "t_end_ns": int(enc_slice.t_end_ns),
                "duration_s": float(max(0, int(enc_slice.t_end_ns) - int(enc_slice.t_start_ns))) / 1e9,
                "approach_type": str(ident["approach_type"]),
                "min_clearance_m": _float(ident["min_clearance_m"]),
                "detour_time_s": _float(enc_res.get("detour_time_s", enc_res.get("cost_ped_delay_s"))),
                "path_deviation_m": _float(enc_res.get("path_deviation_m", enc_res.get("cost_ped_deflection_m"))),
                "encounter_energy_j": _float(enc_res.get("encounter_energy_j", enc_res.get("cost_energy_j"))),
                "energy_surge_ratio": _float(enc_res.get("energy_surge_ratio")),
                "sound_exposure_dose_pa2s": _float(enc_res.get("sound_exposure_dose_pa2s", enc_res.get("cost_sound_pa2s"))),
                "probst_R_robot": _float(enc_res.get("probst_R_robot", enc_res.get("probst_r_robot"))),
                "probst_R_actor2": _float(enc_res.get("probst_R_actor2", enc_res.get("probst_r_target"))),
                "probst_R_time": _float(enc_res.get("probst_R_time", enc_res.get("probst_r_time"))),
                "probst_E_robot": _float(enc_res.get("probst_E_robot", enc_res.get("probst_e_robot"))),
                "probst_E_actor2": _float(enc_res.get("probst_E_actor2", enc_res.get("probst_e_target"))),
                "had_contact": linked_contact is not None,
                "contact_id": None if linked_contact is None else linked_contact.get("contact_id"),
            }
            row.update(_declared_fact_values(enc_res, FactTable.ENCOUNTERS))
            base_reason = self._encounter_null_reason(row, has_reference=has_reference, has_power=has_power)
            row["null_reason"] = _merge_provenance_reasons(base_reason, outcome.provenance, row)
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

    def _pedestrian_fact_frame(
        self,
        registry: MetricRegistry,
        episode: AlignedEpisodeBundle,
        pedestrian_slices: tuple[ScopeSlice, ...],
        encounter_rows: list[dict[str, object]],
    ) -> pl.DataFrame:
        """Per-pedestrian lifetime summaries executed through registered PEDESTRIAN calculators."""
        if not pedestrian_slices:
            return pl.DataFrame(schema=PEDESTRIAN_FACT_SCHEMA)

        intrusions_df = pedestrian_intrusion_counts_frame(episode, float(self.robot_params.robot_radius))
        intrusions_by_ped: dict[str, tuple[int, int]] = {}
        if len(intrusions_df) > 0:
            for ped_id, intimate_cnt, personal_cnt in zip(
                intrusions_df.get_column("ped_id").to_numpy(),
                intrusions_df.get_column("total_intrusions_intimate").to_numpy(),
                intrusions_df.get_column("total_intrusions_personal").to_numpy(),
                strict=True,
            ):
                intrusions_by_ped[str(ped_id)] = (int(intimate_cnt), int(personal_cnt))

        agg_df = self._encounter_aggregates_frame(encounter_rows)
        agg_by_ped: dict[str, tuple[int, float | None, float | None]] = {}
        if len(agg_df) > 0:
            for ped_id, enc_cnt, sound_dose, deflection in zip(
                agg_df.get_column("ped_id").to_numpy(),
                agg_df.get_column("total_encounters").to_numpy(),
                agg_df.get_column("cumulative_sound_dose_pa2s").to_numpy(),
                agg_df.get_column("ped_path_deflection_m").to_numpy(),
                strict=True,
            ):
                agg_by_ped[str(ped_id)] = (
                    int(enc_cnt),
                    None if sound_dose is None or (isinstance(sound_dose, float) and sound_dose != sound_dose) else float(sound_dose),
                    None if deflection is None or (isinstance(deflection, float) and deflection != deflection) else float(deflection),
                )

        rows: list[dict[str, object]] = []
        for ped_slice in pedestrian_slices:
            ped_id = ped_slice.slice_id
            intimate_cnt, personal_cnt = intrusions_by_ped.get(ped_id, (0, 0))
            enc_cnt, sound_dose, deflection = agg_by_ped.get(ped_id, (0, None, None))
            priors: dict[str, Any] = {
                "total_intrusions_intimate": intimate_cnt,
                "total_intrusions_personal": personal_cnt,
                "total_encounters": enc_cnt,
                "cumulative_sound_dose_pa2s": sound_dose,
                "ped_path_deflection_m": deflection,
            }
            outcome: ScopeExecutionOutcome = registry.run_scope(
                MetricScope.PEDESTRIAN,
                ped_slice.window_bundle,
                available_topics=set(ped_slice.window_bundle.topics or {}),
                prior_results=priors,
                pedsim_available=True,
                scope_slice=ped_slice,
            )
            ped_res = outcome.results
            ident = ped_slice.identity
            row: dict[str, object] = {
                "run_id": ident["run_id"],
                "episode_id": int(ident["episode_id"]),
                "env_id": ident["env_id"],
                "ped_id": ped_id,
                "ped_name": ident["ped_name"],
                "ped_policy": ident["ped_policy"],
                "is_blind": bool(ident["is_blind"]),
                "desired_speed": _float(ident["desired_speed"]),
                "radius_m": _float(ident["radius_m"]),
                "spawn_x": _float(ident["spawn_x"]),
                "spawn_y": _float(ident["spawn_y"]),
                "spawn_yaw": _float(ident["spawn_yaw"]),
                "total_distance_m": _float(ped_res.get("total_distance_m")),
                "mean_speed_mps": _float(ped_res.get("mean_speed_mps")),
                "cumulative_sound_dose_pa2s": _float(ped_res.get("cumulative_sound_dose_pa2s")),
                "total_encounters": int(enc_cnt),
                "total_intrusions_intimate": int(ped_res.get("total_intrusions_intimate", intimate_cnt) or 0),
                "total_intrusions_personal": int(ped_res.get("total_intrusions_personal", personal_cnt) or 0),
                "ped_path_deflection_m": _float(ped_res.get("ped_path_deflection_m")),
                "round_trips_completed": int(ped_res.get("round_trips_completed") or 0),
            }
            row.update(_declared_fact_values(ped_res, FactTable.PEDESTRIANS))
            base_reason = _row_null_reason(row, _PEDESTRIAN_NULLABLE, outcome.provenance)
            row["null_reason"] = _merge_provenance_reasons(base_reason, outcome.provenance, row)
            rows.append(row)

        return _frame(rows, PEDESTRIAN_FACT_SCHEMA)

    @staticmethod
    def _encounter_aggregates_frame(
        encounter_rows: list[dict[str, object]],
    ) -> pl.DataFrame:
        """Per-actor aggregates of the encounter facts: count, sound dose and reference-twin deflection."""
        if not encounter_rows:
            return pl.DataFrame(
                schema={
                    "ped_id": pl.String,
                    "total_encounters": pl.Int64,
                    "cumulative_sound_dose_pa2s": pl.Float64,
                    "ped_path_deflection_m": pl.Float64,
                }
            )
        frame = pl.DataFrame(encounter_rows, schema=ENCOUNTER_FACT_SCHEMA)
        return frame.group_by(pl.col("actor_2").alias("ped_id")).agg(
            total_encounters=pl.len().cast(pl.Int64),
            cumulative_sound_dose_pa2s=pl.col("sound_exposure_dose_pa2s").drop_nulls().sum() if len(frame.get_column("sound_exposure_dose_pa2s").drop_nulls()) > 0 else pl.lit(None, dtype=pl.Float64),
            ped_path_deflection_m=pl.col("path_deviation_m").drop_nulls().sum() if len(frame.get_column("path_deviation_m").drop_nulls()) > 0 else pl.lit(None, dtype=pl.Float64),
        )

    def _robot_fact_frame(
        self,
        registry: MetricRegistry,
        robot_slice: ScopeSlice,
        metrics: Mapping[str, Any],
        episode_provenance: Mapping[str, str],
    ) -> pl.DataFrame:
        """Per-robot lifetime summary of work, emissions and drivetrain losses from registered robot calculators."""
        episode = robot_slice.window_bundle
        outcome: ScopeExecutionOutcome = registry.run_scope(
            MetricScope.ROBOT,
            episode,
            available_topics=set(episode.topics or {}),
            prior_results=dict(metrics),
            scope_slice=robot_slice,
        )
        robot_res = outcome.results
        ident = robot_slice.identity

        row: dict[str, object] = {
            "run_id": ident["run_id"],
            "episode_id": int(ident["episode_id"]),
            "env_id": ident["env_id"],
            "robot_id": ident["robot_id"],
            "robot_name": ident["robot_name"],
            "path_length_m": _float(robot_res.get("path_length_m", metrics.get("path_length"))),
            "mean_speed_mps": _float(robot_res.get("mean_speed_mps", metrics.get("velocity_mean"))),
            "max_speed_mps": _float(robot_res.get("max_speed_mps", metrics.get("velocity_max"))),
            "mean_power_w": _float(robot_res.get("mean_power_w")),
            "peak_power_w": _float(robot_res.get("peak_power_w")),
            "mechanical_work_j": _float(robot_res.get("mechanical_work_j")),
            "thermal_energy_j": _float(robot_res.get("thermal_energy_j")),
            "electrical_energy_wh": _float(robot_res.get("electrical_energy_wh")),
            "acoustic_emission_pa2s": _float(robot_res.get("acoustic_emission_pa2s", metrics.get("total_acoustic_dose_pa2s"))),
            "drivetrain_scrub_j": _float(robot_res.get("drivetrain_scrub_j")),
        }
        row.update(_declared_fact_values(robot_res, FactTable.ROBOTS))
        base_reason = _row_null_reason(row, _ROBOT_NULLABLE, outcome.provenance)
        merged = _merge_provenance_reasons(base_reason, outcome.provenance, row)
        row["null_reason"] = _merge_provenance_reasons(merged, episode_provenance, row)
        return _frame([row], ROBOT_FACT_SCHEMA)

    def _edge_case_fact_frame(
        self,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per case step and per scored, missing or unplayed case agent."""
        parts: list[pl.DataFrame] = []
        for kind, key in _EDGE_CASE_LIST_FIELDS:
            elements = _sequence(metrics.get(key))
            if not elements:
                continue
            part = pl.DataFrame(
                {
                    "kind": [kind] * len(elements),
                    "ordinal": pl.int_range(0, len(elements), eager=True),
                    key: pl.Series(key, elements, dtype=pl.String),
                }
            )
            parts.append(part)
        if not parts:
            return pl.DataFrame(schema=EDGE_CASE_FACT_SCHEMA)
        frame = _with_element_columns(pl.concat(parts, how="diagonal_relaxed"), _EDGE_CASE_ELEMENT_COLUMNS).with_columns(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id, dtype=pl.String),
            case_idx=pl.lit(0, dtype=pl.Int64),
            null_reason=_null_reason_expression(_EDGE_CASE_NULLABLE),
        )
        return frame.select(list(EDGE_CASE_FACT_SCHEMA.names())).cast(EDGE_CASE_FACT_SCHEMA)

    def _formation_fact_frame(
        self,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per formation member and per formation agent that was never recorded."""
        parts: list[pl.DataFrame] = []
        for role, key in _FORMATION_LIST_FIELDS:
            elements = _sequence(metrics.get(key))
            if not elements:
                continue
            parts.append(
                pl.DataFrame(
                    {
                        "role": [role] * len(elements),
                        key: pl.Series(key, elements, dtype=pl.String),
                    }
                )
            )
        if not parts:
            return pl.DataFrame(schema=FORMATION_FACT_SCHEMA)
        frame = _with_element_columns(pl.concat(parts, how="diagonal_relaxed"), _FORMATION_ELEMENT_COLUMNS).with_columns(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id, dtype=pl.String),
            formation_idx=pl.lit(0, dtype=pl.Int64),
            null_reason=_null_reason_expression(_FORMATION_NULLABLE),
        )
        return frame.select(list(FORMATION_FACT_SCHEMA.names())).cast(FORMATION_FACT_SCHEMA)

    def _phase_outcome_fact_frame(
        self,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per replayed phase with its outcome token and met flag."""
        outcomes = _sequence(metrics.get("phase_outcomes"))
        if not outcomes:
            return pl.DataFrame(schema=PHASE_OUTCOME_FACT_SCHEMA)
        frame = (
            pl.DataFrame(
                {
                    "phase_idx": pl.int_range(0, len(outcomes), eager=True),
                    "phase_outcomes": pl.Series("phase_outcomes", outcomes, dtype=pl.String),
                }
            )
            .with_columns(
                run_id=pl.lit(run_id, dtype=pl.String),
                episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
                robot_id=pl.lit(robot_id, dtype=pl.String),
                met=pl.col("phase_outcomes") == pl.lit("met"),
            )
            .with_columns(null_reason=_null_reason_expression(_PHASE_OUTCOME_NULLABLE))
        )
        return frame.select(list(PHASE_OUTCOME_FACT_SCHEMA.names())).cast(PHASE_OUTCOME_FACT_SCHEMA)

    def _judge_disagreement_fact_frame(
        self,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per judge disagreement with its text."""
        disagreements = _sequence(metrics.get("judge_disagreements"))
        if not disagreements:
            return pl.DataFrame(schema=JUDGE_DISAGREEMENT_FACT_SCHEMA)
        frame = (
            pl.DataFrame(
                {
                    "disagreement_idx": pl.int_range(0, len(disagreements), eager=True),
                    "judge_disagreements": pl.Series("judge_disagreements", disagreements, dtype=pl.String),
                }
            )
            .with_columns(
                run_id=pl.lit(run_id, dtype=pl.String),
                episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
                robot_id=pl.lit(robot_id, dtype=pl.String),
            )
            .with_columns(null_reason=_null_reason_expression(_JUDGE_DISAGREEMENT_NULLABLE))
        )
        return frame.select(list(JUDGE_DISAGREEMENT_FACT_SCHEMA.names())).cast(JUDGE_DISAGREEMENT_FACT_SCHEMA)

    def _mechanism_state_fact_frame(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """Change-point rows of every door and elevator state series, one flag per mechanism."""
        states = _sequence(metrics.get("mechanism_states"))
        if not states:
            return pl.DataFrame(schema=MECHANISM_STATE_FACT_SCHEMA)
        times = _event_time_ns(episode)
        frames: list[pl.DataFrame] = []
        for state in states:
            if not isinstance(state, Mapping):
                continue
            progress = _sequence(state.get("progress"))
            locked = _sequence(state.get("locked"))
            if not progress:
                continue
            count = len(progress)
            time_column = pl.Series("t_ns", times[:count], dtype=pl.Int64) if len(times) >= count else pl.Series("t_ns", [None] * count, dtype=pl.Int64)
            series = pl.DataFrame(
                {
                    "t_ns": time_column,
                    "progress": pl.Series("progress", progress).cast(pl.Float64),
                    "locked": pl.Series("locked", locked[:count] if locked else [None] * count).cast(pl.Boolean),
                }
            ).with_columns(pl.col("progress").fill_nan(None))
            changed = series.filter((pl.int_range(0, pl.len()) == 0) | pl.col("progress").ne_missing(pl.col("progress").shift(1)) | pl.col("locked").ne_missing(pl.col("locked").shift(1)))
            rows = pl.concat(
                [
                    changed.select(
                        "t_ns",
                        pl.lit(field).alias("field"),
                        pl.col(field).cast(pl.Float64).alias("value"),
                    )
                    for field in _MECHANISM_FIELDS
                ],
                how="vertical",
            ).sort("t_ns", "field", nulls_last=True)
            frame = rows.with_columns(
                entity=pl.lit(str(state.get("entity") or ""), dtype=pl.String),
                level=pl.lit(str(state.get("level") or ""), dtype=pl.String),
                kind=pl.lit(str(state.get("kind") or ""), dtype=pl.String),
                mechanism_states=pl.int_range(0, pl.len()) == 0,
            )
            frames.append(frame)
        if not frames:
            return pl.DataFrame(schema=MECHANISM_STATE_FACT_SCHEMA)
        frame = (
            pl.concat(frames, how="vertical")
            .with_columns(
                run_id=pl.lit(run_id, dtype=pl.String),
                episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
                robot_id=pl.lit(robot_id, dtype=pl.String),
            )
            .with_columns(null_reason=_null_reason_expression(_MECHANISM_STATE_NULLABLE))
        )
        return frame.select(list(MECHANISM_STATE_FACT_SCHEMA.names())).cast(MECHANISM_STATE_FACT_SCHEMA)

    def _collision_fact_frame(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per obstacle the episode hit, with the first contact's kind, time and point."""
        obstacles = _sequence(metrics.get("collision_obstacles"))
        if not obstacles:
            return pl.DataFrame(schema=COLLISION_FACT_SCHEMA)
        identifiers = pl.DataFrame({"collision_obstacles": pl.Series("collision_obstacles", obstacles, dtype=pl.String)}).with_row_index("collision_idx")
        contacts = _table(episode, COLLISION_CONTACTS_TOPIC)
        if contacts is None or not {"obstacle_id", "t_ns"} <= set(contacts.columns):
            firsts = identifiers.with_columns(
                obstacle_kind=pl.lit(None, dtype=pl.String),
                t_ns=pl.lit(None, dtype=pl.Int64),
                x_m=pl.lit(None, dtype=pl.Float64),
                y_m=pl.lit(None, dtype=pl.Float64),
            )
        else:
            firsts = identifiers.join(
                contacts.filter(pl.col("obstacle_id").is_not_null() & (pl.col("obstacle_id") != pl.lit("")))
                .sort("t_ns")
                .group_by("obstacle_id")
                .agg(
                    pl.col("kind").first().alias("obstacle_kind") if "kind" in contacts.columns else pl.lit(None, dtype=pl.String).alias("obstacle_kind"),
                    pl.col("t_ns").first().cast(pl.Int64).alias("t_ns"),
                    pl.col("obstacle_x").first().cast(pl.Float64).alias("x_m") if "obstacle_x" in contacts.columns else pl.lit(None, dtype=pl.Float64).alias("x_m"),
                    pl.col("obstacle_y").first().cast(pl.Float64).alias("y_m") if "obstacle_y" in contacts.columns else pl.lit(None, dtype=pl.Float64).alias("y_m"),
                ),
                left_on="collision_obstacles",
                right_on="obstacle_id",
                how="left",
            ).sort("collision_idx")
        frame = firsts.with_columns(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id, dtype=pl.String),
            collisions=pl.lit(1, dtype=pl.Int64),
        ).with_columns(null_reason=_null_reason_expression(_COLLISION_NULLABLE))
        return frame.select(list(COLLISION_FACT_SCHEMA.names())).cast(COLLISION_FACT_SCHEMA)

    @staticmethod
    def _encounter_id(episode_id: int, robot_token: str, encounter_idx: int) -> str:
        return f"enc-{episode_id:03d}-{robot_token}-{encounter_idx:03d}"

    @staticmethod
    def _contact_id(episode_id: int, robot_token: str, contact_idx: int) -> str:
        return f"con-{episode_id:03d}-{robot_token}-{contact_idx:03d}"
