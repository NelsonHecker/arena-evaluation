from __future__ import annotations

import dataclasses
import pathlib
import typing
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final, Literal

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

import polars as pl

# Topic and parameter constants
ROBOT_TRACKS_TOPIC: Final[str] = "robot_tracks"
PEDESTRIAN_TRACKS_TOPIC: Final[str] = "pedestrian_tracks"
ROBOT_VISIBILITY_PARAM: Final[str] = "humansim.robots_visible"

CrowdCondition = Literal["visible", "blind"]


class NullReason(StrEnum):
    """The closed ``null_reason`` vocabulary of the fact tables (01 Data Interfaces section 5).

    A metric cell is NULL only with a reason drawn from this vocabulary, written
    ``<column>:<code>`` (``<column>:<code>:<parameter>`` for the three parameterised codes in
    :data:`NULL_REASON_PARAMETERISED`). A missing measurement is never a zero.
    """

    TOPIC_MISSING = "topic_missing"
    PEDSIM_UNAVAILABLE = "pedsim_unavailable"
    NO_SAMPLES = "no_samples"
    REFERENCE_MISSING = "reference_missing"
    NO_CONFLICT = "no_conflict"
    ZERO_DENOMINATOR = "zero_denominator"
    NOT_RECORDED = "not_recorded"
    NOT_APPLICABLE = "not_applicable"
    INVALID_EPISODE = "invalid_episode"
    EXCEPTION = "exception"


#: Codes that carry a parameter after the code itself.
NULL_REASON_PARAMETERISED: Final[frozenset[str]] = frozenset({NullReason.TOPIC_MISSING.value, NullReason.INVALID_EPISODE.value, NullReason.EXCEPTION.value})

#: One-line meaning of every vocabulary member, the documentation the schema tool serves.
NULL_REASON_MEANINGS: Final[Mapping[str, str]] = {
    f"{NullReason.TOPIC_MISSING.value}:<topic>": "the recording carries no <topic> stream, so the value was never measured",
    NullReason.PEDSIM_UNAVAILABLE.value: "the scenario spawned no dynamic pedestrians",
    NullReason.NO_SAMPLES.value: "the stream was recorded but holds no sample inside the window",
    NullReason.REFERENCE_MISSING.value: "no robot-free twin episode matched this stage and seed",
    NullReason.NO_CONFLICT.value: "clearance stayed above the encounter distance, so nothing happened",
    NullReason.ZERO_DENOMINATOR.value: "the ratio's denominator is zero inside this window",
    NullReason.NOT_RECORDED.value: "the recording does not carry this parameter or value",
    NullReason.NOT_APPLICABLE.value: "the quantity has no meaning for this row's kind",
    f"{NullReason.INVALID_EPISODE.value}:<why>": "the episode is not rankable; <why> names the defect",
    f"{NullReason.EXCEPTION.value}:<Class>": "the calculator raised <Class>",
}

#: ``<why>`` values of ``invalid_episode:<why>``, the defects that make an episode unrankable.
INVALID_EPISODE_REASONS: Final[Mapping[str, str]] = {
    "pose_teleport": "the ground-truth pose jumps faster than any robot can move, so the trajectory is not physical",
    "incomplete_recording": "the recording has no terminal outcome or reports lost sections",
    "simulator_error": "the simulator ended the episode with FATAL",
    "missing_stream:<topic>": "a stream the metric needs was not recorded",
    "reference_exceeds_path": "the reference geodesic is longer than the travelled path, so path efficiency is out of range",
}

#: Producer tokens the pipeline may still emit and the vocabulary code each one becomes.
LEGACY_NULL_REASON_CODES: Final[Mapping[str, str]] = {
    "series_missing": NullReason.NOT_RECORDED.value,
    "reference_exceeds_path": f"{NullReason.INVALID_EPISODE.value}:reference_exceeds_path",
}


def split_null_reason_item(item: str) -> tuple[str, str, str]:
    """``(column, code, parameter)`` of one ``;``-separated provenance item; ``column`` is empty for a bare code."""
    first, separator, rest = item.partition(":")
    if not separator:
        return "", item, ""
    if first in NULL_REASON_PARAMETERISED:
        return "", first, rest
    code, _, parameter = rest.partition(":")
    return first, code, parameter


def is_known_null_reason(text: str) -> bool:
    """Whether every item of a published ``null_reason`` is a vocabulary code with its parameter."""
    return all(_is_known_item(item) for item in text.split(";") if item)


def _is_known_item(item: str) -> bool:
    """One provenance item is a known code, and a parameterised one carries its parameter."""
    _, code, parameter = split_null_reason_item(item)
    if code in NULL_REASON_PARAMETERISED:
        return bool(parameter)
    return code in NULL_REASON_MEANINGS


def canonical_null_reason_item(item: str) -> str:
    """One provenance item with a legacy producer token replaced by its vocabulary code."""
    column, code, parameter = split_null_reason_item(item)
    replacement = LEGACY_NULL_REASON_CODES.get(code)
    if replacement is None:
        return item
    prefix = f"{column}:" if column else ""
    return f"{prefix}{replacement}:{parameter}" if parameter else f"{prefix}{replacement}"


def canonical_null_reason(text: str) -> str:
    """``;``-joined provenance with every legacy producer token replaced by its vocabulary code."""
    return ";".join(canonical_null_reason_item(item) for item in text.split(";") if item)


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


class FactTable(StrEnum):
    """The star-schema fact tables the scoped evaluator publishes.

    The five scoped tables carry one row per evaluation scope entity or window; the six
    child tables reshape the identity-grain list outputs into 1NF rows.
    """

    EPISODES = "episodes"
    ENCOUNTERS = "encounters"
    CONTACTS = "contacts"
    PEDESTRIANS = "pedestrians"
    ROBOTS = "robots"
    EDGE_CASES = "edge_cases"
    FORMATIONS = "formations"
    PHASE_OUTCOMES = "phase_outcomes"
    JUDGE_DISAGREEMENTS = "judge_disagreements"
    MECHANISM_STATES = "mechanism_states"
    COLLISIONS = "collisions"


SCOPED_FACT_TABLES: Final[tuple[FactTable, ...]] = (
    FactTable.EPISODES,
    FactTable.ENCOUNTERS,
    FactTable.CONTACTS,
    FactTable.PEDESTRIANS,
    FactTable.ROBOTS,
)
CHILD_FACT_TABLES: Final[tuple[FactTable, ...]] = (
    FactTable.EDGE_CASES,
    FactTable.FORMATIONS,
    FactTable.PHASE_OUTCOMES,
    FactTable.JUDGE_DISAGREEMENTS,
    FactTable.MECHANISM_STATES,
    FactTable.COLLISIONS,
)


@dataclasses.dataclass(frozen=True, slots=True)
class MetricDeclaration:
    """Static schema declaration of one fact column published by the evaluator."""

    name: str
    unit: str = ""
    fact_table: FactTable = FactTable.EPISODES
    dtype: pl.DataType = pl.Float64
    description: str = ""
    direction: str = ""  # "higher", "lower", or ""


class CrowdVisibility(typing.NamedTuple):
    """Whether the crowd saw the robot, the resulting condition, and where the answer came from."""

    visible: bool | None
    condition: str | None
    source: str | None


_CROWD_LAUNCH_KEY = "human.robots.visible"


def visible_flag(value: object) -> bool | None:
    """A robots_visible value as a bool, None for anything that is not one."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def launch_robots_visible(manifest: Mapping[str, Any] | None) -> bool | None:
    """robots_visible from the run's cli_args, else from the suite launch block."""
    if not manifest:
        return None
    for arg in manifest.get("cli_args") or []:
        if not isinstance(arg, str) or _CROWD_LAUNCH_KEY not in arg:
            continue
        _, _, raw = arg.partition(":=")
        value = visible_flag(raw) if raw else None
        if value is not None:
            return value
    launch = (manifest.get("suite") or {}).get("launch") or {}
    return visible_flag(launch.get(_CROWD_LAUNCH_KEY)) if isinstance(launch, dict) else None


def stages_set_robots_visible(manifest: Mapping[str, Any] | None) -> bool:
    """Whether any stage of the suite sets robots_visible, so a record-less episode has no single value."""
    for stage in ((manifest or {}).get("suite") or {}).get("stages") or []:
        if not isinstance(stage, dict):
            continue
        humansim = (stage.get("config") or {}).get("humansim") or {}
        if "robots_visible" in humansim:
            return True
    return False


def resolve_crowd_visibility(
    accepted: bool | None,
    manifest: Mapping[str, Any] | None,
    metrics_config: Mapping[str, Any] | None = None,
    *,
    record_present: bool = True,
) -> CrowdVisibility:
    """Crowd visibility of one episode: metrics config, accepted parameter, run manifest, then the default."""
    config = metrics_config or {}
    configured = config.get("crowd_condition")
    if configured in ("blind", "visible"):
        return CrowdVisibility(configured == "visible", configured, "metrics_config")
    for key in ("crowd_visible", "robots_visible"):
        configured_flag = visible_flag(config.get(key))
        if configured_flag is not None:
            return CrowdVisibility(configured_flag, "visible" if configured_flag else "blind", "metrics_config")
    if accepted is not None:
        return CrowdVisibility(accepted, "visible" if accepted else "blind", "episode_param")
    if manifest is None:
        return CrowdVisibility(None, None, None)
    launched = launch_robots_visible(manifest)
    if launched is not None:
        return CrowdVisibility(launched, "visible" if launched else "blind", "launch_arg")
    if not record_present and stages_set_robots_visible(manifest):
        return CrowdVisibility(None, None, None)
    return CrowdVisibility(True, "visible", "default")


def recorded_robot_visibility(params: Mapping[str, Any] | None) -> bool | None:
    """Recorded ``humansim.robots_visible`` parameter, None when the recording carries none."""
    if params is None:
        return None
    value = params.get(ROBOT_VISIBILITY_PARAM)
    if value is not None:
        return bool(value)
    leaf = ROBOT_VISIBILITY_PARAM.rsplit(".", maxsplit=1)[-1]
    for name, parameter in params.items():
        if name.rsplit(".", maxsplit=1)[-1] == leaf:
            return bool(parameter)
    return None


def resolve_crowd_condition(
    episode: AlignedEpisodeBundle,
    metrics_config: Mapping[str, Any] | None = None,
) -> CrowdCondition | None:
    """Crowd visibility of the episode: metrics config, recorded parameter, run metadata, else None when unknown."""
    visibility = resolve_crowd_visibility(recorded_robot_visibility(episode.human_params), None, metrics_config)
    if visibility.visible is not None:
        return "visible" if visibility.visible else "blind"
    run = episode.run
    if run is not None and run.crowd_condition is not None:
        return typing.cast(CrowdCondition, run.crowd_condition)
    return None


_EPISODES: Final[FactTable] = FactTable.EPISODES
_ENCOUNTERS: Final[FactTable] = FactTable.ENCOUNTERS
_CONTACTS: Final[FactTable] = FactTable.CONTACTS
_PEDESTRIANS: Final[FactTable] = FactTable.PEDESTRIANS
_ROBOTS: Final[FactTable] = FactTable.ROBOTS
_EDGE_CASES: Final[FactTable] = FactTable.EDGE_CASES
_FORMATIONS: Final[FactTable] = FactTable.FORMATIONS
_PHASE_OUTCOMES: Final[FactTable] = FactTable.PHASE_OUTCOMES
_JUDGE_DISAGREEMENTS: Final[FactTable] = FactTable.JUDGE_DISAGREEMENTS
_MECHANISM_STATES: Final[FactTable] = FactTable.MECHANISM_STATES
_COLLISIONS: Final[FactTable] = FactTable.COLLISIONS
_EPISODE_TABLE: Final[str] = FactTable.EPISODES.value
_ENCOUNTER_TABLE: Final[str] = FactTable.ENCOUNTERS.value
_CONTACT_TABLE: Final[str] = FactTable.CONTACTS.value
_PEDESTRIAN_TABLE: Final[str] = FactTable.PEDESTRIANS.value
_ROBOT_TABLE: Final[str] = FactTable.ROBOTS.value

# Every metric column the scoped evaluator publishes beyond the pre-port schema.
# This tuple is the single source of truth for the fact schema: FACT_SCHEMAS derives
# each table's metric columns and dtypes from it, and the scoped evaluator copies
# each declared column from the calculator result of the same name.
METRIC_DECLARATIONS: Final[tuple[MetricDeclaration, ...]] = (
    # --- EPISODE scope -> fact_episodes --------------------------------------------------
    MetricDeclaration(name="barn_collided", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_optimal_time", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_score", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_success", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="higher", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_time", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_timeout", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="mean_obstacle_clearance", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="performance metric of the clearance_metrics calculator"),
    MetricDeclaration(name="mean_pedestrian_clearance", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="performance metric of the clearance_metrics calculator"),
    MetricDeclaration(name="doorway_blocking_time", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="passing_rule_compliance", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="quiet_zone_dwell_seconds", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="restricted_zone_entries", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="speed_zone_violation_seconds", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="speed_zone_violations", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="clauses_failed", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="clauses_passed", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="clauses_total", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="clauses_unknown", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="condition_success", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="edge_case_collided", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="lower", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_at", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_fault", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_ped_closing", unit="m/s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_robot_closing", unit="m/s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_with", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_freeze_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_id", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_intrusion_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_clearance_at", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_clearance_m", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_clearance_with", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_ttc_at", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_ttc_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_ttc_with", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_pet_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_pet_with", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_played", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_scope", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_triggered_at", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_formation_arrived", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="higher", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_break_fired_at", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_broke_at", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_form_fired_at", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_formed_at", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_held_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_seen", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_variant", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="aeps_linear_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="holistic metric of the holistic_metrics calculator"),
    MetricDeclaration(name="e_cot", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="holistic metric of the holistic_metrics calculator"),
    MetricDeclaration(name="mechanism_components", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the mechanism_component calculator"),
    MetricDeclaration(name="mechanism_score", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="ecological metric of the mechanism_component calculator"),
    MetricDeclaration(name="alarm_egress_blocking_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="blocked_closing_door", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="blocked_closing_door_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="boarded_before_exit", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="cabin_door_blocking_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="door_freeze_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="door_oscillations", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="door_yield_violations", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="elevator_missed_cycles", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="elevator_queue_jumps", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="elevator_unnecessary_calls", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="elevator_wait_to_board_s", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="entered_full_cabin", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="ped_mechanism_delay_seconds", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="kcsc_wh", unit="Wh", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="social metric of the mutual_accommodation calculator"),
    MetricDeclaration(name="relative_throughput", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="social metric of the mutual_accommodation calculator"),
    MetricDeclaration(name="angle_over_length", unit="rad/m", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="curvature_max", unit="1/m", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="curvature_mean", unit="1/m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="roughness_max", unit="rad/m", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="roughness_mean", unit="rad/m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="entered_over_cap_zone", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="ran_red_signal", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="replan_latency_after_state_change_median", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="replan_latency_after_state_change_p95", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="used_elevator_during_alarm", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="used_elevator_during_alarm_robot", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="elevator_rides", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the semantic_interaction_metrics calculator"),
    MetricDeclaration(name="elevator_rides_robot", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="ecological metric of the semantic_interaction_metrics calculator"),
    MetricDeclaration(name="time_waiting_at_doors", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="ecological metric of the semantic_interaction_metrics calculator"),
    MetricDeclaration(name="time_waiting_at_doors_robot", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="ecological metric of the semantic_interaction_metrics calculator"),
    MetricDeclaration(name="goal_condition_rate", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="judge_agrees", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="higher", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="phases_dropped", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="phases_failed", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="phases_met", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="phases_total", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="unsafe", unit="", fact_table=_EPISODES, dtype=pl.Int64, direction="lower", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="idling_time", unit="s", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="performance metric of the time_metrics calculator"),
    MetricDeclaration(name="time_diff", unit="ns", fact_table=_EPISODES, dtype=pl.Int64, direction="", description="performance metric of the time_metrics calculator"),
    MetricDeclaration(name="ade", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="single-leg tracking: average displacement error against the current-leg Theta* reference path"),
    MetricDeclaration(name="adtw", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="fde", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="single-leg tracking: final displacement error against the current-leg Theta* reference path"),
    MetricDeclaration(name="frechet_distance", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="single-leg tracking: discrete Frechet distance against the current-leg Theta* reference path"),
    MetricDeclaration(name="mhd", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="single-leg tracking: modified Hausdorff distance against the current-leg Theta* reference path"),
    MetricDeclaration(name="path_irregularity", unit="rad/m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="theta_star_length", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="", description="whole-mission route geodesic: sum of per-leg Theta* between the task goals the robot reached; feeds path_efficiency"),
    MetricDeclaration(name="topological_complexity", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="goal_inputs", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="ndtw", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="ne", unit="m", fact_table=_EPISODES, dtype=pl.Float64, direction="lower", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="osr", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="sdtw", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="spl_geodesic", unit="", fact_table=_EPISODES, dtype=pl.Float64, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="stuck", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="lower", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="success_geodesic", unit="", fact_table=_EPISODES, dtype=pl.Boolean, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="pose_frame", unit="", fact_table=_EPISODES, dtype=pl.String, direction="", description="TF parent frame the ground-truth pose came from inside the episode window (map, world, an odom frame or the simulator frame); NULL when the recording carries no transform"),
    # --- ROBOT scope -> fact_robots ----------------------------------------------------
    MetricDeclaration(name="acoustic_cost", unit="s/m", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="acoustic_startle_factor", unit="dBA/s", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="acoustic_surge_index", unit="dBA²", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="l_a10", unit="dBA", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="l_a50", unit="dBA", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="l_a90", unit="dBA", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="l_aeq_t", unit="dBA", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="char_phase_coverage", unit="", fact_table=_ROBOTS, dtype=pl.Float64, direction="", description="ecological metric of the characterization calculator"),
    MetricDeclaration(name="battery_soc_drop_pct", unit="%", fact_table=_ROBOTS, dtype=pl.Float64, direction="", description="ecological metric of the energy calculator"),
    MetricDeclaration(name="battery_soc_final", unit="%", fact_table=_ROBOTS, dtype=pl.Float64, direction="", description="ecological metric of the energy calculator"),
    MetricDeclaration(name="energy_static_wh", unit="Wh", fact_table=_ROBOTS, dtype=pl.Float64, direction="", description="ecological metric of the energy calculator"),
    MetricDeclaration(name="friction_dissipation_j", unit="J", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="kinetic_energy_demand_j", unit="J", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="peak_to_mean_power_ratio", unit="", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="specific_cost_of_transport", unit="", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="standstill_energy_penalty_wh", unit="Wh", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="acceleration_mean", unit="m/s²", fact_table=_ROBOTS, dtype=pl.Float64, direction="", description="smoothed ground-truth acceleration magnitude of the motion_metrics calculator: Savitzky-Golay derivatives of the recorded speed, 1.0 s window, order 2, at most 20 Hz"),
    MetricDeclaration(
        name="jerk_mean", unit="m/s³", fact_table=_ROBOTS, dtype=pl.Float64, direction="lower", description="smoothed ground-truth jerk magnitude of the motion_metrics calculator: Savitzky-Golay derivatives of the recorded speed, 1.0 s window, order 2, at most 20 Hz (the episode jerk_max_mps3 uses the same estimator)"
    ),
    # --- PEDESTRIAN scope -> fact_pedestrians -----------------------------------------------
    MetricDeclaration(name="ped_round_trips_completed", unit="", fact_table=_PEDESTRIANS, dtype=pl.Float64, direction="", description="social metric of the pedestrian_disturbance calculator"),
    MetricDeclaration(name="ped_velocity_delay_ratio", unit="", fact_table=_PEDESTRIANS, dtype=pl.Float64, direction="", description="social metric of the pedestrian_disturbance calculator"),
    MetricDeclaration(name="ped_max_exposure_dba", unit="dBA", fact_table=_PEDESTRIANS, dtype=pl.Float64, direction="lower", description="acoustic_exposure PEDESTRIAN: each pedestrian's own loudest exposure; the episode value is the max over pedestrians (exact reduction)"),
    MetricDeclaration(
        name="ped_leq_exposure_dba",
        unit="dBA",
        fact_table=_PEDESTRIANS,
        dtype=pl.Float64,
        direction="lower",
        description="acoustic_exposure PEDESTRIAN: each pedestrian's own energy-equivalent Leq; the legacy episode value is the energy mean over all pedestrian-samples, so the per-pedestrian mean is a reviewed energy-weighted reduction within 1.3 dB (expected-diff max_rel 0.03); the exact episode-grain quantity is l_aeq_t (fact_robots) / total_acoustic_dose_pa2s (fact_episodes)",
    ),
    MetricDeclaration(
        name="ped_max_startle_factor",
        unit="dBA/s",
        fact_table=_PEDESTRIANS,
        dtype=pl.Float64,
        direction="lower",
        description="acoustic_exposure PEDESTRIAN: each pedestrian's own max rise rate; the episode value is the max over pedestrians, reviewed within 4.69 dBA/s (expected-diff max_rel 0.33) because the legacy episode calculator pairs consecutive frames positionally rather than by pedestrian identity",
    ),
    MetricDeclaration(
        name="worst_case_acoustic_t_ns",
        unit="ns",
        fact_table=_PEDESTRIANS,
        dtype=pl.Int64,
        direction="",
        description="acoustic_exposure PEDESTRIAN: t_ns of this pedestrian's loudest exposure frame; the frame is reconstructed by joining series_acoustic_exposure on (run_id, episode_id, t_ns), where is_worst_frame marks the row and robot_x_m/robot_y_m/source_dba/ped_x_m/ped_y_m/open_doors carry it; the nested worst_case_acoustic_frame struct is not a fact column",
    ),
    # --- ENCOUNTER scope -> fact_encounters -------------------------------------------------
    MetricDeclaration(name="sfm_cumulative_force", unit="N·s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="sfm_peak_force", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="sfm_mean_force", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_cumulative_force", unit="N·s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_peak_force", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_mean_force", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_ped_cumulative_force", unit="N·s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_ped_peak_force", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_ped_mean_force", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="social_force_max", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="social_force_mean", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="social_force_ped_max", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="social_force_ped_mean", unit="N", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="ci_max", unit="", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="ci_mean", unit="", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="time_in_intimate_zone", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="time_in_personal_zone", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="time_in_social_zone", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="time_in_public_zone", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="psi_intimate_events", unit="", fact_table=_ENCOUNTERS, dtype=pl.Int64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="psi_personal_events", unit="", fact_table=_ENCOUNTERS, dtype=pl.Int64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="psi_social_events", unit="", fact_table=_ENCOUNTERS, dtype=pl.Int64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="max_speed_intimate_zone", unit="m/s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="max_speed_personal_zone", unit="m/s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="max_speed_social_zone", unit="m/s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="max_speed_public_zone", unit="m/s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="movement_towards_peds_ratio", unit="", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="tti_min", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="higher", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="tti_mean", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="personal_space_intrusion_integral", unit="m·s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="total_time_looking_at_pedestrians", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the gaze_metrics calculator"),
    MetricDeclaration(name="total_time_looked_at_by_pedestrians", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the gaze_metrics calculator"),
    MetricDeclaration(name="num_pedestrians", unit="", fact_table=_ENCOUNTERS, dtype=pl.Int64, direction="", description="social metric of the proxemics calculator"),
    MetricDeclaration(name="total_time_in_personal_space", unit="s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="social metric of the proxemics calculator"),
    MetricDeclaration(name="avg_velocity_in_personal_space", unit="m/s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="social metric of the proxemics calculator"),
    # --- ENCOUNTER scope -> fact_encounters: windowed acoustics and energy -------------------
    # Extensives are additive over encounter windows (sum); extremes reduce by max; levels and
    # ratios reduce by mean over windows, which is a reviewed approximation for dB and ratio
    # means because those quantities are not additive (the parity expected diff carries the bound).
    MetricDeclaration(name="l_aeq_t", unit="dBA", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="acoustics ENCOUNTER: window energy-equivalent level; the episode value reduces as the mean of window levels (reviewed approximation, dB is not additive)"),
    MetricDeclaration(name="l_a10", unit="dBA", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="acoustics ENCOUNTER: window A-weighted level exceeded 10 % of the window; the episode value reduces as the mean of window levels (reviewed approximation)"),
    MetricDeclaration(name="l_a50", unit="dBA", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="acoustics ENCOUNTER: window median A-weighted level; the episode value reduces as the mean of window levels (reviewed approximation)"),
    MetricDeclaration(name="l_a90", unit="dBA", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="acoustics ENCOUNTER: window A-weighted level exceeded 90 % of the window; the episode value reduces as the mean of window levels (reviewed approximation)"),
    MetricDeclaration(name="acoustic_startle_factor", unit="dBA/s", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="acoustics ENCOUNTER: window max gated rise rate; the episode value is the max over windows"),
    MetricDeclaration(name="acoustic_surge_index", unit="dBA²", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="acoustics ENCOUNTER: window variance of 1 s Leq levels; the episode value reduces as the mean over windows (reviewed approximation)"),
    MetricDeclaration(name="acoustic_cost", unit="s/m", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="acoustics ENCOUNTER: window acoustic energy per window path metre; the episode value reduces as the mean over windows (reviewed approximation)"),
    MetricDeclaration(name="energy_static_wh", unit="Wh", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="energy ENCOUNTER: window static-energy integral; the episode value is the sum over windows"),
    MetricDeclaration(name="energy_mechanical_wh", unit="Wh", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy ENCOUNTER: window mechanical-energy integral; the episode value is the sum over windows"),
    MetricDeclaration(name="energy_thermal_wh", unit="Wh", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy ENCOUNTER: window thermal-energy integral; the episode value is the sum over windows"),
    MetricDeclaration(name="energy_total_wh", unit="Wh", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy ENCOUNTER: window total-energy integral; the episode value is the sum over windows"),
    MetricDeclaration(name="power_peak_w", unit="W", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy ENCOUNTER: window peak total power; the episode value is the max over windows"),
    MetricDeclaration(name="battery_soc_final", unit="%", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="", description="energy ENCOUNTER: window final battery state of charge; the episode value reduces as the mean over windows (reviewed approximation, not additive)"),
    MetricDeclaration(name="battery_soc_drop_pct", unit="%", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy ENCOUNTER: window battery state-of-charge drop; the episode value is the sum over windows"),
    MetricDeclaration(name="specific_cost_of_transport", unit="", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy_extended ENCOUNTER: window cost of transport over the window's own path; the episode value reduces as the mean over windows (reviewed approximation)"),
    MetricDeclaration(name="energy_per_meter", unit="Wh/m", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy_extended ENCOUNTER: window energy per window path metre; the episode value reduces as the mean over windows (reviewed approximation)"),
    MetricDeclaration(name="peak_to_mean_power_ratio", unit="", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy_extended ENCOUNTER: window peak-to-mean power ratio; the episode value reduces as the mean over windows (reviewed approximation)"),
    MetricDeclaration(name="standstill_energy_penalty_wh", unit="Wh", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy_extended ENCOUNTER: window standstill energy; the episode value is the sum over windows"),
    MetricDeclaration(name="kinetic_energy_demand_j", unit="J", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy_extended ENCOUNTER: window kinetic-energy demand; the episode value is the sum over windows"),
    MetricDeclaration(name="friction_dissipation_j", unit="J", fact_table=_ENCOUNTERS, dtype=pl.Float64, direction="lower", description="energy_extended ENCOUNTER: window rolling-friction dissipation; the episode value is the sum over windows"),
    # --- CONTACT scope -> fact_contacts -----------------------------------------------------
    MetricDeclaration(name="hhi_interaction", unit="", fact_table=_CONTACTS, dtype=pl.String, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_hold_onset_s", unit="s", fact_table=_CONTACTS, dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_release_s", unit="s", fact_table=_CONTACTS, dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_episode_s", unit="s", fact_table=_CONTACTS, dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stall_count", unit="", fact_table=_CONTACTS, dtype=pl.Int64, direction="lower", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stall_time_s", unit="s", fact_table=_CONTACTS, dtype=pl.Float64, direction="lower", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stalls_approach", unit="", fact_table=_CONTACTS, dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stalls_hold", unit="", fact_table=_CONTACTS, dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stalls_post", unit="", fact_table=_CONTACTS, dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stalls_rest", unit="", fact_table=_CONTACTS, dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_share_approach", unit="", fact_table=_CONTACTS, dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_share_hold", unit="", fact_table=_CONTACTS, dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_share_post", unit="", fact_table=_CONTACTS, dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_share_rest", unit="", fact_table=_CONTACTS, dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_passes_between", unit="", fact_table=_CONTACTS, dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_min_pair_clearance", unit="m", fact_table=_CONTACTS, dtype=pl.Float64, direction="higher", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_pair_gap_hold", unit="m", fact_table=_CONTACTS, dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="collision_amount", unit="", fact_table=_CONTACTS, dtype=pl.Int64, direction="lower", description="performance metric of the collision_metrics calculator"),
    MetricDeclaration(name="collision_amount_wall", unit="", fact_table=_CONTACTS, dtype=pl.Int64, direction="", description="performance metric of the collision_metrics calculator"),
    MetricDeclaration(name="spl", unit="", fact_table=_CONTACTS, dtype=pl.Float64, direction="higher", description="mission-route success weighted by path length: the reference length is the whole-mission route geodesic, not the final leg"),
    # --- EDGE CASES -> fact_edge_cases (one row per case step / agent) ----------------------
    MetricDeclaration(name="edge_case_steps", unit="", fact_table=_EDGE_CASES, dtype=pl.String, direction="", description="one row per step the case provenance lists; the step name"),
    MetricDeclaration(name="edge_case_agents", unit="", fact_table=_EDGE_CASES, dtype=pl.String, direction="", description="one row per scored case agent; the agent name"),
    MetricDeclaration(name="edge_case_missing", unit="", fact_table=_EDGE_CASES, dtype=pl.String, direction="", description="one row per case agent absent from the recording; the agent name"),
    MetricDeclaration(name="edge_case_unplayed", unit="", fact_table=_EDGE_CASES, dtype=pl.String, direction="", description="one row per encounter agent that never took a stimulus; the agent name"),
    # --- FORMATIONS -> fact_formations (one row per member / absent agent) ------------------
    MetricDeclaration(name="edge_case_formation_members", unit="", fact_table=_FORMATIONS, dtype=pl.String, direction="", description="one row per formation member; the agent name"),
    MetricDeclaration(name="edge_case_formation_absent", unit="", fact_table=_FORMATIONS, dtype=pl.String, direction="", description="one row per formation agent never recorded; the agent name"),
    # --- TASK REPLAY -> fact_phase_outcomes / fact_judge_disagreements ---------------------
    MetricDeclaration(name="phase_outcomes", unit="", fact_table=_PHASE_OUTCOMES, dtype=pl.String, direction="", description="one row per replayed phase; the outcome token"),
    MetricDeclaration(name="judge_disagreements", unit="", fact_table=_JUDGE_DISAGREEMENTS, dtype=pl.String, direction="", description="one row per judge disagreement; the disagreement text"),
    # --- MECHANISM STATES -> fact_mechanism_states (change-point rows) ---------------------
    MetricDeclaration(name="mechanism_states", unit="", fact_table=_MECHANISM_STATES, dtype=pl.Boolean, direction="", description="true on the first change-point row of a mechanism's sampled state series"),
    # --- COLLISIONS -> fact_collisions (one row per hit obstacle) --------------------------
    MetricDeclaration(name="collision_obstacles", unit="", fact_table=_COLLISIONS, dtype=pl.String, direction="", description="one row per obstacle the episode collided with; the obstacle id"),
    MetricDeclaration(name="collisions", unit="", fact_table=_COLLISIONS, dtype=pl.Int64, direction="", description="collision record count contribution, one per fact row"),
)

_BASE_FACT_SCHEMAS: Final[Mapping[str, pl.Schema]] = {
    _EPISODE_TABLE: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "result": pl.String,
            "success": pl.Boolean,
            "time_to_goal_s": pl.Float64,
            "path_length_m": pl.Float64,
            # Whole-mission route geodesic over the travelled path length; see PathEfficiencyCalculator.
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
            # Success conjoined with "no contact of that kind"; NULL while the collision stream is unknown,
            # FALSE as soon as one such contact is recorded, TRUE only when the stream was recorded and empty.
            "success_no_ped_contact": pl.Boolean,
            "success_no_robot_fault": pl.Boolean,
            # Ranking validity of the episode; invalid_reason NULL means valid (08 H1.5).
            "is_valid_for_ranking": pl.Boolean,
            "invalid_reason": pl.String,
            "mar": pl.Float64,
            "pfi": pl.Float64,
            "robot_path_dev_m": pl.Float64,
            "total_acoustic_dose_pa2s": pl.Float64,
            "null_reason": pl.String,
        }
    ),
    _ENCOUNTER_TABLE: pl.Schema(
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
    ),
    _CONTACT_TABLE: pl.Schema(
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
    ),
    _PEDESTRIAN_TABLE: pl.Schema(
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
    ),
    _ROBOT_TABLE: pl.Schema(
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
    ),
    FactTable.EDGE_CASES.value: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "case_idx": pl.Int64,
            "kind": pl.String,
            "ordinal": pl.Int64,
            "null_reason": pl.String,
        }
    ),
    FactTable.FORMATIONS.value: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "formation_idx": pl.Int64,
            "role": pl.String,
            "null_reason": pl.String,
        }
    ),
    FactTable.PHASE_OUTCOMES.value: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "phase_idx": pl.Int64,
            "met": pl.Boolean,
            "null_reason": pl.String,
        }
    ),
    FactTable.JUDGE_DISAGREEMENTS.value: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "disagreement_idx": pl.Int64,
            "null_reason": pl.String,
        }
    ),
    FactTable.MECHANISM_STATES.value: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "entity": pl.String,
            "level": pl.String,
            "kind": pl.String,
            "field": pl.String,
            "t_ns": pl.Int64,
            "value": pl.Float64,
            "null_reason": pl.String,
        }
    ),
    FactTable.COLLISIONS.value: pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "collision_idx": pl.Int64,
            "obstacle_kind": pl.String,
            "t_ns": pl.Int64,
            "x_m": pl.Float64,
            "y_m": pl.Float64,
            "null_reason": pl.String,
        }
    ),
}


def declarations_for(table: FactTable) -> tuple[MetricDeclaration, ...]:
    """Declarations that publish into ``table``, in declaration order."""
    return tuple(declaration for declaration in METRIC_DECLARATIONS if declaration.fact_table is table)


def declared_columns(table: FactTable) -> tuple[str, ...]:
    """Fact columns declared for ``table``, the declaration order of the schema."""
    return tuple(declaration.name for declaration in declarations_for(table))


def _derived_schema(table: FactTable) -> pl.Schema:
    """Base fact schema of ``table`` plus its declared metric columns, provenance string last."""
    base_columns = dict(_BASE_FACT_SCHEMAS[table.value])
    null_reason = base_columns.pop("null_reason")
    for declaration in declarations_for(table):
        if declaration.name in base_columns:
            raise ValueError(f"declaration {declaration.name!r} duplicates a base column of {table.value}")
        base_columns[declaration.name] = declaration.dtype
    base_columns["null_reason"] = null_reason
    return pl.Schema(base_columns)


FACT_TABLE_NAMES: Final[tuple[str, ...]] = tuple(table.value for table in FactTable)
SCOPED_FACT_TABLE_NAMES: Final[tuple[str, ...]] = tuple(table.value for table in SCOPED_FACT_TABLES)
CHILD_FACT_TABLE_NAMES: Final[tuple[str, ...]] = tuple(table.value for table in CHILD_FACT_TABLES)

EPISODE_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.EPISODES)
ENCOUNTER_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.ENCOUNTERS)
CONTACT_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.CONTACTS)
PEDESTRIAN_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.PEDESTRIANS)
ROBOT_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.ROBOTS)
EDGE_CASE_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.EDGE_CASES)
FORMATION_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.FORMATIONS)
PHASE_OUTCOME_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.PHASE_OUTCOMES)
JUDGE_DISAGREEMENT_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.JUDGE_DISAGREEMENTS)
MECHANISM_STATE_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.MECHANISM_STATES)
COLLISION_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(FactTable.COLLISIONS)

FACT_SCHEMAS: Final[dict[str, pl.Schema]] = {table.value: _derived_schema(table) for table in FactTable}

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
    "success_no_ped_contact",
    "success_no_robot_fault",
    "mar",
    "pfi",
    "robot_path_dev_m",
    "total_acoustic_dose_pa2s",
) + declared_columns(FactTable.EPISODES)
# `invalid_reason` is NULL exactly when the episode is valid, so its NULL is a value: it never
# demands a `null_reason`. `is_valid_for_ranking` is computed for every evaluated episode.
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
) + declared_columns(FactTable.ENCOUNTERS)
_CONTACT_NULLABLE: Final[tuple[str, ...]] = (
    "impact_speed_mps",
    "robot_speed_at_impact_mps",
    "obstacle_speed_at_impact_mps",
    "closer_label",
    "fault_attributed_to",
    "robot_closing_share",
) + declared_columns(FactTable.CONTACTS)
_PEDESTRIAN_NULLABLE: Final[tuple[str, ...]] = (
    "ped_policy",
    "is_blind",
    "desired_speed",
    "radius_m",
    "spawn_x",
    "spawn_y",
    "spawn_yaw",
    "mean_speed_mps",
    "cumulative_sound_dose_pa2s",
    "ped_path_deflection_m",
) + declared_columns(FactTable.PEDESTRIANS)
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
) + declared_columns(FactTable.ROBOTS)
# Child-table nullability is structural: every row always carries its own element value,
# so only the derived fields that may genuinely be missing are listed.
_EDGE_CASE_NULLABLE: Final[tuple[str, ...]] = ()
_FORMATION_NULLABLE: Final[tuple[str, ...]] = ()
_PHASE_OUTCOME_NULLABLE: Final[tuple[str, ...]] = ()
_JUDGE_DISAGREEMENT_NULLABLE: Final[tuple[str, ...]] = ()
_MECHANISM_STATE_NULLABLE: Final[tuple[str, ...]] = ("level", "t_ns")
_COLLISION_NULLABLE: Final[tuple[str, ...]] = (
    "obstacle_kind",
    "t_ns",
    "x_m",
    "y_m",
)


_NULLABLE_BY_TABLE: Final[dict[str, tuple[str, ...]]] = {
    FactTable.EPISODES.value: _EPISODE_NULLABLE,
    FactTable.ENCOUNTERS.value: _ENCOUNTER_NULLABLE,
    FactTable.CONTACTS.value: _CONTACT_NULLABLE,
    FactTable.PEDESTRIANS.value: _PEDESTRIAN_NULLABLE,
    FactTable.ROBOTS.value: _ROBOT_NULLABLE,
    FactTable.EDGE_CASES.value: _EDGE_CASE_NULLABLE,
    FactTable.FORMATIONS.value: _FORMATION_NULLABLE,
    FactTable.PHASE_OUTCOMES.value: _PHASE_OUTCOME_NULLABLE,
    FactTable.JUDGE_DISAGREEMENTS.value: _JUDGE_DISAGREEMENT_NULLABLE,
    FactTable.MECHANISM_STATES.value: _MECHANISM_STATE_NULLABLE,
    FactTable.COLLISIONS.value: _COLLISION_NULLABLE,
}


def nullable_columns(table: FactTable) -> tuple[str, ...]:
    """Fact columns that may be NULL, each of which requires a ``null_reason`` when it is."""
    return _NULLABLE_BY_TABLE[table.value]


# Row keys of each fact table. Two child tables have no stable row key: one row per list
# element spread over two element columns (formations) and change-point rows without a time
# axis (mechanism_states); their identity keys are still checked non-null.
FACT_KEY_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    FactTable.EPISODES.value: ("run_id", "episode_id", "robot_id"),
    FactTable.ENCOUNTERS.value: ("run_id", "episode_id", "robot_id", "encounter_id"),
    FactTable.CONTACTS.value: ("run_id", "episode_id", "robot_id", "contact_id"),
    FactTable.PEDESTRIANS.value: ("run_id", "episode_id", "env_id", "ped_id"),
    FactTable.ROBOTS.value: ("run_id", "episode_id", "env_id", "robot_id"),
    FactTable.EDGE_CASES.value: ("run_id", "episode_id", "robot_id", "case_idx", "kind", "ordinal"),
    FactTable.FORMATIONS.value: ("run_id", "episode_id", "robot_id"),
    FactTable.PHASE_OUTCOMES.value: ("run_id", "episode_id", "robot_id", "phase_idx"),
    FactTable.JUDGE_DISAGREEMENTS.value: ("run_id", "episode_id", "robot_id", "disagreement_idx"),
    FactTable.MECHANISM_STATES.value: ("run_id", "episode_id", "robot_id"),
    FactTable.COLLISIONS.value: ("run_id", "episode_id", "robot_id", "collision_idx"),
}

FACT_TABLES_WITHOUT_ROW_KEY: Final[frozenset[str]] = frozenset({FactTable.FORMATIONS.value, FactTable.MECHANISM_STATES.value})


@dataclasses.dataclass(frozen=True, slots=True)
class ScopedFactTables:
    """The eleven typed fact tables of one evaluated episode."""

    episodes: pl.DataFrame
    encounters: pl.DataFrame
    contacts: pl.DataFrame
    pedestrians: pl.DataFrame
    robots: pl.DataFrame
    edge_cases: pl.DataFrame = dataclasses.field(default_factory=lambda: pl.DataFrame(schema=EDGE_CASE_FACT_SCHEMA))
    formations: pl.DataFrame = dataclasses.field(default_factory=lambda: pl.DataFrame(schema=FORMATION_FACT_SCHEMA))
    phase_outcomes: pl.DataFrame = dataclasses.field(default_factory=lambda: pl.DataFrame(schema=PHASE_OUTCOME_FACT_SCHEMA))
    judge_disagreements: pl.DataFrame = dataclasses.field(default_factory=lambda: pl.DataFrame(schema=JUDGE_DISAGREEMENT_FACT_SCHEMA))
    mechanism_states: pl.DataFrame = dataclasses.field(default_factory=lambda: pl.DataFrame(schema=MECHANISM_STATE_FACT_SCHEMA))
    collisions: pl.DataFrame = dataclasses.field(default_factory=lambda: pl.DataFrame(schema=COLLISION_FACT_SCHEMA))

    def by_name(self) -> dict[str, pl.DataFrame]:
        """Fact tables keyed by their star-schema table name."""
        return {
            "episodes": self.episodes,
            "encounters": self.encounters,
            "contacts": self.contacts,
            "pedestrians": self.pedestrians,
            "robots": self.robots,
            "edge_cases": self.edge_cases,
            "formations": self.formations,
            "phase_outcomes": self.phase_outcomes,
            "judge_disagreements": self.judge_disagreements,
            "mechanism_states": self.mechanism_states,
            "collisions": self.collisions,
        }


def write_fact_tables(tables: ScopedFactTables, directory: pathlib.Path) -> None:
    """Write every fact table as ``fact_<name>.parquet`` in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.by_name().items():
        frame.write_parquet(directory / f"fact_{name}.parquet")


def read_fact_tables(directory: pathlib.Path) -> ScopedFactTables | None:
    """Read the fact tables from ``fact_<name>.parquet``; child tables are optional."""
    frames: dict[str, pl.DataFrame] = {}
    for name in SCOPED_FACT_TABLE_NAMES:
        path = directory / f"fact_{name}.parquet"
        if not path.is_file():
            return None
        frames[name] = pl.read_parquet(path)
    for name in CHILD_FACT_TABLE_NAMES:
        path = directory / f"fact_{name}.parquet"
        frames[name] = pl.read_parquet(path) if path.is_file() else pl.DataFrame(schema=FACT_SCHEMAS[name])
    return ScopedFactTables(**frames)
