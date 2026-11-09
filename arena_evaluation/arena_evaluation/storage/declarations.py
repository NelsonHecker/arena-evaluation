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


_EPISODE: Final[MetricScope] = MetricScope.EPISODE
_ENCOUNTER: Final[MetricScope] = MetricScope.ENCOUNTER
_CONTACT: Final[MetricScope] = MetricScope.CONTACT
_ROBOT: Final[MetricScope] = MetricScope.ROBOT
_PEDESTRIAN: Final[MetricScope] = MetricScope.PEDESTRIAN
_EPISODE_TABLE: Final[str] = MetricScope.EPISODE.value
_ENCOUNTER_TABLE: Final[str] = MetricScope.ENCOUNTER.value
_CONTACT_TABLE: Final[str] = MetricScope.CONTACT.value
_ROBOT_TABLE: Final[str] = MetricScope.ROBOT.value
_PEDESTRIAN_TABLE: Final[str] = MetricScope.PEDESTRIAN.value

# Every metric column the scoped evaluator publishes beyond the pre-port schema.
# This tuple is the single source of truth for the fact schema: FACT_SCHEMAS derives
# each table's metric columns and dtypes from it, and the scoped evaluator copies
# each declared column from the calculator result of the same name.
METRIC_DECLARATIONS: Final[tuple[MetricDeclaration, ...]] = (
    # --- EPISODE scope -> fact_episodes --------------------------------------------------
    MetricDeclaration(name="barn_collided", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_optimal_time", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_score", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_success", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="higher", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_time", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="barn_timeout", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="", description="performance metric of the barn_score calculator"),
    MetricDeclaration(name="mean_obstacle_clearance", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="performance metric of the clearance_metrics calculator"),
    MetricDeclaration(name="mean_pedestrian_clearance", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="performance metric of the clearance_metrics calculator"),
    MetricDeclaration(name="doorway_blocking_time", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="passing_rule_compliance", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="quiet_zone_dwell_seconds", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="restricted_zone_entries", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="speed_zone_violation_seconds", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="speed_zone_violations", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the compliance_metrics calculator"),
    MetricDeclaration(name="clauses_failed", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="clauses_passed", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="clauses_total", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="clauses_unknown", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="condition_success", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="", description="ecological metric of the condition_compliance calculator"),
    MetricDeclaration(name="edge_case_collided", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="lower", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_at", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_fault", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_ped_closing", unit="m/s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_robot_closing", unit="m/s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_collision_with", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_freeze_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_id", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_intrusion_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_clearance_at", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_clearance_m", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_clearance_with", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_ttc_at", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_ttc_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_min_ttc_with", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_pet_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_pet_with", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_played", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_scope", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_triggered_at", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_criticality calculator"),
    MetricDeclaration(name="edge_case_formation_arrived", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="higher", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_break_fired_at", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_broke_at", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_form_fired_at", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_formed_at", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_held_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_seen", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="edge_case_formation_variant", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="edge_case metric of the edge_case_formation calculator"),
    MetricDeclaration(name="aeps_linear_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="holistic metric of the holistic_metrics calculator"),
    MetricDeclaration(name="e_cot", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="holistic metric of the holistic_metrics calculator"),
    MetricDeclaration(name="mechanism_components", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the mechanism_component calculator"),
    MetricDeclaration(name="mechanism_score", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="ecological metric of the mechanism_component calculator"),
    MetricDeclaration(name="alarm_egress_blocking_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="blocked_closing_door", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="blocked_closing_door_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="boarded_before_exit", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="cabin_door_blocking_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="door_freeze_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="door_oscillations", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="door_yield_violations", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="elevator_missed_cycles", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="elevator_queue_jumps", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="elevator_unnecessary_calls", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="elevator_wait_to_board_s", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="entered_full_cabin", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="ped_mechanism_delay_seconds", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="ecological metric of the mechanism_metrics calculator"),
    MetricDeclaration(name="kcsc_wh", unit="Wh", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="social metric of the mutual_accommodation calculator"),
    MetricDeclaration(name="relative_throughput", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="social metric of the mutual_accommodation calculator"),
    MetricDeclaration(name="angle_over_length", unit="rad/m", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="curvature_max", unit="1/m", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="curvature_mean", unit="1/m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="roughness_max", unit="rad/m", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="roughness_mean", unit="rad/m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="performance metric of the path_metrics calculator"),
    MetricDeclaration(name="entered_over_cap_zone", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="ran_red_signal", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="replan_latency_after_state_change_median", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="replan_latency_after_state_change_p95", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="used_elevator_during_alarm", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="used_elevator_during_alarm_robot", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the regime_metrics calculator"),
    MetricDeclaration(name="elevator_rides", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the semantic_interaction_metrics calculator"),
    MetricDeclaration(name="elevator_rides_robot", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="ecological metric of the semantic_interaction_metrics calculator"),
    MetricDeclaration(name="time_waiting_at_doors", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="ecological metric of the semantic_interaction_metrics calculator"),
    MetricDeclaration(name="time_waiting_at_doors_robot", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="ecological metric of the semantic_interaction_metrics calculator"),
    MetricDeclaration(name="goal_condition_rate", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="judge_agrees", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="higher", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="phases_dropped", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="phases_failed", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="phases_met", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="phases_total", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="unsafe", unit="", scopes=(_EPISODE,), dtype=pl.Int64, direction="lower", description="performance metric of the task_replay calculator"),
    MetricDeclaration(name="idling_time", unit="s", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="performance metric of the time_metrics calculator"),
    MetricDeclaration(name="time_diff", unit="ns", scopes=(_EPISODE,), dtype=pl.Int64, direction="", description="performance metric of the time_metrics calculator"),
    MetricDeclaration(name="ade", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="adtw", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="fde", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="frechet_distance", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="mhd", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="path_irregularity", unit="rad/m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="theta_star_length", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="topological_complexity", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="naturalness metric of the trajectory_naturalness calculator"),
    MetricDeclaration(name="goal_inputs", unit="", scopes=(_EPISODE,), dtype=pl.String, direction="", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="ndtw", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="ne", unit="m", scopes=(_EPISODE,), dtype=pl.Float64, direction="lower", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="osr", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="sdtw", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="spl_geodesic", unit="", scopes=(_EPISODE,), dtype=pl.Float64, direction="higher", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="stuck", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="lower", description="performance metric of the vln_metrics calculator"),
    MetricDeclaration(name="success_geodesic", unit="", scopes=(_EPISODE,), dtype=pl.Boolean, direction="higher", description="performance metric of the vln_metrics calculator"),
    # --- ROBOT scope -> fact_robots ----------------------------------------------------
    MetricDeclaration(name="acoustic_cost", unit="s/m", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="acoustic_startle_factor", unit="dBA/s", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="acoustic_surge_index", unit="dBAÃƒâ€šÃ‚Â²", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="l_a10", unit="dBA", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="l_a50", unit="dBA", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="l_a90", unit="dBA", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="l_aeq_t", unit="dBA", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the acoustics calculator"),
    MetricDeclaration(name="char_phase_coverage", unit="", scopes=(_ROBOT,), dtype=pl.Float64, direction="", description="ecological metric of the characterization calculator"),
    MetricDeclaration(name="battery_soc_drop_pct", unit="%", scopes=(_ROBOT,), dtype=pl.Float64, direction="", description="ecological metric of the energy calculator"),
    MetricDeclaration(name="battery_soc_final", unit="%", scopes=(_ROBOT,), dtype=pl.Float64, direction="", description="ecological metric of the energy calculator"),
    MetricDeclaration(name="energy_static_wh", unit="Wh", scopes=(_ROBOT,), dtype=pl.Float64, direction="", description="ecological metric of the energy calculator"),
    MetricDeclaration(name="friction_dissipation_j", unit="J", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="kinetic_energy_demand_j", unit="J", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="peak_to_mean_power_ratio", unit="", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="specific_cost_of_transport", unit="", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="standstill_energy_penalty_wh", unit="Wh", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="ecological metric of the energy_extended calculator"),
    MetricDeclaration(name="acceleration_mean", unit="m/sÃƒâ€šÃ‚Â²", scopes=(_ROBOT,), dtype=pl.Float64, direction="", description="performance metric of the motion_metrics calculator"),
    MetricDeclaration(name="jerk_mean", unit="m/sÃƒâ€šÃ‚Â³", scopes=(_ROBOT,), dtype=pl.Float64, direction="lower", description="performance metric of the motion_metrics calculator"),
    # --- PEDESTRIAN scope -> fact_pedestrians -----------------------------------------------
    MetricDeclaration(name="ped_round_trips_completed", unit="", scopes=(_PEDESTRIAN,), dtype=pl.Float64, direction="", description="social metric of the pedestrian_disturbance calculator"),
    MetricDeclaration(name="ped_velocity_delay_ratio", unit="", scopes=(_PEDESTRIAN,), dtype=pl.Float64, direction="", description="social metric of the pedestrian_disturbance calculator"),

    # --- ENCOUNTER scope -> fact_encounters -------------------------------------------------
    MetricDeclaration(name="sfm_cumulative_force", unit="NÂ·s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="sfm_peak_force", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="sfm_mean_force", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_cumulative_force", unit="NÂ·s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_peak_force", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_mean_force", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_ped_cumulative_force", unit="NÂ·s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_ped_peak_force", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="esfm_ped_mean_force", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="social_force_max", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="social_force_mean", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="social_force_ped_max", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="social_force_ped_mean", unit="N", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="ci_max", unit="", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="ci_mean", unit="", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the social_forces calculator"),
    MetricDeclaration(name="time_in_intimate_zone", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="time_in_personal_zone", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="time_in_social_zone", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="time_in_public_zone", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="psi_intimate_events", unit="", scopes=(_ENCOUNTER,), dtype=pl.Int64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="psi_personal_events", unit="", scopes=(_ENCOUNTER,), dtype=pl.Int64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="psi_social_events", unit="", scopes=(_ENCOUNTER,), dtype=pl.Int64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="max_speed_intimate_zone", unit="m/s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="max_speed_personal_zone", unit="m/s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="max_speed_social_zone", unit="m/s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="max_speed_public_zone", unit="m/s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="movement_towards_peds_ratio", unit="", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="tti_min", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="higher", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="tti_mean", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="personal_space_intrusion_integral", unit="mÂ·s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the proxemics_extended calculator"),
    MetricDeclaration(name="total_time_looking_at_pedestrians", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the gaze_metrics calculator"),
    MetricDeclaration(name="total_time_looked_at_by_pedestrians", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the gaze_metrics calculator"),
    MetricDeclaration(name="num_pedestrians", unit="", scopes=(_ENCOUNTER,), dtype=pl.Int64, direction="", description="social metric of the proxemics calculator"),
    MetricDeclaration(name="total_time_in_personal_space", unit="s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="lower", description="social metric of the proxemics calculator"),
    MetricDeclaration(name="avg_velocity_in_personal_space", unit="m/s", scopes=(_ENCOUNTER,), dtype=pl.Float64, direction="", description="social metric of the proxemics calculator"),
    # --- CONTACT scope -> fact_contacts -----------------------------------------------------
    MetricDeclaration(name="hhi_interaction", unit="", scopes=(_CONTACT,), dtype=pl.String, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_hold_onset_s", unit="s", scopes=(_CONTACT,), dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_release_s", unit="s", scopes=(_CONTACT,), dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_episode_s", unit="s", scopes=(_CONTACT,), dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stall_count", unit="", scopes=(_CONTACT,), dtype=pl.Int64, direction="lower", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stall_time_s", unit="s", scopes=(_CONTACT,), dtype=pl.Float64, direction="lower", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stalls_approach", unit="", scopes=(_CONTACT,), dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stalls_hold", unit="", scopes=(_CONTACT,), dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stalls_post", unit="", scopes=(_CONTACT,), dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_stalls_rest", unit="", scopes=(_CONTACT,), dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_share_approach", unit="", scopes=(_CONTACT,), dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_share_hold", unit="", scopes=(_CONTACT,), dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_share_post", unit="", scopes=(_CONTACT,), dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_share_rest", unit="", scopes=(_CONTACT,), dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_passes_between", unit="", scopes=(_CONTACT,), dtype=pl.Int64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_min_pair_clearance", unit="m", scopes=(_CONTACT,), dtype=pl.Float64, direction="higher", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="hhi_pair_gap_hold", unit="m", scopes=(_CONTACT,), dtype=pl.Float64, direction="", description="social metric of the hhi_contact calculator"),
    MetricDeclaration(name="collision_amount", unit="", scopes=(_CONTACT,), dtype=pl.Int64, direction="lower", description="performance metric of the collision_metrics calculator"),
    MetricDeclaration(name="collision_amount_wall", unit="", scopes=(_CONTACT,), dtype=pl.Int64, direction="", description="performance metric of the collision_metrics calculator"),
    MetricDeclaration(name="spl", unit="", scopes=(_CONTACT,), dtype=pl.Float64, direction="higher", description="performance metric of the collision_metrics calculator"),
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
}


def declarations_for(scope: MetricScope) -> tuple[MetricDeclaration, ...]:
    """Declarations that publish into the fact table of ``scope``, in declaration order."""
    return tuple(
        declaration
        for declaration in METRIC_DECLARATIONS
        if scope in declaration.scopes
    )


def declared_columns(scope: MetricScope) -> tuple[str, ...]:
    """Fact columns declared for ``scope``, the declaration order of the schema."""
    return tuple(declaration.name for declaration in declarations_for(scope))


def _derived_schema(scope: MetricScope) -> pl.Schema:
    """Base fact schema of ``scope`` plus its declared metric columns, provenance string last."""
    base_columns = dict(_BASE_FACT_SCHEMAS[scope.value])
    null_reason = base_columns.pop("null_reason")
    for declaration in declarations_for(scope):
        if declaration.name in base_columns:
            raise ValueError(
                f"declaration {declaration.name!r} duplicates a base column of {scope.value}"
            )
        base_columns[declaration.name] = declaration.dtype
    base_columns["null_reason"] = null_reason
    return pl.Schema(base_columns)


FACT_TABLE_NAMES: Final[tuple[str, ...]] = tuple(scope.value for scope in MetricScope)

EPISODE_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(_EPISODE)
ENCOUNTER_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(MetricScope.ENCOUNTER)
CONTACT_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(MetricScope.CONTACT)
PEDESTRIAN_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(_PEDESTRIAN)
ROBOT_FACT_SCHEMA: Final[pl.Schema] = _derived_schema(_ROBOT)

FACT_SCHEMAS: Final[dict[str, pl.Schema]] = {
    name: _derived_schema(MetricScope(name)) for name in FACT_TABLE_NAMES
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
) + declared_columns(_EPISODE)
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
) + declared_columns(_ENCOUNTER)
_CONTACT_NULLABLE: Final[tuple[str, ...]] = (
    "impact_speed_mps",
    "robot_speed_at_impact_mps",
    "obstacle_speed_at_impact_mps",
    "closer_label",
    "robot_closing_share",
) + declared_columns(_CONTACT)
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
) + declared_columns(_PEDESTRIAN)
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
) + declared_columns(_ROBOT)


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
