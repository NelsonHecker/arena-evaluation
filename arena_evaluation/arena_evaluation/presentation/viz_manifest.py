from __future__ import annotations

import pathlib
import typing
import yaml
from pydantic import BaseModel, Field

from ..storage.schemas import PlotSpec


class ManifestGroup(BaseModel):
    """A report section (layout_group → rendered heading)."""
    id: str
    title: str


class SummarySpec(BaseModel):
    """One column of the report's summary table."""
    metric: str
    label: str
    format: str = "{:.2f}"


class VizManifest(BaseModel):
    """
    Declarative manifest configuring which plots to generate for a benchmark.

    Mirrors the suite/contest declaration pattern: named YAML files in
    ``configs/benchmark/manifests/*.yaml``, resolved by :func:`resolve_manifest`.
    """
    manifest_version: str = "1.0"
    name: str | None = None
    title: str | None = None
    description: str | None = None
    # metrics | characterization_samples | characterization_summary | <parquet filename>
    data_source: str = "metrics"
    groups: list[ManifestGroup] = Field(default_factory=list)
    summary: list[SummarySpec] = Field(default_factory=list)
    summary_group_by: list[str] | str | None = None
    units: dict[str, str] = Field(default_factory=dict)
    plots: list[PlotSpec] = Field(default_factory=list)

    @classmethod
    def load(cls, path: pathlib.Path | None) -> "VizManifest":
        """Load manifest from a YAML file path (missing path → default)."""
        if path is None or not pathlib.Path(path).exists():
            return cls.load_default()

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest {path} must be a YAML mapping, got {type(data).__name__}")
        return cls.model_validate(data)

    @classmethod
    def load_default(cls) -> "VizManifest":
        """Load the default ('standard') named manifest."""
        from .manifest_registry import ManifestNotFoundError, find_manifest_file

        p = find_manifest_file("standard")
        if p is None:
            raise ManifestNotFoundError(
                "standard",
                "Manifest 'standard' not found. Install arena_evaluation or check "
                "configs/benchmark/manifests/standard.yaml.",
            )
        return cls.load(p)

    @classmethod
    def _default_manifest(cls) -> "VizManifest":
<<<<<<< HEAD
        """Generate a basic default manifest."""
        return cls(plots=[
            # ── Overview ──────────────────────────────────────────────────────────
            PlotSpec(
                id="radar_local_planner",
                type="radar",
                title="Performance Overview by Local Planner",
                data_key="*",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="overview",
                options={"metrics": ["success", "path_efficiency", "time_to_goal", "collision_amount", "roughness_mean", "jerk_mean"]},
            ),
            PlotSpec(
                id="radar_inter_planner",
                type="radar",
                title="Performance Overview by Inter-Planner",
                data_key="*",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="overview",
                options={"metrics": ["success", "path_efficiency", "time_to_goal", "collision_amount", "roughness_mean", "jerk_mean"]},
            ),
            PlotSpec(
                id="correlation_matrix",
                type="heatmap",
                title="Metrics Correlation Matrix",
                data_key="*",
                layout_group="overview",
            ),
            PlotSpec(
                id="heatmap_success_pivot",
                type="heatmap",
                title="Success Rate Pivot Grid (Local vs Inter Planner)",
                data_key="success",
                layout_group="overview",
                options={"x": "inter_planner", "y": "local_planner"},
            ),

            # ── Safety & Collisions ───────────────────────────────────────────────
            PlotSpec(
                id="bar_success_local",
                type="bar",
                title="Success Rate by Local Planner",
                data_key="success",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="safety",
            ),
            PlotSpec(
                id="bar_success_inter",
                type="bar",
                title="Success Rate by Inter-Planner",
                data_key="success",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="safety",
            ),
            PlotSpec(
                id="box_collisions_local",
                type="box",
                title="Collision Amount by Local Planner",
                data_key="collision_amount",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="safety",
            ),
            PlotSpec(
                id="box_collisions_inter",
                type="box",
                title="Collision Amount by Inter-Planner",
                data_key="collision_amount",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="safety",
            ),

            # ── Efficiency & Time ──────────────────────────────────────────────────
            PlotSpec(
                id="violin_efficiency_local",
                type="violin",
                title="Path Efficiency by Local Planner",
                data_key="path_efficiency",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="efficiency",
            ),
            PlotSpec(
                id="violin_efficiency_inter",
                type="violin",
                title="Path Efficiency by Inter-Planner",
                data_key="path_efficiency",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="efficiency",
            ),
            PlotSpec(
                id="violin_time_local",
                type="violin",
                title="Time to Goal by Local Planner",
                data_key="time_to_goal",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="efficiency",
            ),
            PlotSpec(
                id="violin_time_inter",
                type="violin",
                title="Time to Goal by Inter-Planner",
                data_key="time_to_goal",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="efficiency",
            ),
            PlotSpec(
                id="box_idling_local",
                type="box",
                title="Idling Time by Local Planner",
                data_key="idling_time",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="efficiency",
            ),
            PlotSpec(
                id="box_idling_inter",
                type="box",
                title="Idling Time by Inter-Planner",
                data_key="idling_time",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="efficiency",
            ),
            PlotSpec(
                id="histogram_path_length_local",
                type="histogram",
                title="Path Length Distribution by Local Planner",
                data_key="path_length",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="efficiency",
            ),
            PlotSpec(
                id="histogram_path_length_inter",
                type="histogram",
                title="Path Length Distribution by Inter-Planner",
                data_key="path_length",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="efficiency",
            ),

            # ── Motion Dynamics ──────────────────────────────────────────────────
            PlotSpec(
                id="violin_velocity_local",
                type="violin",
                title="Mean Velocity by Local Planner",
                data_key="velocity_mean",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="motion",
            ),
            PlotSpec(
                id="violin_velocity_inter",
                type="violin",
                title="Mean Velocity by Inter-Planner",
                data_key="velocity_mean",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="motion",
            ),
            PlotSpec(
                id="box_acceleration_local",
                type="box",
                title="Mean Acceleration by Local Planner",
                data_key="acceleration_mean",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="motion",
            ),
            PlotSpec(
                id="box_acceleration_inter",
                type="box",
                title="Mean Acceleration by Inter-Planner",
                data_key="acceleration_mean",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="motion",
            ),
            PlotSpec(
                id="box_jerk_local",
                type="box",
                title="Mean Jerk by Local Planner",
                data_key="jerk_mean",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="motion",
            ),
            PlotSpec(
                id="box_jerk_inter",
                type="box",
                title="Mean Jerk by Inter-Planner",
                data_key="jerk_mean",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="motion",
            ),

            # ── Path Smoothness ──────────────────────────────────────────────────
            PlotSpec(
                id="violin_curvature_local",
                type="violin",
                title="Curvature Mean by Local Planner",
                data_key="curvature_mean",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="smoothness",
            ),
            PlotSpec(
                id="violin_curvature_inter",
                type="violin",
                title="Curvature Mean by Inter-Planner",
                data_key="curvature_mean",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="smoothness",
            ),
            PlotSpec(
                id="violin_roughness_local",
                type="violin",
                title="Roughness Mean by Local Planner",
                data_key="roughness_mean",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="smoothness",
            ),
            PlotSpec(
                id="violin_roughness_inter",
                type="violin",
                title="Roughness Mean by Inter-Planner",
                data_key="roughness_mean",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="smoothness",
            ),

            # ── Social & Pedestrian Interaction ───────────────────────────────────
            PlotSpec(
                id="violin_personal_space_local",
                type="violin",
                title="Time in Personal Space by Local Planner",
                data_key="total_time_in_personal_space",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="social",
            ),
            PlotSpec(
                id="violin_personal_space_inter",
                type="violin",
                title="Time in Personal Space by Inter-Planner",
                data_key="total_time_in_personal_space",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="social",
            ),
            PlotSpec(
                id="box_looking_at_peds_local",
                type="box",
                title="Time Looking at Pedestrians by Local Planner",
                data_key="total_time_looking_at_pedestrians",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="social",
            ),
            PlotSpec(
                id="box_looking_at_peds_inter",
                type="box",
                title="Time Looking at Pedestrians by Inter-Planner",
                data_key="total_time_looking_at_pedestrians",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="social",
            ),
            PlotSpec(
                id="box_looked_at_by_peds_local",
                type="box",
                title="Time Looked at by Pedestrians by Local Planner",
                data_key="total_time_looked_at_by_pedestrians",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="social",
            ),
            PlotSpec(
                id="box_looked_at_by_peds_inter",
                type="box",
                title="Time Looked at by Pedestrians by Inter-Planner",
                data_key="total_time_looked_at_by_pedestrians",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="social",
            ),

            # ── Environment Interaction ──────────────────────────────────────────
            PlotSpec(
                id="box_time_waiting_doors_local",
                type="box",
                title="Time Waiting at Doors by Local Planner",
                data_key="time_waiting_at_doors",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="ecological",
            ),
            PlotSpec(
                id="box_time_waiting_doors_inter",
                type="box",
                title="Time Waiting at Doors by Inter-Planner",
                data_key="time_waiting_at_doors",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="ecological",
            ),
            PlotSpec(
                id="box_elevator_rides_local",
                type="box",
                title="Elevator Rides by Local Planner",
                data_key="elevator_rides",
                differentiate="local_planner",
                auto_differentiate=False,
                layout_group="ecological",
            ),
            PlotSpec(
                id="box_elevator_rides_inter",
                type="box",
                title="Elevator Rides by Inter-Planner",
                data_key="elevator_rides",
                differentiate="inter_planner",
                auto_differentiate=False,
                layout_group="ecological",
            ),

            # ── Compliance ────────────────────────────────────────────────────────
            PlotSpec(
                id="bar_speed_zone_violations",
                type="bar",
                title="Speed Zone Violations",
                data_key="speed_zone_violations",
                differentiate="local_planner",
                layout_group="compliance",
            ),
            PlotSpec(
                id="bar_speed_zone_violation_seconds",
                type="bar",
                title="Speed Zone Violation Seconds",
                data_key="speed_zone_violation_seconds",
                differentiate="local_planner",
                layout_group="compliance",
            ),
            PlotSpec(
                id="bar_quiet_zone_dwell",
                type="bar",
                title="Quiet Zone Dwell (s)",
                data_key="quiet_zone_dwell_seconds",
                differentiate="local_planner",
                layout_group="compliance",
            ),
            PlotSpec(
                id="bar_restricted_entries",
                type="bar",
                title="Restricted Zone Entries",
                data_key="restricted_zone_entries",
                differentiate="local_planner",
                layout_group="compliance",
            ),
            PlotSpec(
                id="bar_doorway_blocking",
                type="bar",
                title="Doorway Blocking Time (s)",
                data_key="doorway_blocking_time",
                differentiate="local_planner",
                layout_group="compliance",
            ),

            # ── Regime Change ────────────────────────────────────────────────────
            PlotSpec(
                id="bar_used_elevator_during_alarm",
                type="bar",
                title="Used Elevator During Alarm",
                data_key="used_elevator_during_alarm",
                differentiate="local_planner",
                layout_group="regime",
            ),
            PlotSpec(
                id="bar_ran_red_signal",
                type="bar",
                title="Ran Red Signal",
                data_key="ran_red_signal",
                differentiate="local_planner",
                layout_group="regime",
            ),
            PlotSpec(
                id="bar_entered_over_cap_zone",
                type="bar",
                title="Entered Over-Cap Zone",
                data_key="entered_over_cap_zone",
                differentiate="local_planner",
                layout_group="regime",
            ),
            PlotSpec(
                id="box_replan_latency_median",
                type="box",
                title="Replan Latency After State Change, Median (s)",
                data_key="replan_latency_after_state_change_median",
                differentiate="local_planner",
                layout_group="regime",
            ),
            PlotSpec(
                id="box_replan_latency_p95",
                type="box",
                title="Replan Latency After State Change, p95 (s)",
                data_key="replan_latency_after_state_change_p95",
                differentiate="local_planner",
                layout_group="regime",
            ),

            # ── Conditions ───────────────────────────────────────────────────────
            PlotSpec(
                id="bar_condition_success",
                type="bar",
                title="Condition Success",
                data_key="condition_success",
                differentiate="local_planner",
                layout_group="conditions",
            ),
            PlotSpec(
                id="bar_clauses_unknown",
                type="bar",
                title="Unknown Clauses",
                data_key="clauses_unknown",
                differentiate="local_planner",
                layout_group="conditions",
            ),

            # ── Detailed Traces ──────────────────────────────────────────────────
            PlotSpec(
                id="trajectory_plot",
                type="trajectory",
                title="Robot Trajectories",
                data_key="path",
                differentiate="local_planner",
                auto_differentiate=False,
                group_by=["stage"],
                layout_group="details",
            ),
            PlotSpec(
                id="pedestrian_trajectory_plot",
                type="trajectory",
                title="Pedestrian Trajectories",
                data_key="pedestrian_path",
                differentiate="local_planner",
                auto_differentiate=False,
                group_by=["stage"],
                layout_group="details",
                options={"overlay_markers": False},
            ),
        ])
=======
        """Backward-compatible alias for :meth:`load_default`."""
        return cls.load_default()
>>>>>>> origin-fork/jazzy
