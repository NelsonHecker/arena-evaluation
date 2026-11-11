"""Shipped report manifests bind declared sources and only columns the catalog serves.

A report is only as trustworthy as its bindings: a plot that names a table nobody publishes or a
column the table does not carry renders nothing, and the reader cannot tell that from a clean
result. This module pins the bindings of the seven shipped manifests against a fixture catalog built
from the writer's own schemas.

Two bindings are still known to miss, and both are pinned rather than hidden:

- **Grain.** The manifests ask the episode grain for metrics the pipeline publishes per encounter,
  per robot, per pedestrian or per contact, and those fact tables carry no planner identity to group
  by. Closing that needs the per-grain projection views of the dataset registry (D6/D7).
- **Missing views.** ``acoustic_exposure_sunburst`` reads ``v_zone_acoustic_summary`` and
  ``encounter_trajectory_slice`` reads ``v_encounter_telemetry``; no catalog publishes either, so
  those panels report a failed check on every build.

``_BINDING_DEBT`` names both per manifest so this check fails when a *new* unresolvable binding
enters a shipped manifest, and also when one is fixed without the debt being updated.
"""

from __future__ import annotations

import pathlib
from typing import Final

import polars as pl
import pytest
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.presentation.data_sources import data_source_table
from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.presentation.viz_manifest import VizManifest
from arena_evaluation.processing.metrics.ecological.characterization import SERIES_SCHEMA as CHARACTERIZATION_SERIES_SCHEMA
from arena_evaluation.processing.metrics.ecological.energy import SERIES_SCHEMA as ENERGY_SERIES_SCHEMA
from arena_evaluation.processing.metrics.performance.clearance_metrics import SERIES_SCHEMA as CLEARANCE_SERIES_SCHEMA
from arena_evaluation.storage.schemas import PlotSpec

__all__ = ["test_every_manifest_names_a_canonical_data_source", "test_every_manifest_summary_column_resolves", "test_manifest_plot_bindings_resolve_or_are_pinned_debt"]

_MANIFEST_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[2] / "configs" / "benchmark" / "manifests"

_SERIES_SCHEMAS: Final[dict[str, pl.Schema]] = {
    "characterization": pl.Schema(CHARACTERIZATION_SERIES_SCHEMA),
    "energy": pl.Schema(ENERGY_SERIES_SCHEMA),
    "clearance": pl.Schema(CLEARANCE_SERIES_SCHEMA),
}

# Every unresolvable binding the shipped manifests still declare: a metric name no episode-grain
# table carries yet, or a view no catalog publishes. Pinned so the set cannot grow unnoticed.
_BINDING_DEBT: Final[dict[str, frozenset[str]]] = {
    "characterization.yaml": frozenset(),
    "ecological.yaml": frozenset(
        {
            "battery_soc_drop_pct",
            "energy_mechanical_wh",
            "energy_static_wh",
            "energy_thermal_wh",
            "friction_dissipation_j",
            "kinetic_energy_demand_j",
            "ped_leq_exposure_dba",
            "ped_max_exposure_dba",
            "ped_max_startle_factor",
            "power_peak_w",
            "specific_cost_of_transport",
            "total_sound_energy_joules",
        }
    ),
    "everything.yaml": frozenset(
        {
            "acceleration_mean",
            "battery_soc_drop_pct",
            "ci_max",
            "ci_mean",
            "collision_amount",
            "collision_amount_pedestrian",
            "collision_amount_static",
            "collision_amount_wall",
            "energy_mechanical_wh",
            "energy_static_wh",
            "energy_thermal_wh",
            "friction_dissipation_j",
            "jerk_mean",
            "kinetic_energy_demand_j",
            "max_speed_intimate_zone",
            "max_speed_personal_zone",
            "movement_towards_peds_ratio",
            "path",
            "ped_leq_exposure_dba",
            "ped_max_exposure_dba",
            "ped_max_startle_factor",
            "ped_path_deflection_m",
            "pedestrian_path",
            "personal_space_intrusion_integral",
            "power_peak_w",
            "psi_intimate_events",
            "psi_personal_events",
            "social_force_ped_max",
            "specific_cost_of_transport",
            "time_in_intimate_zone",
            "total_sound_energy_joules",
            "total_time_in_personal_space",
            "total_time_looked_at_by_pedestrians",
            "total_time_looking_at_pedestrians",
            "tti_mean",
            "tti_min",
            "velocity_max",
        }
    ),
    "failure_modes.yaml": frozenset(
        {
            "acceleration_mean",
            "avg_velocity_in_personal_space",
            "battery_soc_drop_pct",
            "ci_max",
            "ci_mean",
            "collision_amount",
            "collision_amount_pedestrian",
            "collision_amount_static",
            "collision_amount_wall",
            "energy_mechanical_wh",
            "energy_static_wh",
            "energy_thermal_wh",
            "friction_dissipation_j",
            "jerk_mean",
            "kinetic_energy_demand_j",
            "map",
            "max_speed_intimate_zone",
            "max_speed_personal_zone",
            "movement_towards_peds_ratio",
            "path",
            "ped_leq_exposure_dba",
            "ped_max_exposure_dba",
            "ped_max_startle_factor",
            "ped_path_deflection_m",
            "personal_space_intrusion_integral",
            "power_peak_w",
            "psi_intimate_events",
            "psi_personal_events",
            "social_force_ped_max",
            "social_force_ped_mean",
            "specific_cost_of_transport",
            "time_in_intimate_zone",
            "total_sound_energy_joules",
            "total_time_in_personal_space",
            "tti_mean",
            "tti_min",
            "velocity_max",
        }
    ),
    "safety.yaml": frozenset(
        {
            "ci_max",
            "ci_mean",
            "collision_amount",
            "collision_amount_pedestrian",
            "collision_amount_static",
            "collision_amount_wall",
            "jerk_mean",
            "max_speed_intimate_zone",
            "movement_towards_peds_ratio",
            "path",
            "pedestrian_path",
            "personal_space_intrusion_integral",
            "tti_mean",
            "tti_min",
        }
    ),
    "social.yaml": frozenset(
        {
            "avg_velocity_in_personal_space",
            "ci_max",
            "ci_mean",
            "collision_amount",
            "composite_rank",
            "crowd_condition",
            "crowd_density",
            "encounter_type",
            "energy_wh",
            "jerk_mean",
            "max_speed_intimate_zone",
            "max_speed_personal_zone",
            "path",
            "ped_max_exposure_dba",
            "ped_max_startle_factor",
            "ped_path_deflection_m",
            "pedestrian_path",
            "personal_space_intrusion_integral",
            "planner",
            "psi_intimate_events",
            "psi_personal_events",
            "social_force_ped_max",
            "social_force_ped_mean",
            "specific_cost_of_transport",
            "table:v_encounter_telemetry",
            "table:v_zone_acoustic_summary",
            "time_in_intimate_zone",
            "total_time_in_personal_space",
            "total_time_looked_at_by_pedestrians",
            "total_time_looking_at_pedestrians",
        }
    ),
    "standard.yaml": frozenset(
        {
            "acceleration_mean",
            "ci_max",
            "collision_amount",
            "composite_rank",
            "crowd_condition",
            "crowd_density",
            "encounter_type",
            "energy_mechanical_wh",
            "energy_static_wh",
            "energy_thermal_wh",
            "energy_wh",
            "friction_dissipation_j",
            "inter_planner",
            "jerk_mean",
            "kinetic_energy_demand_j",
            "path",
            "ped_leq_exposure_dba",
            "ped_max_exposure_dba",
            "ped_max_startle_factor",
            "pedestrian_path",
            "personal_space_intrusion_integral",
            "planner",
            "social_force_ped_max",
            "specific_cost_of_transport",
            "table:v_zone_acoustic_summary",
            "total_time_in_personal_space",
            "total_time_looked_at_by_pedestrians",
            "total_time_looking_at_pedestrians",
            "tti_min",
        }
    ),
}


def _fixture_catalog(tmp_path: pathlib.Path) -> BenchmarkCatalog:
    """A catalog with the writer's declared fact schemas and the 1NF series tables of this branch."""
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True)
    for stem, schema in _SERIES_SCHEMAS.items():
        pl.DataFrame(schema=schema).write_parquet(metrics / f"series_{stem}.parquet")
    catalog = BenchmarkCatalog.from_directory(tmp_path)
    missing_views = [stem for stem in _SERIES_SCHEMAS if f"v_series_{stem}" not in catalog.table_names()]
    assert not missing_views, f"fixture catalog did not register {missing_views}"
    return catalog


def _manifest_paths() -> list[pathlib.Path]:
    return sorted(_MANIFEST_DIR.glob("*.yaml"))


def _manifest_and_raw(manifest_path: pathlib.Path) -> tuple[VizManifest, dict]:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return VizManifest.model_validate(raw), raw


def _resolved_plots(manifest_path: pathlib.Path) -> list[PlotSpec]:
    """Every plot of a manifest with its recipe merged in, the way the report builder sees it."""
    resolver = DataResolver()
    _, raw = _manifest_and_raw(manifest_path)
    return [resolver.resolve_spec(PlotSpec.model_validate(raw_plot)) for raw_plot in raw.get("plots") or []]


def _plot_source(spec: PlotSpec, manifest: VizManifest) -> str:
    declared = (spec.data.source if spec.data and spec.data.source else None) or spec.data_source or manifest.data_source
    return data_source_table(declared)


def _referenced_columns(spec: PlotSpec) -> set[str]:
    """Columns a plot binds: channels, keys, option metrics, filters, grouping and required columns."""
    columns: set[str] = set()
    if spec.data_key and spec.data_key != "*":
        columns.add(spec.data_key)
    for channel in (spec.channels.x, spec.channels.y, spec.channels.color, spec.channels.size, spec.channels.symbol, spec.channels.facet_col, spec.channels.facet_row, spec.channels.hierarchy):
        if isinstance(channel, str) and channel:
            columns.add(channel)
        elif isinstance(channel, list):
            columns.update(str(item) for item in channel if item)
    for key in ("x", "y", "secondary_data_key"):
        value = spec.options.get(key)
        if isinstance(value, str) and value:
            columns.add(value)
    for key in ("metrics", "columns"):
        for item in spec.options.get(key) or []:
            if isinstance(item, dict):
                metric = item.get("metric")
                if metric:
                    columns.add(str(metric))
            elif isinstance(item, str):
                columns.add(item)
    columns.update(str(key) for key in (spec.filter or {}))
    group_by = spec.group_by if isinstance(spec.group_by, list) else [spec.group_by]
    columns.update(str(item) for item in group_by if item)
    if "differentiate" in spec.model_fields_set and spec.differentiate:
        columns.add(spec.differentiate)
    if spec.checks:
        columns.update(str(item) for item in spec.checks.get("require_columns") or [])
    return columns


def _unresolved_bindings(catalog: BenchmarkCatalog, manifest_path: pathlib.Path) -> set[str]:
    manifest, _raw = _manifest_and_raw(manifest_path)
    tables = set(catalog.table_names())
    unresolved: set[str] = set()
    for spec in _resolved_plots(manifest_path):
        if spec.data is not None and (spec.data.query or spec.data.stored_query or spec.data.evidence_id):
            continue
        source = _plot_source(spec, manifest)
        if source not in tables:
            unresolved.add(f"table:{source}")
            continue
        unresolved.update(_referenced_columns(spec) - set(catalog.get_table(source).collect_schema().names()))
    return unresolved


def test_every_manifest_names_a_canonical_data_source(tmp_path: pathlib.Path) -> None:
    catalog = _fixture_catalog(tmp_path)
    problems: list[str] = []
    for path in _manifest_paths():
        manifest, _raw = _manifest_and_raw(path)
        if manifest.data_source == "metrics":
            problems.append(f"{path.name}: still names the historic 'metrics' alias")
        if data_source_table(manifest.data_source) not in catalog.table_names():
            problems.append(f"{path.name}: data_source {manifest.data_source!r} names no catalog table or declared view")
    assert not problems, "\n".join(problems)


def test_every_manifest_summary_column_resolves(tmp_path: pathlib.Path) -> None:
    catalog = _fixture_catalog(tmp_path)
    problems: list[str] = []
    for path in _manifest_paths():
        manifest, _raw = _manifest_and_raw(path)
        available = set(catalog.get_table(data_source_table(manifest.data_source)).collect_schema().names())
        missing = sorted({summary.metric for summary in manifest.summary} - available)
        if missing:
            problems.append(f"{path.name}: summary metrics {missing}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("manifest_path", _manifest_paths(), ids=lambda path: path.name)
def test_manifest_plot_bindings_resolve_or_are_pinned_debt(tmp_path: pathlib.Path, manifest_path: pathlib.Path) -> None:
    unresolved = _unresolved_bindings(_fixture_catalog(tmp_path), manifest_path)
    pinned = set(_BINDING_DEBT[manifest_path.name])
    assert unresolved == pinned, f"{manifest_path.name}: new {sorted(unresolved - pinned)}, fixed {sorted(pinned - unresolved)}"
