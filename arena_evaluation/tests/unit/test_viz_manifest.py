"""Unit tests for VizManifest YAML parsing/round-trip with the declarative fields."""

import pathlib

import pytest
import yaml

from arena_evaluation.presentation.data_sources import (
    data_source_table,
    declared_data_sources,
    declared_series_stems,
    declared_view_names,
    series_view_name,
)
from arena_evaluation.presentation.manifest_registry import available_manifests, find_manifest_file
from arena_evaluation.presentation.viz_manifest import VizManifest


@pytest.mark.parametrize("manifest_name", ["standard", "safety", "social", "ecological", "characterization", "everything", "failure_modes"])
def test_all_bundled_manifests_load_and_roundtrip(manifest_name: str):
    """Test that all bundled manifests exist, validate schema, have unique IDs, and roundtrip cleanly."""
    p = find_manifest_file(manifest_name)
    assert p is not None, f"Manifest {manifest_name}.yaml must be bundled"

    manifest = VizManifest.load(p)
    assert manifest.name == manifest_name
    # A manifest names a declared table or view; a view nothing publishes fails at load.
    assert data_source_table(manifest.data_source) in declared_data_sources()
    assert manifest.units, f"Manifest {manifest_name} must declare units"
    assert manifest.groups, f"Manifest {manifest_name} must declare layout groups"
    assert len(manifest.plots) > 0, f"Manifest {manifest_name} must have plots"

    # All plot specs must have unique IDs
    ids = [spec.id for spec in manifest.plots]
    assert len(ids) == len(set(ids)), f"Duplicate plot IDs found in {manifest_name}: {[x for x in ids if ids.count(x) > 1]}"

    # YAML round-trip: dump -> re-validate
    dumped = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False)
    again = VizManifest.model_validate(yaml.safe_load(dumped))
    assert again.name == manifest.name
    assert len(again.plots) == len(manifest.plots)


def test_available_manifests_contains_all_manifests():
    """Verify registry discovers all bundled manifests including everything and failure_modes."""
    manifests = available_manifests()
    for expected in ["standard", "safety", "social", "ecological", "characterization", "everything", "failure_modes"]:
        assert expected in manifests, f"Expected manifest '{expected}' in available_manifests()"


def test_old_style_manifest_still_validates():
    manifest = VizManifest.model_validate({"plots": []})
    assert manifest.data_source == "metrics"
    assert manifest.groups == []
    assert manifest.summary == []
    assert manifest.units == {}


def test_characterization_manifest_loads():
    p = find_manifest_file("characterization")
    assert p is not None

    manifest = VizManifest.load(p)
    assert manifest.name == "characterization"
    # The report reads the 1NF series table the pipeline publishes, not the episode summary.
    assert manifest.data_source == series_view_name("characterization")
    assert "characterization" in declared_series_stems()
    assert manifest.data_source in declared_view_names()
    assert manifest.summary_group_by == ["phase_kind", "speed_target_mps"]
    assert manifest.summary  # declarative summary table
    line_specs = [s for s in manifest.plots if s.type == "line"]
    assert line_specs, "characterization manifest must use line charts"
    # Curves aggregate per working point from the long per-sample frame.
    assert any(s.options.get("aggregate") for s in manifest.plots)
    assert manifest.units.get("power_total_w") == "W"
    assert manifest.units.get("dba") == "dBA"


def test_manifest_view_sources_are_declared_by_the_package():
    """Every bundled manifest names a declared source, so a renamed view cannot silently orphan it."""
    for manifest_name in available_manifests():
        manifest_path = find_manifest_file(manifest_name)
        assert manifest_path is not None
        manifest = VizManifest.load(manifest_path)
        assert data_source_table(manifest.data_source) in declared_data_sources(), manifest_name


def test_a_manifest_referencing_an_unknown_view_fails_at_load(tmp_path: pathlib.Path):
    """A view-shaped data source is a declaration: an undeclared view fails at load, not at render."""
    path = tmp_path / "unknown_view.yaml"
    path.write_text(
        yaml.safe_dump({"name": "unknown_view", "data_source": "v_series_not_a_published_table", "plots": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown report data source"):
        VizManifest.load(path)


def test_characterization_manifest_columns_are_published_by_the_series():
    """Pin the report columns to the characterization series schema so the view and manifest cannot drift."""
    from arena_evaluation.processing.metrics.ecological.characterization import SERIES_SCHEMA

    manifest = VizManifest.load(find_manifest_file("characterization"))
    series_columns = set(SERIES_SCHEMA)
    assert set(manifest.units) <= series_columns
    assert {spec.metric for spec in manifest.summary} <= series_columns
    group_by = manifest.summary_group_by
    assert isinstance(group_by, list)
    assert set(group_by) <= series_columns
    assert {spec.data_key for spec in manifest.plots if spec.data_key and spec.data_key != "*"} <= series_columns


def test_standard_manifest_modernized_recipes_and_channels():
    """Verify that standard.yaml uses declarative plot recipes and explicit GoG channels."""
    from arena_evaluation.presentation.resolver import DataResolver

    p = find_manifest_file("standard")
    assert p is not None

    manifest = VizManifest.load(p)
    assert manifest.name == "standard"

    plot_by_id = {s.id: s for s in manifest.plots}

    # 1. Composite plot recipe checks
    assert "overview_table" in plot_by_id
    assert plot_by_id["overview_table"].uses == "scorecard_table"
    assert plot_by_id["overview_table"].layout_group == "overview"
    assert plot_by_id["overview_table"].title == "Benchmark Executive Overview"

    assert "radar_local_planner" in plot_by_id
    assert plot_by_id["radar_local_planner"].uses == "multi_planner_radar"
    assert plot_by_id["radar_local_planner"].layout_group == "overview"
    assert plot_by_id["radar_local_planner"].title == "Performance Overview by Local Planner"

    assert "radar_inter_planner" in plot_by_id
    assert plot_by_id["radar_inter_planner"].uses == "multi_planner_radar"
    assert plot_by_id["radar_inter_planner"].layout_group == "overview"
    assert plot_by_id["radar_inter_planner"].title == "Performance Overview by Inter-Planner"
    assert plot_by_id["radar_inter_planner"].channels.color == "inter_planner"

    assert "bar_collision_breakdown" in plot_by_id
    assert plot_by_id["bar_collision_breakdown"].uses == "closer_attribution_matrix"
    assert plot_by_id["bar_collision_breakdown"].layout_group == "safety"
    assert plot_by_id["bar_collision_breakdown"].title == "Collision Fault Attribution by Obstacle & Crowd Condition"

    assert "encounter_energy_breakdown" in plot_by_id
    assert plot_by_id["encounter_energy_breakdown"].uses == "encounter_energy_breakdown"
    assert plot_by_id["encounter_energy_breakdown"].layout_group == "ecological"

    assert "acoustic_exposure_sunburst" in plot_by_id
    assert plot_by_id["acoustic_exposure_sunburst"].uses == "acoustic_exposure_sunburst"
    assert plot_by_id["acoustic_exposure_sunburst"].layout_group == "social"

    assert "planner_ranking_stability" in plot_by_id
    assert plot_by_id["planner_ranking_stability"].uses == "planner_ranking_stability"
    assert plot_by_id["planner_ranking_stability"].layout_group == "tradeoffs"

    # 2. Verify all recipes resolve cleanly via DataResolver
    resolver = DataResolver()
    for s in manifest.plots:
        resolved = resolver.resolve_spec(s)
        assert resolved.id == s.id
        assert resolved.layout_group == s.layout_group

    # 3. Check atomic metric plots (box, violin, bar, scatter) have explicit GoG channels & data source
    for s in manifest.plots:
        if s.uses is None and s.type in ("box", "violin", "bar", "scatter"):
            assert s.data is not None, f"Plot {s.id} must declare data binding"
            assert s.data.source == "v_episode_summary", f"Plot {s.id} must bind the episode summary view, the only episode-grain source with planner identity"
            assert s.channels.x is not None, f"Plot {s.id} must declare x channel"
            assert s.channels.y is not None, f"Plot {s.id} must declare y channel"


def test_social_manifest_modernized_recipes_and_channels():
    """Verify that social.yaml uses declarative plot recipes and explicit GoG channels."""
    from arena_evaluation.presentation.resolver import DataResolver

    p = find_manifest_file("social")
    assert p is not None

    manifest = VizManifest.load(p)
    assert manifest.name == "social"

    plot_by_id = {s.id: s for s in manifest.plots}

    # 1. Composite plot recipe checks
    assert "social_scorecard_table" in plot_by_id
    assert plot_by_id["social_scorecard_table"].uses == "scorecard_table"
    assert plot_by_id["social_scorecard_table"].layout_group == "overview"
    assert plot_by_id["social_scorecard_table"].title == "Social Benchmark Executive Scorecard"

    assert "social_radar" in plot_by_id
    assert plot_by_id["social_radar"].uses == "multi_planner_radar"
    assert plot_by_id["social_radar"].layout_group == "overview"
    assert plot_by_id["social_radar"].title == "Social Capability Tradeoff Radar"

    assert "social_closer_attribution" in plot_by_id
    assert plot_by_id["social_closer_attribution"].uses == "closer_attribution_matrix"
    assert plot_by_id["social_closer_attribution"].layout_group == "encounters"
    assert plot_by_id["social_closer_attribution"].title == "Collision Fault Attribution by Crowd Condition"

    assert "social_encounter_energy" in plot_by_id
    assert plot_by_id["social_encounter_energy"].uses == "encounter_energy_breakdown"
    assert plot_by_id["social_encounter_energy"].layout_group == "encounters"

    assert "social_encounter_trajectory" in plot_by_id
    assert plot_by_id["social_encounter_trajectory"].uses == "encounter_trajectory_slice"
    assert plot_by_id["social_encounter_trajectory"].layout_group == "encounters"

    assert "social_acoustic_sunburst" in plot_by_id
    assert plot_by_id["social_acoustic_sunburst"].uses == "acoustic_exposure_sunburst"
    assert plot_by_id["social_acoustic_sunburst"].layout_group == "acoustics"

    # 2. Verify all recipes resolve cleanly via DataResolver
    resolver = DataResolver()
    for s in manifest.plots:
        resolved = resolver.resolve_spec(s)
        assert resolved.id == s.id
        assert resolved.layout_group == s.layout_group

    # 3. Check atomic metric plots (box, violin, bar, scatter) have explicit GoG channels & data source
    for s in manifest.plots:
        if s.uses is None and s.type in ("box", "violin", "bar", "scatter"):
            assert s.data is not None, f"Plot {s.id} must declare data binding"
            assert s.data.source == "v_episode_summary", f"Plot {s.id} must bind the episode summary view, the only episode-grain source with planner identity"
            assert s.channels.x is not None, f"Plot {s.id} must declare x channel"
            assert s.channels.y is not None, f"Plot {s.id} must declare y channel"
