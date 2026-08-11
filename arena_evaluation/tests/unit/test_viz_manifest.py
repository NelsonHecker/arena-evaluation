"""Unit tests for VizManifest YAML parsing/round-trip with the declarative fields."""

import pathlib

import yaml

from arena_evaluation.presentation.manifest_registry import find_manifest_file
from arena_evaluation.presentation.viz_manifest import VizManifest


def test_standard_manifest_loads_and_roundtrips():
    p = find_manifest_file("standard")
    assert p is not None, "standard.yaml must be bundled"

    manifest = VizManifest.load(p)
    assert manifest.name == "standard"
    assert manifest.data_source == "metrics"
    assert len(manifest.plots) == 43
    assert len(manifest.groups) == 8
    assert manifest.units  # units declared
    assert all(spec.id for spec in manifest.plots)
    ids = [spec.id for spec in manifest.plots]
    assert len(ids) == len(set(ids))

    # YAML round-trip: dump → re-validate
    dumped = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False)
    again = VizManifest.model_validate(yaml.safe_load(dumped))
    assert again.name == manifest.name
    assert len(again.plots) == len(manifest.plots)


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
    assert manifest.data_source == "metrics"
    assert manifest.summary_group_by == ["timeseries_char_phase_kind"]
    assert manifest.summary  # declarative summary table
    line_specs = [s for s in manifest.plots if s.type == "line"]
    assert line_specs, "characterization manifest must use line charts"
    # Curves aggregate per working point from the long per-sample frame.
    assert any(
        s.options.get("aggregate") for s in manifest.plots
    )
    assert manifest.units.get("timeseries_char_power_total_w") == "W"
    assert manifest.units.get("timeseries_char_dba") == "dBA"
