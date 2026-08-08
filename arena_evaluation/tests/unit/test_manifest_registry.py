"""Unit tests for declarative report-manifest resolution."""

import pathlib

import pytest

from arena_evaluation.presentation import manifest_registry as registry
from arena_evaluation.presentation.viz_manifest import VizManifest


@pytest.fixture
def manifests_dir(tmp_path: pathlib.Path, monkeypatch) -> pathlib.Path:
    """A fake package root with bundled manifests, reachable via source_tree_dir()."""
    root = tmp_path
    d = root / "configs" / "benchmark" / "manifests"
    d.mkdir(parents=True)
    (d / "foo.yaml").write_text(
        "manifest_version: '1.0'\nname: foo\ntitle: Foo Report\ndata_source: metrics\nplots: []\n"
    )
    (d / "standard.yaml").write_text(
        "manifest_version: '1.0'\nname: standard\nplots: []\n"
    )
    monkeypatch.setattr(registry, "share_dir", lambda: None)
    monkeypatch.setattr(registry, "source_tree_dir", lambda: root)
    return d


def test_resolve_by_name(manifests_dir):
    m = registry.resolve_manifest("foo")
    assert isinstance(m, VizManifest)
    assert m.name == "foo"
    assert m.title == "Foo Report"


def test_resolve_by_name_with_yaml_suffix(manifests_dir):
    m = registry.resolve_manifest("foo.yaml")
    assert m.name == "foo"


def test_resolve_inline_dict():
    m = registry.resolve_manifest("{name: inline, plots: []}")
    assert m.name == "inline"
    assert m.plots == []


def test_resolve_inline_list_rejected():
    with pytest.raises(ValueError):
        registry.resolve_manifest("[a, b]")


def test_resolve_explicit_path(manifests_dir):
    p = manifests_dir / "foo.yaml"
    m = registry.resolve_manifest(str(p))
    assert m.name == "foo"


def test_resolve_missing_name_raises(manifests_dir):
    with pytest.raises(registry.ManifestNotFoundError) as exc:
        registry.resolve_manifest("does_not_exist")
    assert "does_not_exist" in str(exc.value)
    assert "foo" in str(exc.value)  # available list included


def test_legacy_benchmark_viz_manifest(tmp_path: pathlib.Path):
    (tmp_path / "viz_manifest.yaml").write_text(
        "manifest_version: '1.0'\nname: legacy\ndata_source: metrics\nplots: []\n"
    )
    m = registry.resolve_manifest(None, benchmark_dir=tmp_path)
    assert m.name == "legacy"


def test_note_file_readback(tmp_path: pathlib.Path, manifests_dir):
    (tmp_path / "report_manifest.yaml").write_text(
        "name: foo\ndata_source: metrics\nn_plots: 0\n"
    )
    m = registry.resolve_manifest(None, benchmark_dir=tmp_path)
    assert m.name == "foo"


def test_stale_note_falls_back_to_default(tmp_path: pathlib.Path, manifests_dir):
    (tmp_path / "report_manifest.yaml").write_text("name: gone\n")
    m = registry.resolve_manifest(None, benchmark_dir=tmp_path)
    assert m.name == "standard"


def test_default_when_nothing_else(tmp_path: pathlib.Path, manifests_dir):
    m = registry.resolve_manifest(None, benchmark_dir=tmp_path)
    assert m.name == "standard"


def test_available_manifests(manifests_dir):
    assert registry.available_manifests() == ["foo", "standard"]
