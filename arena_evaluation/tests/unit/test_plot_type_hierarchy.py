"""Unit tests for the hierarchical breakdown and phylogenetic cladistics renderer."""

from __future__ import annotations

import pathlib

import polars as pl
import yaml

from arena_evaluation.presentation.plot_types.hierarchy import HierarchyRenderer
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.storage.schemas import PlotSpec

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _episodes_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb", "teb", "mppi", "mppi", "rpp", "rpp"],
            "time_to_goal": [12.0, 14.0, 10.0, 11.0, 8.5, 9.0, 13.0, 15.0],
            "path_length": [15.2, 16.0, 14.1, 14.5, 12.0, 12.5, 16.5, 17.0],
            "collision_amount": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            "jerk_mean": [0.45, 0.50, 0.30, 0.35, 0.80, 0.85, 0.40, 0.42],
            "specific_cost_of_transport": [2.1, 2.3, 1.8, 1.9, 3.2, 3.4, 2.5, 2.6],
        }
    )


def _encounters_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "crowd_condition": ["sparse", "sparse", "dense", "dense", "dense"],
            "approach_type": ["head_on", "crossing", "head_on", "crossing", "overtaking"],
            "planner": ["dwb", "teb", "dwb", "mppi", "teb"],
            "encounter_energy_j": [150.0, 120.0, 310.0, 280.0, 210.0],
        }
    )


def _assert_png(path: pathlib.Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == _PNG_MAGIC


# ── Dendrogram Mode (Plotly) ──────────────────────────────────────────────


def test_dendrogram_plotly_renders_html_div():
    spec = PlotSpec(
        id="dendro_test",
        type="hierarchy",
        title="Behavioral Cladistics",
        channels={
            "x": "planner",
            "y": ["time_to_goal", "path_length", "collision_amount", "jerk_mean", "specific_cost_of_transport"],
        },
        options={"mode": "dendrogram", "clustering_method": "ward"},
    )
    renderer = HierarchyRenderer(spec)
    html = renderer.render_plotly(_episodes_df())
    assert html is not None
    assert html.startswith("<div")
    assert "plotly" in html.lower() or "data" in html
    assert "Behavioral Cladistics" in html


def test_dendrogram_plotly_type_dendrogram_direct():
    spec = PlotSpec(
        id="dendro_direct",
        type="dendrogram",
        title="Cladistics Direct",
        channels={
            "x": "planner",
            "y": ["time_to_goal", "path_length"],
        },
    )
    renderer = HierarchyRenderer(spec)
    html = renderer.render_plotly(_episodes_df())
    assert html is not None
    assert html.startswith("<div")


def test_dendrogram_plotly_minmax_normalization():
    spec = PlotSpec(
        id="dendro_minmax",
        type="hierarchy",
        title="MinMax Cladistics",
        channels={"x": "planner", "y": ["time_to_goal", "path_length"]},
        options={"mode": "dendrogram", "normalize": "minmax"},
    )
    html = HierarchyRenderer(spec).render_plotly(_episodes_df())
    assert html is not None
    assert html.startswith("<div")


def test_dendrogram_insufficient_planners_returns_none():
    single_planner_df = pl.DataFrame(
        {
            "planner": ["dwb", "dwb"],
            "time_to_goal": [10.0, 12.0],
            "path_length": [14.0, 15.0],
        }
    )
    spec = PlotSpec(
        id="single",
        type="hierarchy",
        channels={"x": "planner", "y": ["time_to_goal", "path_length"]},
        options={"mode": "dendrogram"},
    )
    assert HierarchyRenderer(spec).render_plotly(single_planner_df) is None


# ── Hierarchical Breakdown (Treemap / Sunburst) ────────────────────────────


def test_hierarchy_treemap_plotly_renders_html_div():
    spec = PlotSpec(
        id="treemap_test",
        type="hierarchy",
        title="Encounter Energy Treemap",
        channels={
            "hierarchy": ["crowd_condition", "approach_type", "planner"],
            "y": "encounter_energy_j",
        },
        options={"mode": "treemap"},
    )
    renderer = HierarchyRenderer(spec)
    html = renderer.render_plotly(_encounters_df())
    assert html is not None
    assert html.startswith("<div")
    assert "Encounter Energy Treemap" in html


def test_hierarchy_sunburst_plotly_renders_html_div():
    spec = PlotSpec(
        id="sunburst_test",
        type="hierarchy",
        title="Encounter Energy Sunburst",
        channels={
            "hierarchy": ["crowd_condition", "approach_type", "planner"],
            "y": "encounter_energy_j",
        },
        options={"mode": "sunburst"},
    )
    renderer = HierarchyRenderer(spec)
    html = renderer.render_plotly(_encounters_df())
    assert html is not None
    assert html.startswith("<div")


def test_hierarchy_path_fallback_channels():
    spec = PlotSpec(
        id="fallback_path",
        type="hierarchy",
        title="Fallback Hierarchy",
        channels={"x": "crowd_condition", "color": "planner", "y": "encounter_energy_j"},
    )
    renderer = HierarchyRenderer(spec)
    html = renderer.render_plotly(_encounters_df())
    assert html is not None
    assert html.startswith("<div")


def test_empty_dataframe_returns_none():
    spec = PlotSpec(id="empty", type="hierarchy", options={"mode": "dendrogram"})
    assert HierarchyRenderer(spec).render_plotly(pl.DataFrame()) is None


# ── Seaborn / Static PNG Rendering ─────────────────────────────────────────


def test_dendrogram_static_renders_png(tmp_path: pathlib.Path):
    out = tmp_path / "dendrogram.png"
    spec = PlotSpec(
        id="dendro_sb",
        type="hierarchy",
        title="Dendrogram Seaborn",
        channels={"x": "planner", "y": ["time_to_goal", "path_length", "jerk_mean"]},
        options={"mode": "dendrogram", "clustering_method": "ward"},
    )
    HierarchyRenderer(spec).render_static(_episodes_df(), out)
    _assert_png(out)


def test_dendrogram_static_clustermap_renders_png(tmp_path: pathlib.Path):
    out = tmp_path / "clustermap.png"
    spec = PlotSpec(
        id="dendro_cm",
        type="hierarchy",
        title="Clustermap Seaborn",
        channels={"x": "planner", "y": ["time_to_goal", "path_length", "jerk_mean"]},
        options={"mode": "dendrogram", "clustermap": True},
    )
    HierarchyRenderer(spec).render_static(_episodes_df(), out)
    _assert_png(out)


def test_hierarchy_static_renders_png(tmp_path: pathlib.Path):
    out = tmp_path / "hierarchy_bars.png"
    spec = PlotSpec(
        id="hier_sb",
        type="hierarchy",
        title="Hierarchy Seaborn",
        channels={
            "hierarchy": ["crowd_condition", "approach_type"],
            "y": "encounter_energy_j",
        },
    )
    HierarchyRenderer(spec).render_static(_encounters_df(), out)
    _assert_png(out)


# ── Dispatcher Integration ────────────────────────────────────────────────


def test_plotly_dispatcher_hierarchy():
    spec = PlotSpec(
        id="disp_hier",
        type="hierarchy",
        channels={"hierarchy": ["crowd_condition", "approach_type"], "y": "encounter_energy_j"},
    )
    html = PlotlyRenderer().render(spec, _encounters_df())
    assert html is not None
    assert html.startswith("<div")


def test_plotly_dispatcher_dendrogram():
    spec = PlotSpec(
        id="disp_dendro",
        type="dendrogram",
        channels={"x": "planner", "y": ["time_to_goal", "path_length"]},
    )
    html = PlotlyRenderer().render(spec, _episodes_df())
    assert html is not None
    assert html.startswith("<div")


def test_static_dispatcher_hierarchy(tmp_path: pathlib.Path):
    out = tmp_path / "disp_hier.png"
    spec = PlotSpec(
        id="disp_sb_hier",
        type="hierarchy",
        channels={"hierarchy": ["crowd_condition", "approach_type"], "y": "encounter_energy_j"},
    )
    StaticRenderer().render(spec, _encounters_df(), out)
    _assert_png(out)


def test_static_dispatcher_dendrogram(tmp_path: pathlib.Path):
    out = tmp_path / "disp_sb_dendro.png"
    spec = PlotSpec(
        id="disp_sb_dendro",
        type="dendrogram",
        channels={"x": "planner", "y": ["time_to_goal", "path_length"]},
    )
    StaticRenderer().render(spec, _episodes_df(), out)
    _assert_png(out)


# ── Declarative YAML Recipes ──────────────────────────────────────────────


def test_declarative_recipes_load_and_render(tmp_path: pathlib.Path):
    decl_dir = pathlib.Path(__file__).parents[2] / "configs" / "declarations" / "plot_declarations"

    # 1. Planner Behavioral Dendrogram
    dendro_path = decl_dir / "planner_behavioral_dendrogram.yaml"
    assert dendro_path.exists()
    dendro_raw = yaml.safe_load(dendro_path.read_text(encoding="utf-8"))
    dendro_spec = PlotSpec(**dendro_raw)
    assert dendro_spec.type == "hierarchy"
    assert dendro_spec.options.get("mode") == "dendrogram"

    dendro_html = PlotlyRenderer().render(dendro_spec, _episodes_df())
    assert dendro_html is not None
    assert dendro_html.startswith("<div")

    dendro_png = tmp_path / "recipe_dendro.png"
    StaticRenderer().render(dendro_spec, _episodes_df(), dendro_png)
    _assert_png(dendro_png)

    # 2. Encounter Hierarchy Treemap
    treemap_path = decl_dir / "encounter_hierarchy_treemap.yaml"
    assert treemap_path.exists()
    treemap_raw = yaml.safe_load(treemap_path.read_text(encoding="utf-8"))
    treemap_spec = PlotSpec(**treemap_raw)
    assert treemap_spec.type == "hierarchy"
    assert treemap_spec.options.get("mode") == "treemap"

    treemap_html = PlotlyRenderer().render(treemap_spec, _encounters_df())
    assert treemap_html is not None
    assert treemap_html.startswith("<div")

    treemap_png = tmp_path / "recipe_treemap.png"
    StaticRenderer().render(treemap_spec, _encounters_df(), treemap_png)
    _assert_png(treemap_png)
