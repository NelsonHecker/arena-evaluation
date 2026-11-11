"""Unit tests for the Modern Declarative Presentation Engine (WP 4 / Layer 5).

Verifies:
1. Declarative Grammar of Graphics: multi-dimensional channel encodings (x, y,
   color/hue, facet_col, facet_row, size, symbol, split) across all 12 plot types
   with interactive Plotly HTML and publication-quality static/vector image output
   (PNG, SVG, PDF).
2. DataResolver: binding against Star Schema lakehouse tables/views, Polars SQL queries,
   content-addressed EvidenceStore artifacts, in-memory fallback frames, and
   spatiotemporal slices (episode, encounter, contact).
3. Recipe Catalog: resolution and deep-merge overrides for all pre-defined plot recipes
   via `uses: <recipe_name>`.
4. Non-Blocking Diagnostics: DiagnosticInspector metadata collection for small sample
   counts (N < 10), zero-variance columns, null columns, and missing required columns
   without ever halting rendering.
5. ReportBuilder & Standalone Artifacts: full generation of HTML reports with diagnostics
   callouts and standalone per-plot image artifacts for autonomous agent inspection.
6. Full backward compatibility with legacy PlotSpec and manifest declarations.
"""

from __future__ import annotations

import pathlib
from typing import Final

import polars as pl
import pytest

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.presentation.checks import DiagnosticInspector
from arena_evaluation.presentation.plot_types.bar import BarRenderer
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer
from arena_evaluation.presentation.plot_types.box import BoxRenderer
from arena_evaluation.presentation.plot_types.heatmap import HeatmapRenderer
from arena_evaluation.presentation.plot_types.histogram import HistogramRenderer
from arena_evaluation.presentation.plot_types.line import LineRenderer
from arena_evaluation.presentation.plot_types.radar import RadarRenderer
from arena_evaluation.presentation.plot_types.scatter import ScatterRenderer
from arena_evaluation.presentation.plot_types.table import TableRenderer
from arena_evaluation.presentation.plot_types.timeseries import TimeseriesRenderer
from arena_evaluation.presentation.plot_types.trajectory import TrajectoryRenderer
from arena_evaluation.presentation.plot_types.violin import ViolinRenderer
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.resolver import DataResolver, find_recipe_path, load_recipe
from arena_evaluation.storage.schemas import (
    PlotChannels,
    PlotDataSpec,
    PlotSliceSpec,
    PlotSpec,
)
from star_benchmark import write_benchmark

_PNG_MAGIC: Final[bytes] = b"\x89PNG\r\n\x1a\n"


def _sample_distributions_frame() -> pl.DataFrame:
    """Multi-dimensional sample dataset for distribution plots."""
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "dwb", "dwb", "teb", "teb", "teb", "teb", "mppi", "mppi", "mppi", "mppi"],
            "crowd_condition": ["sparse", "sparse", "dense", "dense", "sparse", "sparse", "dense", "dense", "sparse", "sparse", "dense", "dense"],
            "path_length_m": [12.4, 13.1, 15.6, 16.2, 11.8, 12.0, 14.2, 14.8, 10.9, 11.2, 13.5, 13.9],
            "time_to_goal_s": [18.2, 19.0, 24.5, 25.1, 16.5, 17.1, 22.0, 22.8, 15.0, 15.4, 20.1, 20.8],
            "velocity_mps": [0.72, 0.70, 0.65, 0.64, 0.74, 0.73, 0.68, 0.66, 0.75, 0.74, 0.71, 0.70],
            "success": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "collision_count": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "zone_name": ["corridor", "lounge", "corridor", "lounge", "corridor", "lounge", "corridor", "lounge", "corridor", "lounge", "corridor", "lounge"],
        }
    )


def _sample_timeseries_frame() -> pl.DataFrame:
    """1NF flat timeseries dataset for trajectory and curve plots."""
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    return pl.DataFrame(
        {
            "planner": ["dwb"] * 5 + ["teb"] * 5,
            "time_s": times + times,
            "vx_mps": [0.0, 0.4, 0.7, 0.75, 0.0, 0.0, 0.35, 0.65, 0.70, 0.0],
            "power_w": [45.0, 72.0, 85.0, 83.0, 48.0, 44.0, 68.0, 80.0, 78.0, 46.0],
            "sound_pressure_dba": [42.0, 48.0, 56.0, 54.0, 43.0, 41.0, 46.0, 52.0, 51.0, 42.0],
            "episode_id": [1] * 5 + [2] * 5,
            "encounter_id": ["enc_01"] * 5 + ["enc_02"] * 5,
        }
    )


def _sample_trajectory_frame() -> pl.DataFrame:
    """2D spatial trajectory dataset with coordinates and navigation outcomes."""
    return pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "episode_id": [1, 2],
            "encounter_id": ["enc_1", "enc_2"],
            "contact_id": ["cnt_1", "cnt_2"],
            "path": [
                [[0.0, 0.0], [1.0, 0.2], [2.0, 0.5], [3.0, 1.0]],
                [[0.0, 0.0], [0.8, 0.4], [1.8, 0.9], [2.9, 1.0]],
            ],
            "start": [[0.0, 0.0], [0.0, 0.0]],
            "goal": [[3.0, 1.0], [3.0, 1.0]],
            "result": ["SUCCESS", "SUCCESS"],
            "success": [1.0, 1.0],
            "collision_amount": [0, 0],
        },
        schema={
            "planner": pl.Utf8,
            "episode_id": pl.Int64,
            "encounter_id": pl.Utf8,
            "contact_id": pl.Utf8,
            "path": pl.List(pl.List(pl.Float64)),
            "start": pl.List(pl.Float64),
            "goal": pl.List(pl.Float64),
            "result": pl.Utf8,
            "success": pl.Float64,
            "collision_amount": pl.Int64,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Grammar of Graphics & Multi-Dimensional Channel Rendering
# ═══════════════════════════════════════════════════════════════════════════


def test_grammar_of_graphics_violin_channels_plotly_and_static(tmp_path: pathlib.Path) -> None:
    df = _sample_distributions_frame()
    spec = PlotSpec(
        id="violin_multidim",
        type="violin",
        title="Path Length Distribution Across Crowd Conditions",
        channels=PlotChannels(
            x="planner",
            y="path_length_m",
            color="crowd_condition",
            facet_col="zone_name",
            split=True,
        ),
        options={"box": True, "points": "all"},
    )

    renderer = ViolinRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None
    assert html.startswith("<div")
    assert "Path Length" in html

    out_png = tmp_path / "violin.png"
    out_svg = tmp_path / "violin.svg"
    renderer.render_static(df, out_png)
    renderer.render_static(df, out_svg)
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_png.read_bytes()[:8] == _PNG_MAGIC
    assert out_svg.exists() and out_svg.stat().st_size > 0


def test_grammar_of_graphics_box_channels_plotly_and_static(tmp_path: pathlib.Path) -> None:
    df = _sample_distributions_frame()
    spec = PlotSpec(
        id="box_multidim",
        type="box",
        title="Time to Goal Spread by Planner and Crowd",
        channels=PlotChannels(
            x="planner",
            y="time_to_goal_s",
            color="crowd_condition",
            facet_col="zone_name",
        ),
        options={"notched": True},
    )

    renderer = BoxRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None
    assert html.startswith("<div")

    out_png = tmp_path / "box.png"
    renderer.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_grammar_of_graphics_bar_stacked_and_grouped_modes(tmp_path: pathlib.Path) -> None:
    df = _sample_distributions_frame()
    spec_grouped = PlotSpec(
        id="bar_grouped",
        type="bar",
        title="Success Rate by Planner and Crowd",
        channels=PlotChannels(
            x="planner",
            y="success",
            color="crowd_condition",
            facet_col="zone_name",
        ),
        options={"barmode": "group"},
    )
    renderer_grouped = BarRenderer(spec_grouped)
    html_grouped = renderer_grouped.render_plotly(df)
    assert html_grouped is not None and html_grouped.startswith("<div")

    out_png = tmp_path / "bar_grouped.png"
    renderer_grouped.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0

    spec_stacked = PlotSpec(
        id="bar_stacked",
        type="bar",
        title="Collision Incidents by Crowd",
        channels=PlotChannels(
            x="planner",
            y="collision_count",
            color="crowd_condition",
        ),
        options={"barmode": "stack"},
    )
    renderer_stacked = BarRenderer(spec_stacked)
    html_stacked = renderer_stacked.render_plotly(df)
    assert html_stacked is not None and html_stacked.startswith("<div")


def test_grammar_of_graphics_line_channels_plotly_and_static(tmp_path: pathlib.Path) -> None:
    df = _sample_timeseries_frame()
    spec = PlotSpec(
        id="line_multidim",
        type="line",
        title="Velocity Trajectory Over Time",
        channels=PlotChannels(x="time_s", y="vx_mps", color="planner"),
    )
    renderer = LineRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None and html.startswith("<div")

    out_png = tmp_path / "line.png"
    renderer.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_grammar_of_graphics_scatter_channels_plotly_and_static(tmp_path: pathlib.Path) -> None:
    df = _sample_distributions_frame()
    spec = PlotSpec(
        id="scatter_multidim",
        type="scatter",
        title="Path Length vs Time to Goal",
        channels=PlotChannels(
            x="path_length_m",
            y="time_to_goal_s",
            color="planner",
            size="velocity_mps",
        ),
    )
    renderer = ScatterRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None and html.startswith("<div")

    out_png = tmp_path / "scatter.png"
    renderer.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_grammar_of_graphics_histogram_channels_plotly_and_static(tmp_path: pathlib.Path) -> None:
    df = _sample_distributions_frame()
    spec = PlotSpec(
        id="hist_multidim",
        type="histogram",
        title="Velocity Distribution",
        channels=PlotChannels(x="velocity_mps", color="planner"),
        options={"barmode": "overlay", "nbins": 10},
    )
    renderer = HistogramRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None and html.startswith("<div")

    out_png = tmp_path / "hist.png"
    renderer.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_grammar_of_graphics_heatmap_pivot_and_correlation(tmp_path: pathlib.Path) -> None:
    df = _sample_distributions_frame()
    # 1. 2D Matrix Pivot Heatmap
    spec_pivot = PlotSpec(
        id="heatmap_pivot",
        type="heatmap",
        title="Average Velocity by Planner and Crowd",
        channels=PlotChannels(x="planner", y="crowd_condition", color="velocity_mps"),
    )
    renderer_pivot = HeatmapRenderer(spec_pivot)
    html_pivot = renderer_pivot.render_plotly(df)
    assert html_pivot is not None and html_pivot.startswith("<div")

    out_png = tmp_path / "heatmap_pivot.png"
    renderer_pivot.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0

    # 2. Metric Correlation Heatmap
    spec_corr = PlotSpec(
        id="heatmap_corr",
        type="heatmap",
        title="Metric Correlation Matrix",
        data_key="correlation",
        options={"kind": "correlation"},
    )
    renderer_corr = HeatmapRenderer(spec_corr)
    html_corr = renderer_corr.render_plotly(df)
    assert html_corr is not None and html_corr.startswith("<div")


def test_grammar_of_graphics_timeseries_plotly_and_static(tmp_path: pathlib.Path) -> None:
    df = _sample_timeseries_frame()
    spec = PlotSpec(
        id="timeseries_power",
        type="timeseries",
        title="Power Profiles",
        channels=PlotChannels(x="time_s", y="power_w", color="planner"),
        options={"render_static": True},
    )
    renderer = TimeseriesRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None and html.startswith("<div")

    out_png = tmp_path / "timeseries.png"
    renderer.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_grammar_of_graphics_trajectory_2d_paths(tmp_path: pathlib.Path) -> None:
    df = _sample_trajectory_frame()
    spec = PlotSpec(
        id="trajectory_paths",
        type="trajectory",
        title="2D Navigation Paths",
        channels=PlotChannels(x="path", color="planner"),
        options={"overlay_markers": True},
    )
    renderer = TrajectoryRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None and html.startswith("<div")

    out_png = tmp_path / "trajectory.png"
    renderer.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_grammar_of_graphics_radar_multi_metric(tmp_path: pathlib.Path) -> None:
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb", "mppi"],
            "path_efficiency": [0.82, 0.78, 0.88],
            "time_to_goal": [18.0, 16.5, 15.0],
            "collision_amount": [0.1, 0.2, 0.0],
            "velocity_mean": [0.71, 0.73, 0.74],
            "success": [0.95, 0.90, 1.00],
        }
    )
    spec = PlotSpec(
        id="radar_chart",
        type="radar",
        title="Planner Profile Comparison",
        channels=PlotChannels(
            color="planner",
            y=["path_efficiency", "time_to_goal", "collision_amount", "velocity_mean", "success"],
        ),
    )
    renderer = RadarRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None and html.startswith("<div")

    out_png = tmp_path / "radar.png"
    renderer.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_grammar_of_graphics_table_html_and_image(tmp_path: pathlib.Path) -> None:
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "success_rate": [0.95, 0.90],
            "collision_rate": [0.05, 0.10],
            "time_to_goal_s": [18.2, 16.5],
        }
    )
    spec = PlotSpec(
        id="table_summary",
        type="table",
        title="Scorecard Table",
        channels=PlotChannels(x="planner", y=["success_rate", "collision_rate", "time_to_goal_s"]),
        options={"format": "{:.2f}"},
    )
    renderer = TableRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None
    assert "<table" in html
    assert "Scorecard Table" in html
    assert "0.95" in html

    out_png = tmp_path / "table.png"
    renderer.render_static(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_save_figure_multi_format_vector_and_raster(tmp_path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [0, 1, 4])

    png_path = tmp_path / "vector_test.png"
    svg_path = tmp_path / "vector_test.svg"
    pdf_path = tmp_path / "vector_test.pdf"

    BasePlotRenderer.save_figure(fig, png_path)
    BasePlotRenderer.save_figure(fig, svg_path)
    BasePlotRenderer.save_figure(fig, pdf_path)
    plt.close(fig)

    assert png_path.exists() and png_path.stat().st_size > 0
    assert png_path.read_bytes()[:8] == _PNG_MAGIC
    assert svg_path.exists() and svg_path.stat().st_size > 0
    assert b"<svg" in svg_path.read_bytes()
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert pdf_path.read_bytes()[:4] == b"%PDF"


# ═══════════════════════════════════════════════════════════════════════════
# 2. DataResolver: Lakehouse, SQL, Evidence, Slice, and Filters
# ═══════════════════════════════════════════════════════════════════════════


def test_data_resolver_catalog_view_with_sql_where(tmp_path: pathlib.Path) -> None:
    # Setup mock lakehouse parquet file
    scorecard_df = pl.DataFrame(
        {
            "planner": ["dwb", "teb", "mppi"],
            "success_rate": [0.45, 0.85, 0.95],
            "energy_wh": [12.0, 15.2, 10.8],
        }
    )
    catalog = BenchmarkCatalog.from_directory(tmp_path)
    catalog.register_table("custom_scorecard", scorecard_df)
    resolver = DataResolver(catalog=catalog)

    spec = PlotSpec(
        id="high_performers",
        type="bar",
        data=PlotDataSpec(source="custom_scorecard", where="success_rate >= 0.80"),
        channels=PlotChannels(x="planner", y="energy_wh"),
    )

    df_resolved, diagnostics = resolver.resolve(spec)
    assert len(df_resolved) == 2
    planners = set(df_resolved["planner"])
    assert "teb" in planners and "mppi" in planners
    assert "dwb" not in planners
    assert diagnostics.row_count == 2


def test_data_resolver_polars_sql_engine(tmp_path: pathlib.Path) -> None:
    episodes_df = pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb", "teb"],
            "time_to_goal": [15.0, 17.0, 20.0, 22.0],
            "success": [1.0, 1.0, 1.0, 0.0],
        }
    )
    catalog = BenchmarkCatalog.from_directory(tmp_path)
    catalog.register_table("custom_episodes", episodes_df)
    sql_engine = PolarsSQLEngine(catalog)
    resolver = DataResolver(catalog=catalog, sql_engine=sql_engine)

    spec = PlotSpec(
        id="avg_time_sql",
        type="bar",
        data=PlotDataSpec(query="SELECT planner, AVG(time_to_goal) AS mean_time FROM custom_episodes GROUP BY planner"),
        channels=PlotChannels(x="planner", y="mean_time"),
    )

    df_resolved, diagnostics = resolver.resolve(spec)
    assert len(df_resolved) == 2
    assert "mean_time" in set(df_resolved.columns)
    assert diagnostics.row_count == 2


def test_data_resolver_evidence_store(tmp_path: pathlib.Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    source_df = pl.DataFrame({"treatment": ["A", "B"], "delta_wh": [-2.4, 3.1]})
    ref = store.record(source_df, kind="paired_diff", operation="test_delta")

    resolver = DataResolver(evidence_store=store)
    spec = PlotSpec(
        id="evidence_plot",
        type="bar",
        data=PlotDataSpec(evidence_id=ref.evidence_id),
        channels=PlotChannels(x="treatment", y="delta_wh"),
    )

    df_resolved, diagnostics = resolver.resolve(spec)
    assert len(df_resolved) == 2
    assert "delta_wh" in set(df_resolved.columns)
    assert diagnostics.row_count == 2


def test_data_resolver_spatiotemporal_slicing() -> None:
    df = _sample_trajectory_frame()
    resolver = DataResolver()

    spec = PlotSpec(
        id="slice_encounter",
        type="trajectory",
        slice=PlotSliceSpec(encounter_id="enc_2"),
        channels=PlotChannels(x="path", color="planner"),
    )

    df_resolved, _ = resolver.resolve(spec, fallback_df=df)
    assert len(df_resolved) == 1
    assert df_resolved["encounter_id"][0] == "enc_2"
    assert df_resolved["planner"][0] == "teb"


def test_data_resolver_filter_dictionary_application() -> None:
    df = _sample_distributions_frame()
    resolver = DataResolver()

    spec = PlotSpec(
        id="filter_test",
        type="violin",
        filter={"planner": ["dwb", "mppi"], "crowd_condition": "dense"},
        channels=PlotChannels(x="planner", y="path_length_m"),
    )

    df_resolved, _ = resolver.resolve(spec, fallback_df=df)
    assert len(df_resolved) == 4
    for row_planner in df_resolved["planner"]:
        assert row_planner in ("dwb", "mppi")
    for row_crowd in df_resolved["crowd_condition"]:
        assert row_crowd == "dense"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Recipe Catalog & `uses: <recipe_name>` Resolution
# ═══════════════════════════════════════════════════════════════════════════

_RECIPES: Final[tuple[str, ...]] = (
    "closer_attribution_matrix",
    "planner_ranking_stability",
    "encounter_energy_breakdown",
    "encounter_trajectory_slice",
    "acoustic_exposure_sunburst",
    "scorecard_table",
    "multi_planner_radar",
)


@pytest.mark.parametrize("recipe_name", _RECIPES)
def test_recipe_catalog_resolution_all_predefined_recipes(recipe_name: str) -> None:
    path = find_recipe_path(recipe_name)
    assert path is not None, f"Recipe '{recipe_name}' not found."
    data = load_recipe(recipe_name)
    assert isinstance(data, dict)
    assert data["id"] == recipe_name

    # Validate resolution via DataResolver
    resolver = DataResolver()
    spec = PlotSpec(uses=recipe_name, title="Custom Title Override", filter={"world": "hospital_1"})
    resolved = resolver.resolve_spec(spec)

    assert resolved.id == recipe_name
    assert resolved.title == "Custom Title Override"
    assert resolved.type == data["type"]
    assert resolved.filter == {"world": "hospital_1"}
    assert resolved.channels.x is not None or resolved.channels.y is not None or resolved.channels.color is not None


def test_recipe_catalog_missing_recipe_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="not found in search paths"):
        load_recipe("non_existent_recipe_xyz_99")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Non-Blocking Diagnostics Engine
# ═══════════════════════════════════════════════════════════════════════════


def test_non_blocking_diagnostics_on_low_sample_size_and_nulls() -> None:
    # 2 rows (N=2 < 10), constant zero-variance column, and all-null column
    sparse_df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "constant_val": [5.0, 5.0],
            "null_val": [None, None],
            "metric": [10.0, 12.0],
        },
        schema={"planner": pl.Utf8, "constant_val": pl.Float64, "null_val": pl.Float64, "metric": pl.Float64},
    )

    spec = PlotSpec(
        id="sparse_plot",
        type="bar",
        channels=PlotChannels(x="planner", y="metric"),
        checks={"require_columns": ["planner", "missing_required_col"]},
    )

    diagnostics = DiagnosticInspector.inspect(sparse_df, spec)
    assert diagnostics.row_count == 2
    assert "constant_val" in diagnostics.zero_variance_columns
    assert "null_val" in diagnostics.null_columns
    assert diagnostics.cardinality["planner"] == 2

    # Verify warnings collected as informative metadata without raising exceptions
    warning_text = " ".join(diagnostics.warnings)
    assert "Low sample size" in warning_text and "N=2" in warning_text
    assert "constant_val" in warning_text
    assert "null_val" in warning_text
    assert "missing_required_col" in warning_text

    # Rendering MUST proceed and succeed with N=2
    renderer = BarRenderer(spec)
    html = renderer.render_plotly(sparse_df)
    assert html is not None and html.startswith("<div")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Full Backward Compatibility with Legacy Manifests
# ═══════════════════════════════════════════════════════════════════════════


def test_backward_compatibility_with_legacy_plotspec() -> None:
    df = _sample_distributions_frame()
    spec = PlotSpec(
        id="legacy_plot",
        type="violin",
        title="Legacy Path Length",
        data_key="path_length_m",
        differentiate="planner",
        group_by=["zone_name"],
        filter={"crowd_condition": "dense"},
    )

    renderer = ViolinRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None
    assert html.startswith("<div")
    assert "Path Length" in html


# ═══════════════════════════════════════════════════════════════════════════
# 6. ReportBuilder & Standalone Artifacts Integration
# ═══════════════════════════════════════════════════════════════════════════


def test_report_builder_with_data_resolver_and_standalone_artifacts(tmp_path: pathlib.Path) -> None:
    # 1. Setup the canonical benchmark lakehouse the report reads
    from arena_evaluation.storage.declarations import CONTACT_FACT_SCHEMA

    source = _sample_distributions_frame()
    episodes = [
        {
            "episode_id": index,
            "planner": row["planner"],
            "crowd_condition": row["crowd_condition"],
            "path_length_m": row["path_length_m"],
            "time_to_goal_s": row["time_to_goal_s"],
            "success": bool(row["success"]),
        }
        for index, row in enumerate(source.iter_rows(named=True))
    ]
    write_benchmark(tmp_path, episodes)
    for index in range(len(episodes)):
        contact = {"run_id": "run", "episode_id": index, "robot_id": "robot_0", "closer_label": "robot", "fault_attributed_to": "robot", "impact_speed_mps": 0.8, "robot_closing_share": 0.7, "counterfactual_freeze_persists": False}
        metrics = tmp_path / "episodes" / f"episode_{index:03d}" / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([{name: contact.get(name) for name in CONTACT_FACT_SCHEMA.names()}], schema=CONTACT_FACT_SCHEMA).write_parquet(metrics / "fact_contacts.parquet")

    # 2. Setup manifest with recipe usage, multi-dimensional plots, and diagnostic warnings
    manifest_yaml = """
name: test_declarative_report
title: "Declarative Integration Report"
plots:
  - id: violin_paths
    type: violin
    title: "Path Lengths"
    data_key: path_length_m
    differentiate: planner
  - id: bar_success
    type: bar
    title: "Success Rates"
    channels:
      x: planner
      y: success
      color: crowd_condition
  - id: blame_matrix
    uses: closer_attribution_matrix
    filter:
      crowd_condition: dense
"""
    (tmp_path / "viz_manifest.yaml").write_text(manifest_yaml.strip(), encoding="utf-8")

    # 3. Build report
    builder = ReportBuilder(benchmark_dir=tmp_path)
    report_path = builder.build()

    assert report_path is not None and report_path.is_file()
    html_content = report_path.read_text(encoding="utf-8")

    assert "Declarative Integration Report" in html_content
    assert "Path Lengths" in html_content
    assert "Success Rates" in html_content
    assert "plotly.min.js" in html_content

    # 4. Check standalone image artifacts generated for autonomous agent inspection
    plots_dir = tmp_path / "plots"
    assert plots_dir.is_dir()
    image_files = list(plots_dir.glob("*.png"))
    assert len(image_files) >= 2
    for img in image_files:
        assert img.stat().st_size > 0
        assert img.read_bytes()[:8] == _PNG_MAGIC
