from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.report_builder import (
    ReportBuilder,
    _default_summary_group_cols,
    _status_table_html,
    data_source_table,
)
from arena_evaluation.presentation.viz_manifest import ManifestGroup, SummarySpec, VizManifest
from arena_evaluation.storage.schemas import PlotSpec
from star_benchmark import write_benchmark


def _rows() -> list[dict[str, object]]:
    return [
        {
            "episode_id": 0,
            "planner": "dwa",
            "stage": "s1",
            "world": "map_empty",
            "success": True,
            "time_to_goal_s": 15.0,
            "path_length_m": 10.0,
            "path_efficiency": 0.9,
            "total_energy_wh": 3.0,
            "total_collisions_ped": 0,
            "total_collisions_static": 0,
        },
        {
            "episode_id": 1,
            "planner": "mppi",
            "stage": "s1",
            "world": "map_empty",
            "success": False,
            "time_to_goal_s": 20.0,
            "path_length_m": 12.0,
            "path_efficiency": 0.7,
            "total_energy_wh": 4.0,
            "total_collisions_ped": 1,
            "total_collisions_static": 0,
        },
    ]


def _benchmark(tmp_path: pathlib.Path) -> pathlib.Path:
    return write_benchmark(tmp_path, _rows())


def test_data_sources_name_canonical_views() -> None:
    assert data_source_table(None) == "v_episode_summary"
    assert data_source_table("metrics") == "v_episode_summary"
    assert data_source_table("v_planner_scorecard") == "v_planner_scorecard"


def test_report_builder_renders_manifest_groups_and_plots(tmp_path: pathlib.Path) -> None:
    benchmark_dir = _benchmark(tmp_path)
    manifest = VizManifest(
        name="standard",
        title="Standard Benchmark Report",
        groups=[ManifestGroup(id="overview", title="Performance Overview"), ManifestGroup(id="efficiency", title="Efficiency Metrics"), ManifestGroup(id="kinematics", title="Kinematic Profiles")],
        plots=[
            PlotSpec(id="radar_overview", type="radar", title="Performance Radar", data_key="*", layout_group="kinematics"),
            PlotSpec(id="violin_path", type="violin", title="Path Length Distribution", data_key="path_length_m", differentiate="planner", layout_group="efficiency"),
            PlotSpec(id="bar_collisions", type="bar", title="Average Collisions", data_key="total_collisions_ped", differentiate="planner", layout_group="efficiency"),
        ],
    )
    report_path = ReportBuilder(benchmark_dir, manifest=manifest).build()

    assert report_path == benchmark_dir / "report.html"
    assert (benchmark_dir / "plotly.min.js").exists()
    content = report_path.read_text()
    assert "Standard Benchmark Report" in content
    assert "Efficiency Metrics" in content
    assert "Path Length Distribution" in content
    assert "Average Collisions" in content
    assert "dwa" in content and "mppi" in content


def test_report_builder_binds_a_plot_to_a_canonical_view(tmp_path: pathlib.Path) -> None:
    benchmark_dir = _benchmark(tmp_path)
    manifest = VizManifest(
        name="scorecard",
        title="Scorecard Report",
        groups=[ManifestGroup(id="efficiency", title="Efficiency Metrics")],
        plots=[
            PlotSpec(
                id="planner_scorecard",
                type="bar",
                title="Planner Scorecard",
                data={"source": "v_planner_scorecard"},
                channels={"x": "planner", "y": "episodes_count"},
                layout_group="efficiency",
            )
        ],
    )
    report_path = ReportBuilder(benchmark_dir, manifest=manifest).build()

    content = report_path.read_text()
    assert "Planner Scorecard" in content
    assert "Plot Quality Gate Rejected" not in content


def test_report_builder_refuses_an_unknown_data_source(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    benchmark_dir = _benchmark(tmp_path)
    manifest = VizManifest(name="legacy", title="Legacy", data_source="characterization_samples", plots=[])
    assert ReportBuilder(benchmark_dir, manifest=manifest).build() is None
    assert "names no Star Schema table or view" in capsys.readouterr().out


def test_summary_leads_with_planner_then_other_varying_dims() -> None:
    df = pl.DataFrame(
        {
            "local_planner": ["dwb", "dwb", "teb", "teb"],
            "stage": ["a", "b", "a", "b"],
            "success": [1.0, 0.9, 0.8, 0.6],
        }
    )
    assert _default_summary_group_cols(df) == ["local_planner", "stage"]
    assert _default_summary_group_cols(df.rename({"local_planner": "planner"}).drop("stage")) == ["planner"]


def test_summary_keeps_the_breakdown_of_a_single_planner_sweep() -> None:
    df = pl.DataFrame({"local_planner": ["dwb"] * 3, "world": ["a", "b", "c"], "success": [1.0, 0.5, 0.0]})
    assert _default_summary_group_cols(df) == ["local_planner", "world"]


def test_all_null_identity_columns_are_not_grouped_on() -> None:
    df = pl.DataFrame({"local_planner": [None, None], "robot_model": ["jackal", "husky"], "success": [1.0, 0.5]}, schema_overrides={"local_planner": pl.Utf8})
    assert _default_summary_group_cols(df) == ["robot_model"]


def test_summary_without_any_identity_aggregates_all_runs() -> None:
    df = pl.DataFrame({"success": [1.0, 0.5]})
    assert _default_summary_group_cols(df) == []

    manifest = VizManifest(name="summary_test", title="Summary Test", summary=[SummarySpec(metric="success", label="Success", format="{:.0%}")], plots=[])
    html = ReportBuilder(pathlib.Path("/nonexistent"), manifest=manifest)._generate_summary_table_manifest(df, manifest)
    assert "All runs" in html
    assert html.count("<tr") == 2
    assert "75%" in html


def test_partially_null_grouping_keys_render_as_unknown() -> None:
    df = pl.DataFrame({"local_planner": ["dwb", None], "success": [1.0, 0.5]})
    manifest = VizManifest(name="summary_test", title="Summary Test", summary=[SummarySpec(metric="success", label="Success", format="{:.0%}")], plots=[])
    html = ReportBuilder(pathlib.Path("/nonexistent"), manifest=manifest)._generate_summary_table_manifest(df, manifest)
    assert ">unknown<" in html
    assert ">None<" not in html


def test_per_plot_note_rendered_below_plot(tmp_path: pathlib.Path) -> None:
    from arena_evaluation.presentation.report_builder import _render_markdown_light

    assert "<strong>bold</strong>" in _render_markdown_light("**bold** insight")
    assert '<a href="https://x" target="_blank">link</a>' in _render_markdown_light("[link](https://x)")
    assert "<script>" not in _render_markdown_light("<script>alert(1)</script>")

    benchmark_dir = _benchmark(tmp_path)
    (benchmark_dir / "notes.yaml").write_text("- {label: Insight, value: '**Key** finding'}\n")
    manifest = VizManifest(
        name="note_test",
        title="Note Test",
        groups=[ManifestGroup(id="overview", title="Overview")],
        plots=[
            PlotSpec(id="p1", type="bar", title="P1", data_key="path_length_m", options={"note": "Inline **note**"}, layout_group="overview"),
            PlotSpec(id="p2", type="bar", title="P2", data_key="path_length_m", options={"notes_key": "Insight"}, layout_group="overview"),
        ],
    )
    content = ReportBuilder(benchmark_dir, manifest=manifest).build().read_text()
    assert content.count("class='plot-note'") == 2
    assert "Inline <strong>note</strong>" in content
    assert "<strong>Key</strong> finding" in content


def test_status_table_lists_only_unevaluated_rows() -> None:
    df = pl.DataFrame(
        {
            "stage": ["s1", "s1", "s2"],
            "planner": ["p", "p", "p"],
            "episode_id": [0, 1, 2],
            "robot_id": ["r", "r", "r"],
            "success": [True, None, None],
            "result": ["GOAL_REACHED", "COLLISION", "GOAL_REACHED"],
            "null_reason": [None, "no_samples", "no_samples"],
        }
    )
    out = _status_table_html(df)
    assert "2 of 3" in out
    assert "COLLISION" in out
    assert "<td>0</td>" not in out


def test_status_table_empty_when_every_episode_is_evaluated() -> None:
    assert _status_table_html(pl.DataFrame({"episode_id": [0], "success": [True]}, schema={"episode_id": pl.Int64, "success": pl.Boolean})) == ""


def test_report_builder_reads_a_consolidated_series_table(tmp_path: pathlib.Path) -> None:
    """A 1NF series table is a first-class catalog source: long-format samples, never list columns."""
    benchmark_dir = _benchmark(tmp_path)
    pl.DataFrame({"planner": ["dwa", "mppi"], "power_w": [10.0, 20.0]}).write_parquet(benchmark_dir / "metrics" / "series_characterization.parquet")
    manifest = VizManifest(
        name="series",
        title="Series Report",
        groups=[ManifestGroup(id="efficiency", title="Efficiency Metrics")],
        plots=[
            PlotSpec(
                id="series_power",
                type="bar",
                title="Series Power",
                data={"source": "series_characterization"},
                channels={"x": "planner", "y": "power_w"},
                layout_group="efficiency",
            )
        ],
    )
    content = ReportBuilder(benchmark_dir, manifest=manifest).build().read_text()
    assert "Series Power" in content
    assert "Plot Quality Gate Rejected" not in content


def test_report_builder_names_a_missing_series_source(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    benchmark_dir = _benchmark(tmp_path)
    manifest = VizManifest(name="series", title="Series", data_source="series_energy", plots=[])
    assert ReportBuilder(benchmark_dir, manifest=manifest).build() is None
    assert "series data source needs the pipeline to write metrics/series_*.parquet" in capsys.readouterr().out
