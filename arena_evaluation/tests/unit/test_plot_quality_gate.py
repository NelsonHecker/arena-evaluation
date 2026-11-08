"""Unit tests for the automated plot quality gate and agent visual inspection loop."""

from __future__ import annotations

import json
import pathlib

import polars as pl

from arena_evaluation.presentation.checks import DiagnosticInspector
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.viz_manifest import VizManifest
from arena_evaluation.storage.schemas import PlotChannels, PlotSpec


def test_empty_dataframe_rejection() -> None:
    """Empty DataFrames must fail the quality gate with zero row diagnostics."""
    empty_df = pl.DataFrame(
        {
            "planner": pl.Series([], dtype=pl.Utf8),
            "metric": pl.Series([], dtype=pl.Float64),
        }
    )
    spec = PlotSpec(
        id="empty_plot",
        type="bar",
        channels=PlotChannels(x="planner", y="metric"),
    )

    diagnostics = DiagnosticInspector.inspect(empty_df, spec)

    assert diagnostics.status == "FAILED"
    assert diagnostics.sample_count == 0
    assert diagnostics.row_count == 0
    assert "Resolved DataFrame contains 0 rows" in diagnostics.issues


def test_all_null_column_rejection() -> None:
    """Columns where values are 100% null must be rejected with FAILED status."""
    null_df = pl.DataFrame(
        {
            "planner": [f"planner_{i}" for i in range(12)],
            "metric": [None] * 12,
        },
        schema={"planner": pl.Utf8, "metric": pl.Float64},
    )
    spec = PlotSpec(
        id="null_plot",
        type="bar",
        channels=PlotChannels(x="planner", y="metric"),
    )

    diagnostics = DiagnosticInspector.inspect(null_df, spec)

    assert diagnostics.status == "FAILED"
    assert diagnostics.null_ratio == 1.0
    assert any("100% NULL across all 12 rows" in issue for issue in diagnostics.issues)


def test_all_nan_column_rejection() -> None:
    """Continuous float columns that are 100% NaN must be treated as null and rejected."""
    nan_df = pl.DataFrame(
        {
            "planner": [f"planner_{i}" for i in range(12)],
            "metric": [float("nan")] * 12,
        },
        schema={"planner": pl.Utf8, "metric": pl.Float64},
    )
    spec = PlotSpec(
        id="nan_plot",
        type="bar",
        channels=PlotChannels(x="planner", y="metric"),
    )

    diagnostics = DiagnosticInspector.inspect(nan_df, spec)

    assert diagnostics.status == "FAILED"
    assert diagnostics.null_ratio == 1.0
    assert any("100% NULL across all 12 rows" in issue for issue in diagnostics.issues)


def test_zero_variance_warning_detection() -> None:
    """Constant continuous metrics must trigger zero_variance flag and WARNING status."""
    zero_var_df = pl.DataFrame(
        {
            "planner": [f"planner_{i}" for i in range(15)],
            "metric": [42.0] * 15,
        },
        schema={"planner": pl.Utf8, "metric": pl.Float64},
    )
    spec = PlotSpec(
        id="zero_var_plot",
        type="scatter",
        channels=PlotChannels(x="planner", y="metric"),
    )

    diagnostics = DiagnosticInspector.inspect(zero_var_df, spec)

    assert diagnostics.status == "WARNING"
    assert diagnostics.zero_variance is True
    assert diagnostics.null_ratio == 0.0
    assert diagnostics.sample_count == 15
    assert any("zero variance" in issue.lower() for issue in diagnostics.issues)


def test_valid_data_pass_state() -> None:
    """Sufficient samples with variance and valid values must pass with PASSED status."""
    valid_df = pl.DataFrame(
        {
            "planner": [f"planner_{i}" for i in range(15)],
            "metric": [float(i * 1.5) for i in range(15)],
        },
        schema={"planner": pl.Utf8, "metric": pl.Float64},
    )
    spec = PlotSpec(
        id="valid_plot",
        type="bar",
        channels=PlotChannels(x="planner", y="metric"),
    )

    diagnostics = DiagnosticInspector.inspect(valid_df, spec)

    assert diagnostics.status == "PASSED"
    assert diagnostics.zero_variance is False
    assert diagnostics.null_ratio == 0.0
    assert diagnostics.sample_count == 15
    assert len(diagnostics.issues) == 0


def test_missing_required_columns_rejection() -> None:
    """Missing required columns from spec.checks must trigger FAILED status."""
    df = pl.DataFrame(
        {
            "planner": [f"planner_{i}" for i in range(12)],
            "metric": [float(i) for i in range(12)],
        }
    )
    spec = PlotSpec(
        id="req_check_plot",
        type="bar",
        channels=PlotChannels(x="planner", y="metric"),
        checks={"require_columns": ["planner", "speed_limit_mps"]},
    )

    diagnostics = DiagnosticInspector.inspect(df, spec)

    assert diagnostics.status == "FAILED"
    assert any("speed_limit_mps" in issue for issue in diagnostics.issues)


def test_quality_gate_failed_card_in_report_and_skips_static(tmp_path: pathlib.Path) -> None:
    """Failed quality gate renders an HTML rejection card and skips static rendering."""
    df = pl.DataFrame(
        {
            "planner": ["dwb", "mppi"],
            "success": [1.0, 0.5],
            "broken_metric": [float("nan"), float("nan")],
        }
    )
    df.write_parquet(tmp_path / "combined_metrics.parquet")

    manifest = VizManifest(
        plots=[
            PlotSpec(
                id="plot_broken",
                type="bar",
                title="Broken Metric Plot",
                channels=PlotChannels(x="planner", y="broken_metric"),
            )
        ]
    )

    builder = ReportBuilder(benchmark_dir=tmp_path, manifest=manifest)
    report_file = builder.build()

    assert report_file is not None and report_file.exists()
    content = report_file.read_text(encoding="utf-8")

    assert "plot-gate-card gate-failed" in content
    assert "[Plot Quality Gate Rejected]" in content
    assert "100% NULL" in content

    # Static plot must NOT be created when gate fails
    static_png = tmp_path / "plots" / "plot_broken.png"
    assert not static_png.exists()


def test_agent_annotation_from_notes_json(tmp_path: pathlib.Path) -> None:
    """Agent insights from notes.json are rendered in HTML callouts below the plot."""
    df = pl.DataFrame(
        {
            "planner": [f"planner_{i}" for i in range(12)],
            "path_length": [float(10.0 + i) for i in range(12)],
        }
    )
    df.write_parquet(tmp_path / "combined_metrics.parquet")

    notes = {
        "plot_path_length": "DWB shows highest SPL despite near-miss surge",
    }
    _ = (tmp_path / "notes.json").write_text(json.dumps(notes), encoding="utf-8")

    manifest = VizManifest(
        plots=[
            PlotSpec(
                id="plot_path_length",
                type="bar",
                title="Path Length by Planner",
                channels=PlotChannels(x="planner", y="path_length"),
            )
        ]
    )

    builder = ReportBuilder(benchmark_dir=tmp_path, manifest=manifest)
    report_file = builder.build()

    assert report_file is not None and report_file.exists()
    content = report_file.read_text(encoding="utf-8")

    assert '<div class="agent-annotation">' in content
    assert '<span class="badge">Agent Insight</span>' in content
    assert "DWB shows highest SPL despite near-miss surge" in content


def test_agent_annotation_from_insights_json_list(tmp_path: pathlib.Path) -> None:
    """Agent insights from insights.json list structure are rendered in HTML callouts."""
    df = pl.DataFrame(
        {
            "planner": [f"planner_{i}" for i in range(12)],
            "time_to_goal": [float(5.0 + i) for i in range(12)],
        }
    )
    df.write_parquet(tmp_path / "combined_metrics.parquet")

    insights = [
        {
            "plot_id": "plot_ttg",
            "insight": "Regulated Pure Pursuit exhibits minimal corridor oscillation",
        }
    ]
    _ = (tmp_path / "insights.json").write_text(json.dumps(insights), encoding="utf-8")

    manifest = VizManifest(
        plots=[
            PlotSpec(
                id="plot_ttg",
                type="bar",
                title="Time to Goal",
                channels=PlotChannels(x="planner", y="time_to_goal"),
            )
        ]
    )

    builder = ReportBuilder(benchmark_dir=tmp_path, manifest=manifest)
    report_file = builder.build()

    assert report_file is not None and report_file.exists()
    content = report_file.read_text(encoding="utf-8")

    assert '<div class="agent-annotation">' in content
    assert '<span class="badge">Agent Insight</span>' in content
    assert "Regulated Pure Pursuit exhibits minimal corridor oscillation" in content
