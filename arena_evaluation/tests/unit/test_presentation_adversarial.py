"""Adversarial presentation tests: hostile frames and manifests must never crash the report."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import VizManifest
from arena_evaluation.storage.schemas import PlotSpec
from star_benchmark import write_benchmark


def _build(benchmark_dir: pathlib.Path, rows: list[dict[str, object]], plots: list[PlotSpec]) -> None:
    write_benchmark(benchmark_dir, rows)
    ReportBuilder(benchmark_dir, manifest=VizManifest(plots=plots)).build()


def test_report_builder_empty_star_schema(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A benchmark with no episodes still assembles a report with a rejected-plot card."""
    _build(tmp_path, [], [PlotSpec(id="test_bar", type="bar", title="Empty", data_key="path_length_m")])

    assert "failed quality gate: Resolved DataFrame contains 0 rows" in capsys.readouterr().out
    assert (tmp_path / "report.html").exists()


def test_report_builder_skips_a_plot_with_a_missing_metric(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing metric fails its own plot only; the report and the other plots still build."""
    rows = [
        {"episode_id": 0, "planner": "A", "path_length_m": 10.0},
        {"episode_id": 1, "planner": "B", "path_length_m": 12.0},
    ]
    _build(
        tmp_path,
        rows,
        [
            PlotSpec(id="test1", type="bar", title="T", data_key="path_length_m", differentiate="planner"),
            PlotSpec(id="test2", type="violin", title="T", data_key="missing_col", differentiate="planner"),
        ],
    )

    out = capsys.readouterr().out
    assert "Plot 'test2' failed quality gate" in out
    assert "Plot 'test1' failed quality gate" not in out
    assert (tmp_path / "report.html").exists()


def test_renderers_survive_an_empty_frame(tmp_path: pathlib.Path) -> None:
    """Fuzzer: directly feed an empty frame to every renderer."""
    df = pl.DataFrame()
    plotly_renderer = PlotlyRenderer()
    static_renderer = StaticRenderer()

    for plot_type in ["violin", "box", "bar", "trajectory", "radar", "scatter", "histogram", "heatmap", "timeseries", "line", "table"]:
        spec = PlotSpec(id=f"test_{plot_type}", type=plot_type, title="T", data_key="k")
        plotly_renderer.render(spec, df)
        static_renderer.render(spec, df, tmp_path / f"{plot_type}.png")


def test_report_builder_handles_many_rows(tmp_path: pathlib.Path) -> None:
    """20k rows go through the catalog without the report path materialising pandas."""
    rows = [{"episode_id": index, "planner": "A" if index % 2 else "B", "path_length_m": float(index % 5), "success": index % 2 == 0} for index in range(20_000)]
    _build(tmp_path, rows, [PlotSpec(id="huge_box", type="box", title="Huge", data_key="path_length_m", differentiate="planner")])

    assert (tmp_path / "plots" / "huge_box.png").exists()
    assert (tmp_path / "report.html").exists()


def test_conflicting_dimensions_still_render_a_report(tmp_path: pathlib.Path) -> None:
    """A differentiate column or group key that does not exist must not abort the report."""
    rows = [
        {"episode_id": 0, "planner": "A", "path_length_m": 1.0},
        {"episode_id": 1, "planner": "B", "path_length_m": 2.0},
        {"episode_id": 2, "planner": "C", "path_length_m": 3.0},
    ]
    _build(
        tmp_path,
        rows,
        [
            PlotSpec(id="test_missing_diff", type="bar", title="T", data_key="path_length_m", differentiate="non_existent_diff"),
            PlotSpec(id="test_missing_group", type="bar", title="T", data_key="path_length_m", group_by=["missing_group"]),
        ],
    )

    assert (tmp_path / "report.html").exists()


def test_static_box_renders_each_planner(tmp_path: pathlib.Path) -> None:
    df = pl.DataFrame({"planner": ["A", "B"], "val": [1, 2]})
    png_path = tmp_path / "test.png"

    StaticRenderer().render(PlotSpec(id="test", type="box", title="T", data_key="val", differentiate="planner"), df, png_path)

    assert png_path.stat().st_size > 0


def test_malformed_metric_types_do_not_raise() -> None:
    """Data keys pointing at unexpected types still fail only inside their own render."""
    df = pl.DataFrame(
        {
            "planner": ["A", "B"],
            "val": ["string1", "string2"],
            "path": [{"x": 1}, {"y": 2}],
        }
    )

    plotly_renderer = PlotlyRenderer()
    plotly_renderer.render(PlotSpec(id="test_violin", type="violin", title="T", data_key="val", differentiate="planner"), df)
    plotly_renderer.render(PlotSpec(id="test_traj", type="trajectory", title="T", data_key="path", differentiate="planner"), df)
