"""A plot that fails to render is a failed check in the build result and the report appendix."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.checks import CheckStage
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.viz_manifest import VizManifest
from arena_evaluation.storage.schemas import PlotChannels, PlotDataSpec, PlotSpec
from star_benchmark import write_benchmark


def _benchmark(tmp_path: pathlib.Path) -> pathlib.Path:
    write_benchmark(
        tmp_path,
        [
            {"episode_id": 0, "planner": "dwb", "success": True, "time_to_goal_s": 10.0, "is_reference": False},
            {"episode_id": 1, "planner": "mppi", "success": False, "time_to_goal_s": 20.0, "is_reference": False},
        ],
    )
    return tmp_path


def _build(tmp_path: pathlib.Path, plots: list[PlotSpec]) -> tuple[ReportBuilder, str]:
    builder = ReportBuilder(benchmark_dir=_benchmark(tmp_path), manifest=VizManifest(name="checks", title="Checks", plots=plots))
    report_path = builder.build()
    assert report_path is not None and report_path.exists()
    return builder, report_path.read_text(encoding="utf-8")


def test_a_plot_whose_data_cannot_be_loaded_is_a_failed_check(tmp_path: pathlib.Path) -> None:
    builder, html = _build(tmp_path, [PlotSpec(id="no_query", type="bar", data=PlotDataSpec(stored_query="not_a_stored_query"), channels=PlotChannels(x="planner", y="success"))])

    assert builder.build_checks.plots_total == 1
    assert builder.build_checks.failed_count == 1
    assert builder.build_checks.passed_count == 0
    failures = builder.build_checks.for_plot("no_query")
    assert [failure.stage for failure in failures] == [CheckStage.LOAD]
    assert "not_a_stored_query" in failures[0].detail
    assert "Appendix — Failed Checks" in html
    assert "not_a_stored_query" in html


def test_a_static_render_failure_is_a_failed_check(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(
        id="text_as_number",
        type="violin",
        title="Text As Number",
        data=PlotDataSpec(source="v_episode_summary"),
        channels=PlotChannels(x="planner", y="planner"),
    )
    builder, html = _build(tmp_path, [spec])

    assert builder.build_checks.failed_count == 1
    failures = builder.build_checks.for_plot("text_as_number")
    assert [failure.stage for failure in failures] == [CheckStage.STATIC]
    assert "planner" in failures[0].detail
    assert "text_as_number" in html.split("Appendix — Failed Checks")[1]
    assert "<table class=\"dataframe\">" in html.split("Appendix — Failed Checks")[1]


def test_an_acoustic_error_is_a_failed_check_carrying_its_message(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(
        id="acoustic_no_map",
        type="acoustic_field",
        title="Acoustic Field",
        data=PlotDataSpec(query="SELECT 'map_that_no_world_declares' AS map, 1.0 AS ped_max_exposure_dba, 0 AS episode_id, '' AS robot_id, '' AS run_id"),
    )
    builder, html = _build(tmp_path, [spec])

    failures = builder.build_checks.for_plot("acoustic_no_map")
    assert {failure.stage for failure in failures} == {CheckStage.STATIC, CheckStage.INTERACTIVE}
    assert all("map_that_no_world_declares" in failure.detail for failure in failures)
    assert "map_that_no_world_declares" in html


def test_a_quality_gate_rejection_is_a_failed_check(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(id="all_null", type="bar", title="All Null", data=PlotDataSpec(source="v_episode_summary"), channels=PlotChannels(x="planner", y="min_ped_clearance_m"))
    builder, html = _build(tmp_path, [spec])

    failures = builder.build_checks.for_plot("all_null")
    assert [failure.stage for failure in failures] == [CheckStage.QUALITY]
    assert builder.build_checks.failed_count == 1
    assert "Plot Quality Gate Rejected" in html


def test_a_rendered_plot_counts_as_passed_and_leaves_no_appendix(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(id="good_plot", type="bar", title="Success", data=PlotDataSpec(source="v_episode_summary"), channels=PlotChannels(x="planner", y="success"))
    builder, html = _build(tmp_path, [spec])

    assert builder.build_checks.plots_total == 1
    assert builder.build_checks.failed_count == 0
    assert builder.build_checks.passed_count == 1
    assert builder.build_checks.failures == ()
    assert "Appendix — Failed Checks" not in html
    assert (tmp_path / "plots" / "good_plot.png").exists()


def test_the_build_result_reports_the_counts_of_a_run(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    plots = [
        PlotSpec(id="good", type="bar", data=PlotDataSpec(source="v_episode_summary"), channels=PlotChannels(x="planner", y="success")),
        PlotSpec(id="bad", type="bar", data=PlotDataSpec(stored_query="not_a_stored_query"), channels=PlotChannels(x="planner", y="success")),
    ]
    builder, _html = _build(tmp_path, plots)

    assert (builder.build_checks.plots_total, builder.build_checks.passed_count, builder.build_checks.failed_count) == (2, 1, 1)
    assert "Plot checks: 2 plots, 1 passed, 1 failed" in capsys.readouterr().out
