"""Unit tests for radar charts."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.radar import RadarRenderer, _normalized_matrix
from arena_evaluation.storage.schemas import PlotSpec

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_DEFAULT_METRICS = ["success", "path_efficiency", "total_collisions_ped"]


def _radar_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "success": [0.8, 1.0],
            "path_efficiency": [0.5, 0.9],
            "total_collisions_ped": [1.0, 2.0],
            "time_to_goal_s": [10.0, 20.0],
            "mean_velocity_mps": [0.7, 0.9],
        },
        schema={
            "planner": pl.Utf8,
            "success": pl.Float64,
            "path_efficiency": pl.Float64,
            "total_collisions_ped": pl.Float64,
            "time_to_goal_s": pl.Float64,
            "mean_velocity_mps": pl.Float64,
        },
    )


def _radar_spec(**options) -> PlotSpec:
    options.setdefault("metrics", list(_DEFAULT_METRICS))
    return PlotSpec(id="rad", type="radar", title="Radar Overview", data_key="*", differentiate="planner", options=options)


def test_radar_plotly_happy_path():
    html = RadarRenderer(_radar_spec()).render_plotly(_radar_df())
    assert html is not None
    assert html.startswith("<div")
    assert "scatterpolar" in html


def test_radar_plotly_fewer_than_three_metrics_returns_none():
    spec = _radar_spec(metrics=["success", "path_efficiency"])
    assert RadarRenderer(spec).render_plotly(_radar_df()) is None


def test_radar_plotly_null_metrics_are_filtered_out():
    df = _radar_df().with_columns(pl.lit(None).alias("bogus_a"), pl.lit(None).alias("bogus_b"))
    spec = _radar_spec(metrics=["success", "bogus_a", "bogus_b"])
    assert RadarRenderer(spec).render_plotly(df) is None


def test_radar_plotly_missing_diff_col_returns_none():
    assert RadarRenderer(_radar_spec()).render_plotly(_radar_df().drop("planner")) is None


def test_radar_plotly_all_null_diff_col_renders_null_group():
    df = _radar_df().with_columns(pl.lit(None, dtype=pl.Utf8).alias("planner"))
    html = RadarRenderer(_radar_spec()).render_plotly(df)
    assert html is not None and "scatterpolar" in html


@pytest.mark.parametrize(
    ("metric", "values", "expected"),
    [
        ("success", [0.0, 0.0], [1.0, 1.0]),
        ("success", [0.4, 0.8], [0.5, 1.0]),
        ("total_collisions_ped", [1.0, 2.0], [1.0, 0.5]),
        ("total_collisions_ped", [-1.0, 1.0], [2.0, 0.0]),
        ("total_collisions_ped", [-2.0, 0.0], [1.0, 0.0]),
    ],
)
def test_normalized_matrix_puts_the_best_group_at_one(metric: str, values: list[float], expected: list[float]) -> None:
    normalized = _normalized_matrix(pl.DataFrame({metric: values}), [metric], use_log_scale=False)
    assert normalized[:, 0].tolist() == pytest.approx(expected)


def test_normalized_matrix_scales_after_the_log() -> None:
    normalized = _normalized_matrix(pl.DataFrame({"success": [1.0, 3.0]}), ["success"], use_log_scale=True)
    assert normalized[:, 0].tolist() == pytest.approx([0.5, 1.0])


def test_normalized_matrix_leaves_a_missing_metric_at_zero() -> None:
    frame = pl.DataFrame({"a": [1.0, None], "b": [2.0, 4.0]}, schema={"a": pl.Float64, "b": pl.Float64})
    normalized = _normalized_matrix(frame, ["a", "b"], use_log_scale=False)
    assert normalized.shape == (2, 2)
    assert np.isfinite(normalized).all()


def test_radar_plotly_draws_the_normalized_values():
    spec = _radar_spec(metrics=["success", "path_efficiency", "total_collisions_ped"])
    html = RadarRenderer(spec).render_plotly(_radar_df())
    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    radii = {trace["name"]: dict(zip(trace["theta"], trace["r"], strict=True)) for trace in traces}
    assert radii["dwb"]["Success"] == pytest.approx(0.8)
    assert radii["teb"]["Success"] == pytest.approx(1.0)
    assert radii["dwb"]["Total Collisions Ped"] == pytest.approx(1.0)
    assert radii["teb"]["Total Collisions Ped"] == pytest.approx(0.5)


def test_radar_plotly_log_scale_option():
    spec = _radar_spec(use_log_scale=True)
    html = RadarRenderer(spec).render_plotly(_radar_df())
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_single_planner_row():
    html = RadarRenderer(_radar_spec()).render_plotly(_radar_df().slice(0, 1))
    assert html is not None and "dwb" in html


def test_radar_plotly_units_in_labels():
    html = RadarRenderer(_radar_spec(), units={"total_collisions_ped": "n"}).render_plotly(_radar_df())
    assert html is not None
    assert "Total Collisions Ped [n]" in html


def test_radar_static_happy_path(tmp_path):
    out = tmp_path / "radar.png"
    RadarRenderer(_radar_spec()).render_static(_radar_df(), out)
    assert out.exists() and out.read_bytes()[:8] == _PNG_MAGIC


def test_radar_static_missing_diff_writes_nothing(tmp_path):
    out = tmp_path / "radar.png"
    RadarRenderer(_radar_spec()).render_static(_radar_df().drop("planner"), out)
    assert not out.exists()


def test_radar_static_too_few_metrics_writes_nothing(tmp_path):
    out = tmp_path / "radar.png"
    RadarRenderer(_radar_spec(metrics=["success", "path_efficiency"])).render_static(_radar_df(), out)
    assert not out.exists()


def test_radar_static_log_scale(tmp_path):
    out = tmp_path / "radar.png"
    RadarRenderer(_radar_spec(use_log_scale=True)).render_static(_radar_df(), out)
    assert out.exists() and out.stat().st_size > 0


def test_radar_static_metric_crossing_zero(tmp_path):
    out = tmp_path / "radar.png"
    df = _radar_df().with_columns(pl.Series("total_collisions_ped", [-1.0, 1.0]))
    RadarRenderer(_radar_spec()).render_static(df, out)
    assert out.exists() and out.stat().st_size > 0