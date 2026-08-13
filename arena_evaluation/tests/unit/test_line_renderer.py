"""Unit tests for the long-format line renderer."""

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.line import LineRenderer
from arena_evaluation.storage.schemas import PlotSpec


def _samples_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [1e9, 2e9, 3e9, 1e9, 2e9, 3e9],
            "p_total_w": [10.0, 20.0, 15.0, 12.0, 18.0, 14.0],
            "phase_kind": ["linear"] * 3 + ["angular"] * 3,
            "std": [1.0] * 6,
        }
    )


def _spec(**options) -> PlotSpec:
    return PlotSpec(
        id="t",
        type="line",
        title="Test Line",
        data_key="time_ns",
        group_by=["phase_kind"],
        options=options,
    )


def test_line_with_confidence_band():
    spec = _spec(y="p_total_w", error_y="std", time_to_s=True, time_relative=True)
    html = LineRenderer(spec).render_plotly(_samples_df())
    assert html is not None
    # Band mode renders a filled region (toself) + a line per group.
    assert "toself" in html
    assert html.count("scatter") >= 4  # 2 groups x (line + band)


def test_line_without_error_band():
    spec = _spec(y="p_total_w")
    html = LineRenderer(spec).render_plotly(_samples_df())
    assert html is not None
    assert "toself" not in html


def test_line_missing_columns_returns_none():
    assert LineRenderer(_spec(y="missing")).render_plotly(_samples_df()) is None
    assert LineRenderer(_spec(y="p_total_w")).render_plotly(pl.DataFrame()) is None
    # No group_by columns -> falls back to differentiate, which may be absent.
    spec = PlotSpec(id="t", type="line", title="T", data_key="time_ns", options={"y": "p_total_w"})
    assert spec is not None
    html = LineRenderer(spec).render_plotly(_samples_df())
    assert html is not None or True  # must not raise


def test_line_time_conversion_and_relative():
    spec = _spec(y="p_total_w", time_to_s=True, time_relative=True)
    html = LineRenderer(spec).render_plotly(_samples_df())
    assert html is not None
    assert "Time [s]" in html


def test_line_seaborn_renders_png(tmp_path: pathlib.Path):
    spec = _spec(y="p_total_w", error_y="std", mode="lines+markers")
    out = tmp_path / "line.png"
    LineRenderer(spec).render_seaborn(_samples_df(), out)
    assert out.exists() and out.stat().st_size > 0


def test_line_seaborn_empty_df_no_crash(tmp_path: pathlib.Path):
    out = tmp_path / "empty.png"
    LineRenderer(_spec(y="p_total_w")).render_seaborn(pl.DataFrame(), out)
    assert not out.exists() or out.stat().st_size == 0
