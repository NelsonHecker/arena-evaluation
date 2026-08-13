"""Unit tests for the declarative table plot type."""

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.table import TableRenderer, _load_notes
from arena_evaluation.storage.schemas import PlotSpec


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "local_planner": ["dwb", "teb", "dwb", "teb"],
            "success": [1.0, 0.5, 0.75, 0.25],
            "time_to_goal": [15.0, 20.0, 18.0, 22.0],
        }
    )


def _spec(**options) -> PlotSpec:
    return PlotSpec(id="t", type="table", title="Overview", data_key="*", options=options)


def test_data_derived_rows():
    spec = _spec(
        group_by=["local_planner"],
        columns=[
            {"metric": "success", "label": "Success", "format": "{:.0%}"},
            {"metric": "time_to_goal", "label": "Avg Time", "format": "{:.1f}"},
        ],
    )
    html = TableRenderer(spec).render_plotly(_df())
    assert html is not None
    assert "<table" in html
    assert "dwb" in html and "teb" in html
    assert "88%" in html  # mean of 1.0 and 0.75
    assert "38%" in html  # teb mean of 0.5 and 0.25
    assert "Avg Time" in html


def test_notes_from_file(tmp_path: pathlib.Path):
    notes = tmp_path / "notes.yaml"
    notes.write_text("- {label: Conclusion, value: DWB wins}\n- {label: Best planner, value: dwb}\n")
    renderer = TableRenderer(_spec(notes="notes.yaml"))
    renderer.run_dir = tmp_path
    html = renderer.render_plotly(_df())
    assert html is not None
    assert "Conclusion" in html and "DWB wins" in html
    assert "Best planner" in html


def test_notes_inline_list():
    renderer = TableRenderer(_spec(notes=[{"label": "Run", "value": "20260808-x"}, {"label": "n", "value": "3"}]))
    html = renderer.render_plotly(_df())
    assert html is not None
    assert "20260808-x" in html and "3" in html


def test_load_notes_text_lines():
    rows = _load_notes("First: 1\nSecond: two\n# comment\nplain line", None)
    assert {"label": "First", "value": "1"} in rows
    assert {"label": "Second", "value": "two"} in rows
    assert {"label": "", "value": "plain line"} in rows


def test_empty_df_no_crash():
    assert TableRenderer(_spec()).render_plotly(pl.DataFrame()) is None
    # No group_by/columns -> no data rows, no notes -> None.
    assert TableRenderer(_spec()).render_plotly(_df()) is None
