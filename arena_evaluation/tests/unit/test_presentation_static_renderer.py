"""Unit tests for the static-artifact dispatcher (StaticRenderer)."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from PIL import Image

from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.storage.schemas import PlotChannels, PlotSpec

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_ALL_PLOT_TYPES = [
    "violin",
    "box",
    "bar",
    "trajectory",
    "radar",
    "scatter",
    "histogram",
    "heatmap",
    "line",
    "acoustic_field",
    "acoustic_field_animation",
    "timeseries",
    "table",
    "hierarchy",
    "dendrogram",
]


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb", "teb"],
            "episode_id": [1, 2, 1, 2],
            "success": [True, False, True, True],
            "time_to_goal_s": [40.0, 90.0, 38.0, 52.0],
            "path_efficiency": [0.9, 0.5, 0.8, 0.7],
            "total_energy_wh": [10.0, 20.0, 11.0, 13.0],
            "total_collisions_ped": [0, 2, 0, 1],
        }
    )


def _spec(plot_type: str, **channels: object) -> PlotSpec:
    return PlotSpec(id=f"p_{plot_type}", type=plot_type, title=f"Title {plot_type}", data_key="success", channels=PlotChannels(**channels))


def _assert_png(path: pathlib.Path) -> tuple[int, int]:
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == _PNG_MAGIC
    with Image.open(path) as image:
        return image.size


def test_all_plot_types_registered() -> None:
    assert set(StaticRenderer().renderers) == set(_ALL_PLOT_TYPES)


def test_units_and_gif_flags_default_off() -> None:
    renderer = StaticRenderer()
    assert renderer.generate_gifs is False
    assert renderer.units == {}
    assert StaticRenderer(generate_gifs=True, units={"success": "%"}).generate_gifs is True
    assert StaticRenderer(units={"success": "%"}).units == {"success": "%"}


def test_render_unknown_type_writes_nothing(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "nothing.png"
    spec = PlotSpec(id="x", type="bogus_type", title="X", data_key="success")
    assert StaticRenderer().render(spec, _df(), out) is None
    assert not out.exists()


@pytest.mark.parametrize(
    ("plot_type", "channels", "options"),
    [
        ("bar", {}, {}),
        ("box", {}, {}),
        ("violin", {}, {}),
        ("histogram", {"x": "time_to_goal_s"}, {}),
        ("scatter", {"x": "time_to_goal_s", "y": "time_to_goal_s"}, {}),
        ("line", {"x": "episode_id", "y": "time_to_goal_s"}, {"aggregate": True}),
        ("radar", {"color": "planner"}, {"metrics": ["path_efficiency", "total_energy_wh", "total_collisions_ped"]}),
        ("heatmap", {}, {"metrics": ["success", "time_to_goal_s", "path_efficiency"]}),
        ("timeseries", {"x": "episode_id", "y": "time_to_goal_s"}, {}),
        ("table", {"x": "planner"}, {"columns": [{"metric": "success", "label": "Success", "format": "{:.0%}"}]}),
    ],
)
def test_dispatch_writes_a_png_of_the_requested_plot(tmp_path: pathlib.Path, plot_type: str, channels: dict, options: dict) -> None:
    out = tmp_path / f"{plot_type}.png"
    spec = _spec(plot_type, **channels)
    spec = spec.model_copy(update={"options": options, "data_key": "*" if plot_type in {"radar", "heatmap"} else "success"})
    StaticRenderer().render(spec, _df(), out)
    width, height = _assert_png(out)
    assert min(width, height) > 100


@pytest.mark.parametrize(("suffix", "magic"), [("svg", b"<svg"), ("pdf", b"%PDF")])
def test_dispatch_honours_the_requested_vector_format(tmp_path: pathlib.Path, suffix: str, magic: bytes) -> None:
    out = tmp_path / f"bar.{suffix}"
    StaticRenderer().render(_spec("bar"), _df(), out)
    payload = out.read_bytes()
    assert out.stat().st_size > 0
    assert magic in payload[:400] if suffix == "svg" else payload.startswith(magic)