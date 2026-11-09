from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.storage.schemas import PlotSpec


class StaticRenderer:
    """Dispatches a plot specification to the renderer class that draws its static artifact."""

    def __init__(self, *, generate_gifs: bool = False, units: dict[str, str] | None = None) -> None:
        self.generate_gifs = generate_gifs
        self.units = units or {}
        from .plot_types import (
            AcousticFieldAnimationRenderer,
            AcousticFieldRenderer,
            BarRenderer,
            BoxRenderer,
            HeatmapRenderer,
            HierarchyRenderer,
            HistogramRenderer,
            LineRenderer,
            RadarRenderer,
            ScatterRenderer,
            TableRenderer,
            TimeseriesRenderer,
            TrajectoryRenderer,
            ViolinRenderer,
        )

        self.renderers = {
            "violin": ViolinRenderer,
            "box": BoxRenderer,
            "bar": BarRenderer,
            "trajectory": TrajectoryRenderer,
            "radar": RadarRenderer,
            "scatter": ScatterRenderer,
            "histogram": HistogramRenderer,
            "heatmap": HeatmapRenderer,
            "timeseries": TimeseriesRenderer,
            "line": LineRenderer,
            "table": TableRenderer,
            "acoustic_field": AcousticFieldRenderer,
            "acoustic_field_animation": AcousticFieldAnimationRenderer,
            "hierarchy": HierarchyRenderer,
            "dendrogram": HierarchyRenderer,
        }

        from .color_utils import set_global_color_palette

        set_global_color_palette()

    def render(self, spec: PlotSpec, df: pl.DataFrame, out_path: pathlib.Path, run_dir: pathlib.Path | None = None) -> None:
        renderer_cls = self.renderers.get(spec.type)
        if not renderer_cls:
            return

        renderer = renderer_cls(spec, units=self.units)
        renderer.run_dir = run_dir
        if renderer_cls.__name__ == "TrajectoryRenderer":
            renderer.generate_gifs = self.generate_gifs
        renderer.render_static(df, out_path)
