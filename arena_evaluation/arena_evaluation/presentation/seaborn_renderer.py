from __future__ import annotations

import pathlib
import polars as pl
from ..storage.schemas import PlotSpec

class SeabornRenderer:
    """Dispatches plot rendering to the correct static PNG class."""
    
    def __init__(self, generate_gifs: bool = False, units: dict[str, str] | None = None):
        self.generate_gifs = generate_gifs
        self.units = units or {}
        from .plot_types import (
            ViolinRenderer,
            BoxRenderer,
            BarRenderer,
            TrajectoryRenderer,
            RadarRenderer,
            ScatterRenderer,
            HistogramRenderer,
            HeatmapRenderer,
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
        }
        
        from .color_utils import set_global_color_palette
        set_global_color_palette()

    def render(self, spec: PlotSpec, df: pl.DataFrame, out_path: pathlib.Path, run_dir: pathlib.Path | None = None) -> None:
        renderer_cls = self.renderers.get(spec.type)
        if not renderer_cls:
            return
            
        renderer = renderer_cls(spec, units=self.units)
        if hasattr(renderer, "run_dir") or renderer_cls.__name__ == "TrajectoryRenderer":
            renderer.run_dir = run_dir
            if renderer_cls.__name__ == "TrajectoryRenderer":
                renderer.generate_gifs = self.generate_gifs
        renderer.render_seaborn(df, out_path)
