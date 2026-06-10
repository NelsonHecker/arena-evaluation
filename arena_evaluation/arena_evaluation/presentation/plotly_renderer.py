from __future__ import annotations

import polars as pl
from ..storage.schemas import PlotSpec

class PlotlyRenderer:
    """Dispatches plot rendering to the correct Plotly class."""
    
    def __init__(self):
        from .plot_types import (
            ViolinRenderer,
            BoxRenderer,
            BarRenderer,
            TrajectoryRenderer,
            RadarRenderer,
            ScatterRenderer,
            HistogramRenderer,
        )
        
        self.renderers = {
            "violin": ViolinRenderer,
            "box": BoxRenderer,
            "bar": BarRenderer,
            "trajectory": TrajectoryRenderer,
            "radar": RadarRenderer,
            "scatter": ScatterRenderer,
            "histogram": HistogramRenderer,
        }

    def render(self, spec: PlotSpec, df: pl.DataFrame, run_dir: pathlib.Path | None = None) -> str | list[str] | None:
        renderer_cls = self.renderers.get(spec.type)
        if not renderer_cls:
            return None
            
        renderer = renderer_cls(spec)
        if hasattr(renderer, "run_dir") or renderer_cls.__name__ == "TrajectoryRenderer":
            renderer.run_dir = run_dir
        return renderer.render_plotly(df)
