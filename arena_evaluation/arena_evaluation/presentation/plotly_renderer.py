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
        )
        
        self.renderers = {
            "violin": ViolinRenderer,
            "box": BoxRenderer,
            "bar": BarRenderer,
            "trajectory": TrajectoryRenderer,
            "radar": RadarRenderer,
        }

    def render(self, spec: PlotSpec, df: pl.DataFrame) -> str | None:
        renderer_cls = self.renderers.get(spec.type)
        if not renderer_cls:
            return None
            
        renderer = renderer_cls(spec)
        return renderer.render_plotly(df)
