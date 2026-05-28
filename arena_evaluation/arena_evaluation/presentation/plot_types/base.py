from __future__ import annotations

from abc import ABC, abstractmethod
import polars as pl
import pathlib

from ...storage.schemas import PlotSpec


class BasePlotRenderer(ABC):
    """
    Abstract Base Class for all plot renderers.
    """
    PLOT_TYPE: str = ""

    def __init__(self, spec: PlotSpec):
        self.spec = spec

    @abstractmethod
    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """
        Render interactive plot using Plotly.
        Returns the HTML string representation of the plot.
        """
        pass

    @abstractmethod
    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """
        Render static plot using Seaborn/Matplotlib.
        Saves the PNG to out_path.
        """
        pass
        
    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply filters defined in the PlotSpec."""
        if not self.spec.filter:
            return df
            
        res_df = df
        for k, v in self.spec.filter.items():
            if k in res_df.columns:
                res_df = res_df.filter(pl.col(k) == v)
        return res_df
