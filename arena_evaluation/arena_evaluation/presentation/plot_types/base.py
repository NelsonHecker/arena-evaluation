# SESSION SNAPSHOT (2026-08-10) — base.py with run_dir attr + list-filter support.
# Path: src/Arena/arena_evaluation/arena_evaluation/arena_evaluation/presentation/plot_types/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import polars as pl
import pathlib

from ...storage.schemas import PlotSpec
from ..dimension_detector import resolve_differentiate


class BasePlotRenderer(ABC):
    """
    Abstract Base Class for all plot renderers.
    """
    PLOT_TYPE: str = ""

    def __init__(self, spec: PlotSpec, units: dict[str, str] | None = None):
        self.spec = spec
        self.units = units or {}
        self.run_dir: pathlib.Path | None = None

    def format_label(self, label: str, data_key: str) -> str:
        """Format the label with the unit associated with data_key."""
        unit = self.units.get(data_key)
        if unit:
            return f"{label} [{unit}]"
        return label

    @abstractmethod
    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        """
        Render interactive plot using Plotly.
        Returns the HTML string representation of the plot, or a list of HTML strings.
        """
        pass

    @abstractmethod
    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """
        Render static plot using Seaborn/Matplotlib.
        Saves the PNG to out_path.
        """
        pass

    def resolve_diff_col(self, df: pl.DataFrame) -> tuple[str, pl.DataFrame]:
        return resolve_differentiate(self.spec, df)

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply filters defined in the PlotSpec.

        Scalar values match equality; list/tuple/set values match membership
        (e.g. ``filter: {planner: [dwb, teb]}`` selects only those runs).
        """
        if not self.spec.filter:
            return df

        res_df = df
        for k, v in self.spec.filter.items():
            if k not in res_df.columns:
                continue
            if isinstance(v, (list, tuple, set)):
                res_df = res_df.filter(pl.col(k).is_in(list(v)))
            else:
                res_df = res_df.filter(pl.col(k) == v)
        return res_df
