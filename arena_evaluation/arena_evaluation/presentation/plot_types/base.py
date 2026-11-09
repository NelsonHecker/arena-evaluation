from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from arena_evaluation.presentation.dimension_detector import resolve_differentiate
from arena_evaluation.storage.schemas import PlotSpec

if TYPE_CHECKING:
    import matplotlib.figure

    from arena_evaluation.presentation.checks import PlotDiagnostics

__all__ = [
    "BasePlot",
    "BasePlotRenderer",
    "finish_axes",
    "label_values",
    "new_figure",
    "numeric_values",
    "palette_color",
    "series_values",
]


def numeric_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """Finite Float64 values of one column as a NumPy array, nulls and NaN dropped."""
    return frame[column].cast(pl.Float64).fill_nan(None).drop_nulls().to_numpy()


def series_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """Float64 values of one column, NaN where the value is null, length preserved."""
    return frame[column].cast(pl.Float64).to_numpy()


def label_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """String labels of one column as a NumPy array, nulls rendered as ``unknown``."""
    return frame[column].cast(pl.Utf8).fill_null("unknown").to_numpy()


def new_figure(*, width_in: float = 10.0, height_in: float = 6.0) -> tuple[plt.Figure, plt.Axes]:
    """A fresh Matplotlib figure and its single axes."""
    return plt.subplots(figsize=(width_in, height_in))


def finish_axes(ax: plt.Axes, *, title: str, x_label: str, y_label: str, grid_axis: str = "y") -> None:
    """Apply the shared static-plot style and axis labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5, axis=grid_axis)
    ax.set_title(title, pad=12, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def palette_color(palette: Sequence[str], index: int) -> str:
    """The palette color at ``index``, cycling through the palette."""
    return palette[index % len(palette)]


class BasePlotRenderer(ABC):
    """Base class for all plot renderers."""

    PLOT_TYPE: str = ""

    def __init__(self, spec: PlotSpec, units: dict[str, str] | None = None):
        self.spec = spec
        self.units = units or {}
        self.run_dir: pathlib.Path | None = None
        self.diagnostics: PlotDiagnostics | None = None

    def format_label(self, label: str, data_key: str) -> str:
        """Format the label with the unit associated with data_key."""
        unit = self.units.get(data_key)
        if unit:
            return f"{label} [{unit}]"
        return label

    def resolve_channel(self, channel: str, fallback: str | None = None) -> str | None:
        """Resolve a channel string name (e.g. 'x', 'y', 'color', 'facet_col', 'size', 'symbol')."""
        ch = self.spec.channels
        val: str | list[str] | None = None
        match channel:
            case "x":
                val = ch.x
            case "y":
                val = ch.y
            case "color" | "hue":
                val = ch.color
            case "facet_col":
                val = ch.facet_col
            case "facet_row":
                val = ch.facet_row
            case "size":
                val = ch.size
            case "symbol":
                val = ch.symbol
            case "hierarchy":
                val = ch.hierarchy
            case _:
                val = None

        if val is not None:
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str) and val[0].strip():
                return val[0].strip()
        return fallback

    def resolve_channel_list(self, channel: str, fallback: list[str] | str | None = None) -> list[str]:
        """Resolve a channel that can be a list of columns (e.g. multi-metric y)."""
        ch = self.spec.channels
        val: str | list[str] | None = None
        match channel:
            case "x":
                val = ch.x
            case "y":
                val = ch.y
            case "color" | "hue":
                val = ch.color
            case "hierarchy":
                val = ch.hierarchy
            case _:
                val = None

        if isinstance(val, list):
            return [str(v).strip() for v in val if v]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        if fallback is not None:
            return [fallback.strip()] if isinstance(fallback, str) else [str(f).strip() for f in fallback if f]
        return []

    @staticmethod
    def save_figure(fig: matplotlib.figure.Figure, out_path: pathlib.Path, dpi: int = 300) -> None:
        """Save matplotlib figure to vector (PDF/SVG) or high-res image (PNG)."""
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    @abstractmethod
    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        """Render an interactive Plotly plot as an HTML string, or a list of them."""
        pass

    @abstractmethod
    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib plot to ``out_path`` as PNG, SVG or PDF."""
        pass

    def resolve_diff_col(self, df: pl.DataFrame) -> tuple[str, pl.DataFrame]:
        return resolve_differentiate(self.spec, df)

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply scalar and sequence filter predicates from PlotSpec."""
        if not self.spec.filter:
            return df

        res_df = df
        schema = res_df.schema
        for k, v in self.spec.filter.items():
            if schema.get(k) is None:
                continue
            if schema[k] == pl.List:
                if isinstance(v, (list, tuple, set)):
                    v_list = list(v)
                    if v_list:
                        res_df = res_df.filter(pl.any_horizontal([pl.col(k).list.contains(x) for x in v_list]))
                    else:
                        res_df = res_df.filter(pl.lit(False))
                else:
                    res_df = res_df.filter(pl.col(k).list.contains(v))
            elif isinstance(v, (list, tuple, set)):
                res_df = res_df.filter(pl.col(k).is_in(list(v)))
            else:
                res_df = res_df.filter(pl.col(k) == v)
        return res_df

    def _apply_row_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply filters to scalar/exploded columns after list expansion."""
        if not self.spec.filter:
            return df

        res_df = df
        schema = res_df.schema
        for k, v in self.spec.filter.items():
            if schema.get(k) is not None and schema[k] != pl.List:
                if isinstance(v, (list, tuple, set)):
                    res_df = res_df.filter(pl.col(k).is_in(list(v)))
                else:
                    res_df = res_df.filter(pl.col(k) == v)
        return res_df


BasePlot = BasePlotRenderer