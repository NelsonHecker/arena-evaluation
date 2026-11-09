from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette

from .base import BasePlotRenderer, finish_axes, new_figure, palette_color, series_values

__all__ = ["ScatterRenderer"]

_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


class ScatterRenderer(BasePlotRenderer):
    """Scatter plot renderer supporting multi-dimensional channels (size, symbol, facets)."""

    PLOT_TYPE = "scatter"

    def _prepare(self, df: pl.DataFrame) -> tuple[pl.DataFrame, str, str, str | None, str | None, str | None, str | None, str | None] | None:
        """Filtered frame plus the resolved x, y, color, size, symbol and facet channels."""
        df_filtered = self._apply_filters(df)

        x_col = self.resolve_channel("x", fallback=self.spec.data_key)
        y_col = self.resolve_channel("y", fallback=self.spec.options.get("y"))

        schema = df_filtered.schema
        if not y_col or not x_col or schema.get(x_col) is None or schema.get(y_col) is None:
            return None

        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            if df_filtered.schema.get(diff_col) is None:
                return None
            color_col = diff_col

        size_col = self.resolve_channel("size", fallback=self.spec.options.get("size"))
        if size_col and df_filtered.schema.get(size_col) is None:
            size_col = None

        symbol_col = self.resolve_channel("symbol", fallback=self.spec.options.get("symbol"))
        if symbol_col and df_filtered.schema.get(symbol_col) is None:
            symbol_col = None

        facet_col = self.resolve_channel("facet_col")
        if facet_col and df_filtered.schema.get(facet_col) is None:
            facet_col = None

        facet_row = self.resolve_channel("facet_row")
        if facet_row and df_filtered.schema.get(facet_row) is None:
            facet_row = None

        filter_keys = [k for k in (self.spec.filter or {}) if df_filtered.schema.get(k) is not None]
        keep = [c for c in [x_col, y_col, color_col, size_col, symbol_col, facet_col, facet_row, *filter_keys] if c and df_filtered.schema.get(c) is not None]
        unique_keep = list(dict.fromkeys(keep))
        list_cols = [c for c in unique_keep if df_filtered.schema[c] == pl.List]
        selected = df_filtered.select(unique_keep).explode(list_cols) if list_cols else df_filtered.select(unique_keep)
        selected = self._apply_row_filters(selected)

        max_points = int(self.spec.options.get("max_points_per_trace", 5000))
        if max_points > 0 and len(selected) > max_points:
            selected = selected.gather_every(max(1, len(selected) // max_points))

        if selected.is_empty() or selected[y_col].is_null().all():
            return None
        return selected, x_col, y_col, color_col, size_col, symbol_col, facet_col, facet_row

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        prepared = self._prepare(df)
        if prepared is None:
            return None
        selected, x_col, y_col, color_col, size_col, symbol_col, facet_col, facet_row = prepared
        labels = {
            x_col: self.format_label(x_col.replace("_", " ").title(), x_col),
            y_col: self.format_label(y_col.replace("_", " ").title(), y_col),
        }
        if color_col:
            labels[color_col] = color_col.lstrip("_").replace("_", " ").title()

        fig = px.scatter(
            selected,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            symbol=symbol_col,
            facet_col=facet_col,
            facet_row=facet_row,
            labels=labels,
            template="plotly_white",
            opacity=0.75,
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=50, r=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        prepared = self._prepare(df)
        if prepared is None:
            return
        selected, x_col, y_col, color_col, size_col, symbol_col, _facet_col, _facet_row = prepared
        plotted = selected.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
        if plotted.is_empty():
            return

        palette = get_color_palette()
        fig, ax = new_figure()
        symbol_markers = self._symbol_markers(plotted, symbol_col)

        if color_col:
            groups = plotted.partition_by(color_col, as_dict=True)
            labels = [str(key[0]) for key in groups]
        else:
            groups = {(): plotted}
            labels = [""]

        for index, (label, group) in enumerate(zip(labels, groups.values(), strict=True)):
            sizes = self._size_scale(group, size_col) if size_col else 30.0
            marker = "o"
            if symbol_col:
                symbols = group[symbol_col].drop_nulls().unique().to_list()
                marker = symbol_markers.get(str(symbols[0]), "o") if symbols else "o"
            ax.scatter(
                series_values(group, x_col),
                series_values(group, y_col),
                s=sizes,
                alpha=0.75,
                marker=marker,
                label=label or None,
                color=palette_color(palette, index),
            )
        if color_col:
            ax.legend(frameon=False)
        finish_axes(
            ax,
            title=self.spec.title,
            x_label=self.format_label(x_col.replace("_", " ").title(), x_col),
            y_label=self.format_label(y_col.replace("_", " ").title(), y_col),
            grid_axis="both",
        )
        self.save_figure(fig, out_path)
        plt.close(fig)

    @staticmethod
    def _symbol_markers(plotted: pl.DataFrame, symbol_col: str | None) -> dict[str, str]:
        """Marker glyph per symbol value, cycling the glyph table."""
        if not symbol_col:
            return {}
        names = sorted(str(name) for name in plotted[symbol_col].drop_nulls().unique().to_list())
        return {name: _MARKERS[index % len(_MARKERS)] for index, name in enumerate(names)}

    def _size_scale(self, plotted: pl.DataFrame, size_col: str | None) -> float | np.ndarray:
        if not size_col:
            return 30.0
        sizes = series_values(plotted, size_col)
        if sizes.size == 0:
            return 30.0
        span = float(np.nanmax(sizes) - np.nanmin(sizes))
        if span == 0.0:
            return 30.0
        return 20.0 + 180.0 * (sizes - np.nanmin(sizes)) / span