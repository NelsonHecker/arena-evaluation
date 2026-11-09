from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette

from .base import BasePlotRenderer, finish_axes, new_figure, numeric_values, palette_color

__all__ = ["HistogramRenderer"]


class HistogramRenderer(BasePlotRenderer):
    """Histogram and density curve renderer supporting multi-dimensional channels."""

    PLOT_TYPE = "histogram"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        x_col = self.resolve_channel("x", fallback=self.spec.data_key)
        schema = df_filtered.schema
        if not x_col or schema.get(x_col) is None:
            return None

        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        if not color_col or df_filtered.schema.get(color_col) is None:
            return None

        binned = self._bin_counts(df_filtered, x_col, color_col)
        if binned is None:
            return None
        centers, series = binned
        color_arg = color_col if len(series) > 1 or series[0][0] else None
        counts_df = (
            pl.concat([pl.DataFrame({"bin_center": centers, "count": counts, color_col: label}) for label, counts in series])
            if color_arg
            else pl.DataFrame({"bin_center": centers, "count": series[0][1]})
        )

        fig = px.area(counts_df, x="bin_center", y="count", color=color_arg, template="plotly_white")
        fig.update_traces(opacity=float(self.spec.options.get("opacity", 0.6)), line=dict(shape="spline", smoothing=0.8))
        fig.update_layout(
            xaxis_title=self.format_label(x_col.replace("_", " ").title(), x_col),
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=50, r=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def _bin_counts(self, df_filtered: pl.DataFrame, x_col: str, color_col: str) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]] | None:
        """Shared bin centers and per-color counts for the histogram and the density curve."""
        values = numeric_values(df_filtered, x_col)
        if values.size == 0:
            return None

        num_bins = int(self.spec.options.get("nbins", 15))
        low = float(values.min())
        high = float(values.max())
        if low == high:
            low -= 1.0
            high += 1.0
        bins = np.linspace(low, high, num_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2.0

        grouped = df_filtered.group_by(color_col, maintain_order=True).agg(pl.col(x_col).cast(pl.Float64).drop_nulls().alias("values"))
        series: list[tuple[str, np.ndarray]] = []
        for row in grouped.iter_rows(named=True):
            counts, _ = np.histogram(np.asarray(row["values"], dtype=float), bins=bins)
            series.append((str(row[color_col]), counts))
        return centers, series

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        x_col = self.resolve_channel("x", fallback=self.spec.data_key)
        schema = df_filtered.schema
        if not x_col or schema.get(x_col) is None:
            return

        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        if not color_col or df_filtered.schema.get(color_col) is None:
            return

        binned = self._bin_counts(df_filtered, x_col, color_col)
        if binned is None:
            return
        centers, series = binned
        palette = get_color_palette()

        fig, ax = new_figure()
        for index, (label, counts) in enumerate(series):
            color = palette_color(palette, index)
            ax.plot(centers, counts, label=label or None, marker="o", linewidth=2, color=color)
            ax.fill_between(centers, counts, alpha=0.25, color=color)
        if any(label for label, _ in series):
            ax.legend(frameon=False)
        finish_axes(
            ax,
            title=self.spec.title,
            x_label=self.format_label(x_col.replace("_", " ").title(), x_col),
            y_label="Count",
            grid_axis="both",
        )
        self.save_figure(fig, out_path)
        plt.close(fig)