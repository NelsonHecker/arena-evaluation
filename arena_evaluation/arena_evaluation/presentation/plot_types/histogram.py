from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl

from .base import BasePlotRenderer

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

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        color_col = self.resolve_channel("color", fallback=diff_col)
        if not color_col or df_filtered.schema.get(color_col) is None:
            return None

        keep = [c for c in [x_col, color_col] if c and df_filtered.schema.get(c) is not None]
        pdf = df_filtered.select(list(dict.fromkeys(keep))).to_pandas().dropna(subset=[x_col])
        if pdf.empty:
            return None

        num_bins = int(self.spec.options.get("nbins", 15))
        opacity = float(self.spec.options.get("opacity", 0.6))

        global_min = float(pdf[x_col].min())
        global_max = float(pdf[x_col].max())
        if global_min == global_max:
            global_min -= 1.0
            global_max += 1.0

        bins = np.linspace(global_min, global_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        if color_col in pdf.columns:
            binned_dfs = []
            for name, group in pdf.groupby(color_col, observed=False):
                counts, _ = np.histogram(group[x_col], bins=bins)
                df_group = pd.DataFrame({"bin_center": bin_centers, "count": counts, color_col: name})
                binned_dfs.append(df_group)
            counts_df = pd.concat(binned_dfs)
            color_arg = color_col
        else:
            counts, _ = np.histogram(pdf[x_col], bins=bins)
            counts_df = pd.DataFrame({"bin_center": bin_centers, "count": counts})
            color_arg = None

        fig = px.area(
            counts_df,
            x="bin_center",
            y="count",
            color=color_arg,
            template="plotly_white",
        )

        fig.update_traces(opacity=opacity, line=dict(shape="spline", smoothing=0.8))
        fig.update_layout(
            xaxis_title=self.format_label(x_col.replace("_", " ").title(), x_col),
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        x_col = self.resolve_channel("x", fallback=self.spec.data_key)
        schema = df_filtered.schema
        if not x_col or schema.get(x_col) is None:
            return

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        color_col = self.resolve_channel("color", fallback=diff_col)
        if not color_col or df_filtered.schema.get(color_col) is None:
            return

        keep = [c for c in [x_col, color_col] if c and df_filtered.schema.get(c) is not None]
        pdf = df_filtered.select(list(dict.fromkeys(keep))).to_pandas().dropna(subset=[x_col])
        if pdf.empty:
            return

        import matplotlib.pyplot as plt

        num_bins = int(self.spec.options.get("nbins", 15))
        global_min = float(pdf[x_col].min())
        global_max = float(pdf[x_col].max())
        if global_min == global_max:
            global_min -= 1.0
            global_max += 1.0

        bins = np.linspace(global_min, global_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        if color_col in pdf.columns:
            binned_dfs = []
            for name, group in pdf.groupby(color_col, observed=False):
                counts, _ = np.histogram(group[x_col], bins=bins)
                df_group = pd.DataFrame({"bin_center": bin_centers, "count": counts, color_col: name})
                binned_dfs.append(df_group)
            counts_df = pd.concat(binned_dfs)
            hue_arg = color_col
        else:
            counts, _ = np.histogram(pdf[x_col], bins=bins)
            counts_df = pd.DataFrame({"bin_center": bin_centers, "count": counts})
            hue_arg = None

        fig, ax = plt.subplots(figsize=(10, 6))
        if hue_arg:
            for name, group in counts_df.groupby(hue_arg, observed=False):
                ax.plot(group["bin_center"], group["count"], label=name, marker="o", linewidth=2)
                ax.fill_between(group["bin_center"], group["count"], alpha=0.3)
            ax.legend()
        else:
            ax.plot(counts_df["bin_center"], counts_df["count"], marker="o", linewidth=2)
            ax.fill_between(counts_df["bin_center"], counts_df["count"], alpha=0.3)

        ax.set_title(self.spec.title)
        ax.set_xlabel(self.format_label(x_col.replace("_", " ").title(), x_col))
        ax.set_ylabel("Count")
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
