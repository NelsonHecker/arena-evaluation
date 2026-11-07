from __future__ import annotations

import pathlib

import plotly.express as px
import polars as pl

from .base import BasePlotRenderer

__all__ = ["ScatterRenderer"]


class ScatterRenderer(BasePlotRenderer):
    """Scatter plot renderer supporting multi-dimensional channels (size, symbol, facets)."""

    PLOT_TYPE = "scatter"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        x_col = self.resolve_channel("x", fallback=self.spec.data_key)
        y_col = self.resolve_channel("y", fallback=self.spec.options.get("y"))

        schema = df_filtered.schema
        if not y_col or not x_col or schema.get(x_col) is None or schema.get(y_col) is None:
            return None

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None
        color_col = self.resolve_channel("color", fallback=diff_col)

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
        if list_cols:
            df_filtered = df_filtered.select(unique_keep).explode(list_cols)
        else:
            df_filtered = df_filtered.select(unique_keep)
        df_filtered = self._apply_row_filters(df_filtered)

        pdf = df_filtered.to_pandas()
        if pdf.empty or bool(pdf[y_col].isna().all()):
            return None

        max_points = int(self.spec.options.get("max_points_per_trace", 5000))
        if max_points > 0 and len(pdf) > max_points:
            stride = max(1, len(pdf) // max_points)
            pdf = pdf.iloc[::stride]

        labels = {
            x_col: self.format_label(x_col.replace("_", " ").title(), x_col),
            y_col: self.format_label(y_col.replace("_", " ").title(), y_col),
        }
        if color_col:
            labels[color_col] = color_col.lstrip("_").replace("_", " ").title()

        fig = px.scatter(
            pdf,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            symbol=symbol_col,
            facet_col=facet_col,
            facet_row=facet_row,
            labels=labels,
            template="plotly_white",
            opacity=0.7,
        )

        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)

        x_col = self.resolve_channel("x", fallback=self.spec.data_key)
        y_col = self.resolve_channel("y", fallback=self.spec.options.get("y"))

        schema = df_filtered.schema
        if not y_col or not x_col or schema.get(x_col) is None or schema.get(y_col) is None:
            return

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return
        color_col = self.resolve_channel("color", fallback=diff_col)

        size_col = self.resolve_channel("size", fallback=self.spec.options.get("size"))
        if size_col and df_filtered.schema.get(size_col) is None:
            size_col = None

        symbol_col = self.resolve_channel("symbol", fallback=self.spec.options.get("symbol"))
        if symbol_col and df_filtered.schema.get(symbol_col) is None:
            symbol_col = None

        keep = [c for c in [x_col, y_col, color_col, size_col, symbol_col] if c and df_filtered.schema.get(c) is not None]
        unique_keep = list(dict.fromkeys(keep))
        list_cols = [c for c in unique_keep if df_filtered.schema[c] == pl.List]
        if list_cols:
            df_filtered = df_filtered.select(unique_keep).explode(list_cols)
        else:
            df_filtered = df_filtered.select(unique_keep)
        df_filtered = self._apply_row_filters(df_filtered)

        pdf = df_filtered.to_pandas()
        if pdf.empty or bool(pdf[y_col].isna().all()):
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=pdf,
            x=x_col,
            y=y_col,
            hue=color_col,
            size=size_col,
            style=symbol_col,
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(self.spec.title)
        ax.set_xlabel(self.format_label(x_col.replace("_", " ").title(), x_col))
        ax.set_ylabel(self.format_label(y_col.replace("_", " ").title(), y_col))
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
