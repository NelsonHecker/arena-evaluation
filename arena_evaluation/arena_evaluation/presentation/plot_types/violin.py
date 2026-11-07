from __future__ import annotations

import pathlib

import plotly.express as px
import polars as pl

from .base import BasePlotRenderer

__all__ = ["ViolinRenderer"]


class ViolinRenderer(BasePlotRenderer):
    """Violin plot renderer with multi-dimensional channels and split distributions."""

    PLOT_TYPE = "violin"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        y_col = self.resolve_channel("y", fallback=self.spec.data_key)
        if not y_col or df_filtered.schema.get(y_col) is None:
            return None

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        color_col = self.resolve_channel("color", fallback=diff_col)
        if not color_col or df_filtered.schema.get(color_col) is None:
            return None

        x_col = self.resolve_channel("x")
        if x_col is not None and df_filtered.schema.get(x_col) is None:
            x_col = None

        facet_col = self.resolve_channel("facet_col")
        facet_row = self.resolve_channel("facet_row")

        keep = [c for c in [y_col, x_col, color_col, facet_col, facet_row] if c and df_filtered.schema.get(c) is not None]
        pdf = df_filtered.select(list(dict.fromkeys(keep))).to_pandas()
        if pdf.empty or bool(pdf[y_col].isna().all()):
            return None

        box_opt = bool(self.spec.options.get("box", True))
        points_opt = self.spec.options.get("points", "all")

        labels = {
            y_col: self.format_label(y_col.replace("_", " ").title(), y_col),
            color_col: color_col.lstrip("_").replace("_", " ").title(),
        }
        if x_col:
            labels[x_col] = self.format_label(x_col.replace("_", " ").title(), x_col)

        fig = px.violin(
            pdf,
            x=x_col,
            y=y_col,
            color=color_col,
            facet_col=facet_col,
            facet_row=facet_row,
            box=box_opt,
            points=points_opt,
            labels=labels,
            template="plotly_white",
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        y_col = self.resolve_channel("y", fallback=self.spec.data_key)
        if not y_col or df_filtered.schema.get(y_col) is None:
            return

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        color_col = self.resolve_channel("color", fallback=diff_col)
        if not color_col or df_filtered.schema.get(color_col) is None:
            return

        x_col = self.resolve_channel("x")
        if x_col is not None and df_filtered.schema.get(x_col) is None:
            x_col = None

        keep = [c for c in [y_col, x_col, color_col] if c and df_filtered.schema.get(c) is not None]
        pdf = df_filtered.select(list(dict.fromkeys(keep))).to_pandas()
        if pdf.empty or bool(pdf[y_col].isna().all()):
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        split_opt = self.spec.channels.split or bool(self.spec.options.get("split", False))
        box_opt = bool(self.spec.options.get("box", True))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=pdf,
            x=x_col,
            y=y_col,
            hue=color_col,
            split=split_opt if (x_col and color_col and split_opt) else False,
            inner="box" if box_opt else None,
            ax=ax,
        )
        ax.set_title(self.spec.title)
        ax.set_ylabel(self.format_label(y_col.replace("_", " ").title(), y_col))
        if x_col:
            ax.set_xlabel(self.format_label(x_col.replace("_", " ").title(), x_col))
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
