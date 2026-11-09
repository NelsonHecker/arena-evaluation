from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette

from .base import BasePlotRenderer, finish_axes, new_figure, palette_color

__all__ = ["BoxRenderer"]


class BoxRenderer(BasePlotRenderer):
    """Box plot renderer with multi-dimensional channels and faceting support."""

    PLOT_TYPE = "box"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        y_col = self.resolve_channel("y", fallback=self.spec.data_key)
        if not y_col or df_filtered.schema.get(y_col) is None:
            return None

        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        if not color_col or df_filtered.schema.get(color_col) is None:
            return None

        x_col = self.resolve_channel("x")
        if x_col is not None and df_filtered.schema.get(x_col) is None:
            x_col = None

        facet_col = self.resolve_channel("facet_col")
        facet_row = self.resolve_channel("facet_row")

        keep = [c for c in [y_col, x_col, color_col, facet_col, facet_row] if c and df_filtered.schema.get(c) is not None]
        selected = df_filtered.select(list(dict.fromkeys(keep)))
        if selected.is_empty() or selected[y_col].is_null().all():
            return None

        labels = {
            y_col: self.format_label(y_col.replace("_", " ").title(), y_col),
            color_col: color_col.lstrip("_").replace("_", " ").title(),
        }
        if x_col:
            labels[x_col] = self.format_label(x_col.replace("_", " ").title(), x_col)

        fig = px.box(
            selected,
            x=x_col,
            y=y_col,
            color=color_col,
            facet_col=facet_col,
            facet_row=facet_row,
            notched=bool(self.spec.options.get("notched", False)),
            points=self.spec.options.get("points", False),
            labels=labels,
            template="plotly_white",
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=50, r=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        y_col = self.resolve_channel("y", fallback=self.spec.data_key)
        if not y_col or df_filtered.schema.get(y_col) is None:
            return

        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        if not color_col or df_filtered.schema.get(color_col) is None:
            return

        x_col = self.resolve_channel("x")
        if x_col is not None and df_filtered.schema.get(x_col) is None:
            x_col = None

        category_col = x_col if x_col is not None else color_col
        series_col = color_col if color_col != category_col else None
        group_cols = [category_col] if series_col is None else [category_col, series_col]
        grouped = df_filtered.group_by(group_cols).agg(pl.col(y_col).cast(pl.Float64).drop_nulls().alias("values"))
        if grouped.is_empty():
            return

        lookup: dict[tuple[str, str], np.ndarray] = {}
        for row in grouped.iter_rows(named=True):
            category = str(row[category_col])
            series_name = str(row[series_col]) if series_col else ""
            lookup[(category, series_name)] = np.asarray(row["values"], dtype=float)

        categories = sorted({key[0] for key in lookup})
        series = sorted({key[1] for key in lookup}) if series_col else [""]
        palette = get_color_palette()
        fig, ax = new_figure()
        width = 0.8 / max(1, len(series))
        drawn = False
        for series_index, series_name in enumerate(series):
            positions: list[float] = []
            boxes: list[np.ndarray] = []
            for category_index, category in enumerate(categories):
                values = lookup.get((category, series_name))
                if values is None or values.size == 0:
                    continue
                positions.append(category_index + (series_index - (len(series) - 1) / 2.0) * width)
                boxes.append(values)
            if not boxes:
                continue
            drawn = True
            artists = ax.boxplot(boxes, positions=positions, widths=width, patch_artist=True, showfliers=False, label=series_name or None)
            color = palette_color(palette, series_index)
            for patch in artists["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

        if not drawn:
            plt.close(fig)
            return
        ax.set_xticks(np.arange(len(categories)), categories)
        if series_col is not None:
            ax.legend(frameon=False)
        finish_axes(
            ax,
            title=self.spec.title,
            x_label=self.format_label(category_col.lstrip("_").replace("_", " ").title(), category_col),
            y_label=self.format_label(y_col.replace("_", " ").title(), y_col),
        )
        self.save_figure(fig, out_path)
        plt.close(fig)