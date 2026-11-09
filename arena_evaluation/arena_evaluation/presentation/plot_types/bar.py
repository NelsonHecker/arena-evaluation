from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl
from plotly.graph_objects import Figure

from arena_evaluation.presentation.color_utils import get_color_palette

from .base import BasePlotRenderer, finish_axes, label_values, new_figure, palette_color, series_values

__all__ = ["BarRenderer"]


class BarRenderer(BasePlotRenderer):
    """Bar plot renderer supporting grouped, stacked, and multi-metric charts."""

    PLOT_TYPE = "bar"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y", fallback=self.spec.data_key)
        color_col = self.resolve_channel("color")
        facet_col = self.resolve_channel("facet_col")
        facet_row = self.resolve_channel("facet_row")

        if not x_col and not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            x_col = diff_col
            color_col = diff_col
        elif not x_col and color_col:
            x_col = color_col

        if not x_col or df_filtered.schema.get(x_col) is None:
            return None

        is_stacked = bool(self.spec.options.get("stacked", False) or self.spec.options.get("barmode") == "stack")
        barmode = self.spec.options.get("barmode", "stack" if is_stacked else "group")

        if self.spec.options.get("stacked", False) and not self.spec.options.get("metrics"):
            if not isinstance(self.spec.channels.y, list) or not self.spec.channels.y:
                return None

        metrics = self.spec.options.get("metrics", [])
        if not metrics and isinstance(self.spec.channels.y, list):
            metrics = self.spec.channels.y

        # Mode A: Multi-metric composition
        if metrics:
            schema = df_filtered.schema
            valid_metrics = [m for m in metrics if schema.get(m) is not None]
            if not valid_metrics:
                return None

            agg_exprs = [pl.col(m).mean().alias(m) for m in valid_metrics]
            grouped = df_filtered.group_by(x_col).agg(agg_exprs)
            if grouped.is_empty():
                return None

            if self.spec.options.get("normalize", True):
                total = pl.sum_horizontal([pl.col(m) for m in valid_metrics])
                grouped = grouped.with_columns([(pl.col(m) / total * 100.0).fill_nan(None).alias(m) for m in valid_metrics])

            fig = px.bar(
                grouped,
                x=x_col,
                y=valid_metrics,
                template="plotly_white",
                barmode=barmode,
                labels={
                    "value": "Percentage (%)" if self.spec.options.get("normalize", True) else "Value",
                    "variable": "Component",
                    x_col: x_col.lstrip("_").replace("_", " ").title(),
                },
            )
            return self._finish_plotly(fig)

        schema = df_filtered.schema
        if not y_col or schema.get(y_col) is None:
            return None

        # Mode B: Categorical label counting
        if schema[y_col] in (pl.Utf8, pl.Categorical):
            grp = [c for c in [x_col, color_col, facet_col, facet_row] if c and schema.get(c) is not None]
            unique_grp = list(dict.fromkeys(grp))
            count_df = df_filtered.group_by(unique_grp).agg(pl.len().alias("count"))
            if is_stacked and self.spec.options.get("normalize", False):
                count_df = count_df.with_columns((pl.col("count") / pl.col("count").sum().over(x_col) * 100.0).alias("share_pct"))
                plot_y = "share_pct"
                y_label = "Share (%)"
            else:
                plot_y = "count"
                y_label = "Count"

            if count_df.is_empty():
                return None

            fig = px.bar(
                count_df,
                x=x_col,
                y=plot_y,
                color=color_col if color_col != x_col else None,
                facet_col=facet_col,
                facet_row=facet_row,
                barmode=barmode,
                template="plotly_white",
                labels={
                    plot_y: y_label,
                    x_col: x_col.lstrip("_").replace("_", " ").title(),
                },
            )
            return self._finish_plotly(fig)

        # Mode C: Numeric metric (pre-aggregated or raw observations)
        grp = [c for c in [x_col, color_col, facet_col, facet_row] if c and schema.get(c) is not None]
        unique_grp = list(dict.fromkeys(grp))

        max_rows_per_group = df_filtered.group_by(unique_grp).agg(pl.len().alias("cnt"))["cnt"].max() if unique_grp else 1
        is_pre_aggregated = max_rows_per_group == 1

        labels = {
            y_col: self.format_label(y_col.replace("_", " ").title(), y_col),
            x_col: x_col.lstrip("_").replace("_", " ").title(),
        }
        if color_col:
            labels[color_col] = color_col.lstrip("_").replace("_", " ").title()

        if is_pre_aggregated:
            selected = df_filtered.select(list(dict.fromkeys(unique_grp + [y_col])))
            fig = px.bar(
                selected,
                x=x_col,
                y=y_col,
                color=color_col if color_col != x_col else None,
                facet_col=facet_col,
                facet_row=facet_row,
                barmode=barmode,
                template="plotly_white",
                labels=labels,
            )
        else:
            grouped = self._grouped_with_error(df_filtered, unique_grp, y_col)
            if grouped.is_empty():
                return None
            labels["mean"] = labels[y_col]
            fig = px.bar(
                grouped,
                x=x_col,
                y="mean",
                color=color_col if color_col and color_col in grouped.schema and color_col != x_col else None,
                facet_col=facet_col,
                facet_row=facet_row,
                barmode=barmode,
                error_y="err_plus",
                error_y_minus="err_minus",
                template="plotly_white",
                labels=labels,
            )

        return self._finish_plotly(fig)

    @staticmethod
    def _finish_plotly(fig: Figure) -> str:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=50, r=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def _grouped_with_error(self, df_filtered: pl.DataFrame, group_col: str | list[str], metric_col: str) -> pl.DataFrame:
        """Compute per-group mean with vectorized error margins (Wilson score for bool, std otherwise)."""
        z = 1.96
        key = metric_col
        schema = df_filtered.schema
        cols = [group_col] if isinstance(group_col, str) else group_col
        if schema[key] == pl.Boolean:
            p = pl.col(key).cast(pl.Float64).mean()
            n = pl.col(key).count()
            denom = 1.0 + (z * z) / n
            center = (p + (z * z) / (2.0 * n)) / denom
            radicand = (p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)
            half = z * radicand.sqrt() / denom
            return df_filtered.group_by(cols).agg(
                [
                    p.alias("mean"),
                    (center + half - p).clip(lower_bound=0.0).alias("err_plus"),
                    (p - (center - half)).clip(lower_bound=0.0).alias("err_minus"),
                ]
            )

        return df_filtered.group_by(cols).agg(
            [
                pl.col(key).mean().alias("mean"),
                pl.col(key).std().fill_null(0.0).alias("err_plus"),
                pl.col(key).std().fill_null(0.0).alias("err_minus"),
            ]
        )

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)

        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y", fallback=self.spec.data_key)
        color_col = self.resolve_channel("color")

        if not x_col and not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            x_col = diff_col
            color_col = diff_col
        elif not x_col and color_col:
            x_col = color_col

        if not x_col or df_filtered.schema.get(x_col) is None:
            return

        metrics = self.spec.options.get("metrics", [])
        if not metrics and isinstance(self.spec.channels.y, list):
            metrics = self.spec.channels.y

        fig, ax = new_figure()
        palette = get_color_palette()
        drawn = self._draw_multi_metric(ax, df_filtered, x_col, metrics, palette) if metrics else (self._draw_single_metric(ax, df_filtered, x_col, y_col, color_col, palette) if y_col else False)
        if not drawn:
            plt.close(fig)
            return

        finish_axes(
            ax,
            title=self.spec.title,
            x_label=self.format_label(x_col.lstrip("_").replace("_", " ").title(), x_col),
            y_label=self.format_label(y_col.replace("_", " ").title(), y_col) if y_col else "Value",
        )
        self.save_figure(fig, out_path)
        plt.close(fig)

    def _draw_multi_metric(self, ax: plt.Axes, df_filtered: pl.DataFrame, x_col: str, metrics: list[str], palette: list[str]) -> bool:
        schema = df_filtered.schema
        valid_metrics = [m for m in metrics if schema.get(m) is not None]
        if not valid_metrics:
            return False
        grouped = df_filtered.group_by(x_col).agg([pl.col(m).mean().alias(m) for m in valid_metrics])
        if grouped.is_empty():
            return False
        if self.spec.options.get("normalize", True):
            total = pl.sum_horizontal([pl.col(m) for m in valid_metrics])
            grouped = grouped.with_columns([(pl.col(m) / total * 100.0).fill_nan(None).alias(m) for m in valid_metrics])

        categories = label_values(grouped, x_col)
        positions = np.arange(len(categories))
        stacks = [np.nan_to_num(series_values(grouped, m)) for m in valid_metrics]
        ax.stackplot(positions, *stacks, labels=[m.replace("_", " ").title() for m in valid_metrics], colors=[palette_color(palette, i) for i in range(len(stacks))])
        ax.set_xticks(positions, categories)
        ax.legend(frameon=False)
        return True

    def _draw_single_metric(self, ax: plt.Axes, df_filtered: pl.DataFrame, x_col: str, y_col: str, color_col: str | None, palette: list[str]) -> bool:
        schema = df_filtered.schema
        if schema.get(y_col) is None or schema.get(x_col) is None:
            return False
        split_by_color = bool(color_col and color_col != x_col and schema.get(color_col) is not None)
        color_name = color_col if split_by_color else None
        group_cols = [x_col, color_name] if color_name else [x_col]
        grouped = self._grouped_with_error(df_filtered, group_cols, y_col)
        if grouped.is_empty():
            return False

        if color_name:
            mean_matrix = grouped.pivot(on=color_name, index=x_col, values="mean")
            error_matrix = grouped.pivot(on=color_name, index=x_col, values="err_plus")
            series_labels = sorted(name for name in mean_matrix.columns if name != x_col)
        else:
            mean_matrix = grouped
            error_matrix = grouped
            series_labels = [""]

        categories = label_values(mean_matrix, x_col)
        positions = np.arange(len(categories))
        width = 0.8 / len(series_labels)
        stacked = bool(self.spec.options.get("stacked", False) or self.spec.options.get("barmode") == "stack")

        bottom = np.zeros(len(categories))
        for index, label in enumerate(series_labels):
            column = label if split_by_color else "mean"
            means = np.nan_to_num(series_values(mean_matrix, column))
            errors = np.nan_to_num(series_values(error_matrix, column))
            color = palette_color(palette, index)
            if stacked:
                ax.bar(positions, means, 0.8, bottom=bottom, label=label or None, color=color)
                bottom = bottom + means
            else:
                offset = (index - (len(series_labels) - 1) / 2.0) * width
                ax.bar(positions + offset, means, width, yerr=errors, capsize=2, label=label or None, color=color)

        ax.set_xticks(positions, categories)
        if split_by_color:
            ax.legend(frameon=False)
        return True
