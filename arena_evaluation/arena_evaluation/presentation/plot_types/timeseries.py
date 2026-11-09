from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import plotly.express as px
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer, finish_axes

__all__ = ["TimeseriesRenderer"]


class TimeseriesRenderer(BasePlotRenderer):
    """Timeseries renderer for 1NF long-format temporal curves (speed, energy, power, acoustics)."""

    PLOT_TYPE = "timeseries"

    def _resolve(self, df: pl.DataFrame) -> tuple[pl.DataFrame, str, list[str], str | None] | None:
        """Filtered frame, its time column, the plotted value columns and the legend column."""
        df_filtered = self._apply_filters(df)

        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        options = self.spec.options or {}
        metrics = options.get("metrics", [])
        if not metrics:
            channel_y = self.spec.channels.y
            if channel_y:
                metrics = channel_y if isinstance(channel_y, list) else [channel_y]
            elif self.spec.data_key:
                metrics = [self.spec.data_key]
            else:
                return None

        schema = df_filtered.schema
        valid_metrics = [str(metric) for metric in metrics if schema.get(str(metric)) is not None]
        if not valid_metrics:
            return None

        x_col = self.resolve_channel("x", fallback=str(options.get("x", "time_s")))
        if not x_col or schema.get(x_col) is None:
            return None

        if schema.get("is_reference") is not None and not options.get("include_reference", False):
            df_filtered = df_filtered.filter(~pl.col("is_reference").fill_null(False))
            if df_filtered.is_empty():
                return None

        return df_filtered, x_col, valid_metrics, color_col

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        prepared = self._resolve(df)
        if prepared is None:
            return None
        df_filtered, x_col, metrics, color_col = prepared

        fig = px.line(
            df_filtered,
            x=x_col,
            y=metrics[0] if len(metrics) == 1 else metrics,
            color=color_col if color_col and color_col != x_col else None,
            template="plotly_white",
            labels={x_col: self.format_label(x_col.replace("timeseries_", "").replace("_", " ").title(), x_col)},
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=50, r=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        prepared = self._resolve(df)
        if prepared is None:
            return
        df_filtered, x_col, metrics, color_col = prepared
        palette = get_color_palette()
        dashes = ["-", "--", ":", "-."]

        fig, ax = plt.subplots(figsize=(10, 6))
        if color_col and color_col != x_col and df_filtered.schema.get(color_col) is not None:
            grouped = df_filtered.partition_by(color_col, maintain_order=True, as_dict=True)
            groups = [(str(key[0]), frame) for key, frame in grouped.items()]
        else:
            groups = [("", df_filtered)]

        plotted = False
        for group_index, (group_name, frame) in enumerate(groups):
            ordered = frame.sort(x_col)
            times = ordered[x_col].cast(pl.Float64).to_numpy()
            for metric_index, metric in enumerate(metrics):
                values = ordered[metric].cast(pl.Float64).to_numpy()
                label = group_name if len(metrics) == 1 else f"{group_name} · {metric.replace('_', ' ')}"
                ax.plot(times, values, label=label or None, color=palette[group_index % len(palette)], linestyle=dashes[metric_index % len(dashes)])
                plotted = True
        if not plotted:
            plt.close(fig)
            return

        finish_axes(
            ax,
            title=self.spec.title,
            x_label=self.format_label(x_col.replace("timeseries_", "").replace("_", " ").title(), x_col),
            y_label=self.format_label(metrics[0].replace("timeseries_", "").replace("_", " ").title(), metrics[0]),
            grid_axis="both",
        )
        ax.legend(frameon=False)
        self.save_figure(fig, out_path)
        plt.close(fig)