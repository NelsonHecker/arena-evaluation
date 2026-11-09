from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer, label_values

__all__ = ["RadarRenderer"]

_DEFAULT_METRICS: tuple[str, ...] = (
    "path_efficiency",
    "time_to_goal_s",
    "total_collisions_ped",
    "mean_velocity_mps",
    "jerk_max_mps3",
)

_POSITIVE_METRICS = frozenset(
    {
        "success",
        "success_rate",
        "success_no_robot_fault",
        "spl",
        "spl_geodesic",
        "success_geodesic",
        "osr",
        "ndtw",
        "sdtw",
        "path_efficiency",
        "relative_throughput",
        "passing_rule_compliance",
        "mean_velocity_mps",
        "max_speed_mps",
        "mar",
        "min_obstacle_clearance_m",
        "min_ped_clearance_m",
    }
)


def _normalized_matrix(grouped: pl.DataFrame, metrics: list[str], *, use_log_scale: bool) -> np.ndarray:
    """Scale every metric so 1 is the best group, inverting the metrics where lower is better."""
    matrix = grouped.select([pl.col(metric).cast(pl.Float64).fill_null(0.0) for metric in metrics]).to_numpy().copy()
    if use_log_scale:
        matrix = np.log1p(matrix)
    for column_index, metric in enumerate(metrics):
        column = matrix[:, column_index]
        high = float(column.max())
        low = float(column.min())
        if high == 0.0 and low == 0.0:
            column[:] = 1.0
        elif metric in _POSITIVE_METRICS:
            column[:] = column / high if high > 0.0 else 1.0
        elif low > 0.0:
            column[:] = low / column
        elif high == 0.0:
            column[:] = column / low
        else:
            column[:] = 1.0 - column / high
    return matrix


def _radar_metrics(renderer: BasePlotRenderer) -> list[str]:
    options = renderer.spec.options or {}
    metrics = options.get("metrics", [])
    if not metrics and renderer.spec.channels.y:
        metrics = renderer.spec.channels.y if isinstance(renderer.spec.channels.y, list) else [renderer.spec.channels.y]
    return list(metrics) if metrics else list(_DEFAULT_METRICS)


class RadarRenderer(BasePlotRenderer):
    """Radar and capability tradeoff profile renderer."""

    PLOT_TYPE = "radar"

    def _prepare(self, df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, list[str], str] | None:
        """Grouped metric means, their normalized matrix, the metric names and the legend column."""
        df_filtered = self._apply_filters(df)
        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        schema = df_filtered.schema
        if not color_col or schema.get(color_col) is None:
            return None

        valid_metrics = [metric for metric in _radar_metrics(self) if schema.get(metric) is not None and not df_filtered[metric].is_null().all()]
        if len(valid_metrics) < 3:
            return None

        grouped = df_filtered.group_by(color_col, maintain_order=True).agg([pl.col(metric).mean().alias(metric) for metric in valid_metrics])
        if grouped.is_empty():
            return None
        normalized = _normalized_matrix(grouped, valid_metrics, use_log_scale=bool(self.spec.options.get("use_log_scale", False)))
        return grouped, normalized, valid_metrics, color_col

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        prepared = self._prepare(df)
        if prepared is None:
            return None
        grouped, normalized, metrics, color_col = prepared
        formatted_metrics = [self.format_label(metric.replace("_", " ").title(), metric) for metric in metrics]
        labels = formatted_metrics + [formatted_metrics[0]]
        names = label_values(grouped, color_col)

        fig = go.Figure()
        for row_index, name in enumerate(names):
            values = normalized[row_index].tolist()
            values.append(values[0])
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill="toself" if self.spec.options.get("fill", False) else None,
                    line=dict(width=2.5),
                    name=str(name),
                    opacity=0.85,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=50, r=50),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        prepared = self._prepare(df)
        if prepared is None:
            return
        grouped, normalized, metrics, color_col = prepared
        formatted_labels = [self.format_label(metric.replace("_", " ").title(), metric) for metric in metrics]
        angles = np.linspace(0.0, 2.0 * np.pi, len(metrics), endpoint=False).tolist()
        angles.append(angles[0])
        names = label_values(grouped, color_col)
        palette = get_color_palette()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        for row_index, name in enumerate(names):
            values = normalized[row_index].tolist()
            values.append(values[0])
            color = palette[row_index % len(palette)]
            ax.plot(angles, values, linewidth=2, label=str(name), color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(formatted_labels, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7, color="grey")
        ax.set_title(self.spec.title, size=13, pad=20, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
        self.save_figure(fig, out_path)
        plt.close(fig)
