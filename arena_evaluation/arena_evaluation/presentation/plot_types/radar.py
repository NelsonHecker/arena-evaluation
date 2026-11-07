from __future__ import annotations

import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer

if typing.TYPE_CHECKING:
    import pandas as pd


_POSITIVE_METRICS = frozenset(
    {
        "success",
        "success_rate",
        "spl",
        "spl_geodesic",
        "success_geodesic",
        "osr",
        "ndtw",
        "sdtw",
        "path_efficiency",
        "relative_throughput",
        "passing_rule_compliance",
        "velocity_mean",
        "velocity_max",
    }
)


def _normalize(grouped: pd.DataFrame, metrics: list[str], *, use_log_scale: bool) -> pd.DataFrame:
    """Scale each metric so 1 is the best group, inverting the metrics where lower is better."""
    normalized = grouped.copy()
    if use_log_scale:
        for m in metrics:
            normalized[m] = np.log1p(normalized[m])

    for m in metrics:
        max_val = normalized[m].max()
        min_val = normalized[m].min()

        if max_val == 0 and min_val == 0:
            normalized[m] = 1.0
        elif m in _POSITIVE_METRICS:
            normalized[m] = normalized[m] / max_val if max_val > 0 else 1.0
        elif min_val > 0:
            normalized[m] = min_val / normalized[m]
        elif max_val == 0:
            normalized[m] = normalized[m] / min_val
        else:
            normalized[m] = 1.0 - (normalized[m] / max_val)
    return normalized


class RadarRenderer(BasePlotRenderer):
    """Radar and capability tradeoff profile renderer."""

    PLOT_TYPE = "radar"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        schema = df_filtered.schema
        if not color_col or schema.get(color_col) is None:
            return None

        metrics = self.spec.options.get("metrics", [])
        if not metrics and self.spec.channels.y:
            metrics = self.spec.channels.y if isinstance(self.spec.channels.y, list) else [self.spec.channels.y]
        if not metrics:
            metrics = ["path_efficiency", "time_to_goal", "collision_amount", "roughness_mean", "jerk_mean"]

        valid_metrics = [m for m in metrics if schema.get(m) is not None and not df_filtered[m].is_null().all()]
        if len(valid_metrics) < 3:
            return None

        grouped = df_filtered.group_by(color_col).agg([pl.col(m).mean().alias(m) for m in valid_metrics]).to_pandas()
        grouped = grouped.fillna(0.0)

        if grouped.empty:
            return None

        normalized = _normalize(grouped, valid_metrics, use_log_scale=self.spec.options.get("use_log_scale", False))

        fig = go.Figure()

        for _, row in normalized.iterrows():
            values = [row[m] for m in valid_metrics]
            values.append(values[0])
            formatted_metrics = [self.format_label(m.replace("_", " ").title(), m) for m in valid_metrics]
            labels = formatted_metrics + [formatted_metrics[0]]

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill="toself" if self.spec.options.get("fill", False) else None,
                    line=dict(width=2.5),
                    name=str(row[color_col]),
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

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        schema = df_filtered.schema
        if not color_col or schema.get(color_col) is None:
            return

        metrics = self.spec.options.get("metrics", [])
        if not metrics and self.spec.channels.y:
            metrics = self.spec.channels.y if isinstance(self.spec.channels.y, list) else [self.spec.channels.y]
        if not metrics:
            metrics = ["path_efficiency", "time_to_goal", "collision_amount", "roughness_mean", "jerk_mean"]
        valid_metrics = [m for m in metrics if schema.get(m) is not None and not df_filtered[m].is_null().all()]
        if len(valid_metrics) < 3:
            return

        grouped = df_filtered.group_by(color_col).agg([pl.col(m).mean().alias(m) for m in valid_metrics]).to_pandas()
        grouped = grouped.fillna(0.0)
        if grouped.empty:
            return

        normalized = _normalize(grouped, valid_metrics, use_log_scale=self.spec.options.get("use_log_scale", False))

        num_vars = len(valid_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        formatted_labels = [self.format_label(m.replace("_", " ").title(), m) for m in valid_metrics]

        palette = get_color_palette()
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for idx, (_, row) in enumerate(normalized.iterrows()):
            values = [row[m] for m in valid_metrics]
            values += values[:1]
            color = palette[idx % len(palette)]
            ax.plot(angles, values, linewidth=2, label=str(row[color_col]), color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(formatted_labels, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7, color="grey")

        ax.set_title(self.spec.title, size=13, pad=20, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)

        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
