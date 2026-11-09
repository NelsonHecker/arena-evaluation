from __future__ import annotations

import dataclasses
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl

from .base import BasePlotRenderer, label_values, new_figure

__all__ = ["HeatmapRenderer"]

_CORRELATION_CANDIDATES: tuple[str, ...] = (
    "success",
    "time_to_goal",
    "path_length",
    "path_efficiency",
    "total_energy_wh",
    "mean_velocity_mps",
    "max_acceleration_mps2",
    "jerk_max_mps3",
    "min_obstacle_clearance_m",
    "min_ped_clearance_m",
    "total_collisions_ped",
    "total_collisions_static",
    "mar",
    "pfi",
    "robot_path_dev_m",
    "total_acoustic_dose_pa2s",
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _HeatmapMatrix:
    """One rendered heatmap: matrix values, axis tick labels and the value label."""

    values: np.ndarray
    x_labels: list[str]
    y_labels: list[str]
    value_label: str


class HeatmapRenderer(BasePlotRenderer):
    """2D matrix, correlation, and rank-swap heatmap renderer."""

    PLOT_TYPE = "heatmap"

    def _is_correlation(self) -> bool:
        data_key = self.spec.data_key or ""
        return data_key == "*" or data_key.lower() == "correlation" or self.spec.options.get("kind") == "correlation"

    def _correlation_matrix(self, df_filtered: pl.DataFrame) -> _HeatmapMatrix | None:
        options = self.spec.options or {}
        schema = df_filtered.schema
        candidates: list[str]
        if options.get("metrics"):
            candidates = list(options["metrics"])
        elif self.spec.channels.y:
            candidates = self.spec.channels.y if isinstance(self.spec.channels.y, list) else [self.spec.channels.y]
        else:
            candidates = list(_CORRELATION_CANDIDATES)

        valid_cols = [column for column in candidates if schema.get(column) is not None and not df_filtered[column].is_null().all()]
        if len(valid_cols) < 2:
            valid_cols = [column for column, dtype in schema.items() if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32) and not df_filtered[column].is_null().all()]
        if len(valid_cols) < 2:
            return None

        correlation = df_filtered.select(valid_cols).corr()
        labels = [column.replace("_", " ").title() for column in correlation.columns]
        return _HeatmapMatrix(values=correlation.to_numpy(), x_labels=labels, y_labels=labels, value_label="Correlation")

    def _pivot_matrix(self, df_filtered: pl.DataFrame) -> _HeatmapMatrix | None:
        options = self.spec.options or {}
        x_col = self.resolve_channel("x", fallback=str(options.get("x", "planner")))
        y_col = self.resolve_channel("y", fallback=str(options.get("y", "crowd_condition")))
        metric = self.resolve_channel("color", fallback=self.spec.data_key)

        schema = df_filtered.schema
        if not metric or schema.get(metric) is None or not x_col or schema.get(x_col) is None or not y_col or schema.get(y_col) is None:
            return None

        grouped = df_filtered.group_by([y_col, x_col]).agg(pl.col(metric).cast(pl.Float64).mean().alias("val"))
        if grouped.is_empty():
            return None

        pivot = grouped.pivot(on=x_col, index=y_col, values="val")
        x_labels = [str(column) for column in pivot.columns if column != y_col]
        y_labels = [str(label) for label in label_values(pivot, y_col)]
        values = pivot.select(x_labels).to_numpy()
        return _HeatmapMatrix(values=values, x_labels=x_labels, y_labels=y_labels, value_label=self.format_label(metric.replace("_", " ").title(), metric))

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return None

        matrix = self._correlation_matrix(df_filtered) if self._is_correlation() else self._pivot_matrix(df_filtered)
        if matrix is None:
            return None

        scale = "RdBu" if self._is_correlation() else str(self.spec.options.get("colorscale", "Viridis"))
        fig = px.imshow(
            matrix.values,
            x=matrix.x_labels,
            y=matrix.y_labels,
            color_continuous_scale=scale,
            color_continuous_midpoint=0 if self._is_correlation() else None,
            range_color=[-1, 1] if self._is_correlation() else None,
            labels=dict(color=matrix.value_label),
            aspect="auto",
        )
        if matrix.values.size <= 100:
            fig.update_traces(texttemplate="%{z:.2f}")
        fig.update_layout(
            template="plotly_white",
            height=max(520, len(matrix.y_labels) * 40),
            yaxis=dict(tickmode="linear", dtick=1),
            xaxis=dict(tickmode="linear", dtick=1),
            margin=dict(t=40, b=40, l=50, r=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return

        matrix = self._correlation_matrix(df_filtered) if self._is_correlation() else self._pivot_matrix(df_filtered)
        if matrix is None:
            return

        if self._is_correlation():
            cmap = "RdBu_r"
            limits: dict[str, float] = {"vmin": -1.0, "vmax": 1.0}
        else:
            cmap = str(self.spec.options.get("colorscale", "viridis")).lower()
            cmap = cmap if cmap in plt.colormaps() else "viridis"
            finite = matrix.values[np.isfinite(matrix.values)]
            limits = {"vmin": float(finite.min()), "vmax": float(finite.max())} if finite.size else {}

        fig, ax = new_figure(width_in=10.0, height_in=8.0)
        image = ax.imshow(matrix.values, cmap=cmap, aspect="auto", **limits)
        fig.colorbar(image, ax=ax, label=matrix.value_label)

        annotate = bool(self.spec.options.get("annotate", True))
        if annotate and matrix.values.size <= 100:
            for row_index in range(matrix.values.shape[0]):
                for column_index in range(matrix.values.shape[1]):
                    ax.text(column_index, row_index, f"{matrix.values[row_index, column_index]:.2f}", ha="center", va="center", fontsize=8)

        ax.set_xticks(np.arange(len(matrix.x_labels)), matrix.x_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(matrix.y_labels)), matrix.y_labels)
        ax.set_title(self.spec.title, pad=12, fontsize=12, fontweight="bold")
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
