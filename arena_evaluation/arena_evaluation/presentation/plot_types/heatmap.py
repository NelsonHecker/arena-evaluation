from __future__ import annotations

import pathlib

import numpy as np
import plotly.express as px
import polars as pl

from .base import BasePlotRenderer

__all__ = ["HeatmapRenderer"]


class HeatmapRenderer(BasePlotRenderer):
    """2D matrix, correlation, and rank-swap heatmap renderer."""

    PLOT_TYPE = "heatmap"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return None

        is_correlation = self.spec.data_key == "*" or self.spec.data_key.lower() == "correlation" or self.spec.options.get("kind") == "correlation"

        if is_correlation:
            metrics_opt = self.spec.options.get("metrics")
            if metrics_opt:
                candidates = list(metrics_opt)
            elif self.spec.channels.y:
                candidates = self.spec.channels.y if isinstance(self.spec.channels.y, list) else [self.spec.channels.y]
            else:
                candidates = [
                    "success",
                    "collision_amount",
                    "time_to_goal",
                    "path_length",
                    "path_efficiency",
                    "velocity_mean",
                    "velocity_max",
                    "acceleration_mean",
                    "jerk_mean",
                    "curvature_mean",
                    "roughness_mean",
                    "angle_over_length",
                    "total_time_looking_at_pedestrians",
                    "total_time_looked_at_by_pedestrians",
                    "total_time_in_personal_space",
                    "avg_velocity_in_personal_space",
                ]
            schema = df_filtered.schema
            valid_cols = [col for col in candidates if schema.get(col) is not None and not df_filtered[col].is_null().all()]
            if len(valid_cols) < 2:
                valid_cols = [col for col, dtype in schema.items() if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32) and not df_filtered[col].is_null().all()]
            if len(valid_cols) < 2:
                return None

            pdf = df_filtered.select(valid_cols).to_pandas().astype(float)
            corr_matrix = pdf.corr()

            labels = [c.replace("_", " ").title() for c in corr_matrix.columns]

            fig = px.imshow(
                corr_matrix.values,
                x=labels,
                y=labels,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                range_color=[-1, 1],
                labels=dict(color="Correlation"),
                aspect="auto",
            )
            fig.update_layout(
                template="plotly_white",
                height=max(600, len(labels) * 40),
                yaxis=dict(tickmode="linear", dtick=1),
                xaxis=dict(tickmode="linear", dtick=1),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=40, b=40, l=50, r=30),
            )
            return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

        # 2D Pivot or Matrix Heatmap
        x_col = self.resolve_channel("x", fallback=self.spec.options.get("x", "planner"))
        y_col = self.resolve_channel("y", fallback=self.spec.options.get("y", "crowd_condition"))
        metric = self.resolve_channel("color", fallback=self.spec.data_key)

        schema = df_filtered.schema
        if not metric or schema.get(metric) is None or not x_col or schema.get(x_col) is None or not y_col or schema.get(y_col) is None:
            return None

        grouped = df_filtered.group_by([y_col, x_col]).agg(pl.col(metric).mean().alias("val")).to_pandas()
        if grouped.empty:
            return None

        pivot_df = grouped.pivot(index=y_col, columns=x_col, values="val")

        fig = px.imshow(
            pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            color_continuous_scale=self.spec.options.get("colorscale", "Viridis"),
            labels=dict(
                color=self.format_label(metric.replace("_", " ").title(), metric),
                x=x_col.replace("_", " ").title(),
                y=y_col.replace("_", " ").title(),
            ),
            aspect="auto",
        )
        if pivot_df.size <= 100:
            fig.update_traces(text=np.round(pivot_df.values, 2), texttemplate="%{text}")
        fig.update_layout(
            template="plotly_white",
            height=max(500, len(pivot_df.index) * 40),
            yaxis=dict(tickmode="linear", dtick=1),
            xaxis=dict(tickmode="linear", dtick=1),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=50, r=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns

        is_correlation = self.spec.data_key == "*" or self.spec.data_key.lower() == "correlation" or self.spec.options.get("kind") == "correlation"

        fig, ax = plt.subplots(figsize=(10, 8))

        if is_correlation:
            metrics_opt = self.spec.options.get("metrics")
            if metrics_opt:
                candidates = list(metrics_opt)
            elif self.spec.channels.y:
                candidates = self.spec.channels.y if isinstance(self.spec.channels.y, list) else [self.spec.channels.y]
            else:
                candidates = [
                    "success",
                    "collision_amount",
                    "time_to_goal",
                    "path_length",
                    "path_efficiency",
                    "velocity_mean",
                    "velocity_max",
                    "acceleration_mean",
                    "jerk_mean",
                    "curvature_mean",
                    "roughness_mean",
                    "angle_over_length",
                    "total_time_looking_at_pedestrians",
                    "total_time_looked_at_by_pedestrians",
                    "total_time_in_personal_space",
                    "avg_velocity_in_personal_space",
                ]
            schema = df_filtered.schema
            valid_cols = [col for col in candidates if schema.get(col) is not None and not df_filtered[col].is_null().all()]
            if len(valid_cols) < 2:
                valid_cols = [col for col, dtype in schema.items() if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32) and not df_filtered[col].is_null().all()]
            if len(valid_cols) < 2:
                plt.close(fig)
                return

            pdf = df_filtered.select(valid_cols).to_pandas().astype(float)
            corr_matrix = pdf.corr()
            labels = [c.replace("_", " ").title() for c in corr_matrix.columns]

            cmap = mpl.colormaps["RdBu_r"].with_extremes(bad=(0, 0, 0, 0))
            sns.heatmap(
                corr_matrix,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                ax=ax,
            )
            ax.set_title(self.spec.title, pad=12, fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)
            fig.tight_layout()
            self.save_figure(fig, out_path)
            plt.close(fig)
        else:
            x_col = self.resolve_channel("x", fallback=self.spec.options.get("x", "planner"))
            y_col = self.resolve_channel("y", fallback=self.spec.options.get("y", "crowd_condition"))
            metric = self.resolve_channel("color", fallback=self.spec.data_key)

            schema = df_filtered.schema
            if not metric or schema.get(metric) is None or not x_col or schema.get(x_col) is None or not y_col or schema.get(y_col) is None:
                plt.close(fig)
                return

            grouped = df_filtered.group_by([y_col, x_col]).agg(pl.col(metric).mean().alias("val")).to_pandas()
            if grouped.empty:
                plt.close(fig)
                return

            pivot_df = grouped.pivot(index=y_col, columns=x_col, values="val")

            cmap_name = str(self.spec.options.get("colorscale", "viridis")).lower()
            cmap = mpl.colormaps.get(cmap_name, mpl.colormaps["viridis"]).with_extremes(bad=(0, 0, 0, 0))
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                cbar_kws={"label": self.format_label(metric.replace("_", " ").title(), metric)},
                ax=ax,
            )
            ax.set_title(self.spec.title, pad=12, fontsize=12, fontweight="bold")
            ax.set_xlabel(x_col.replace("_", " ").title())
            ax.set_ylabel(y_col.replace("_", " ").title())
            fig.tight_layout()
            self.save_figure(fig, out_path)
            plt.close(fig)
