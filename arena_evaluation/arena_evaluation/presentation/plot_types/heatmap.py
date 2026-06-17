from __future__ import annotations

import pathlib
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from .base import BasePlotRenderer


class HeatmapRenderer(BasePlotRenderer):
    PLOT_TYPE = "heatmap"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return None

        # Determine if this is a correlation heatmap or a pivot heatmap
        is_correlation = self.spec.data_key == "*" or self.spec.data_key.lower() == "correlation"

        if is_correlation:
            # List of numeric metrics to compute correlation for
            candidates = [
                "success", "collision_amount", "time_to_goal", "path_length", 
                "path_efficiency", "velocity_mean", "velocity_max",
                "acceleration_mean", "jerk_mean", "curvature_mean", 
                "roughness_mean", "angle_over_length",
                "total_time_looking_at_pedestrians", "total_time_looked_at_by_pedestrians",
                "total_time_in_personal_space", "avg_velocity_in_personal_space"
            ]
            valid_cols = [col for col in candidates if col in df_filtered.columns and not df_filtered[col].is_null().all()]
            if len(valid_cols) < 2:
                return None
            
            # Compute correlation matrix
            pdf = df_filtered.select(valid_cols).to_pandas().astype(float)
            corr_matrix = pdf.corr()
            
            # Format titles
            labels = [c.replace("_", " ").title() for c in corr_matrix.columns]
            
            fig = px.imshow(
                corr_matrix.values,
                x=labels,
                y=labels,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                range_color=[-1, 1],
                title=self.spec.title,
                labels=dict(color="Correlation"),
                aspect="auto"
            )
            fig.update_layout(
                template="plotly_white",
                height=max(600, len(labels) * 40),
                yaxis=dict(tickmode="linear", dtick=1),
                xaxis=dict(tickmode="linear", dtick=1)
            )
            return fig.to_html(full_html=False, include_plotlyjs=False)
            
        else:
            # Pivot heatmap: e.g. local_planner vs inter_planner for a given metric (spec.data_key)
            metric = self.spec.data_key
            if metric not in df_filtered.columns:
                return None
                
            x_col = self.spec.options.get("x", "inter_planner")
            y_col = self.spec.options.get("y", "local_planner")
            
            if x_col not in df_filtered.columns or y_col not in df_filtered.columns:
                return None
                
            # Aggregate and pivot
            grouped = (
                df_filtered
                .group_by([y_col, x_col])
                .agg(pl.col(metric).mean().alias("val"))
                .to_pandas()
            )
            if grouped.empty:
                return None
                
            pivot_df = grouped.pivot(index=y_col, columns=x_col, values="val")
            
            fig = px.imshow(
                pivot_df.values,
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                color_continuous_scale="Viridis",
                title=self.spec.title,
                labels=dict(color=metric.replace("_", " ").title(), x=x_col.replace("_", " ").title(), y=y_col.replace("_", " ").title()),
                aspect="auto"
            )
            # Add text values inside cells for readability if there aren't too many cells
            if pivot_df.size <= 100:
                fig.update_traces(
                    text=np.round(pivot_df.values, 2),
                    texttemplate="%{text}"
                )
            fig.update_layout(
                template="plotly_white",
                height=max(500, len(pivot_df.index) * 40),
                yaxis=dict(tickmode="linear", dtick=1),
                xaxis=dict(tickmode="linear", dtick=1)
            )
            return fig.to_html(full_html=False, include_plotlyjs=False)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        is_correlation = self.spec.data_key == "*" or self.spec.data_key.lower() == "correlation"

        plt.figure(figsize=(10, 8))

        if is_correlation:
            candidates = [
                "success", "collision_amount", "time_to_goal", "path_length", 
                "path_efficiency", "velocity_mean", "velocity_max",
                "acceleration_mean", "jerk_mean", "curvature_mean", 
                "roughness_mean", "angle_over_length",
                "total_time_looking_at_pedestrians", "total_time_looked_at_by_pedestrians",
                "total_time_in_personal_space", "avg_velocity_in_personal_space"
            ]
            valid_cols = [col for col in candidates if col in df_filtered.columns and not df_filtered[col].is_null().all()]
            if len(valid_cols) < 2:
                plt.close()
                return
            
            pdf = df_filtered.select(valid_cols).to_pandas().astype(float)
            corr_matrix = pdf.corr()
            labels = [c.replace("_", " ").title() for c in corr_matrix.columns]

            sns.heatmap(
                corr_matrix,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                center=0,
                square=True
            )
            plt.title(self.spec.title)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
        else:
            metric = self.spec.data_key
            if metric not in df_filtered.columns:
                plt.close()
                return
                
            x_col = self.spec.options.get("x", "inter_planner")
            y_col = self.spec.options.get("y", "local_planner")
            
            if x_col not in df_filtered.columns or y_col not in df_filtered.columns:
                plt.close()
                return
                
            grouped = (
                df_filtered
                .group_by([y_col, x_col])
                .agg(pl.col(metric).mean().alias("val"))
                .to_pandas()
            )
            if grouped.empty:
                plt.close()
                return
                
            pivot_df = grouped.pivot(index=y_col, columns=x_col, values="val")
            
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                cbar_kws={'label': metric.replace("_", " ").title()}
            )
            plt.title(self.spec.title)
            plt.xlabel(x_col.replace("_", " ").title())
            plt.ylabel(y_col.replace("_", " ").title())
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
