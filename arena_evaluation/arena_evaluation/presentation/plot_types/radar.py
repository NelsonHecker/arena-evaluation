from __future__ import annotations

import pathlib
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from .base import BasePlotRenderer


class RadarRenderer(BasePlotRenderer):
    PLOT_TYPE = "radar"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        diff_col = self.spec.differentiate or "planner"
        
        if diff_col not in df_filtered.columns:
            return None
            
        # Define the metrics to include in the radar chart
        metrics = self.spec.options.get("metrics", ["path_efficiency", "time_to_goal", "collision_amount", "roughness_mean", "jerk_mean"])
        
        # Check if metrics exist and are not entirely Null
        valid_metrics = []
        for m in metrics:
            if m in df_filtered.columns and not df_filtered[m].is_null().all():
                valid_metrics.append(m)
        
        if len(valid_metrics) < 3:
            return None
            
        # Group by diff_col and calculate means
        grouped = df_filtered.group_by(diff_col).agg([
            pl.col(m).mean().alias(m) for m in valid_metrics
        ]).to_pandas()
        
        if grouped.empty:
            return None
            
        # Normalize data so they fit nicely on a 0-1 radar chart
        # We need to invert some metrics (lower is better for time, collisions, roughness, jerk)
        # Higher is better for path_efficiency
        
        normalized = grouped.copy()
        for m in valid_metrics:
            max_val = normalized[m].max()
            min_val = normalized[m].min()
            
            if max_val == min_val:
                normalized[m] = 1.0 # If all same, perfect score
            else:
                if m == "path_efficiency":
                    # Higher is better: normal scale
                    normalized[m] = (normalized[m] - min_val) / (max_val - min_val)
                else:
                    # Lower is better: invert scale
                    normalized[m] = 1.0 - ((normalized[m] - min_val) / (max_val - min_val))
                    
        fig = go.Figure()
        
        for _, row in normalized.iterrows():
            values = [row[m] for m in valid_metrics]
            # Close the polygon
            values.append(values[0])
            
            labels = valid_metrics + [valid_metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=row[diff_col],
                opacity=0.6
            ))
            
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=self.spec.title,
            template="plotly_white",
            colorway=px.colors.qualitative.Pastel
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        # Static radar charts are complex in seaborn/matplotlib without lots of boilerplate.
        # Fallback to bar chart of the normalized metrics.
        pass
