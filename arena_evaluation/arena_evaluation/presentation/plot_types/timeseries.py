from __future__ import annotations

import pathlib
import polars as pl
import plotly.graph_objects as go

from .base import BasePlotRenderer


class TimeseriesRenderer(BasePlotRenderer):
    PLOT_TYPE = "timeseries"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        diff_col, df_filtered = self.resolve_diff_col(df_filtered)

        metrics = self.spec.options.get("metrics", [])
        if not metrics:
            if self.spec.data_key:
                metrics = [self.spec.data_key]
            else:
                return None
                
        # Must have timeseries_time_s or similar as X axis
        x_col = self.spec.options.get("x", "timeseries_time_s")

        if x_col not in df_filtered.columns:
            return None

        # Check if all needed metrics are in the dataframe
        valid_metrics = [m for m in metrics if m in df_filtered.columns]
        if not valid_metrics:
            return None

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        fig = go.Figure()
        
        planners = pdf[diff_col].unique() if diff_col in pdf.columns else ["unknown"]
        colors = px.colors.qualitative.Plotly if hasattr(px, "colors") else ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        import plotly.express as px
        colors = px.colors.qualitative.Plotly
        
        # Build traces
        for planner_idx, planner in enumerate(planners):
            planner_df = pdf[pdf[diff_col] == planner] if diff_col in pdf.columns else pdf
            
            for _, row in planner_df.iterrows():
                episode_val = row.get("episode", "unknown")
                legend_group = f"{planner} - Ep {episode_val}"
                
                x_data = row.get(x_col)
                if x_data is None or not isinstance(x_data, (list, tuple)):
                    continue
                    
                # Base color for this planner
                base_color = colors[planner_idx % len(colors)]
                
                for m_idx, metric in enumerate(valid_metrics):
                    y_data = row.get(metric)
                    if y_data is None or not isinstance(y_data, (list, tuple)):
                        continue
                        
                    # Opacity and line styling to distinguish metrics within the same episode group
                    dash_styles = ["solid", "dash", "dot", "dashdot"]
                    dash = dash_styles[m_idx % len(dash_styles)]
                    
                    showlegend = True if m_idx == 0 else False
                    
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="lines",
                        name=f"{legend_group}" if m_idx == 0 else metric,
                        legendgroup=legend_group,
                        line=dict(color=base_color, dash=dash, width=2),
                        hovertemplate=f"Ep {episode_val}<br>{metric}: %{{y}}<extra></extra>",
                        showlegend=showlegend
                    ))

        fig.update_layout(
            title=self.spec.title,
            template="plotly_white",
            xaxis_title=self.format_label(x_col.replace("timeseries_", "").replace("_", " ").title(), x_col),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(r=150)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        pass  # Timeseries is too complex for static seaborn out-of-the-box, rely on plotly
