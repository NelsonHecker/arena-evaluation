from __future__ import annotations

import pathlib

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from .base import BasePlotRenderer

__all__ = ["TimeseriesRenderer"]


class TimeseriesRenderer(BasePlotRenderer):
    """Timeseries renderer for temporal curves (speed, energy, power, acoustics)."""

    PLOT_TYPE = "timeseries"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        color_col = self.resolve_channel("color")
        if not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            color_col = diff_col

        metrics = self.spec.options.get("metrics", [])
        if not metrics:
            ch_y = self.spec.channels.y
            if ch_y:
                metrics = ch_y if isinstance(ch_y, list) else [ch_y]
            elif self.spec.data_key:
                metrics = [self.spec.data_key]
            else:
                return None

        # Check for matching metric columns or aliases
        valid_metrics: list[str] = []
        schema = df_filtered.schema
        for m in metrics:
            if schema.get(m) is not None:
                valid_metrics.append(m)
            elif schema.get(f"timeseries_{m}") is not None:
                valid_metrics.append(f"timeseries_{m}")
            elif schema.get(m.replace("timeseries_", "")) is not None:
                valid_metrics.append(m.replace("timeseries_", ""))

        if not valid_metrics:
            return None

        x_col = self.resolve_channel("x", fallback=self.spec.options.get("x", "timeseries_time_s"))
        if not x_col or schema.get(x_col) is None:
            # Fallback to time_s if available
            if schema.get("time_s") is not None:
                x_col = "time_s"
            else:
                return None

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        # Filter out reference runs from main planner traces if requested
        if schema.get("is_reference") is not None and not self.spec.options.get("include_reference", False):
            pdf = pdf[~pdf["is_reference"].fillna(False)]
            if pdf.empty:
                return None

        first_val = df_filtered[x_col][0] if not df_filtered.is_empty() else None
        is_nested = isinstance(first_val, (list, tuple, np.ndarray)) or schema[x_col] == pl.List or any(schema[m] == pl.List for m in valid_metrics)

        # 1NF flat timeseries rows
        if not is_nested:
            fig = px.line(
                pdf,
                x=x_col,
                y=valid_metrics[0] if len(valid_metrics) == 1 else valid_metrics,
                color=color_col if color_col and color_col in pdf.columns and color_col != x_col else None,
                template="plotly_white",
                labels={
                    x_col: self.format_label(x_col.replace("timeseries_", "").replace("_", " ").title(), x_col),
                },
            )
            fig.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=40, b=40, l=50, r=30),
            )
            return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

        # Nested list/array columns
        fig = go.Figure()
        planners = pdf[color_col].unique() if color_col and color_col in pdf.columns else ["unknown"]
        colors = px.colors.qualitative.Plotly

        for planner_idx, planner in enumerate(planners):
            planner_df = pdf[pdf[color_col] == planner] if color_col and color_col in pdf.columns else pdf

            for _, row in planner_df.iterrows():
                episode_val = row.get("episode", "unknown")
                legend_group = f"{planner} - Ep {episode_val}"
                base_color = colors[planner_idx % len(colors)]

                x_raw = row.get(x_col)
                if x_raw is None or isinstance(x_raw, (int, float, str, bool)):
                    continue
                try:
                    x_data = np.array(x_raw, dtype=float)
                except Exception:
                    continue
                if x_data.ndim != 1 or len(x_data) == 0:
                    continue

                for m_idx, metric in enumerate(valid_metrics):
                    y_raw = row.get(metric)
                    if y_raw is None:
                        continue
                    try:
                        y_data = np.array(y_raw, dtype=float)
                    except Exception:
                        continue

                    if y_data.ndim != 1 or len(y_data) == 0:
                        continue

                    if len(x_data) != len(y_data):
                        cur_x = np.linspace(x_data[0], x_data[-1], len(y_data)) if len(y_data) > 1 else x_data[: len(y_data)]
                    else:
                        cur_x = x_data

                    dash_styles = ["solid", "dash", "dot", "dashdot"]
                    dash = dash_styles[m_idx % len(dash_styles)]

                    m_label = self.format_label(metric.replace("timeseries_", "").replace("_", " ").title(), metric)
                    fig.add_trace(
                        go.Scatter(
                            x=cur_x.tolist(),
                            y=y_data.tolist(),
                            mode="lines",
                            name=f"{legend_group}" if m_idx == 0 else m_label,
                            legendgroup=legend_group,
                            line=dict(color=base_color, dash=dash, width=2),
                            hovertemplate=f"Ep {episode_val}<br>Time: %{{x:.2f}}s<br>{m_label}: %{{y:.3f}}<extra></extra>",
                            showlegend=bool(m_idx == 0),
                        )
                    )

        fig.update_layout(
            template="plotly_white",
            xaxis_title=self.format_label(x_col.replace("timeseries_", "").replace("_", " ").title(), x_col),
            yaxis_title=self.format_label(valid_metrics[0].replace("timeseries_", "").replace("_", " ").title(), valid_metrics[0]),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(r=150),
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        pass
