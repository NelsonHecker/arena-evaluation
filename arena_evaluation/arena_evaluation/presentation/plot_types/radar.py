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

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        # Define the metrics to include in the radar chart
        metrics = self.spec.options.get(
            "metrics",
            ["path_efficiency", "time_to_goal", "collision_amount", "roughness_mean", "jerk_mean"],
        )

        # Check if metrics exist and are not entirely Null
        valid_metrics = [
            m for m in metrics
            if m in df_filtered.columns and not df_filtered[m].is_null().all()
        ]

        if len(valid_metrics) < 3:
            return None

        # Group by diff_col and calculate means
        grouped = (
            df_filtered
            .group_by(diff_col)
            .agg([pl.col(m).mean().alias(m) for m in valid_metrics])
            .to_pandas()
        )
        grouped = grouped.fillna(0.0)

        if grouped.empty:
            return None

        normalized = grouped.copy()

        use_log_scale = self.spec.options.get("use_log_scale", False)
        if use_log_scale:
            for m in valid_metrics:
                # Apply log1p (ln(1+x)) to handle zeros and strictly dampen outliers
                normalized[m] = np.log1p(normalized[m])

        # Metrics where higher values are better (should not be inverted)
        positive_metrics = {"success", "success_rate", "path_efficiency", "velocity_mean", "velocity_max"}

        for m in valid_metrics:
            max_val = normalized[m].max()
            min_val = normalized[m].min()

            if max_val == 0 and min_val == 0:
                normalized[m] = 1.0
            else:
                if m in positive_metrics:
                    normalized[m] = normalized[m] / max_val if max_val > 0 else 1.0
                else:
                    if min_val > 0:
                        normalized[m] = min_val / normalized[m]
                    else:
                        normalized[m] = 1.0 - (normalized[m] / max_val)

        fig = go.Figure()

        for _, row in normalized.iterrows():
            values = [row[m] for m in valid_metrics]
            values.append(values[0])
            formatted_metrics = [self.format_label(m.replace("_", " ").title(), m) for m in valid_metrics]
            labels = formatted_metrics + [formatted_metrics[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill=None,
                line=dict(width=3),
                name=row[diff_col],
                opacity=1.0,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=self.spec.title,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        # Static radar charts are complex in seaborn/matplotlib without lots of boilerplate.
        # Fallback to bar chart of the normalized metrics.
        pass
