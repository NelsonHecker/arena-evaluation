from __future__ import annotations

import pathlib
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

from .base import BasePlotRenderer


class BarRenderer(BasePlotRenderer):
    PLOT_TYPE = "bar"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        if self.spec.data_key not in df_filtered.columns:
            return None
            
        # Group and calculate mean/std
        diff_col = self.spec.differentiate or "planner"
        if diff_col not in df_filtered.columns:
            return None
            
        grouped = df_filtered.group_by(diff_col).agg([
            pl.col(self.spec.data_key).mean().alias("mean"),
            pl.col(self.spec.data_key).std().alias("std")
        ]).to_pandas()
        
        if grouped.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=grouped[diff_col],
            y=grouped["mean"],
            error_y=dict(type='data', array=grouped["std"]),
            name=self.spec.title
        ))
        
        fig.update_layout(
            title=self.spec.title,
            xaxis_title=diff_col.title(),
            yaxis_title=self.spec.data_key.replace("_", " ").title(),
            template="plotly_white",
            colorway=px.colors.qualitative.Pastel
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if self.spec.data_key not in df_filtered.columns:
            return
            
        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=pdf,
            y=self.spec.data_key,
            hue=self.spec.differentiate or "planner",
            errorbar="sd"
        )
        plt.title(self.spec.title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
