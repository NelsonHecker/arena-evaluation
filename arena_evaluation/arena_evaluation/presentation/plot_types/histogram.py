from __future__ import annotations

import pathlib
import polars as pl
import plotly.express as px

from .base import BasePlotRenderer


class HistogramRenderer(BasePlotRenderer):
    PLOT_TYPE = "histogram"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        
        x_col = self.spec.data_key
        diff_col = self.spec.differentiate or "planner"
        
        if x_col not in df_filtered.columns:
            return None
            
        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        barmode = self.spec.options.get("barmode", "overlay")

        fig = px.histogram(
            pdf,
            x=x_col,
            color=diff_col,
            title=self.spec.title,
            barmode=barmode,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title="Count"
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        
        x_col = self.spec.data_key
        diff_col = self.spec.differentiate or "planner"
        
        if x_col not in df_filtered.columns:
            return
            
        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=pdf,
            x=x_col,
            hue=diff_col,
            kde=True,
            element="step" if self.spec.options.get("barmode") == "overlay" else "bars"
        )
        plt.title(self.spec.title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
