from __future__ import annotations

import pathlib
import polars as pl
import plotly.express as px

from .base import BasePlotRenderer


class BoxRenderer(BasePlotRenderer):
    PLOT_TYPE = "box"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        if self.spec.data_key not in df_filtered.columns:
            return None
            
        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        fig = px.box(
            pdf,
            y=self.spec.data_key,
            color=self.spec.differentiate or "planner",
            title=self.spec.title,
            labels={
                self.spec.data_key: self.spec.data_key.replace("_", " ").title(),
                self.spec.differentiate or "planner": (self.spec.differentiate or "planner").title()
            },
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if self.spec.data_key not in df_filtered.columns:
            return
            
        pdf = df_filtered.to_pandas()
        if pdf.empty or pdf[self.spec.data_key].isna().all():
            return
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=pdf,
            y=self.spec.data_key,
            hue=self.spec.differentiate or "planner"
        )
        plt.title(self.spec.title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
