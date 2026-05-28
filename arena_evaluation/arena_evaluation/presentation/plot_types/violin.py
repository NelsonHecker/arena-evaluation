from __future__ import annotations

import pathlib
import polars as pl
import plotly.express as px

from .base import BasePlotRenderer


class ViolinRenderer(BasePlotRenderer):
    PLOT_TYPE = "violin"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        if self.spec.data_key not in df_filtered.columns:
            return None
            
        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        fig = px.violin(
            pdf,
            y=self.spec.data_key,
            color=self.spec.differentiate or "planner",
            box=True,
            points="all",
            title=self.spec.title,
            labels={
                self.spec.data_key: self.spec.data_key.replace("_", " ").title(),
                self.spec.differentiate or "planner": (self.spec.differentiate or "planner").title()
            }
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
        sns.violinplot(
            data=pdf,
            y=self.spec.data_key,
            hue=self.spec.differentiate or "planner",
            inner="box"
        )
        plt.title(self.spec.title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
