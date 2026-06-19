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

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        grouped = (
            df_filtered
            .group_by(diff_col)
            .agg([
                pl.col(self.spec.data_key).mean().alias("mean"),
                pl.col(self.spec.data_key).std().alias("std"),
            ])
            .to_pandas()
        )

        if grouped.empty:
            return None

        fig = px.bar(
            grouped,
            x=diff_col,
            y="mean",
            color=diff_col,
            error_y="std",
            template="plotly_white",
            title=self.spec.title,
            labels={
                "mean": self.spec.data_key.replace("_", " ").title(),
                diff_col: diff_col.lstrip("_").replace("_", " ").title()
            }
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if self.spec.data_key not in df_filtered.columns:
            return

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=pdf,
            x=diff_col,
            y=self.spec.data_key,
            hue=diff_col,
            errorbar="sd",
            legend=False,
        )
        plt.title(self.spec.title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
