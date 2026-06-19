from __future__ import annotations

import pathlib
import polars as pl
import plotly.express as px

from .base import BasePlotRenderer


class ScatterRenderer(BasePlotRenderer):
    PLOT_TYPE = "scatter"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        x_col = self.spec.data_key
        y_col = self.spec.options.get("y")

        if not y_col or x_col not in df_filtered.columns or y_col not in df_filtered.columns:
            return None

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        fig = px.scatter(
            pdf,
            x=x_col,
            y=y_col,
            color=diff_col,
            title=self.spec.title,
            template="plotly_white",
            opacity=0.7,
        )

        fig.update_layout(
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=y_col.replace("_", " ").title(),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)

        x_col = self.spec.data_key
        y_col = self.spec.options.get("y")

        if not y_col or x_col not in df_filtered.columns or y_col not in df_filtered.columns:
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
        sns.scatterplot(
            data=pdf,
            x=x_col,
            y=y_col,
            hue=diff_col,
            alpha=0.7,
        )
        plt.title(self.spec.title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

