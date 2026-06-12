from __future__ import annotations

import pathlib
import polars as pl
import plotly.express as px

from .base import BasePlotRenderer


class HistogramRenderer(BasePlotRenderer):
    PLOT_TYPE = "histogram"

    def _prepare_binned_data(self, df_filtered: pl.DataFrame, x_col: str) -> tuple[pd.DataFrame, list[str]] | None:
        import numpy as np
        import pandas as pd

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        # Clean/extract valid numeric values
        vals = pdf[x_col].dropna().to_numpy()
        if len(vals) == 0:
            return None

        num_bins = self.spec.options.get("nbins", 10)
        if num_bins < 1:
            num_bins = 10

        edges = np.percentile(vals, np.linspace(0, 100, num_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5])

        bin_labels = []
        for i in range(len(edges) - 1):
            val1, val2 = edges[i], edges[i+1]
            if float(val1).is_integer() and float(val2).is_integer():
                lbl = f"[{int(val1)}, {int(val2)})" if i < len(edges) - 2 else f"[{int(val1)}, {int(val2)}]"
            else:
                lbl = f"[{val1:.2f}, {val2:.2f})" if i < len(edges) - 2 else f"[{val1:.2f}, {val2:.2f}]"
            bin_labels.append(lbl)

        edges_cut = edges.copy()
        edges_cut[0] -= 1e-9
        edges_cut[-1] += 1e-9

        pdf[f"{x_col}_bin"] = pd.cut(pdf[x_col], bins=edges_cut, include_lowest=True, labels=bin_labels)
        pdf = pdf.dropna(subset=[f"{x_col}_bin"])
        
        return pdf, bin_labels

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        x_col = self.spec.data_key
        if x_col not in df_filtered.columns:
            return None

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        res = self._prepare_binned_data(df_filtered, x_col)
        if res is None:
            return None
        pdf, bin_labels = res

        barmode = self.spec.options.get("barmode", "group")
        opacity = self.spec.options.get("opacity", 1.0)

        fig = px.histogram(
            pdf,
            x=f"{x_col}_bin",
            color=diff_col,
            title=self.spec.title,
            barmode=barmode,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            opacity=opacity,
            category_orders={f"{x_col}_bin": bin_labels},
        )

        fig.update_layout(
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title="Count",
            xaxis=dict(categoryorder="array", categoryarray=bin_labels),
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)

        x_col = self.spec.data_key
        if x_col not in df_filtered.columns:
            return

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return

        res = self._prepare_binned_data(df_filtered, x_col)
        if res is None:
            return
        pdf, bin_labels = res

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=pdf,
            x=f"{x_col}_bin",
            hue=diff_col,
            order=bin_labels,
            palette="pastel",
        )
        plt.title(self.spec.title)
        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
