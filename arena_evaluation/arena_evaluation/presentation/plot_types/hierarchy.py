from __future__ import annotations

import pathlib

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import polars as pl
import scipy.cluster.hierarchy as sch

from .base import BasePlot

__all__ = ["HierarchyRenderer"]


class HierarchyRenderer(BasePlot):
    """Hierarchical breakdown and phylogenetic cladistics renderer."""

    PLOT_TYPE = "hierarchy"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render interactive Plotly dendrogram, treemap, or sunburst."""
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return None

        is_dendrogram = (
            self.spec.type == "dendrogram"
            or str(self.spec.options.get("mode", "")).lower() == "dendrogram"
        )

        if is_dendrogram:
            return self._render_plotly_dendrogram(df_filtered)
        return self._render_plotly_hierarchy(df_filtered)

    def _render_plotly_dendrogram(self, df: pl.DataFrame) -> str | None:
        planner_col = self.resolve_channel("x", fallback=self.spec.options.get("x", "planner"))
        if not planner_col or df.schema.get(planner_col) is None:
            return None

        metrics = self._resolve_metrics(df, planner_col)
        if not metrics:
            return None

        agg_df = (
            df.group_by(planner_col)
            .agg([pl.col(m).mean().alias(m) for m in metrics])
            .sort(planner_col)
        )
        if agg_df.height < 2:
            return None

        agg_df = agg_df.with_columns([pl.col(m).fill_null(0.0) for m in metrics])
        # agent-contracts: ignore dendrogram label serialization
        planner_names = [str(p) for p in agg_df[planner_col].to_list()]
        matrix = agg_df.select(metrics).to_numpy()

        norm_matrix = self._normalize_matrix(matrix)
        method = str(self.spec.options.get("clustering_method", "ward"))
        orientation = str(self.spec.options.get("orientation", "left"))

        try:
            fig = ff.create_dendrogram(
                norm_matrix,
                orientation=orientation,
                labels=planner_names,
                linkagefun=lambda x: sch.linkage(x, method=method),
            )
        except Exception:
            fig = ff.create_dendrogram(
                norm_matrix,
                orientation=orientation,
                labels=planner_names,
            )

        title = self.spec.title or "Planner Phenotypic Behavioral Cladistics"
        fig.update_layout(
            template="plotly_dark",
            title=title,
            margin=dict(t=50, b=40, l=120, r=40),
            height=max(500, len(planner_names) * 35),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _render_plotly_hierarchy(self, df: pl.DataFrame) -> str | None:
        path_cols = self._resolve_hierarchy_path(df)
        if not path_cols:
            return None

        values_col = self.resolve_channel("y", fallback=self.spec.data_key)
        if values_col and df.schema.get(values_col) is None:
            values_col = None

        color_col = self.resolve_channel("color")
        if color_col and df.schema.get(color_col) is None:
            color_col = None

        df_clean = df.drop_nulls(subset=path_cols)
        if values_col:
            df_clean = df_clean.drop_nulls(subset=[values_col])
            df_clean = df_clean.with_columns(pl.col(values_col).clip(lower_bound=0.0))

        if df_clean.is_empty():
            return None

        select_cols = list(
            dict.fromkeys(
                path_cols
                + ([values_col] if values_col else [])
                + ([color_col] if color_col else [])
            )
        )
        pdf = df_clean.select(select_cols).to_pandas()
        if pdf.empty:
            return None

        mode = str(self.spec.options.get("mode", "treemap")).lower()
        is_sunburst = mode == "sunburst" or self.spec.type == "sunburst"

        if is_sunburst:
            fig = px.sunburst(
                pdf,
                path=path_cols,
                values=values_col,
                color=color_col,
                title=self.spec.title or None,
            )
        else:
            fig = px.treemap(
                pdf,
                path=path_cols,
                values=values_col,
                color=color_col,
                title=self.spec.title or None,
            )

        fig.update_layout(
            template="plotly_white",
            margin=dict(t=50 if self.spec.title else 30, b=30, l=30, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_seaborn(self, df: pl.DataFrame, path: pathlib.Path) -> None:
        """Render static dendrogram, clustered heatmap, or grouped bar chart."""
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return

        is_dendrogram = (
            self.spec.type == "dendrogram"
            or str(self.spec.options.get("mode", "")).lower() == "dendrogram"
        )

        if is_dendrogram:
            self._render_seaborn_dendrogram(df_filtered, path)
        else:
            self._render_seaborn_hierarchy(df_filtered, path)

    def _render_seaborn_dendrogram(self, df: pl.DataFrame, path: pathlib.Path) -> None:
        import matplotlib.pyplot as plt

        planner_col = self.resolve_channel("x", fallback=self.spec.options.get("x", "planner"))
        if not planner_col or df.schema.get(planner_col) is None:
            return

        metrics = self._resolve_metrics(df, planner_col)
        if not metrics:
            return

        agg_df = (
            df.group_by(planner_col)
            .agg([pl.col(m).mean().alias(m) for m in metrics])
            .sort(planner_col)
        )
        if agg_df.height < 2:
            return

        agg_df = agg_df.with_columns([pl.col(m).fill_null(0.0) for m in metrics])
        # agent-contracts: ignore dendrogram label serialization
        planner_names = [str(p) for p in agg_df[planner_col].to_list()]
        matrix = agg_df.select(metrics).to_numpy()

        norm_matrix = self._normalize_matrix(matrix)
        method = str(self.spec.options.get("clustering_method", "ward"))

        try:
            linkage_matrix = sch.linkage(norm_matrix, method=method)
        except Exception:
            linkage_matrix = sch.linkage(norm_matrix, method="complete")

        if self.spec.options.get("clustermap", False):
            import pandas as pd
            import seaborn as sns

            df_clust = pd.DataFrame(
                norm_matrix,
                index=planner_names,
                columns=[m.replace("_", " ").title() for m in metrics],
            )
            g = sns.clustermap(
                df_clust,
                row_linkage=linkage_matrix,
                col_cluster=len(metrics) > 1,
                cmap=self.spec.options.get("colorscale", "viridis"),
            )
            if self.spec.title:
                g.figure.suptitle(self.spec.title, y=1.02, fontsize=12, fontweight="bold")
            self.save_figure(g.figure, path)
            plt.close(g.figure)
        else:
            fig, ax = plt.subplots(figsize=(10, max(5, len(planner_names) * 0.4)))
            sch.dendrogram(linkage_matrix, labels=planner_names, orientation="left", ax=ax)
            ax.set_title(self.spec.title or "Planner Behavioral Cladistics", pad=12, fontsize=12, fontweight="bold")
            ax.set_xlabel("Distance")
            fig.tight_layout()
            self.save_figure(fig, path)
            plt.close(fig)

    def _render_seaborn_hierarchy(self, df: pl.DataFrame, path: pathlib.Path) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        path_cols = self._resolve_hierarchy_path(df)
        if not path_cols:
            return

        x_col = path_cols[0]
        hue_col = path_cols[1] if len(path_cols) > 1 else None

        values_col = self.resolve_channel("y", fallback=self.spec.data_key)
        schema = df.schema
        if values_col and schema.get(values_col) is None:
            values_col = None

        grp = [c for c in [x_col, hue_col] if c]
        is_numeric = (
            values_col is not None
            and schema.get(values_col) is not None
            and schema[values_col] in (
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
                pl.UInt64,
                pl.UInt32,
                pl.Int16,
                pl.Int8,
                pl.UInt16,
                pl.UInt8,
            )
        )

        if values_col and is_numeric:
            agg_df = df.group_by(grp).agg(pl.col(values_col).sum().alias(values_col))
            plot_val = values_col
            y_label = self.format_label(values_col.replace("_", " ").title(), values_col)
        else:
            agg_df = df.group_by(grp).agg(pl.len().alias("count"))
            plot_val = "count"
            y_label = "Count"

        pdf = agg_df.to_pandas()
        if pdf.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=pdf, x=x_col, y=plot_val, hue=hue_col, ax=ax)
        ax.set_title(self.spec.title or "Hierarchical Breakdown", pad=12, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_label)
        if hue_col:
            ax.legend(title=hue_col.replace("_", " ").title())
        fig.tight_layout()
        self.save_figure(fig, path)
        plt.close(fig)

    def _resolve_metrics(self, df: pl.DataFrame, planner_col: str) -> list[str]:
        raw_metrics: list[str] = []
        opt_metrics = self.spec.options.get("metrics")
        if opt_metrics:
            raw_metrics = list(opt_metrics) if isinstance(opt_metrics, (list, tuple)) else [str(opt_metrics)]
        elif isinstance(self.spec.channels.y, list):
            raw_metrics = [str(m) for m in self.spec.channels.y if m]
        elif isinstance(self.spec.channels.y, str) and self.spec.channels.y.strip():
            raw_metrics = [self.spec.channels.y.strip()]

        schema = df.schema
        metrics = [m for m in raw_metrics if schema.get(m) is not None]
        if not metrics:
            candidates = [
                "time_to_goal",
                "path_length",
                "collision_amount",
                "jerk_mean",
                "specific_cost_of_transport",
                "success",
                "velocity_mean",
            ]
            metrics = [m for m in candidates if schema.get(m) is not None]
        if not metrics:
            metrics = [
                col
                for col, dtype in schema.items()
                if col != planner_col and dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)
            ]
        return metrics

    def _resolve_hierarchy_path(self, df: pl.DataFrame) -> list[str]:
        raw_hierarchy = (
            self.spec.channels.hierarchy
            or self.spec.options.get("path")
            or [self.spec.channels.x, self.spec.channels.color]
        )
        path_cols: list[str] = []
        if isinstance(raw_hierarchy, (list, tuple)):
            for item in raw_hierarchy:
                if isinstance(item, str) and item.strip():
                    path_cols.append(item.strip())
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, str) and sub.strip():
                            path_cols.append(sub.strip())
        elif isinstance(raw_hierarchy, str) and raw_hierarchy.strip():
            path_cols.append(raw_hierarchy.strip())

        schema = df.schema
        return [c for c in path_cols if schema.get(c) is not None]

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        norm_mode = str(self.spec.options.get("normalize", "zscore")).lower()
        if norm_mode == "minmax":
            col_min = matrix.min(axis=0)
            col_max = matrix.max(axis=0)
            col_diff = col_max - col_min
            col_diff = np.where(col_diff == 0.0, 1.0, col_diff)
            norm_matrix = (matrix - col_min) / col_diff
        else:
            col_mean = matrix.mean(axis=0)
            col_std = matrix.std(axis=0)
            col_std = np.where(col_std == 0.0, 1.0, col_std)
            norm_matrix = (matrix - col_mean) / col_std
        return np.nan_to_num(norm_matrix, nan=0.0)
