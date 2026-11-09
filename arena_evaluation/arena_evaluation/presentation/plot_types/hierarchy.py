from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import polars as pl
import scipy.cluster.hierarchy as sch

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlot, finish_axes, label_values, new_figure, palette_color

__all__ = ["HierarchyRenderer"]


class HierarchyRenderer(BasePlot):
    """Hierarchical breakdown and phylogenetic cladistics renderer."""

    PLOT_TYPE = "hierarchy"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render interactive Plotly dendrogram, treemap, or sunburst."""
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return None

        is_dendrogram = self.spec.type == "dendrogram" or str(self.spec.options.get("mode", "")).lower() == "dendrogram"
        return self._render_plotly_dendrogram(df_filtered) if is_dendrogram else self._render_plotly_hierarchy(df_filtered)

    def _render_plotly_dendrogram(self, df: pl.DataFrame) -> str | None:
        planner_col = self.resolve_channel("x", fallback=str(self.spec.options.get("x", "planner")))
        if not planner_col or df.schema.get(planner_col) is None:
            return None

        metrics = self._resolve_metrics(df, planner_col)
        if not metrics:
            return None

        agg_df = df.group_by(planner_col).agg([pl.col(metric).mean().alias(metric) for metric in metrics]).sort(planner_col)
        if agg_df.height < 2:
            return None

        agg_df = agg_df.with_columns([pl.col(metric).fill_null(0.0) for metric in metrics])
        planner_names = label_values(agg_df, planner_col).tolist()
        normalized = self._normalize_matrix(agg_df.select(metrics).to_numpy())
        method = str(self.spec.options.get("clustering_method", "ward"))

        try:
            fig = ff.create_dendrogram(
                normalized,
                orientation=str(self.spec.options.get("orientation", "left")),
                labels=planner_names,
                linkagefun=lambda matrix: sch.linkage(matrix, method=method),
            )
        except ValueError:
            fig = ff.create_dendrogram(normalized, labels=planner_names)

        fig.update_layout(
            template="plotly_dark",
            title=self.spec.title or "Planner Phenotypic Behavioral Cladistics",
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
            df_clean = df_clean.drop_nulls(subset=[values_col]).with_columns(pl.col(values_col).clip(lower_bound=0.0))
        if df_clean.is_empty():
            return None

        select_cols = list(dict.fromkeys(path_cols + ([values_col] if values_col else []) + ([color_col] if color_col else [])))
        selected = df_clean.select(select_cols)
        mode = str(self.spec.options.get("mode", "treemap")).lower()
        fig = (
            px.sunburst(selected, path=path_cols, values=values_col, color=color_col, title=self.spec.title or None)
            if mode == "sunburst"
            else px.treemap(selected, path=path_cols, values=values_col, color=color_col, title=self.spec.title or None)
        )
        fig.update_layout(template="plotly_white", margin=dict(t=50 if self.spec.title else 30, b=30, l=30, r=30))
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static dendrogram, clustered heatmap, or grouped bar chart."""
        df_filtered = self._apply_filters(df)
        if df_filtered.is_empty():
            return

        is_dendrogram = self.spec.type == "dendrogram" or str(self.spec.options.get("mode", "")).lower() == "dendrogram"
        if is_dendrogram:
            self._render_static_dendrogram(df_filtered, out_path)
        else:
            self._render_static_hierarchy(df_filtered, out_path)

    def _render_static_dendrogram(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        planner_col = self.resolve_channel("x", fallback=str(self.spec.options.get("x", "planner")))
        if not planner_col or df.schema.get(planner_col) is None:
            return

        metrics = self._resolve_metrics(df, planner_col)
        if not metrics:
            return

        agg_df = df.group_by(planner_col).agg([pl.col(metric).mean().alias(metric) for metric in metrics]).sort(planner_col)
        if agg_df.height < 2:
            return

        agg_df = agg_df.with_columns([pl.col(metric).fill_null(0.0) for metric in metrics])
        planner_names = label_values(agg_df, planner_col).tolist()
        matrix = self._normalize_matrix(agg_df.select(metrics).to_numpy())
        method = str(self.spec.options.get("clustering_method", "ward"))
        try:
            linkage = sch.linkage(matrix, method=method)
        except ValueError:
            linkage = sch.linkage(matrix, method="complete")

        if self.spec.options.get("clustermap", False):
            self._render_static_clustermap(matrix, planner_names, metrics, linkage, out_path)
            return

        fig, ax = plt.subplots(figsize=(10, max(5, len(planner_names) * 0.4)))
        sch.dendrogram(linkage, labels=planner_names, orientation="left", ax=ax)
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)

    def _render_static_clustermap(self, matrix: np.ndarray, planner_names: list[str], metrics: list[str], linkage: np.ndarray, out_path: pathlib.Path) -> None:
        """Row dendrogram beside the normalized metric heatmap, in the leaf order the linkage implies."""
        leaves = sch.dendrogram(linkage, no_plot=True)["leaves"]
        ordered = matrix[leaves]
        ordered_names = [planner_names[index] for index in leaves]
        labels = [metric.replace("_", " ").title() for metric in metrics]

        fig, (dendrogram_ax, heatmap_ax) = plt.subplots(
            1,
            2,
            figsize=(12, max(5, len(planner_names) * 0.5)),
            gridspec_kw={"width_ratios": [1, 3]},
        )
        sch.dendrogram(linkage, labels=planner_names, orientation="left", ax=dendrogram_ax)
        image = heatmap_ax.imshow(ordered, cmap=str(self.spec.options.get("colorscale", "viridis")), aspect="auto")
        heatmap_ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        heatmap_ax.set_yticks(np.arange(len(ordered_names)), ordered_names)
        fig.colorbar(image, ax=heatmap_ax, label="z-score")
        fig.suptitle(self.spec.title or "Planner Behavioral Cladistics", fontsize=12, fontweight="bold")
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)

    def _render_static_hierarchy(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        path_cols = self._resolve_hierarchy_path(df)
        if not path_cols:
            return

        x_col = path_cols[0]
        hue_col = path_cols[1] if len(path_cols) > 1 else None
        values_col = self.resolve_channel("y", fallback=self.spec.data_key)
        schema = df.schema
        if values_col and schema.get(values_col) is None:
            values_col = None

        group_cols = [x_col] if hue_col is None else [x_col, hue_col]
        numeric_values_col = values_col is not None and schema.get(values_col) is not None and schema[values_col].is_numeric()
        if numeric_values_col and values_col is not None:
            agg_df = df.group_by(group_cols).agg(pl.col(values_col).sum().alias("value"))
            y_label = self.format_label(values_col.replace("_", " ").title(), values_col)
        else:
            agg_df = df.group_by(group_cols).agg(pl.len().alias("value"))
            y_label = "Count"
        if agg_df.is_empty():
            return

        categories = label_values(agg_df.select(x_col).unique(), x_col).tolist()
        categories.sort()
        if hue_col is None:
            series = [""]
            frames = agg_df.partition_by(x_col, as_dict=True)
            lookup = {(str(key[0]), ""): float(frame["value"][0]) for key, frame in frames.items()}
        else:
            series = sorted(label_values(agg_df.select(hue_col).unique(), hue_col).tolist())
            cells = agg_df.partition_by([x_col, hue_col], as_dict=True)
            lookup = {(str(frame[x_col][0]), str(frame[hue_col][0])): float(frame["value"][0]) for frame in cells.values()}

        positions = np.arange(len(categories))
        palette = get_color_palette()
        width = 0.8 / max(1, len(series))
        fig, ax = new_figure()
        for series_index, series_name in enumerate(series):
            heights = [lookup.get((category, series_name), 0.0) for category in categories]
            offset = (series_index - (len(series) - 1) / 2.0) * width
            ax.bar(positions + offset, heights, width, label=series_name or None, color=palette_color(palette, series_index))

        ax.set_xticks(positions, categories, rotation=30, ha="right")
        if hue_col is not None:
            ax.legend(frameon=False, title=hue_col.replace("_", " ").title())
        finish_axes(ax, title=self.spec.title or "Hierarchical Breakdown", x_label=x_col.replace("_", " ").title(), y_label=y_label)
        self.save_figure(fig, out_path)
        plt.close(fig)

    def _resolve_metrics(self, df: pl.DataFrame, planner_col: str) -> list[str]:
        raw_metrics: list[str] = []
        opt_metrics = self.spec.options.get("metrics")
        if opt_metrics:
            raw_metrics = list(opt_metrics) if isinstance(opt_metrics, (list, tuple)) else [str(opt_metrics)]
        elif isinstance(self.spec.channels.y, list):
            raw_metrics = [str(metric) for metric in self.spec.channels.y if metric]
        elif isinstance(self.spec.channels.y, str) and self.spec.channels.y.strip():
            raw_metrics = [self.spec.channels.y.strip()]

        schema = df.schema
        metrics = [metric for metric in raw_metrics if schema.get(metric) is not None]
        if not metrics:
            candidates = ["time_to_goal_s", "path_length_m", "total_collisions_ped", "jerk_max_mps3", "total_energy_wh", "success", "mean_velocity_mps"]
            metrics = [metric for metric in candidates if schema.get(metric) is not None]
        if not metrics:
            metrics = [column for column, dtype in schema.items() if column != planner_col and dtype.is_numeric()]
        return metrics

    def _resolve_hierarchy_path(self, df: pl.DataFrame) -> list[str]:
        raw_hierarchy = self.spec.channels.hierarchy or self.spec.options.get("path") or [self.spec.channels.x, self.spec.channels.color]
        path_cols: list[str] = []
        if isinstance(raw_hierarchy, (list, tuple)):
            for item in raw_hierarchy:
                if isinstance(item, str) and item.strip():
                    path_cols.append(item.strip())
                elif isinstance(item, list):
                    path_cols.extend(sub.strip() for sub in item if isinstance(sub, str) and sub.strip())
        elif isinstance(raw_hierarchy, str) and raw_hierarchy.strip():
            path_cols.append(raw_hierarchy.strip())

        schema = df.schema
        return [column for column in path_cols if schema.get(column) is not None]

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        norm_mode = str(self.spec.options.get("normalize", "zscore")).lower()
        if norm_mode == "minmax":
            column_min = matrix.min(axis=0)
            column_max = matrix.max(axis=0)
            column_diff = np.where(column_max - column_min == 0.0, 1.0, column_max - column_min)
            norm_matrix = (matrix - column_min) / column_diff
        else:
            column_mean = matrix.mean(axis=0)
            column_std = np.where(matrix.std(axis=0) == 0.0, 1.0, matrix.std(axis=0))
            norm_matrix = (matrix - column_mean) / column_std
        return np.nan_to_num(norm_matrix, nan=0.0)