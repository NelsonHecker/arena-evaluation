from __future__ import annotations

import pathlib

import plotly.express as px
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette

from .base import BasePlotRenderer

__all__ = ["BarRenderer"]


class BarRenderer(BasePlotRenderer):
    """Bar plot renderer supporting grouped, stacked, and multi-metric charts."""

    PLOT_TYPE = "bar"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y", fallback=self.spec.data_key)
        color_col = self.resolve_channel("color")
        facet_col = self.resolve_channel("facet_col")
        facet_row = self.resolve_channel("facet_row")

        if not x_col and not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            x_col = diff_col
            color_col = diff_col
        elif not x_col and color_col:
            x_col = color_col

        if not x_col or df_filtered.schema.get(x_col) is None:
            return None

        is_stacked = bool(self.spec.options.get("stacked", False) or self.spec.options.get("barmode") == "stack")
        barmode = self.spec.options.get("barmode", "stack" if is_stacked else "group")

        metrics = self.spec.options.get("metrics", [])
        if not metrics and isinstance(self.spec.channels.y, list):
            metrics = self.spec.channels.y

        # Mode A: Multi-metric composition
        if metrics:
            schema = df_filtered.schema
            valid_metrics = [m for m in metrics if schema.get(m) is not None]
            if not valid_metrics:
                return None

            agg_exprs = [pl.col(m).mean().alias(m) for m in valid_metrics]
            grouped = df_filtered.group_by(x_col).agg(agg_exprs).to_pandas()
            if grouped.empty:
                return None

            if self.spec.options.get("normalize", False):
                total = grouped[valid_metrics].sum(axis=1)
                total_mask = total > 0
                grouped.loc[total_mask, valid_metrics] = grouped.loc[total_mask, valid_metrics].div(total[total_mask], axis=0) * 100.0

            fig = px.bar(
                grouped,
                x=x_col,
                y=valid_metrics,
                template="plotly_white",
                barmode=barmode,
                labels={
                    "value": "Percentage (%)" if self.spec.options.get("normalize") else "Value",
                    "variable": "Metric",
                    x_col: x_col.lstrip("_").replace("_", " ").title(),
                },
            )
            fig.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=40, b=40, l=50, r=30),
            )
            return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

        schema = df_filtered.schema
        if not y_col or schema.get(y_col) is None:
            return None

        # Mode B: Categorical label counting
        if schema[y_col] in (pl.Utf8, pl.Categorical):
            grp = [c for c in [x_col, color_col, facet_col, facet_row] if c and schema.get(c) is not None]
            count_df = df_filtered.group_by(list(dict.fromkeys(grp))).agg(pl.len().alias("count"))
            if is_stacked and self.spec.options.get("normalize", False):
                count_df = count_df.with_columns((pl.col("count") / pl.col("count").sum().over(x_col) * 100.0).alias("share_pct"))
                plot_y = "share_pct"
                y_label = "Share (%)"
            else:
                plot_y = "count"
                y_label = "Count"

            pdf = count_df.to_pandas()
            if pdf.empty:
                return None

            fig = px.bar(
                pdf,
                x=x_col,
                y=plot_y,
                color=color_col if color_col != x_col else None,
                facet_col=facet_col if facet_col and facet_col in count_df.columns else None,
                facet_row=facet_row if facet_row and facet_row in count_df.columns else None,
                barmode=barmode,
                template="plotly_white",
                labels={
                    plot_y: y_label,
                    x_col: x_col.lstrip("_").replace("_", " ").title(),
                },
            )
            fig.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=40, b=40, l=50, r=30),
            )
            return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

        # Mode C: Numeric metric (pre-aggregated or raw observations)
        grp = [c for c in [x_col, color_col, facet_col, facet_row] if c and schema.get(c) is not None]
        unique_grp = list(dict.fromkeys(grp))

        # Check if already pre-aggregated (no duplicates on grouping keys)
        max_rows_per_group = df_filtered.group_by(unique_grp).agg(pl.len().alias("cnt"))["cnt"].max() if unique_grp else 1
        is_pre_aggregated = max_rows_per_group == 1

        labels = {
            y_col: self.format_label(y_col.replace("_", " ").title(), y_col),
            x_col: x_col.lstrip("_").replace("_", " ").title(),
        }
        if color_col:
            labels[color_col] = color_col.lstrip("_").replace("_", " ").title()

        if is_pre_aggregated:
            pdf = df_filtered.select(list(dict.fromkeys(unique_grp + [y_col]))).to_pandas()
            fig = px.bar(
                pdf,
                x=x_col,
                y=y_col,
                color=color_col if color_col != x_col else None,
                facet_col=facet_col if facet_col and facet_col in pdf.columns else None,
                facet_row=facet_row if facet_row and facet_row in pdf.columns else None,
                barmode=barmode,
                template="plotly_white",
                labels=labels,
            )
        else:
            grouped = self._grouped_with_error(df_filtered, unique_grp, y_col).to_pandas()
            if grouped.empty:
                return None
            labels["mean"] = labels[y_col]
            fig = px.bar(
                grouped,
                x=x_col,
                y="mean",
                color=color_col if color_col and color_col in grouped.columns and color_col != x_col else None,
                facet_col=facet_col if facet_col and facet_col in grouped.columns else None,
                facet_row=facet_row if facet_row and facet_row in grouped.columns else None,
                barmode=barmode,
                error_y="err_plus",
                error_y_minus="err_minus",
                template="plotly_white",
                labels=labels,
            )

        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=50, r=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def _grouped_with_error(self, df_filtered: pl.DataFrame, group_col: str | list[str], metric_col: str) -> pl.DataFrame:
        """Compute per-group mean with vectorized error margins (Wilson score for bool, std otherwise)."""
        z = 1.96
        key = metric_col
        schema = df_filtered.schema
        cols = [group_col] if isinstance(group_col, str) else group_col
        if schema[key] == pl.Boolean:
            p = pl.col(key).cast(pl.Float64).mean()
            n = pl.col(key).count()
            denom = 1.0 + (z * z) / n
            center = (p + (z * z) / (2.0 * n)) / denom
            radicand = (p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)
            half = z * radicand.sqrt() / denom
            return df_filtered.group_by(cols).agg(
                [
                    p.alias("mean"),
                    (center + half - p).clip(lower_bound=0.0).alias("err_plus"),
                    (p - (center - half)).clip(lower_bound=0.0).alias("err_minus"),
                ]
            )

        return df_filtered.group_by(cols).agg(
            [
                pl.col(key).mean().alias("mean"),
                pl.col(key).std().fill_null(0.0).alias("err_plus"),
                pl.col(key).std().fill_null(0.0).alias("err_minus"),
            ]
        )

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)

        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y", fallback=self.spec.data_key)
        color_col = self.resolve_channel("color")

        if not x_col and not color_col:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            x_col = diff_col
            color_col = diff_col
        elif not x_col and color_col:
            x_col = color_col

        schema = df_filtered.schema
        if not y_col or schema.get(y_col) is None or not x_col or schema.get(x_col) is None:
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 6))
        palette = get_color_palette() if color_col else None

        if schema[y_col] == pl.Boolean:
            grp_cols = [x_col] if (not color_col or color_col == x_col) else [x_col, color_col]
            grouped = self._grouped_with_error(df_filtered, grp_cols, y_col).to_pandas()
            sns.barplot(
                data=grouped,
                x=x_col,
                y="mean",
                hue=color_col if color_col != x_col else None,
                palette=palette,
                ax=ax,
            )
        else:
            keep = [c for c in [y_col, x_col, color_col] if c and schema.get(c) is not None]
            pdf = df_filtered.select(list(dict.fromkeys(keep))).to_pandas()
            if pdf.empty:
                plt.close(fig)
                return
            sns.barplot(
                data=pdf,
                x=x_col,
                y=y_col,
                hue=color_col if color_col != x_col else None,
                palette=palette,
                errorbar="sd",
                ax=ax,
            )

        sns.despine(fig=fig, top=True, right=True)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.set_title(self.spec.title, pad=12, fontsize=12, fontweight="bold")
        ax.set_ylabel(self.format_label(y_col.replace("_", " ").title(), y_col))
        ax.set_xlabel(self.format_label(x_col.replace("_", " ").title(), x_col))
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
