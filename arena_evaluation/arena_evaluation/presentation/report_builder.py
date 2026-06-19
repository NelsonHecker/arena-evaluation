from __future__ import annotations

import pathlib
from typing import Any
import datetime
import jinja2
import polars as pl

from .viz_manifest import VizManifest
from .plotly_renderer import PlotlyRenderer
from .seaborn_renderer import SeabornRenderer
from ..processing.parquet_store import ParquetStore


class ReportBuilder:
    """
    Generates the final interactive HTML report and static PNG plots.
    """

    def __init__(
        self,
        benchmark_dir: pathlib.Path,
        *,
        output_dir: pathlib.Path | None = None,
        generate_gifs: bool = False,
    ):
        self.benchmark_dir = pathlib.Path(benchmark_dir)
        # output_dir defaults to benchmark_dir so existing callers are unaffected
        self.output_dir = pathlib.Path(output_dir) if output_dir else self.benchmark_dir
        self.plots_dir = self.output_dir / "plots"
        self.report_path = self.output_dir / "report.html"
        self.manifest_path = self.benchmark_dir / "viz_manifest.yaml"
        self.generate_gifs = generate_gifs

    # ------------------------------------------------------------------
    # Multi-source use
    # ------------------------------------------------------------------

    @classmethod
    def from_dirs(
        cls,
        source_dirs: list[pathlib.Path],
        output_dir: pathlib.Path,
        *,
        manifest_path: pathlib.Path | None = None,
        generate_gifs: bool = False,
    ) -> "ReportBuilder":
        """
        Build a ReportBuilder that merges data from multiple source directories.
        """
        instance = cls.__new__(cls)
        instance.benchmark_dir = source_dirs[0] if source_dirs else output_dir
        instance.output_dir = pathlib.Path(output_dir)
        instance.plots_dir = instance.output_dir / "plots"
        instance.report_path = instance.output_dir / "report.html"
        instance.generate_gifs = generate_gifs

        # Find manifest: explicit → first source with one → default
        if manifest_path and pathlib.Path(manifest_path).exists():
            instance.manifest_path = pathlib.Path(manifest_path)
        else:
            instance.manifest_path = None  # type: ignore[assignment]
            for src in source_dirs:
                candidate = pathlib.Path(src) / "viz_manifest.yaml"
                if candidate.exists():
                    instance.manifest_path = candidate
                    break
            if not instance.manifest_path:
                # Sentinel: VizManifest.load falls back to default when file absent
                instance.manifest_path = output_dir / "viz_manifest.yaml"

        # Pre-merge the DataFrames from all source dirs
        instance._merged_df = cls._load_and_merge(source_dirs)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_parquet(directory: pathlib.Path) -> pathlib.Path | None:
        """Return the best available metrics parquet in *directory*, or None."""
        for name in ("combined_metrics.parquet", "metrics.parquet"):
            p = directory / name
            if p.exists():
                return p
        return None

    @classmethod
    def _load_and_merge(cls, source_dirs: list[pathlib.Path]) -> pl.DataFrame | None:
        """Load and concatenate parquet files from multiple source directories."""
        dfs: list[pl.DataFrame] = []
        for src in source_dirs:
            p = cls._best_parquet(pathlib.Path(src))
            if p is None:
                print(f"  [warn] No metrics parquet found in {src}, skipping.")
                continue
            try:
                df, _ = ParquetStore.read(p)
                dfs.append(df)
                print(f"  Loaded {len(df)} rows from {p}")
            except Exception as e:
                print(f"  [warn] Failed to read {p}: {e}")
        if not dfs:
            return None
        if len(dfs) == 1:
            return dfs[0]
        return pl.concat(dfs, how="diagonal_relaxed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Execute the report building process."""
        # Resolve DataFrame: use pre-merged if available (multi-source), else load from disk
        if hasattr(self, "_merged_df") and self._merged_df is not None:
            df = self._merged_df
        else:
            combined_metrics_path = self.benchmark_dir / "combined_metrics.parquet"
            metrics_path = self.benchmark_dir / "metrics.parquet"

            target_path = None
            if combined_metrics_path.exists():
                target_path = combined_metrics_path
            elif metrics_path.exists():
                target_path = metrics_path

            if target_path is None:
                print(
                    f"Cannot generate report: neither combined_metrics.parquet nor "
                    f"metrics.parquet found in {self.benchmark_dir}."
                )
                return

            df, _ = ParquetStore.read(target_path)

        # Ensure local_planner and inter_planner are present (for backward compatibility with older runs)
        if "planner" in df.columns:
            if "local_planner" not in df.columns or "inter_planner" not in df.columns:
                from .dimension_detector import split_planner_name
                # Use split_planner_name to populate the columns
                lp_list = []
                ip_list = []
                for p_val in df["planner"].to_list():
                    lp, ip = split_planner_name(p_val)
                    lp_list.append(lp)
                    ip_list.append(ip)
                df = df.with_columns([
                    pl.Series("local_planner", lp_list),
                    pl.Series("inter_planner", ip_list)
                ])

        manifest = VizManifest.load(self.manifest_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        from ..processing.metrics.registry import MetricRegistry
        units = MetricRegistry.get_all_units()

        plotly_renderer = PlotlyRenderer(units=units)
        seaborn_renderer = SeabornRenderer(generate_gifs=self.generate_gifs, units=units)

        html_plots = []

        for spec in manifest.plots:
            # Check if metric data is available for this plot
            if spec.data_key != "*":
                if spec.data_key not in df.columns:
                    print(f"Skipping plot '{spec.id}': data key '{spec.data_key}' not found in metrics.")
                    continue
                if df[spec.data_key].null_count() == len(df):
                    print(f"Skipping plot '{spec.id}': data key '{spec.data_key}' has no calculated data (all values null).")
                    continue

            # Generate static PNG
            png_path = self.plots_dir / f"{spec.id}.png"
            try:
                seaborn_renderer.render(spec, df, png_path, run_dir=self.output_dir)
            except Exception as e:
                print(f"Warning: Failed to render static plot {spec.id}: {e}")

            # Generate interactive HTML chunk
            try:
                html_chunk = plotly_renderer.render(spec, df, run_dir=self.output_dir)
                if html_chunk:
                    if isinstance(html_chunk, list):
                        for chunk in html_chunk:
                            html_plots.append((spec.layout_group, chunk, spec.title))
                    else:
                        html_plots.append((spec.layout_group, html_chunk, spec.title))
            except Exception as e:
                print(f"Warning: Failed to render interactive plot {spec.id}: {e}")

        # Generate summary table
        summary_html = self._generate_summary_table(df)

        # Derive a display name for the report header
        report_title = self.output_dir.name

        # Write final HTML
        html_content = self._assemble_html(summary_html, html_plots, report_title)
        with open(self.report_path, "w") as f:
            f.write(html_content)

        # Write local plotly.min.js to keep html file size small but work completely offline
        import plotly.offline
        js_path = self.output_dir / "plotly.min.js"
        with open(js_path, "w") as f:
            f.write(plotly.offline.get_plotlyjs())

        print(f"Report generated successfully: {self.report_path}")
        print(f"Static plots saved to: {self.plots_dir}")

    def _generate_summary_table(self, df: pl.DataFrame) -> str:
        if "planner" not in df.columns:
            return ""

        # Determine which identity dimensions vary — group by all of them for the summary
        from .dimension_detector import detect_varying_dims, IDENTITY_COLS
        varying = detect_varying_dims(df)
        group_cols = varying if varying else ["planner"]

        # Only include identity cols that actually exist
        group_cols = [c for c in group_cols if c in df.columns]
        if not group_cols:
            group_cols = ["planner"]

        # Group by planner and calculate success rate, avg time, avg path length dynamically
        agg_exprs = []
        if "success" in df.columns and not df["success"].is_null().all():
            agg_exprs.append(pl.col("success").mean().alias("success_rate"))
        if "time_to_goal" in df.columns and not df["time_to_goal"].is_null().all():
            agg_exprs.append(pl.col("time_to_goal").mean().alias("avg_time"))
        if "path_length" in df.columns and not df["path_length"].is_null().all():
            agg_exprs.append(pl.col("path_length").mean().alias("avg_path_length"))
        if "collision_amount" in df.columns and not df["collision_amount"].is_null().all():
            agg_exprs.append(pl.col("collision_amount").mean().alias("avg_collisions"))

        if not agg_exprs:
            return ""

        summary = df.group_by(group_cols).agg(agg_exprs).sort(group_cols).to_pandas()

        # Format columns
        import pandas as pd
        if "success_rate" in summary.columns:
            summary["success_rate"] = summary["success_rate"].map(
                lambda x: f"{x * 100:.1f}%" if not pd.isna(x) else "N/A"
            )
        if "avg_time" in summary.columns:
            summary["avg_time"] = summary["avg_time"].map(
                lambda x: f"{x:.2f}s" if not pd.isna(x) else "N/A"
            )
        if "avg_path_length" in summary.columns:
            summary["avg_path_length"] = summary["avg_path_length"].map(
                lambda x: f"{x:.2f}m" if not pd.isna(x) else "N/A"
            )
        if "avg_collisions" in summary.columns:
            summary["avg_collisions"] = summary["avg_collisions"].map(
                lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
            )

        return summary.to_html(index=False, classes="dataframe")

    def _assemble_html(
        self,
        summary_html: str,
        plot_htmls: list[tuple[str | None, str, str]],
        benchmark_id: str
    ) -> str:
        """Template for the final HTML report using Jinja2."""
        # Group plots by layout_group
        grouped_plots = {}
        ordered_groups = []
        for group, html, title in plot_htmls:
            group_id = group if group else "details"
            # Overview plots are handled separately
            if group_id == "overview":
                continue
            if group_id not in grouped_plots:
                grouped_plots[group_id] = []
                ordered_groups.append(group_id)
            grouped_plots[group_id].append({
                "title": title,
                "html": html
            })

        group_titles = {
            "efficiency": "Efficiency Metrics",
            "safety": "Safety & Collision Metrics",
            "motion": "Motion Dynamics Metrics",
            "smoothness": "Path Smoothness Metrics",
            "social": "Social & Pedestrian Interaction",
            "details": "Detailed Run Traces & Analysis",
            "robot_analysis": "Robot Model Performance",
            "stage_analysis": "Stage Level Performance",
        }

        plot_groups = []
        for group_id in ordered_groups:
            plots = grouped_plots[group_id]
            title = group_titles.get(group_id, group_id.replace("_", " ").title())
            plot_groups.append({
                "id": group_id,
                "title": title,
                "plots": plots
            })

        # Gather overview plots
        overview_plots = [html for group, html, title in plot_htmls if group == "overview"]

        # Load Jinja2 template relative to this file
        template_dir = pathlib.Path(__file__).parent
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))
        template = env.get_template("report_template.html.j2")

        generated_on = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return template.render(
            benchmark_id=benchmark_id,
            report_dir=str(self.output_dir.resolve()),
            generated_on=generated_on,
            summary_html=summary_html,
            overview_plots=overview_plots,
            plot_groups=plot_groups
        )


