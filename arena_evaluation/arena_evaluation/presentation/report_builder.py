from __future__ import annotations

import pathlib
import datetime
import typing

import jinja2
import polars as pl

from .viz_manifest import VizManifest
from .manifest_registry import resolve_manifest
from .plotly_renderer import PlotlyRenderer
from .seaborn_renderer import SeabornRenderer
from ..processing.parquet_store import ParquetStore

# Data source name -> parquet filename. Any other data_source string ending in
# ".parquet" is used verbatim as the filename in the benchmark/output dir.
_DATA_FILES = {
    "metrics": None,
}

# Legacy group headings, used verbatim when a manifest declares no groups.
_LEGACY_GROUP_TITLES = {
    "efficiency": "Efficiency Metrics",
    "safety": "Safety & Collision Metrics",
    "motion": "Motion Dynamics Metrics",
    "smoothness": "Path Smoothness Metrics",
    "social": "Social & Pedestrian Interaction",
    "details": "Detailed Run Traces & Analysis",
    "robot_analysis": "Robot Model Performance",
    "stage_analysis": "Stage Level Performance",
}


def data_file_for(data_source: str | None) -> str | None:
    """Map a manifest data_source to a parquet filename (None = legacy search)."""
    if not data_source:
        return None
    if data_source in _DATA_FILES:
        return _DATA_FILES[data_source]
    if data_source.endswith(".parquet"):
        return data_source
    return None


class ReportBuilder:
    """Generates the final interactive HTML report and static PNG plots."""

    def __init__(
        self,
        benchmark_dir: pathlib.Path,
        *,
        output_dir: pathlib.Path | None = None,
        generate_gifs: bool = False,
        manifest: VizManifest | str | None = None,
    ):
        self.benchmark_dir = pathlib.Path(benchmark_dir)
        self.output_dir = pathlib.Path(output_dir) if output_dir else self.benchmark_dir
        self.plots_dir = self.output_dir / "plots"
        self.report_path = self.output_dir / "report.html"
        self.manifest_path = self.benchmark_dir / "viz_manifest.yaml"
        self.generate_gifs = generate_gifs
        # Explicit manifest object (from_dirs sets this); build() falls back to
        # VizManifest.load(self.manifest_path) when None (keeps mock-based
        # tests working).
        self._manifest_obj = manifest if isinstance(manifest, VizManifest) else None
        self._source_frames: dict[str, pl.DataFrame] = {}

    @classmethod
    def from_dirs(
        cls,
        source_dirs: list[pathlib.Path],
        output_dir: pathlib.Path,
        *,
        manifest_path: pathlib.Path | None = None,
        manifest: VizManifest | str | None = None,
        generate_gifs: bool = False,
    ) -> "ReportBuilder":
        """Build a ReportBuilder that merges data from multiple source directories."""
        instance = cls.__new__(cls)
        instance.benchmark_dir = source_dirs[0] if source_dirs else output_dir
        instance.output_dir = pathlib.Path(output_dir)
        instance.plots_dir = instance.output_dir / "plots"
        instance.report_path = instance.output_dir / "report.html"
        instance.generate_gifs = generate_gifs
        instance._source_frames = {}
        instance._manifest_obj = None

        if manifest is not None:
            instance._manifest_obj = (
                manifest
                if isinstance(manifest, VizManifest)
                else resolve_manifest(manifest, benchmark_dir=instance.benchmark_dir)
            )
            instance.manifest_path = None
        else:
            if manifest_path and pathlib.Path(manifest_path).exists():
                instance.manifest_path = pathlib.Path(manifest_path)
            else:
                instance.manifest_path = None
                for src in source_dirs:
                    candidate = pathlib.Path(src) / "viz_manifest.yaml"
                    if candidate.exists():
                        instance.manifest_path = candidate
                        break
                if not instance.manifest_path:
                    instance.manifest_path = output_dir / "viz_manifest.yaml"

        data_file = None
        if instance._manifest_obj is not None:
            data_file = data_file_for(instance._manifest_obj.data_source)
        instance._merged_df = cls._load_and_merge(source_dirs, data_file=data_file)
        return instance

    @staticmethod
    def _best_parquet(directory: pathlib.Path) -> pathlib.Path | None:
        """Return the best available metrics parquet in *directory*, or None."""
        for name in ("combined_metrics.parquet", "metrics.parquet"):
            p = directory / name
            if p.exists():
                return p
        return None

    @classmethod
    def _load_and_merge(
        cls,
        source_dirs: list[pathlib.Path],
        data_file: str | None = None,
    ) -> pl.DataFrame | None:
        """Load and concatenate parquet files from multiple source directories."""
        dfs: list[pl.DataFrame] = []
        for src in source_dirs:
            if data_file:
                p = pathlib.Path(src) / data_file
                if not p.exists():
                    print(f"  [warn] No {data_file} found in {src}, skipping.")
                    continue
            else:
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


    def _load_primary_frame(self, manifest: VizManifest) -> pl.DataFrame | None:
        """Load the manifest's primary data frame (from benchmark_dir)."""
        data_file = data_file_for(manifest.data_source)
        if data_file:
            target_path = self.benchmark_dir / data_file
            if not target_path.exists():
                print(
                    f"Cannot generate report: '{data_file}' (data_source="
                    f"{manifest.data_source!r}) not found in {self.benchmark_dir}."
                )
                return None
            df, _ = ParquetStore.read(target_path)
            return df

        combined_path = self.benchmark_dir / "combined_metrics.parquet"
        metrics_path = self.benchmark_dir / "metrics.parquet"
        target_path = None
        if combined_path.exists():
            target_path = combined_path
        elif metrics_path.exists():
            target_path = metrics_path
        if target_path is None:
            print(
                f"Cannot generate report: neither combined_metrics.parquet nor "
                f"metrics.parquet found in {self.benchmark_dir}."
            )
            return None
        df, _ = ParquetStore.read(target_path)
        return df

    def _frame_for_spec(
        self,
        spec,
        manifest: VizManifest,
        df: pl.DataFrame,
        df_contestants: pl.DataFrame,
    ) -> pl.DataFrame | None:
        """Pick the frame for one plot (per-plot data_source override support)."""
        src = spec.data_source
        if src is None or src == manifest.data_source:
            return df if spec.type in ("trajectory", "timeseries", "line", "table") else df_contestants
        if src not in self._source_frames:
            data_file = data_file_for(src)
            if not data_file:
                print(f"Skipping plot '{spec.id}': unknown data source '{src}'.")
                return None
            p = self.benchmark_dir / data_file
            if not p.exists():
                print(f"Skipping plot '{spec.id}': '{data_file}' not found in {self.benchmark_dir}.")
                return None
            self._source_frames[src] = ParquetStore.read(p)[0]
        return self._source_frames[src]


    def build(self) -> None:
        """Execute the report building process."""
        manifest = self._manifest_obj if self._manifest_obj is not None else VizManifest.load(self.manifest_path)

        if hasattr(self, "_merged_df") and self._merged_df is not None:
            df = self._merged_df
        else:
            df = self._load_primary_frame(manifest)
            if df is None:
                return

        if "planner" in df.columns:
            if "local_planner" not in df.columns or "inter_planner" not in df.columns:
                from .dimension_detector import split_planner_name
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

        # Separate contestant evaluation runs from reference runs (metrics only;
        # characterization frames have no is_reference column -> no-op).
        if "is_reference" in df.columns:
            df_contestants = df.filter(pl.col("is_reference").is_null() | (pl.col("is_reference") == False))
            if len(df_contestants) == 0:
                df_contestants = df
        else:
            df_contestants = df

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        from ..processing.metrics.registry import MetricRegistry
        units = MetricRegistry.get_all_units()
        units.update(manifest.units or {})

        plotly_renderer = PlotlyRenderer(units=units)
        seaborn_renderer = SeabornRenderer(generate_gifs=self.generate_gifs, units=units)

        html_plots = []

        for spec in manifest.plots:
            plot_df = self._frame_for_spec(spec, manifest, df, df_contestants)
            if plot_df is None:
                continue

            if spec.data_key != "*":
                if spec.data_key not in plot_df.columns:
                    print(f"Skipping plot '{spec.id}': data key '{spec.data_key}' not found in metrics.")
                    continue
                if plot_df[spec.data_key].null_count() == len(plot_df):
                    print(f"Skipping plot '{spec.id}': data key '{spec.data_key}' has no calculated data (all values null).")
                    continue

            png_path = self.plots_dir / f"{spec.id}.png"
            try:
                seaborn_renderer.render(spec, plot_df, png_path, run_dir=self.output_dir)
            except Exception as e:
                print(f"Warning: Failed to render static plot {spec.id}: {e}")

            try:
                html_chunk = plotly_renderer.render(spec, plot_df, run_dir=self.output_dir)
                if html_chunk:
                    if isinstance(html_chunk, list):
                        for chunk in html_chunk:
                            html_plots.append((spec.layout_group, chunk, spec.title))
                    else:
                        html_plots.append((spec.layout_group, html_chunk, spec.title))
            except Exception as e:
                print(f"Warning: Failed to render interactive plot {spec.id}: {e}")

        if manifest.summary:
            summary_html = self._generate_summary_table_manifest(df, manifest)
        else:
            summary_html = self._generate_summary_table(df_contestants)

        report_title = manifest.title or self.output_dir.name

        html_content = self._assemble_html(summary_html, html_plots, report_title, manifest)
        with open(self.report_path, "w") as f:
            f.write(html_content)

        import plotly.offline
        js_path = self.output_dir / "plotly.min.js"
        with open(js_path, "w") as f:
            f.write(plotly.offline.get_plotlyjs())

        self._write_note_file(manifest)

        print(f"Report generated successfully: {self.report_path}")
        print(f"Static plots saved to: {self.plots_dir}")

    def _write_note_file(self, manifest: VizManifest) -> None:
        """Best-effort note recording which manifest produced this report."""
        if not manifest.name:
            return
        try:
            import yaml
            note = {
                "name": manifest.name,
                "title": manifest.title,
                "data_source": manifest.data_source,
                "n_plots": len(manifest.plots),
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            (self.output_dir / "report_manifest.yaml").write_text(
                yaml.safe_dump(note, sort_keys=False)
            )
        except Exception as e:
            print(f"Warning: failed to write report_manifest.yaml note: {e}")


    def _generate_summary_table(self, df: pl.DataFrame) -> str:
        """Legacy summary table (used when the manifest declares no summary)."""
        if "planner" not in df.columns:
            return ""

        from .dimension_detector import detect_varying_dims, IDENTITY_COLS
        varying = detect_varying_dims(df)
        group_cols = varying if varying else ["planner"]

        group_cols = [c for c in group_cols if c in df.columns]
        if not group_cols:
            group_cols = ["planner"]

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

    def _generate_summary_table_manifest(self, df: pl.DataFrame, manifest: VizManifest) -> str:
        """Declarative summary table: one column per SummarySpec."""
        group_by = manifest.summary_group_by
        if isinstance(group_by, str):
            group_by = [group_by]
        group_cols = [c for c in (group_by or []) if c in df.columns]

        if not group_cols:
            from .dimension_detector import detect_varying_dims
            varying = detect_varying_dims(df)
            group_cols = [c for c in varying if c in df.columns]

        if not group_cols:
            return ""

        # Wide per-episode list columns (e.g. timeseries_char_*) are exploded
        # in lockstep so per-sample metrics aggregate correctly per group.
        list_cols = [c for c in [*group_cols, *(s.metric for s in manifest.summary)] if c in df.columns and df.schema[c] == pl.List]
        if list_cols:
            df = df.explode(list_cols)

        agg_exprs = []
        for spec in manifest.summary:
            if spec.metric in df.columns and not df[spec.metric].is_null().all():
                agg_exprs.append(pl.col(spec.metric).mean().alias(spec.metric))
        if not agg_exprs:
            return ""

        summary = df.group_by(group_cols).agg(agg_exprs).sort(group_cols).to_pandas()

        import pandas as pd
        spec_by_metric = {s.metric: s for s in manifest.summary}
        rename = {}
        for metric in summary.columns:
            if metric in group_cols:
                continue
            spec = spec_by_metric.get(metric)
            if spec is None:
                continue
            fmt = spec.format
            summary[metric] = summary[metric].map(
                lambda x: fmt.format(x) if not pd.isna(x) else "N/A"
            )
            rename[metric] = spec.label
        summary = summary.rename(columns=rename)

        return summary.to_html(index=False, classes="dataframe")

    def _assemble_html(
        self,
        summary_html: str,
        plot_htmls: list[tuple[str | None, str, str]],
        report_title: str,
        manifest: VizManifest,
    ) -> str:
        """Template for the final HTML report using Jinja2."""
        grouped_plots = {}
        ordered_groups = []
        for group, html, title in plot_htmls:
            group_id = group if group else "details"
            if group_id == "overview":
                continue
            if group_id not in grouped_plots:
                grouped_plots[group_id] = []
                ordered_groups.append(group_id)
            grouped_plots[group_id].append({
                "title": title,
                "html": html
            })

        group_titles = {g.id: g.title for g in manifest.groups} or _LEGACY_GROUP_TITLES

        plot_groups = []
        for group_id in ordered_groups:
            plots = grouped_plots[group_id]
            title = group_titles.get(group_id, group_id.replace("_", " ").title())
            plot_groups.append({
                "id": group_id,
                "title": title,
                "plots": plots
            })

        overview_plots = [html for group, html, title in plot_htmls if group == "overview"]

        template_dir = pathlib.Path(__file__).parent
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))
        template = env.get_template("report_template.html.j2")

        generated_on = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return template.render(
            benchmark_id=report_title,
            report_dir=str(self.output_dir.resolve()),
            generated_on=generated_on,
            summary_html=summary_html,
            overview_plots=overview_plots,
            plot_groups=plot_groups
        )
