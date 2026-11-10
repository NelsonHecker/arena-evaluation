from __future__ import annotations

import dataclasses
import datetime
import html as html_escape
import pathlib
import typing

import jinja2
import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.presentation.checks import PlotDiagnostics
from arena_evaluation.presentation.data_sources import EPISODE_SUMMARY_VIEW, data_source_table
from arena_evaluation.presentation.manifest_registry import resolve_manifest
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import VizManifest
from arena_evaluation.processing.metrics.registry import MetricRegistry

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import PlotSpec

_EPISODE_TABLE: typing.Final[str] = EPISODE_SUMMARY_VIEW


def _has_values(df: pl.DataFrame, col: str) -> bool:
    """Check if column exists and contains at least one non-null value."""
    return col in df.columns and bool(df[col].is_not_null().any())


_STATUS_COLS = ("stage", "planner", "episode_id", "robot_id", "result", "null_reason")


def _status_table_html(df: pl.DataFrame) -> str:
    """Rows of the canonical episode summary that were never evaluated, empty when every row was."""
    rows = df.filter(pl.col("success").is_null())
    if rows.is_empty():
        return ""
    rows = rows.select(*_STATUS_COLS).sort("stage", "episode_id")
    head = "".join(f"<th>{html_escape.escape(name.replace('_', ' '))}</th>" for name in _STATUS_COLS)
    body = "".join("<tr>" + "".join(f"<td>{html_escape.escape('' if value is None else str(value))}</td>" for value in row) + "</tr>" for row in rows.iter_rows())
    return f"<h3>Episodes not fully evaluated ({len(rows)} of {len(df)})</h3><table class=\"dataframe\"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _default_summary_group_cols(df: pl.DataFrame) -> list[str]:
    """Determine grouping columns for summary tables."""
    from .dimension_detector import detect_varying_dims

    lead = next((c for c in ("local_planner", "planner") if _has_values(df, c)), None)
    rest = [c for c in detect_varying_dims(df) if c != lead and _has_values(df, c)]
    return ([lead] if lead else []) + rest


def _aggregate(df: pl.DataFrame, group_cols: list[str], agg_exprs: list[pl.Expr]) -> pl.DataFrame:
    """Grouped aggregate, or a single ``All runs`` row when nothing identifies a run."""
    if group_cols:
        return df.group_by(group_cols).agg(agg_exprs).sort(group_cols)
    out = df.select(agg_exprs)
    return out.select(pl.lit("All runs", dtype=pl.String).alias("runs"), *[pl.col(name) for name in out.columns])


def _summary_html(table: pl.DataFrame, formats: dict[str, str], labels: dict[str, str]) -> str:
    """Render a grouped summary as an HTML table, one formatted cell per metric and group key."""
    head = "".join(f"<th>{html_escape.escape(labels.get(name, name))}</th>" for name in table.columns)
    body = []
    for row in table.iter_rows(named=True):
        cells = []
        for name, value in row.items():
            if isinstance(value, float):
                cells.append(f"<td>{html_escape.escape(_format_metric(value, formats.get(name)))}</td>")
            else:
                cells.append(f"<td>{html_escape.escape('' if value is None else str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="dataframe"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _format_metric(value: float, template: str | None) -> str:
    """Format one aggregate cell, N/A when it is missing or the column declares no format."""
    if value != value:
        return "N/A"
    return template.format(value) if template else f"{value:.3f}"


def _labelled(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Make grouping keys renderable: nulls become an explicit label.

    List columns are left alone; they are exploded before they are grouped on.
    """
    scalar = [c for c in group_cols if c in df.columns and df.schema[c] != pl.List]
    if not scalar:
        return df
    return df.with_columns([pl.col(c).cast(pl.Utf8).fill_null("unknown") for c in scalar])


def _render_markdown_light(text: str) -> str:
    """Minimal safe markdown -> HTML for agent notes.

    Escapes everything first, then applies: **bold**, *italic*, `code`,
    [text](url) links, and newline -> <br>.
    """
    import html as _html
    import re as _re

    out = _html.escape(text, quote=True)
    out = _re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', out)
    out = _re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = _re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    out = _re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = out.replace("\n", "<br>")
    return out


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _CatalogSources:
    """The benchmark catalogs one report reads, one catalog per source directory."""

    catalogs: tuple[BenchmarkCatalog, ...]

    def __post_init__(self) -> None:
        if not self.catalogs:
            raise ValueError("A report needs at least one benchmark catalog.")

    def frame(self, table: str) -> pl.DataFrame | None:
        """Concatenate the named Star Schema table or view across every source, None when no catalog knows it."""
        frames = [catalog.get_table(table).collect() for catalog in self.catalogs if table in catalog.table_names()]
        if not frames:
            return None
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")


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
        if isinstance(manifest, (str, pathlib.Path)):
            p = pathlib.Path(manifest)
            if p.exists():
                self.manifest_path = p
        elif not self.manifest_path.exists():
            for alt in ("manifest.yaml", "viz_manifest.yml", "manifest.yml"):
                p = self.benchmark_dir / alt
                if p.exists():
                    self.manifest_path = p
                    break
        self.generate_gifs = generate_gifs
        self._manifest_obj = manifest if isinstance(manifest, VizManifest) else None
        self._sources = _CatalogSources(catalogs=(BenchmarkCatalog.from_directory(self.benchmark_dir),))
        self._source_frames: dict[str, pl.DataFrame] = {}
        self._merged_df: pl.DataFrame | None = None
        self._agent_notes: dict[str, str] | None = None

    @classmethod
    def from_dirs(
        cls,
        source_dirs: list[pathlib.Path],
        output_dir: pathlib.Path,
        *,
        manifest_path: pathlib.Path | None = None,
        manifest: VizManifest | str | None = None,
        generate_gifs: bool = False,
    ) -> ReportBuilder:
        """Build a ReportBuilder that merges data from multiple source directories."""
        instance = cls.__new__(cls)
        instance.benchmark_dir = source_dirs[0] if source_dirs else output_dir
        instance.output_dir = pathlib.Path(output_dir)
        instance.plots_dir = instance.output_dir / "plots"
        instance.report_path = instance.output_dir / "report.html"
        instance.generate_gifs = generate_gifs
        instance._source_frames = {}
        instance._manifest_obj = None
        instance._agent_notes = None

        if manifest is not None:
            instance._manifest_obj = manifest if isinstance(manifest, VizManifest) else resolve_manifest(manifest, benchmark_dir=instance.benchmark_dir)
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

        source_paths = tuple(pathlib.Path(src) for src in source_dirs) or (instance.output_dir,)
        instance._sources = _CatalogSources(catalogs=tuple(BenchmarkCatalog.from_directory(path) for path in source_paths))
        table = data_source_table(instance._manifest_obj.data_source) if instance._manifest_obj is not None else data_source_table(None)
        instance._merged_df = instance._sources.frame(table)
        return instance

    def _load_primary_frame(self, manifest: VizManifest) -> pl.DataFrame | None:
        """Load the manifest's primary frame from its Star Schema table or view."""
        table = data_source_table(manifest.data_source)
        frame = self._sources.frame(table)
        if frame is None:
            print(f"Cannot generate report: data_source={manifest.data_source!r} names no Star Schema table or view; a series data source needs the pipeline to write metrics/series_*.parquet.")
        return frame

    def _frame_for_spec(
        self,
        spec: PlotSpec,
        manifest: VizManifest,
        df: pl.DataFrame,
        df_contestants: pl.DataFrame,
    ) -> pl.DataFrame | None:
        """Pick the frame for one plot (per-plot data_source override support)."""
        src = spec.data_source
        if src is None or src == manifest.data_source:
            return df if spec.type in ("trajectory", "timeseries", "line", "table") else df_contestants
        table = data_source_table(src)
        if table not in self._source_frames:
            frame = self._sources.frame(table)
            if frame is None:
                print(f"Skipping plot '{spec.id}': data source '{src}' names no Star Schema table or view.")
                return None
            self._source_frames[table] = frame
        return self._source_frames[table]

    def build(self) -> pathlib.Path | None:
        """Execute the report building process."""
        manifest = self._manifest_obj if self._manifest_obj is not None else VizManifest.load(self.manifest_path)

        if self._merged_df is not None:
            df = self._merged_df
        else:
            df = self._load_primary_frame(manifest)
            if df is None:
                return

        # Separate contestant evaluation runs from reference runs. The canonical
        # episode summary always carries is_reference; other catalog tables may not.
        if "is_reference" in df.columns:
            df_contestants = df.filter(pl.col("is_reference").is_null() | (~pl.col("is_reference")))
            if len(df_contestants) == 0:
                df_contestants = df
        else:
            df_contestants = df

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        units = MetricRegistry.get_all_units()
        units.update(manifest.units or {})

        plotly_renderer = PlotlyRenderer(units=units)
        static_renderer = StaticRenderer(generate_gifs=self.generate_gifs, units=units)

        html_plots = []

        resolver = DataResolver(catalog=self._sources.catalogs[0], benchmark_dir=self.benchmark_dir)

        for spec in manifest.plots:
            resolved_spec = resolver.resolve_spec(spec)
            fallback_df = self._frame_for_spec(resolved_spec, manifest, df, df_contestants)

            try:
                plot_df, diagnostics = resolver.resolve(resolved_spec, fallback_df=fallback_df)
            except Exception as e:
                print(f"Skipping plot '{resolved_spec.id}': {e}")
                continue

            if diagnostics.status == "FAILED":
                issues_str = "; ".join(diagnostics.issues) if diagnostics.issues else "Quality gate rejected"
                print(f"Warning: Plot '{resolved_spec.id}' failed quality gate: {issues_str}")
                gate_card = f'<div class="plot-gate-card gate-failed"><strong>[Plot Quality Gate Rejected]</strong> {html_escape.escape(issues_str)}</div>'
                note_html = self._plot_note_html(resolved_spec)
                html_plots.append((resolved_spec.layout_group, gate_card + note_html, resolved_spec.title))
                continue

            diag_html = self._diagnostics_html(diagnostics)

            png_path = self.plots_dir / f"{resolved_spec.id}.png"
            try:
                static_renderer.render(resolved_spec, plot_df, png_path, run_dir=self.output_dir)
            except Exception as e:
                print(f"Warning: Failed to render static plot {resolved_spec.id}: {e}")

            if resolved_spec.options.get("vector_svg", False):
                svg_path = self.plots_dir / f"{resolved_spec.id}.svg"
                try:
                    static_renderer.render(resolved_spec, plot_df, svg_path, run_dir=self.output_dir)
                except Exception as e:
                    print(f"Warning: Failed to render vector plot {resolved_spec.id}: {e}")

            try:
                html_chunk = plotly_renderer.render(resolved_spec, plot_df, run_dir=self.output_dir)
                note_html = self._plot_note_html(resolved_spec)
                full_note = diag_html + note_html
                if html_chunk:
                    if isinstance(html_chunk, list):
                        for chunk in html_chunk:
                            html_plots.append((resolved_spec.layout_group, chunk + full_note, resolved_spec.title))
                    else:
                        html_plots.append((resolved_spec.layout_group, html_chunk + full_note, resolved_spec.title))
            except Exception as e:
                print(f"Warning: Failed to render interactive plot {resolved_spec.id}: {e}")

        if manifest.summary:
            summary_html = self._generate_summary_table_manifest(df, manifest)
        else:
            summary_html = self._generate_summary_table(df_contestants)
        if data_source_table(manifest.data_source) == _EPISODE_TABLE:
            summary_html = _status_table_html(df_contestants) + summary_html

        report_title = manifest.title or self.output_dir.name

        html_content = self._assemble_html(summary_html, html_plots, report_title, manifest)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        import plotly.offline

        js_path = self.output_dir / "plotly.min.js"
        with open(js_path, "w", encoding="utf-8") as f:
            f.write(plotly.offline.get_plotlyjs())

        self._write_note_file(manifest)

        print(f"Report generated successfully: {self.report_path}")
        print(f"Static plots saved to: {self.plots_dir}")
        return self.report_path

    def _load_agent_notes(self) -> dict[str, str]:
        """Load plot annotations and insights from notes.json or insights.json."""
        if self._agent_notes is not None:
            return self._agent_notes

        import json

        notes: dict[str, str] = {}
        search_dirs = [self.benchmark_dir]
        if self.output_dir != self.benchmark_dir:
            search_dirs.append(self.output_dir)

        for sdir in search_dirs:
            for filename in ("notes.json", "insights.json"):
                path = sdir / filename
                if not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, str):
                                notes[str(k)] = v
                            elif isinstance(v, dict):
                                text = v.get("insight") or v.get("note") or v.get("text")
                                if text:
                                    notes[str(k)] = str(text)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                plot_id = item.get("plot_id") or item.get("id") or item.get("label")
                                text = item.get("insight") or item.get("note") or item.get("value") or item.get("text")
                                if plot_id and text:
                                    notes[str(plot_id)] = str(text)
                except Exception as e:
                    print(f"Warning: Failed to load agent notes from {path}: {e}")

        self._agent_notes = notes
        return notes

    def _plot_note_html(self, spec: PlotSpec) -> str:
        """Per-plot agent note and diagnostic insight, rendered under the plot."""
        opts = spec.options or {}
        note = opts.get("note")
        if note is None:
            key = opts.get("notes_key")
            if key:
                try:
                    import yaml as _yaml

                    notes_path = self.benchmark_dir / "notes.yaml"
                    if notes_path.is_file():
                        data = _yaml.safe_load(notes_path.read_text(encoding="utf-8"))
                        rows = data if isinstance(data, list) else ([{"label": str(k), "value": str(v)} for k, v in (data or {}).items()])
                        for row in rows:
                            if isinstance(row, dict) and str(row.get("label", "")) == str(key):
                                note = row.get("value")
                                break
                except Exception:
                    note = None

        chunks: list[str] = []
        if note:
            chunks.append(f"<div class='plot-note'>{_render_markdown_light(str(note))}</div>")

        agent_notes = self._load_agent_notes()
        if spec.id and spec.id in agent_notes:
            agent_text = agent_notes[spec.id]
            chunks.append(f'<div class="agent-annotation"><span class="badge">Agent Insight</span> {_render_markdown_light(str(agent_text))}</div>')

        return "".join(chunks)

    @staticmethod
    def _diagnostics_html(diagnostics: PlotDiagnostics) -> str:
        """Render non-blocking diagnostic warnings as informative note banner."""
        if diagnostics.status == "WARNING" and diagnostics.issues:
            issues_str = "; ".join(diagnostics.issues)
            return f'<div class="plot-gate-card gate-warning"><strong>[Plot Quality Gate Warning]</strong> {html_escape.escape(issues_str)}</div>'
        return ""

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
                "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            (self.output_dir / "report_manifest.yaml").write_text(yaml.safe_dump(note, sort_keys=False), encoding="utf-8")
        except Exception as e:
            print(f"Warning: failed to write report_manifest.yaml note: {e}")

    def _generate_summary_table(self, df: pl.DataFrame) -> str:
        """Legacy summary table (used when the manifest declares no summary).

        Groups planner-first, then by the other varying identity columns.
        """
        group_cols = _default_summary_group_cols(df)
        metric_cols = [c for c in ["success", "time_to_goal", "path_length"] if c in df.columns and not df[c].is_null().all()]
        needed = [c for c in [*group_cols, *metric_cols] if c in df.columns]
        if needed:
            df = df.select(needed)
        df = _labelled(df, group_cols)

        agg_exprs = []
        if "success" in df.columns and not df["success"].is_null().all():
            agg_exprs.append(pl.col("success").mean().alias("success_rate"))
        if "time_to_goal" in df.columns and not df["time_to_goal"].is_null().all():
            agg_exprs.append(pl.col("time_to_goal").mean().alias("avg_time"))
        if "path_length" in df.columns and not df["path_length"].is_null().all():
            agg_exprs.append(pl.col("path_length").mean().alias("avg_path_length"))
        if not agg_exprs:
            return ""

        summary = _aggregate(df, group_cols, agg_exprs)
        formats = {
            "success_rate": "{:.1%}",
            "avg_time": "{:.2f}s",
            "avg_path_length": "{:.2f}m",
        }
        labels = {"success_rate": "Success Rate", "avg_time": "Avg Time", "avg_path_length": "Avg Path Length"}
        return _summary_html(summary, formats, labels)

    def _generate_summary_table_manifest(self, df: pl.DataFrame, manifest: VizManifest) -> str:
        """Declarative summary table: one column per SummarySpec.

        Grouping: manifest.summary_group_by wins, otherwise planner-first plus
        the other varying identity columns.
        """
        group_by = manifest.summary_group_by
        if isinstance(group_by, str):
            group_by = [group_by]
        group_cols = [c for c in (group_by or []) if c in df.columns]

        if not group_cols:
            group_cols = _default_summary_group_cols(df)

        needed = [c for c in [*group_cols, *(s.metric for s in manifest.summary)] if c in df.columns]
        list_cols = [c for c in needed if df.schema[c] == pl.List]
        if list_cols:
            df = df.select(needed).explode(list_cols)
        else:
            df = df.select(needed) if needed else df
        df = _labelled(df, group_cols)

        agg_exprs = []
        for spec in manifest.summary:
            if spec.metric in df.columns and not df[spec.metric].is_null().all():
                agg_exprs.append(pl.col(spec.metric).mean().alias(spec.metric))
        if not agg_exprs:
            return ""

        summary = _aggregate(df, group_cols, agg_exprs)
        spec_by_metric = {spec.metric: spec for spec in manifest.summary}
        formats = {metric: spec.format for metric, spec in spec_by_metric.items() if metric not in group_cols}
        labels = {metric: spec.label for metric, spec in spec_by_metric.items() if metric not in group_cols}
        return _summary_html(summary, formats, labels)

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
            grouped_plots[group_id].append({"title": title, "html": html})

        group_titles = {group.id: group.title for group in manifest.groups}

        plot_groups = []
        for group_id in ordered_groups:
            plots = grouped_plots[group_id]
            title = group_titles.get(group_id, group_id.replace("_", " ").title())
            plot_groups.append({"id": group_id, "title": title, "plots": plots})

        overview_plots = [html for group, html, title in plot_htmls if group == "overview"]

        template_dir = pathlib.Path(__file__).parent
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))
        template = env.get_template("report_template.html.j2")

        generated_on = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return template.render(benchmark_id=report_title, report_dir=str(self.output_dir.resolve()), generated_on=generated_on, summary_html=summary_html, overview_plots=overview_plots, plot_groups=plot_groups)
