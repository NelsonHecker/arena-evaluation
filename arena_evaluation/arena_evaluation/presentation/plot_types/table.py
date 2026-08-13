"""Declarative HTML table plot type.

Renders a table in the report from two manifest-declared sources:

1. **Data-derived columns** - ``options.columns`` aggregates the metrics frame
   per ``group_by`` (mean of each column, formatted), like the summary table
   but as a standalone plot.

2. **Agent-written notes** - ``options.notes`` pulls in free-form rows written
   by an agent (e.g. through MCP tools) into ``notes.yaml`` in the benchmark
   dir, or inline YAML / ``"key: value"`` text lines.

Example manifest spec:

```yaml
- id: overview_table
  type: table
  title: Benchmark Overview
  data_key: "*"
  layout_group: overview
  options:
    group_by: [local_planner]
    columns:
      - {metric: success, label: Success, format: "{:.0%}"}
      - {metric: time_to_goal, label: Avg Time, format: "{:.1f}"}
    notes: notes.yaml          # or inline [{label: ..., value: ...}]
```
"""

from __future__ import annotations

import pathlib
import typing

import polars as pl

from .base import BasePlotRenderer

if typing.TYPE_CHECKING:
    from ..report_builder import ReportBuilder  # noqa: F401


def _load_notes(notes, benchmark_dir: pathlib.Path | None) -> list[dict[str, str]]:
    """Normalize the notes source into [{label, value}, ...].

    Accepts: a list of {label, value} dicts, a YAML file path (resolved
    against the benchmark dir), or a text file with ``key: value`` lines.
    """
    rows: list[dict[str, str]] = []
    if notes is None:
        return rows
    if isinstance(notes, list):
        for item in notes:
            if isinstance(item, dict):
                rows.append(
                    {
                        "label": str(item.get("label", item.get("key", ""))),
                        "value": str(item.get("value", "")),
                    }
                )
        return rows
    if isinstance(notes, str):
        notes = notes.strip()
        if not notes:
            return rows

        path: pathlib.Path | None = None
        candidate = pathlib.Path(notes)
        if candidate.is_file():
            path = candidate
        elif benchmark_dir is not None and (benchmark_dir / candidate).is_file():
            path = benchmark_dir / candidate

        if path is not None:
            text = path.read_text()
        else:
            text = notes  # inline YAML or text

        try:
            import yaml

            data = yaml.safe_load(text)
            if isinstance(data, list):
                return _load_notes(data, benchmark_dir)
            if isinstance(data, dict):
                return [{"label": str(k), "value": str(v)} for k, v in data.items()]
        except Exception:
            pass

        # Fallback: "key: value" text lines (agent-friendly free text).
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                rows.append({"label": key.strip(), "value": value.strip()})
            else:
                rows.append({"label": "", "value": line})
    return rows


class TableRenderer(BasePlotRenderer):
    PLOT_TYPE = "table"
    run_dir: pathlib.Path | None = None  # set by the renderer dispatcher (notes resolution)

    def _columns(self) -> list[dict[str, str]]:
        cols = (self.spec.options or {}).get("columns") or []
        return [
            {"metric": str(c.get("metric", "")), "label": str(c.get("label", c.get("metric", ""))),
             "format": str(c.get("format", "{:.2f}"))}
            for c in cols
            if isinstance(c, dict) and c.get("metric")
        ]

    def _data_rows(self, df: pl.DataFrame) -> list[list[str]]:
        """Aggregate the metrics frame per group_by into label/value rows."""
        group_by = self.spec.options.get("group_by") or []
        if isinstance(group_by, str):
            group_by = [group_by]
        group_cols = [g for g in group_by if g in df.columns]
        cols = self._columns()
        if not group_cols or not cols:
            return []

        list_cols = [c for c in [*group_cols, *(c["metric"] for c in cols)] if c in df.columns and df.schema[c] == pl.List]
        if list_cols:
            df = df.explode(list_cols)

        agg = [
            pl.col(c["metric"]).mean().alias(f"__m{i}__")
            for i, c in enumerate(cols)
            if c["metric"] in df.columns
        ]
        if not agg:
            return []
        grouped = df.group_by(group_cols).agg(agg).sort(group_cols)

        rows: list[list[str]] = []
        for row in grouped.iter_rows(named=True):
            key = " / ".join(str(row.get(g, "")) for g in group_cols)
            values = []
            for i, c in enumerate(cols):
                val = row.get(f"__m{i}__")
                try:
                    values.append(c["format"].format(float(val)) if val is not None else "N/A")
                except (TypeError, ValueError):
                    values.append("N/A")
            rows.append([key, *values])
        return rows

    def _render_html(self, df: pl.DataFrame, benchmark_dir: pathlib.Path | None) -> str | None:
        opts = self.spec.options or {}
        cols = self._columns()
        group_cols = opts.get("group_by") or []
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        group_cols = [g for g in group_cols if g in df.columns]

        header = [*[g.replace("_", " ").title() for g in group_cols], *[c["label"] for c in cols]]
        if not header:
            header = ["Label", "Value"]
        rows = self._data_rows(df)

        note_rows = _load_notes(opts.get("notes"), benchmark_dir)
        if note_rows:
            if rows:
                rows.append(["—" * max(len(h), 1) for h in header])
            rows.extend([[n["label"], n["value"]] for n in note_rows])

        if not rows:
            return None

        html = ["<table class='dataframe' style='border-collapse: collapse; margin: 8px 0;'>"]
        html.append("<thead><tr>" + "".join(f"<th style='border:1px solid #ccc; padding:4px 10px; background:#f5f5f5;'>{h}</th>" for h in header) + "</tr></thead>")
        html.append("<tbody>")
        for row in rows:
            cells = row + [""] * (len(header) - len(row))
            html.append("<tr>" + "".join(f"<td style='border:1px solid #ccc; padding:4px 10px;'>{c}</td>" for c in cells) + "</tr>")
        html.append("</tbody></table>")
        return "".join(html)

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        return self._render_html(df_filtered, self.run_dir)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        pass  # Tables render as HTML only (like timeseries).
