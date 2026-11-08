"""HTML and static image table renderer for aggregated metric columns and notes."""

from __future__ import annotations

import html as html_escape
import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.base import BasePlotRenderer

__all__ = ["TableRenderer"]


def _load_notes(notes: object | None, benchmark_dir: pathlib.Path | None) -> list[dict[str, str]]:
    """Normalize notes input into a list of label/value mappings."""
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
        if "\n" not in notes and len(notes) < 256:
            try:
                candidate = pathlib.Path(notes)
                if candidate.is_file():
                    path = candidate
                elif benchmark_dir is not None and (benchmark_dir / candidate).is_file():
                    path = benchmark_dir / candidate
            except (OSError, ValueError):
                path = None

        if path is not None:
            text = path.read_text(encoding="utf-8")
        else:
            text = notes

        try:
            import yaml

            data = yaml.safe_load(text)
            if isinstance(data, list):
                return _load_notes(data, benchmark_dir)
            if isinstance(data, dict):
                return [{"label": str(k), "value": str(v)} for k, v in data.items()]
        except Exception:
            pass

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


def _render_notes_callout(notes_rows: list[dict[str, str]]) -> str:
    """Render agent notes as a standalone callout section."""
    if not notes_rows:
        return ""
    parts = [
        "<div class='notes-callout' style='margin: 12px 0; padding: 12px; background: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 4px;'>",
        "<div class='notes-callout-title' style='font-weight: 600; margin-bottom: 6px;'>Analysis Notes</div>",
    ]
    for note in notes_rows:
        label = html_escape.escape(note.get("label", ""), quote=True)
        value = html_escape.escape(note.get("value", ""), quote=True).replace("\n", "<br>")
        if label:
            parts.append(f"<div class='notes-callout-row'><span class='notes-callout-label' style='font-weight: 600;'>{label}: </span><span class='notes-callout-value'>{value}</span></div>")
        else:
            parts.append(f"<div class='notes-callout-row notes-callout-row-plain'><span class='notes-callout-value'>{value}</span></div>")
    parts.append("</div>")
    return "".join(parts)


class TableRenderer(BasePlotRenderer):
    """HTML and image table renderer for scorecard tables and agent findings."""

    PLOT_TYPE = "table"

    def _columns(self, df: pl.DataFrame | None = None) -> list[dict[str, str]]:
        opts = self.spec.options or {}
        declared_cols = opts.get("columns") or []
        if declared_cols:
            return [
                {
                    "metric": str(c.get("metric", "")),
                    "label": str(c.get("label", c.get("metric", ""))),
                    "format": str(c.get("format", "{:.2f}")),
                }
                for c in declared_cols
                if isinstance(c, dict) and c.get("metric")
            ]

        # Grammar of graphics fallback: channels.y
        y_ch = self.spec.channels.y
        if y_ch:
            metric_list = y_ch if isinstance(y_ch, list) else [y_ch]
            return [
                {
                    "metric": str(m),
                    "label": str(m).replace("_", " ").title(),
                    "format": "{:.2f}",
                }
                for m in metric_list
                if m
            ]

        # Auto-infer columns from DataFrame schema if not specified
        if df is not None:
            group_cols = set(self._resolve_group_cols(df))
            inferred = []
            for col, dtype in df.schema.items():
                if col in group_cols:
                    continue
                fmt = "{:.2f}" if dtype in (pl.Float64, pl.Float32) else "{}"
                inferred.append(
                    {
                        "metric": col,
                        "label": col.replace("_", " ").title(),
                        "format": fmt,
                    }
                )
            return inferred

        return []

    def _agent_rows(self) -> list[dict[str, str]]:
        rows = (self.spec.options or {}).get("rows") or []
        out: list[dict[str, str]] = []
        for item in rows:
            if isinstance(item, dict):
                out.append(
                    {
                        "label": str(item.get("label", item.get("key", ""))),
                        "value": str(item.get("value", "")),
                    }
                )
        return out

    def _resolve_group_cols(self, df: pl.DataFrame) -> list[str]:
        opts = self.spec.options or {}
        group_by = opts.get("group_by") or self.spec.channels.x or self.spec.group_by or []
        if isinstance(group_by, str):
            group_by = [group_by]
        schema = df.schema
        return [g for g in group_by if schema.get(g) is not None]

    def _data_rows(self, df: pl.DataFrame) -> tuple[list[str], list[list[str]]]:
        group_cols = self._resolve_group_cols(df)
        cols = self._columns(df)
        if not cols:
            return [], []

        schema = df.schema
        all_target_cols = [*group_cols, *(c["metric"] for c in cols)]
        list_cols = [c for c in all_target_cols if schema.get(c) is not None and schema[c] == pl.List]
        if list_cols:
            need = [c for c in all_target_cols if schema.get(c) is not None]
            df = df.select(need).explode(list_cols)
        df = self._apply_row_filters(df)

        header = [g.replace("_", " ").title() for g in group_cols] + [c["label"] for c in cols]

        # Case 1: Grouped data (aggregate mean if multiple observations per group)
        if group_cols:
            max_cnt = df.group_by(group_cols).agg(pl.len().alias("cnt"))["cnt"].max() if not df.is_empty() else 0
            if max_cnt == 1:
                # Already pre-aggregated
                rows: list[list[str]] = []
                for row in df.to_dicts():
                    vals = []
                    for c in cols:
                        raw = row.get(c["metric"])
                        try:
                            vals.append(c["format"].format(float(raw)) if raw is not None else "N/A")
                        except (TypeError, ValueError):
                            vals.append(str(raw) if raw is not None else "N/A")
                    rows.append([str(row.get(g, "")) for g in group_cols] + vals)
                return header, rows

            agg = [pl.col(c["metric"]).mean().alias(f"__m{i}__") for i, c in enumerate(cols) if schema.get(c["metric"]) is not None]
            if not agg:
                return header, []
            grouped = df.group_by(group_cols).agg(agg).sort(group_cols)

            rows = []
            for row in grouped.to_dicts():
                values = []
                for i, c in enumerate(cols):
                    val = row.get(f"__m{i}__")
                    try:
                        values.append(c["format"].format(float(val)) if val is not None else "N/A")
                    except (TypeError, ValueError):
                        values.append("N/A")
                rows.append([str(row.get(g, "")) for g in group_cols] + values)
            return header, rows

        # Case 2: Flat table without grouping
        rows = []
        for row in df.to_dicts():
            values = []
            for c in cols:
                raw = row.get(c["metric"])
                try:
                    values.append(c["format"].format(float(raw)) if raw is not None else "N/A")
                except (TypeError, ValueError):
                    values.append(str(raw) if raw is not None else "N/A")
            rows.append(values)
        return header, rows

    def _render_data_table(self, df: pl.DataFrame) -> str | None:
        header, rows = self._data_rows(df)
        if not header or not rows:
            return None

        html = ["<table class='dataframe' style='border-collapse: collapse; margin: 8px 0; width: 100%; font-family: sans-serif; font-size: 0.9em;'>"]
        html.append("<thead><tr>" + "".join(f"<th style='border: 1px solid #e2e8f0; padding: 6px 12px; background: #f1f5f9; text-align: left;'>{h}</th>" for h in header) + "</tr></thead>")
        html.append("<tbody>")
        for idx, row in enumerate(rows):
            bg = "#ffffff" if idx % 2 == 0 else "#f8fafc"
            cells = row + [""] * (len(header) - len(row))
            html.append(f"<tr style='background: {bg};'>" + "".join(f"<td style='border: 1px solid #e2e8f0; padding: 6px 12px;'>{c}</td>" for c in cells) + "</tr>")
        html.append("</tbody></table>")
        return "".join(html)

    def _render_agent_table(self) -> str | None:
        rows = self._agent_rows()
        if not rows:
            return None
        parts = [
            "<table class='dataframe' style='border-collapse: collapse; margin: 8px 0; width: 100%; font-family: sans-serif; font-size: 0.9em;'>",
            "<thead><tr><th style='border: 1px solid #e2e8f0; padding: 6px 12px; background: #f1f5f9; text-align: left;'>Label</th><th style='border: 1px solid #e2e8f0; padding: 6px 12px; background: #f1f5f9; text-align: left;'>Value</th></tr></thead>",
            "<tbody>",
        ]
        for idx, row in enumerate(rows):
            bg = "#ffffff" if idx % 2 == 0 else "#f8fafc"
            label = html_escape.escape(row.get("label", ""), quote=True)
            value = html_escape.escape(row.get("value", ""), quote=True).replace("\n", "<br>")
            parts.append(f"<tr style='background: {bg};'><td style='border: 1px solid #e2e8f0; padding: 6px 12px; font-weight: 600;'>{label}</td><td style='border: 1px solid #e2e8f0; padding: 6px 12px;'>{value}</td></tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    def _render_html(self, df: pl.DataFrame, benchmark_dir: pathlib.Path | None) -> str | None:
        opts = self.spec.options or {}
        data_table = self._render_data_table(df)
        agent_table = self._render_agent_table()
        note_rows = _load_notes(opts.get("notes"), benchmark_dir)
        notes_callout = _render_notes_callout(note_rows)

        pieces = [p for p in (data_table, agent_table, notes_callout) if p]
        if not pieces:
            return None
        title_html = f"<h4 class='table-title' style='margin: 12px 0 6px 0; font-weight: 600;'>{html_escape.escape(self.spec.title)}</h4>" if self.spec.title else ""
        return title_html + "".join(pieces)

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        return self._render_html(df_filtered, self.run_dir)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        header, rows = self._data_rows(df_filtered)

        if not rows:
            agent_rows = self._agent_rows()
            if not agent_rows:
                return
            header = ["Label", "Value"]
            rows = [[r.get("label", ""), r.get("value", "")] for r in agent_rows]

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_rows = len(rows)
        n_cols = len(header)
        fig_w = max(6.0, n_cols * 2.0)
        fig_h = max(2.5, n_rows * 0.4 + 1.2)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        ax.set_title(self.spec.title, pad=12, fontsize=12, fontweight="bold")

        table = ax.table(
            cellText=rows,
            colLabels=header,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        for (r, _), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#ffffff" if r % 2 == 1 else "#f8fafc")

        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
