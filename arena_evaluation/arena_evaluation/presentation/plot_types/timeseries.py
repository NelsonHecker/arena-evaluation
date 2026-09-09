from __future__ import annotations

import pathlib
import polars as pl
import plotly.graph_objects as go

from .base import BasePlotRenderer


class TimeseriesRenderer(BasePlotRenderer):
    PLOT_TYPE = "timeseries"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        import plotly.express as px
        import numpy as np

        df_filtered = self._apply_filters(df)
        diff_col, df_filtered = self.resolve_diff_col(df_filtered)

        metrics = self.spec.options.get("metrics", [])
        if not metrics:
            if self.spec.data_key:
                metrics = [self.spec.data_key]
            else:
                return None

        # Check for matching metric columns or aliases
        valid_metrics = []
        for m in metrics:
            if m in df_filtered.columns:
                valid_metrics.append(m)
            elif f"timeseries_{m}" in df_filtered.columns:
                valid_metrics.append(f"timeseries_{m}")
            elif m.replace("timeseries_", "") in df_filtered.columns:
                valid_metrics.append(m.replace("timeseries_", ""))

        if not valid_metrics:
            return None

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        # Filter out reference runs from main planner traces
        if "is_reference" in pdf.columns and not self.spec.options.get("include_reference", False):
            pdf = pdf[~pdf["is_reference"].fillna(False)]
            if pdf.empty:
                return None

        x_col = self.spec.options.get("x", "timeseries_time_s")
        if x_col not in df_filtered.columns:
            return None

        # Row-per-sample scalar columns (e.g. hero_timeseries.parquet): one
        # trace per metric over the full columns. Wide-format list columns
        # (row-per-episode) fall through to the per-planner path below.
        if df_filtered.schema[x_col] != pl.List:
            return self._render_plotly_scalar(df_filtered, x_col, valid_metrics)

        fig = go.Figure()
        planners = pdf[diff_col].unique() if diff_col in pdf.columns else ["unknown"]
        colors = px.colors.qualitative.Plotly

        traces_added = 0
        for planner_idx, planner in enumerate(planners):
            planner_df = pdf[pdf[diff_col] == planner] if diff_col in pdf.columns else pdf

            for _, row in planner_df.iterrows():
                episode_val = row.get("episode", "unknown")
                legend_group = f"{planner} - Ep {episode_val}"
                base_color = colors[planner_idx % len(colors)]

                x_raw = row.get(x_col)
                if x_raw is None or isinstance(x_raw, (int, float, str, bool)):
                    continue
                try:
                    x_data = np.array(x_raw, dtype=float)
                except Exception:
                    continue
                if x_data.ndim != 1 or len(x_data) == 0:
                    continue

                for m_idx, metric in enumerate(valid_metrics):
                    y_raw = row.get(metric)
                    if y_raw is None:
                        continue
                    try:
                        y_data = np.array(y_raw, dtype=float)
                    except Exception:
                        continue

                    if y_data.ndim != 1 or len(y_data) == 0:
                        continue

                    # If sizes differ, interpolate or match lengths
                    if len(x_data) != len(y_data):
                        cur_x = np.linspace(x_data[0], x_data[-1], len(y_data)) if len(y_data) > 1 else x_data[:len(y_data)]
                    else:
                        cur_x = x_data

                    dash_styles = ["solid", "dash", "dot", "dashdot"]
                    dash = dash_styles[m_idx % len(dash_styles)]
                    showlegend = True if m_idx == 0 else False

                    m_label = self.format_label(metric.replace("timeseries_", "").replace("_", " ").title(), metric)
                    fig.add_trace(go.Scatter(
                        x=cur_x.tolist(),
                        y=y_data.tolist(),
                        mode="lines",
                        name=f"{legend_group}" if m_idx == 0 else m_label,
                        legendgroup=legend_group,
                        line=dict(color=base_color, dash=dash, width=2),
                        hovertemplate=f"Ep {episode_val}<br>Time: %{{x:.2f}}s<br>{m_label}: %{{y:.3f}}<extra></extra>",
                        showlegend=showlegend
                    ))
                    traces_added += 1

        fig.update_layout(
            template="plotly_white",
            xaxis_title=self.format_label(x_col.replace("timeseries_", "").replace("_", " ").title(), x_col),
            yaxis_title=self.format_label(valid_metrics[0].replace("timeseries_", "").replace("_", " ").title(), valid_metrics[0]),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(r=150)
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def _render_plotly_scalar(self, df: pl.DataFrame, x_col: str, valid_metrics: list[str]) -> str | None:
        """Plotly rendering for row-per-sample scalar frames (one trace per metric)."""
        import plotly.express as px
        import numpy as np

        x_data = df[x_col].to_numpy().astype(float, copy=False)
        if len(x_data) == 0:
            return None

        options_colors = self.spec.options.get("colors", {})
        palette = px.colors.qualitative.Plotly
        stacked = bool(self.spec.options.get("stacked", False))

        def _aligned(y_raw: np.ndarray):
            if len(y_raw) == len(x_data):
                return x_data, y_raw
            cur_x = np.linspace(x_data[0], x_data[-1], len(y_raw)) if len(y_raw) > 1 else x_data[: len(y_raw)]
            return cur_x, y_raw

        fig = go.Figure()

        for m_idx, m in enumerate(valid_metrics):
            y_raw = df[m].to_numpy().astype(float, copy=False)
            xs, ys = _aligned(y_raw)
            label = self.format_label(m.replace("timeseries_", "").replace("_", " ").title(), m)
            color = options_colors.get(m, palette[m_idx % len(palette)])
            fig.add_trace(go.Scatter(
                x=xs.tolist(), y=ys.tolist(), mode="lines", name=label,
                line=dict(color=color, width=2.2),
                stackgroup="one" if stacked else None,
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
            ))

        twin_cols = [c for c in self.spec.options.get("twin", []) if c in df.columns]
        twin_fill = bool(self.spec.options.get("twin_fill", False))
        for t_idx, c in enumerate(twin_cols):
            y_raw = df[c].to_numpy().astype(float, copy=False)
            xs, ys = _aligned(y_raw)
            label = self.format_label(c.replace("timeseries_", "").replace("_", " ").title(), c)
            color = options_colors.get(c, palette[(len(valid_metrics) + t_idx) % len(palette)])
            fig.add_trace(go.Scatter(
                x=xs.tolist(), y=ys.tolist(), mode="lines", name=label,
                line=dict(color=color, width=2.2, dash="dash"),
                yaxis="y2",
                fill="tozeroy" if twin_fill else None,
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)" if twin_fill else None,
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
            ))

        hline = self.spec.options.get("hline")
        if hline:
            # Plotly dash enum uses names, not matplotlib style codes
            style = str(hline.get("style", "dash"))
            dash_map = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
            fig.add_hline(
                y=float(hline.get("value", 0.0)),
                line_dash=dash_map.get(style, "dash"),
                line_color=hline.get("color", "#e11d48"),
                yref="y2" if str(hline.get("axis", "left")).lower() == "right" else "y",
                annotation_text=str(hline.get("label", "")),
                annotation_font_color=hline.get("color", "#e11d48"),
                annotation_font_size=10,
            )

        shade = self.spec.options.get("shade_window")
        if shade:
            fig.add_vrect(
                x0=float(shade.get("start", 0.0)), x1=float(shade.get("end", 0.0)),
                fillcolor=shade.get("color", "#ffd166"),
                opacity=float(shade.get("alpha", 0.22)),
                line_width=0,
                annotation_text=str(shade.get("label", "")),
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="#7a5c00",
            )

        ylabel = self.spec.options.get("ylabel")
        if not ylabel and valid_metrics:
            unit = self.units.get(valid_metrics[0], "")
            base = valid_metrics[0].replace("timeseries_", "").replace("_", " ").title()
            ylabel = f"{base} [{unit}]" if unit else base
        twin_ylabel = self.spec.options.get("twin_ylabel")
        if not twin_ylabel and twin_cols:
            unit = self.units.get(twin_cols[0], "")
            base = twin_cols[0].replace("timeseries_", "").replace("_", " ").title()
            twin_ylabel = f"{base} [{unit}]" if unit else base

        # Zero-base both axes by default (same reasoning as the seaborn path);
        # a cutoff line beyond the data widens the axis; twin_same_scale locks
        # the twin axis to the primary axis' exact range.
        range_mode = "tozero" if bool(self.spec.options.get("zero_base", True)) else "normal"
        y_range = None
        if hline:
            data_top = max((float(df[m].max()) for m in valid_metrics if df[m].n_unique() > 0), default=0.0)
            h_val = float(hline.get("value", 0.0))
            if str(hline.get("axis", "left")).lower() != "right" and h_val > data_top:
                y_range = [0.0, h_val * 1.05]
        twin_matches = "y" if bool(self.spec.options.get("twin_same_scale", False)) else None
        fig.update_layout(
            template="plotly_white",
            title=dict(text=self.spec.title or "", font=dict(size=14)),
            xaxis_title=self.spec.options.get("xlabel", "Time (s)"),
            yaxis=dict(title=ylabel, rangemode=range_mode, range=y_range),
            yaxis2=dict(title=twin_ylabel or "", overlaying="y", side="right", showgrid=False, rangemode=range_mode, matches=twin_matches),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=56, r=64, b=52, l=64),
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Static PNG time series (publication-style).

        Supports both row-per-sample scalar frames (e.g. a pre-aggregated
        hero_timeseries.parquet) and row-per-episode frames with list columns
        (combined_metrics timeseries_* columns).

        Options:
          x: time column (default "timeseries_time_s")
          metrics: [columns] for the primary (left) axis
          stacked: true -> stackplot for the primary metrics
          colors: {column: hex} fixed series colors
          twin: [columns] plotted on a secondary (right) axis
          twin_fill: true -> soft area fill under twin-axis lines
          hline: {value, label, color, style, axis} horizontal reference line
          shade_window: {start, end, label, color, alpha} annotated band
          ylabel / twin_ylabel: axis labels (defaults from units map)
          figsize: [w, h] in inches (default [7.2, 2.8]); dpi (default 300)
          linewidth: float (default 1.8)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        df_filtered = self._apply_filters(df)
        if len(df_filtered) == 0:
            return

        metrics = self.spec.options.get("metrics", [])
        if not metrics:
            if self.spec.data_key:
                metrics = [self.spec.data_key]
            else:
                return

        valid: list[str] = []
        for m in metrics:
            if m in df_filtered.columns:
                valid.append(m)
            elif f"timeseries_{m}" in df_filtered.columns:
                valid.append(f"timeseries_{m}")
            elif m.replace("timeseries_", "") in df_filtered.columns:
                valid.append(m.replace("timeseries_", ""))
        if not valid:
            return

        x_col = self.spec.options.get("x", "timeseries_time_s")
        if x_col not in df_filtered.columns:
            return

        # ---- data assembly -------------------------------------------------
        list_mode = df_filtered.schema[x_col] == pl.List

        def _col_series(col: str):
            if list_mode:
                out: list[np.ndarray] = []
                for raw in df_filtered[col].to_list():
                    if raw is None:
                        continue
                    arr = np.array(raw, dtype=float)
                    if arr.ndim == 1 and len(arr):
                        out.append(arr)
                return out
            vals = df_filtered[col].to_numpy()
            return [vals.astype(float, copy=False)] if len(vals) else []

        x_series = _col_series(x_col)
        y_series = {m: _col_series(m) for m in valid}
        if not x_series or not any(len(s) for s in y_series.values()):
            return

        twin_cols = [c for c in self.spec.options.get("twin", []) if c in df_filtered.columns]
        twin_series = {c: _col_series(c) for c in twin_cols}

        def _paired(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Trim/interpolate y to the x grid when lengths differ."""
            if len(xs) == len(ys):
                return xs, ys
            cur_x = np.linspace(xs[0], xs[-1], len(ys)) if len(ys) > 1 else xs[: len(ys)]
            return cur_x, ys

        # ---- figure --------------------------------------------------------
        figsize = tuple(float(v) for v in self.spec.options.get("figsize", [7.2, 2.8]))
        dpi = int(self.spec.options.get("dpi", 300))
        lw = float(self.spec.options.get("linewidth", 1.8))

        fig, ax = plt.subplots(figsize=figsize)
        palette = ["#1f77b4", "#8a8f98", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
        colors = self.spec.options.get("colors", {})
        stacked = bool(self.spec.options.get("stacked", False))

        # Primary axis: pick the first x series that aligns with each metric's series
        if stacked:
            # One stackplot accumulating ALL primary metrics (first series per
            # metric), so bands stack instead of overwriting each other.
            x_ref = None
            stacks = []
            stack_labels = []
            stack_colors = []
            for m_idx, m in enumerate(valid):
                series = y_series.get(m, [])
                if not series:
                    continue
                xs, ys_arr = _paired(x_series[0], series[0])
                if x_ref is None or len(xs) >= len(x_ref):
                    x_ref = xs
                stacks.append(ys_arr)
                stack_labels.append(m.replace("timeseries_", "").replace("_", " ").title())
                stack_colors.append(colors.get(m, palette[m_idx % len(palette)]))
            if x_ref is not None:
                ax.stackplot(x_ref, stacks, labels=stack_labels, colors=stack_colors, alpha=0.9, linewidth=0)
        else:
            for m_idx, m in enumerate(valid):
                series = y_series.get(m, [])
                if not series:
                    continue
                color = colors.get(m, palette[m_idx % len(palette)])
                label = m.replace("timeseries_", "").replace("_", " ").title()
                for xs, ys_arr in zip(x_series, series):
                    xs, ys_arr = _paired(xs, ys_arr)
                    ax.plot(xs, ys_arr, label=label, color=color, linewidth=lw)
                if bool(self.spec.options.get("fill", False)):
                    for xs, ys_arr in zip(x_series, series):
                        xs, ys_arr = _paired(xs, ys_arr)
                        ax.fill_between(xs, ys_arr, color=color, alpha=0.15, linewidth=0)

        ylabel = self.spec.options.get("ylabel")
        if not ylabel and valid:
            unit = self.units.get(valid[0], "")
            base = valid[0].replace("timeseries_", "").replace("_", " ").title()
            ylabel = f"{base} [{unit}]" if unit else base
        ax.set_ylabel(ylabel, fontsize=9)

        # ---- twin axis -----------------------------------------------------
        twin_ax = None
        if twin_series:
            twin_ax = ax.twinx()
            for m_idx, c in enumerate(twin_cols):
                color = colors.get(c, palette[(len(valid) + m_idx) % len(palette)])
                label = c.replace("timeseries_", "").replace("_", " ").title()
                for xs, ys_arr in zip(x_series, twin_series.get(c, [])):
                    xs, ys_arr = _paired(xs, ys_arr)
                    twin_ax.plot(xs, ys_arr, label=label, color=color, linewidth=lw, linestyle="--")
                    if bool(self.spec.options.get("twin_fill", False)):
                        twin_ax.fill_between(xs, ys_arr, color=color, alpha=0.12, linewidth=0)
            twin_ylabel = self.spec.options.get("twin_ylabel")
            if not twin_ylabel and twin_cols:
                unit = self.units.get(twin_cols[0], "")
                base = twin_cols[0].replace("timeseries_", "").replace("_", " ").title()
                twin_ylabel = f"{base} [{unit}]" if unit else base
            twin_ax.set_ylabel(twin_ylabel, fontsize=9)

        # ---- reference line -------------------------------------------------
        hline = self.spec.options.get("hline")
        if hline:
            target = ax
            if str(hline.get("axis", "left")).lower() == "right" and twin_ax is not None:
                target = twin_ax
            target.axhline(
                float(hline.get("value", 0.0)),
                color=hline.get("color", "#e11d48"),
                linestyle=hline.get("style", "--"),
                linewidth=hline.get("linewidth", 1.2),
            )
            if hline.get("label"):
                target.text(
                    0.995, 0.88, f"  {hline['label']}", transform=target.transAxes,
                    ha="right", va="top", fontsize=8, color=hline.get("color", "#e11d48"),
                )

        # ---- annotated shade window ----------------------------------------
        shade = self.spec.options.get("shade_window")
        if shade:
            ax.axvspan(
                float(shade.get("start", 0.0)), float(shade.get("end", 0.0)),
                color=shade.get("color", "#ffd166"), alpha=float(shade.get("alpha", 0.22)),
                linewidth=0,
            )
            if shade.get("label"):
                ax.text(
                    float(shade.get("start", 0.0)) + 0.3, 0.955, f"  {shade['label']}",
                    transform=ax.get_xaxis_transform(), ha="left", va="top",
                    fontsize=8, fontweight="bold", color="#7a5c00",
                )

        # ---- polish ----------------------------------------------------------
        # Zero-base both axes by default so twin-axis series can never read as
        # exceeding the primary series purely through axis scaling.
        if bool(self.spec.options.get("zero_base", True)):
            ax.set_ylim(bottom=0.0)
            if twin_ax is not None:
                twin_ax.set_ylim(bottom=0.0)
        # A reference line beyond the data range (e.g. the 100 dBA cutoff)
        # must widen the axis so it stays visible.
        if hline:
            target = ax
            if str(hline.get("axis", "left")).lower() == "right" and twin_ax is not None:
                target = twin_ax
            h_val = float(hline.get("value", 0.0))
            if h_val > target.get_ylim()[1]:
                target.set_ylim(top=h_val * 1.05)
        # Optionally lock the twin axis to the primary axis' exact scale so
        # series on both axes are directly comparable at a glance.
        if twin_ax is not None and bool(self.spec.options.get("twin_same_scale", False)):
            twin_ax.set_ylim(ax.get_ylim())
        ax.set_xlabel(self.spec.options.get("xlabel", "Time (s)"), fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
        if twin_ax is not None:
            twin_ax.tick_params(labelsize=8)
            twin_ax.grid(False)

        handles, labels = ax.get_legend_handles_labels()
        if twin_ax is not None:
            h2, l2 = twin_ax.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
        if handles:
            ax.legend(
                handles, labels, loc="upper right", frameon=False, fontsize=8,
                ncol=min(len(handles), 3), handlelength=1.6, borderaxespad=0.6,
            )

        ax.set_title(self.spec.title or "", fontsize=10, fontweight="bold", loc="left")
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
