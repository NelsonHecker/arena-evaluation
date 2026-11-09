"""Long-format line chart renderer supporting grouped traces and error bands."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer, finish_axes

__all__ = ["LineRenderer"]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _LineConfig:
    """Resolved line-plot channels and options shared by both renderers."""

    x_col: str
    y_col: str
    group_cols: tuple[str, ...]
    error_col: str | None
    error_style: str
    mode: str
    time_to_s: bool
    time_relative: bool
    max_traces: int
    max_points_per_trace: int
    smooth: bool
    smooth_window: int
    error_mode: str


class LineRenderer(BasePlotRenderer):
    PLOT_TYPE = "line"

    def _prepare(self, df: pl.DataFrame) -> tuple[pl.DataFrame, _LineConfig] | None:
        """Validate channels and return the plotted frame plus the resolved configuration."""
        df_filtered = self._apply_filters(df)
        options: Mapping[str, Any] = self.spec.options or {}
        y_col = self.resolve_channel("y") or _option_str(options, "y")
        x_col = self.resolve_channel("x") or self.spec.data_key
        if options.get("y") and y_col == x_col:
            y_col = _option_str(options, "y")
        if not y_col or not x_col:
            return None
        schema = df_filtered.schema
        if schema.get(x_col) is None or schema.get(y_col) is None:
            return None

        color_col = self.resolve_channel("color")
        group_by = self.spec.group_by
        if isinstance(group_by, str):
            group_by = [group_by]
        group_cols = [g for g in (group_by or []) if schema.get(g) is not None]
        if not group_cols:
            if color_col and schema.get(color_col) is not None:
                group_cols = [color_col]
            else:
                diff_col, df_filtered = self.resolve_diff_col(df_filtered)
                if df_filtered.schema.get(diff_col) is not None:
                    group_cols = [diff_col]

        error_col = _option_str(options, "error_y") or None
        if error_col and df_filtered.schema.get(error_col) is None:
            error_col = None

        filter_keys = [k for k in (self.spec.filter or {}) if df_filtered.schema.get(k) is not None]
        keep = list(dict.fromkeys([c for c in [x_col, y_col, *group_cols, error_col, *filter_keys] if c]))
        list_cols = [c for c in keep if df_filtered.schema[c] == pl.List]
        if list_cols:
            df_filtered = df_filtered.select(keep).explode(list_cols)
        df_filtered = self._apply_row_filters(df_filtered)

        bin_x = options.get("bin_x")
        if bin_x is not None and float(bin_x) > 0:
            width = float(bin_x)
            df_filtered = df_filtered.with_columns(((pl.col(x_col) / width).round() * width).round(4).alias(x_col))

        trim_p = float(options.get("trim_percentile", 0.0) or (1.0 - float(options.get("inliers", 1.0))))
        if 0.0 < trim_p < 0.5 and len(df_filtered) > 0:
            group_keys = [c for c in [x_col, *group_cols] if df_filtered.schema.get(c) is not None]
            if group_keys:
                low = trim_p / 2.0
                high = 1.0 - low
                df_filtered = df_filtered.filter((pl.col(y_col) >= pl.col(y_col).quantile(low).over(group_keys)) & (pl.col(y_col) <= pl.col(y_col).quantile(high).over(group_keys)))

        error_mode = _option_str(options, "error_mode", "std").lower()
        if options.get("aggregate") and len(df_filtered) > 0:
            agg_cols = [x_col, *group_cols]
            reduce_mode = _option_str(options, "reduce", "mean")
            if reduce_mode == "leq":
                df_filtered = df_filtered.group_by(agg_cols).agg((10.0 * (10.0 ** (pl.col(y_col) / 10.0)).mean().log10()).alias(y_col))
            elif reduce_mode == "max":
                df_filtered = df_filtered.group_by(agg_cols).agg(pl.col(y_col).max().alias(y_col))
            else:
                if error_mode in ("ci95", "ci_95", "ci"):
                    df_filtered = df_filtered.group_by(agg_cols).agg(
                        pl.col(y_col).mean().alias(y_col),
                        (1.96 * pl.col(y_col).std() / pl.col(y_col).count().sqrt()).fill_null(0.0).alias("__std__"),
                    )
                elif error_mode == "sem":
                    df_filtered = df_filtered.group_by(agg_cols).agg(
                        pl.col(y_col).mean().alias(y_col),
                        (pl.col(y_col).std() / pl.col(y_col).count().sqrt()).fill_null(0.0).alias("__std__"),
                    )
                else:
                    df_filtered = df_filtered.group_by(agg_cols).agg(
                        pl.col(y_col).mean().alias(y_col),
                        pl.col(y_col).std().alias("__std__"),
                    )
                error_col = "__std__"

        selected = df_filtered.select(
            pl.col(x_col),
            pl.col(y_col),
            *[pl.col(name) for name in group_cols],
            *([pl.col(error_col)] if error_col else []),
        )
        if selected.is_empty():
            return None

        config = _LineConfig(
            x_col=x_col,
            y_col=y_col,
            group_cols=tuple(group_cols),
            error_col=error_col,
            error_style=_option_str(options, "error_style", "band"),
            mode=_option_str(options, "mode", "lines"),
            time_to_s=_option_bool(options, "time_to_s"),
            time_relative=_option_bool(options, "time_relative"),
            max_traces=_option_int(options, "max_traces", 40),
            max_points_per_trace=_option_int(options, "max_points_per_trace", 5000),
            smooth=_option_bool(options, "smooth"),
            smooth_window=_option_int(options, "smooth_window", 3),
            error_mode=error_mode,
        )
        return selected, config

    @staticmethod
    def _trace_parts(frame: pl.DataFrame, group_cols: tuple[str, ...], max_traces: int) -> list[tuple[tuple[object, ...], pl.DataFrame]]:
        """One (group key, frame) pair per trace, capped at ``max_traces``."""
        if not group_cols:
            return [((), frame)]
        parts: list[tuple[tuple[object, ...], pl.DataFrame]] = []
        for key, group in frame.group_by(list(group_cols), maintain_order=True):
            if len(parts) >= max_traces:
                break
            parts.append((key, group))
        return parts

    @staticmethod
    def _trace_data(frame: pl.DataFrame, config: _LineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Downsample + transform one trace's frame -> (x, y, err) arrays."""
        if config.max_points_per_trace > 0 and frame.height > config.max_points_per_trace:
            frame = frame.gather_every(max(1, frame.height // config.max_points_per_trace))

        x = frame[config.x_col].cast(pl.Float64).to_numpy()
        y = frame[config.y_col].cast(pl.Float64).to_numpy()
        err = frame[config.error_col].cast(pl.Float64).to_numpy() if config.error_col else None

        if config.time_to_s:
            x = x / 1e9
        if config.time_relative and len(x) > 0:
            x = x - x.min()

        order = x.argsort()
        x, y = x[order], y[order]
        if err is not None:
            err = err[order]

        if config.smooth and len(y) >= 3:
            width = max(2, min(config.smooth_window, len(y)))
            kernel = np.ones(width) / width
            pad_left = width // 2
            pad_right = width - 1 - pad_left
            y = np.convolve(np.pad(y, (pad_left, pad_right), mode="edge"), kernel, mode="valid")
            if err is not None:
                err = np.convolve(np.pad(err, (pad_left, pad_right), mode="edge"), kernel, mode="valid")

        return x, y, err

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        prepared = self._prepare(df)
        if prepared is None:
            return None
        selected, config = prepared
        parts = self._trace_parts(selected, config.group_cols, config.max_traces)

        band_suffix = "± 95% CI" if config.error_mode in ("ci95", "ci_95", "ci") else ("± 1σ (smoothed)" if config.smooth else "± 1σ")
        fig = go.Figure()
        colors = _color_cycle()

        for trace_index, (key, part) in enumerate(parts):
            x, y, err = self._trace_data(part, config)
            if len(x) == 0:
                continue
            label = " / ".join(str(value) for value in key) if config.group_cols else config.y_col
            color = colors[trace_index % len(colors)]

            if config.error_col is not None and err is not None:
                err = np.nan_to_num(err, nan=0.0)
                non_negative = bool(np.all(y >= 0))
                if config.error_style == "bars":
                    arrayminus = np.minimum(err, y) if non_negative else err
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode=config.mode,
                            name=label,
                            line=dict(color=color),
                            error_y=dict(type="data", array=err, arrayminus=arrayminus, symmetric=False, visible=True),
                        )
                    )
                else:
                    y_low = np.maximum(0.0, y - err) if non_negative else y - err
                    y_high = y + err
                    fig.add_trace(go.Scatter(x=x, y=y, mode=config.mode, name=label, line=dict(color=color)))
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x, x[::-1]]),
                            y=np.concatenate([y_high, y_low[::-1]]),
                            fill="toself",
                            fillcolor=_with_alpha(color, 0.2),
                            line=dict(width=0),
                            name=f"{label} {band_suffix}",
                            legendgroup=label,
                            showlegend=False,
                        )
                    )
            else:
                fig.add_trace(go.Scatter(x=x, y=y, mode=config.mode, name=label, line=dict(color=color)))

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time [s]" if config.time_to_s else self.format_label(config.x_col.replace("_", " ").title(), config.x_col),
            yaxis_title=self.format_label(config.y_col.replace("_", " ").title(), config.y_col),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        prepared = self._prepare(df)
        if prepared is None:
            return
        selected, config = prepared
        parts = self._trace_parts(selected, config.group_cols, config.max_traces)
        colors = _color_cycle()
        marker = "o" if "markers" in config.mode else None

        fig, ax = plt.subplots(figsize=(10, 6))
        for trace_index, (key, part) in enumerate(parts):
            x, y, err = self._trace_data(part, config)
            if len(x) == 0:
                continue
            label = " / ".join(str(value) for value in key) if config.group_cols else config.y_col
            color = colors[trace_index % len(colors)]

            if config.error_col is not None and err is not None:
                err = np.nan_to_num(err, nan=0.0)
                non_negative = bool(np.all(y >= 0))
                if config.error_style == "bars":
                    yerr = [np.minimum(err, y), err] if non_negative else err
                    ax.errorbar(x, y, yerr=yerr, label=label, color=color, marker=marker, capsize=2)
                else:
                    y_low = np.maximum(0.0, y - err) if non_negative else y - err
                    ax.plot(x, y, label=label, color=color, marker=marker)
                    ax.fill_between(x, y_low, y + err, color=color, alpha=0.2)
            else:
                ax.plot(x, y, label=label, color=color, marker=marker)

        finish_axes(
            ax,
            title=self.spec.title,
            x_label="Time [s]" if config.time_to_s else self.format_label(config.x_col.replace("_", " ").title(), config.x_col),
            y_label=self.format_label(config.y_col.replace("_", " ").title(), config.y_col),
            grid_axis="both",
        )
        if config.group_cols:
            ax.legend(frameon=False)
        self.save_figure(fig, out_path)
        plt.close(fig)


def _option_str(options: Mapping[str, Any], key: str, default: str = "") -> str:
    value = options.get(key)
    return str(value) if value is not None else default


def _option_bool(options: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = options.get(key)
    return bool(value) if value is not None else default


def _option_int(options: Mapping[str, Any], key: str, default: int) -> int:
    value = options.get(key)
    return int(value) if value is not None else default


def _color_cycle() -> list[str]:
    palette = get_color_palette()
    return list(palette) if palette else ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _with_alpha(hex_color: str, alpha: float) -> str:
    """``rgba(...)`` string for a ``#rrggbb`` color, the first palette color when it cannot be parsed."""
    try:
        red, green, blue = (int(hex_color.lstrip("#")[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return f"rgba(31,119,180,{alpha})"
    return f"rgba({red},{green},{blue},{alpha})"