"""Standalone hero-figure generator (no report manifest involved).

Renders the two hero panels as separate publication PNGs:

  hero_power.png     Clean stat line (Total Energy, Energy per Meter) +
                     stacked decomposed power trace (mechanical base, static
                     mid, heat spikes top) with pedestrian-encounter markers.

  hero_acoustic.png  Clean stat line (Emitted L_Aeq, Max Received) +
                     solid dual-trace decibel plot (emitted vs received on a
                     shared scale) with red door open/close event markers.

No smoothing is applied anywhere; traces are the raw per-sample telemetry.

Usage:
  python -m arena_evaluation.paper.hero_figure \
    --benchmark-dir <run-or-name> --episode episode_001 [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


# Muted publication palette (same family as the ecobench_hero manifest)
GREY = "#aab3bd"
BLUE = "#2a6f97"
TERRA = "#c05a4e"
TEAL = "#357f6e"
AMBER = "#d9a05b"
RED = "#c0392b"
INK = "#2b2f36"
MUTED = "#8a919b"

# 170 x 60 mm banner (full text width x compact strip)
BANNER_FIGSIZE = (170.0 / 25.4, 60.0 / 25.4)


def _stat_line(ax, x, value: str, label: str, accent: str) -> None:
    """Plain, unboxed stat: big bold value with a small caps label beneath."""
    ax.text(x, 0.60, value, transform=ax.transAxes, ha="center", va="center",
            fontsize=14, fontweight="bold", color=accent)
    ax.text(x, 0.16, label.upper(), transform=ax.transAxes, ha="center", va="center",
            fontsize=6.5, color=MUTED)


def _base_style(ax, ylabel: str, xmax: float) -> None:
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#9aa0a8")
    ax.spines["bottom"].set_color("#9aa0a8")


def _legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8,
                  ncol=min(len(handles), 3), handlelength=1.6, borderaxespad=0.6)


def _encounter_windows(bench: pathlib.Path, episode_id: str, t_ns: np.ndarray, t_s: np.ndarray,
                       threshold_m: float = 1.2):
    """(start, end) time windows where the robot is within threshold of a ped."""
    import polars as pl
    from arena_evaluation.presentation.plot_types.acoustic_field import AcousticFieldRenderer
    from arena_evaluation.paper.hero_timeseries import _nearest_ped_per_sample

    ep_df = AcousticFieldRenderer._load_episode_data(bench, episode_id)
    peds_path = bench / "episodes" / episode_id / "topics" / "peds.parquet"
    if ep_df is None or not peds_path.is_file():
        return []
    ped_rows = _nearest_ped_per_sample(pl.read_parquet(peds_path))
    if not len(ped_rows):
        return []

    ped_t = ped_rows["time_ns"].to_numpy()
    ped_x = ped_rows["ped_x"].to_numpy()
    ped_y = ped_rows["ped_y"].to_numpy()
    rx = ep_df["pos_x_gt"].to_numpy()
    ry = ep_df["pos_y_gt"].to_numpy()

    dist = np.full(len(t_ns), np.nan)
    for i, tn in enumerate(t_ns):
        j = int(np.searchsorted(ped_t, tn, side="right")) - 1
        if j < 0 or ped_x[j] is None:
            continue
        dist[i] = float(np.hypot(rx[i] - ped_x[j], ry[i] - ped_y[j]))

    windows = []
    inside = False
    for i in range(len(t_s)):
        d = dist[i]
        if not np.isnan(d) and d <= threshold_m:
            if not inside:
                windows.append([float(t_s[i]), float(t_s[i])])
                inside = True
            windows[-1][1] = float(t_s[i])
        else:
            inside = False
    return [(w[0], w[1]) for w in windows if w[1] - w[0] > 0.5]


def render_power_figure(
    t: np.ndarray,
    p_static: np.ndarray,
    p_mech: np.ndarray,
    p_heat: np.ndarray,
    p_total: np.ndarray,
    out_png: pathlib.Path,
    *,
    xmax: float = 40.0,
    energy_per_meter: float | None = None,
    encounters: list | None = None,
    dpi: int = 300,
) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total_wh = float(np.trapezoid(p_total, t) / 3600.0)

    fig = plt.figure(figsize=BANNER_FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 3.0], hspace=0.06)

    cards = fig.add_subplot(gs[0])
    cards.set_xlim(0, 1)
    cards.set_ylim(0, 1)
    cards.axis("off")
    _stat_line(cards, 0.25, f"{total_wh:.2f} Wh", "Total Energy", BLUE)
    _stat_line(cards, 0.75, f"{energy_per_meter:.3f} Wh/m" if energy_per_meter is not None else "— Wh/m",
               "Energy per Meter", TERRA)

    ax = fig.add_subplot(gs[1])
    ax.stackplot(
        t, p_mech, p_static, p_heat,
        colors=[BLUE, GREY, TERRA], alpha=0.9, linewidth=0,
    )

    # Pedestrian-encounter markers: light bands where the robot is within the
    # personal band of a pedestrian.
    if encounters:
        for lo, hi in encounters:
            ax.axvspan(lo, hi, color=RED, alpha=0.08, linewidth=0)

    _base_style(ax, "Power (W)", xmax)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.965, bottom=0.16)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def render_acoustic_figure(
    t: np.ndarray,
    emitted: np.ndarray,
    exposure: np.ndarray,
    out_png: pathlib.Path,
    *,
    xmax: float = 40.0,
    dpi: int = 300,
) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = emitted[np.isfinite(emitted)]
    leq = float(10.0 * np.log10(np.mean(10.0 ** (valid / 10.0)))) if len(valid) else float("nan")
    max_recv = float(np.nanmax(exposure)) if len(exposure) else float("nan")

    fig = plt.figure(figsize=BANNER_FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 3.0], hspace=0.06)

    cards = fig.add_subplot(gs[0])
    cards.set_xlim(0, 1)
    cards.set_ylim(0, 1)
    cards.axis("off")
    _stat_line(cards, 0.25, f"{leq:.1f} dBA", "Emitted L_Aeq", TEAL)
    _stat_line(cards, 0.75, f"{max_recv:.1f} dBA", "Max Received", AMBER)

    ax = fig.add_subplot(gs[1])
    ax.plot(t, emitted, color=TEAL, linewidth=2.0)
    ax.plot(t, exposure, color=AMBER, linewidth=2.0)

    ax.set_ylim(0.0, 70.0)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])

    _base_style(ax, "Level (dBA)", xmax)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.965, bottom=0.16)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the two hero-figure panels as standalone PNGs.")
    ap.add_argument("--benchmark-dir", required=True, help="Benchmark run directory or name")
    ap.add_argument("--episode", default="episode_001", help="Hero episode (default: episode_001)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None, help="Output directory (default: <benchmark>/plots)")
    ap.add_argument("--xmax", type=float, default=40.0, help="Time axis maximum in seconds (default: 40)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    import polars as pl

    from arena_evaluation.paper.hero_timeseries import build_hero_timeseries, _resolve_benchmark_dir, _normalize_episode
    from arena_evaluation.presentation.plot_types.acoustic_field import AcousticFieldRenderer

    bench = _resolve_benchmark_dir(args.benchmark_dir)
    ep_id = _normalize_episode(args.episode)
    parquet = bench / "hero_timeseries.parquet"
    if not parquet.exists():
        build_hero_timeseries(bench, ep_id, parquet)
    df = pl.read_parquet(parquet)

    t = df["t_s"].to_numpy().astype(float)
    mask = t <= args.xmax + 1e-6
    t_c, p_stat, p_mech, p_heat, p_total = (
        t[mask],
        df["p_static_w"].to_numpy().astype(float)[mask],
        df["p_mech_w"].to_numpy().astype(float)[mask],
        df["p_heat_w"].to_numpy().astype(float)[mask],
        df["p_total_w"].to_numpy().astype(float)[mask],
    )
    emitted = df["emitted_dba"].to_numpy().astype(float)[mask]
    exposure = df["exposure_dba"].to_numpy().astype(float)[mask]

    # Energy per meter from the episode metrics (falls back to None)
    energy_per_meter = None
    metrics_path = bench / "combined_metrics.parquet"
    if metrics_path.is_file():
        mdf = pl.read_parquet(metrics_path)
        ep_num = int(ep_id.replace("episode_", ""))
        rows = mdf.filter(pl.col("episode") == ep_num)
        if len(rows) and "path_length" in rows.columns:
            pl_val = rows["path_length"][0]
            total = float(np.trapezoid(p_total, t_c) / 3600.0)
            if pl_val is not None and total > 0.0:
                energy_per_meter = total / float(pl_val)

    # Pedestrian encounters from the episode topics
    ep_df = AcousticFieldRenderer._load_episode_data(bench, ep_id)
    t_ns = ep_df["time_ns"].to_numpy() if ep_df is not None else np.array([], dtype=np.int64)
    t_ns = t_ns[mask][: len(t_c)]
    encounters = _encounter_windows(bench, ep_id, t_ns, t_c)

    out_dir = args.out_dir or (bench / "plots")
    p1 = render_power_figure(t_c, p_stat, p_mech, p_heat, p_total, out_dir / "hero_power.png",
                             xmax=args.xmax, energy_per_meter=energy_per_meter,
                             encounters=encounters, dpi=args.dpi)
    p2 = render_acoustic_figure(t_c, emitted, exposure, out_dir / "hero_acoustic.png",
                                xmax=args.xmax, dpi=args.dpi)
    print(f"[OK] {p1}  (encounters: {len(encounters)})")
    print(f"[OK] {p2}")


if __name__ == "__main__":
    main()
