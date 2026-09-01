#!/usr/bin/env python3
"""CLI Script to generate and assemble publication figures (Figures 2, 3, 4, 5).

Usage:
    python scripts/generate_paper_figures.py --fig 2 --benchmark-dir /path/to/run --out-dir ./figures/
    python scripts/generate_paper_figures.py --all --mock --out-dir ./figures/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import polars as pl

# Support direct invocation
script_dir = pathlib.Path(__file__).resolve().parent
pkg_dir = script_dir.parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

from arena_evaluation.presentation.figure_grid_assembler import (
    assemble_figure2_cost_of_crowding,
    assemble_figure3_spatial_acoustics,
    assemble_figure4_office_disturbance,
    assemble_figure5_fleet_autonomy,
)


def load_benchmark_data(benchmark_dir: pathlib.Path | None) -> pl.DataFrame:
    """Load combined_metrics or metrics parquet, or return empty df for mock synthesis."""
    if benchmark_dir and benchmark_dir.is_dir():
        for name in ("combined_metrics.parquet", "metrics.parquet"):
            p = benchmark_dir / name
            if p.exists():
                print(f"[info] Loading benchmark data from {p}")
                return pl.read_parquet(p)
    print("[info] Using physics-grounded synthetic scenario telemetry.")
    return pl.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ICRA 2027 Publication Figures")
    parser.add_argument("--fig", choices=["2", "3", "4", "5", "all"], default="all", help="Figure number to assemble")
    parser.add_argument("--benchmark-dir", type=pathlib.Path, default=None, help="Benchmark run directory containing parquet telemetry")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("./paper_figures"), help="Output directory for generated HTML and PNG figures")
    parser.add_argument("--mock", action="store_true", help="Force synthetic visualization test pipeline")

    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame() if args.mock else load_benchmark_data(args.benchmark_dir)

    figs_to_build = ["2", "3", "4", "5"] if args.fig == "all" else [args.fig]

    for f in figs_to_build:
        if f == "2":
            print("[figure 2] Assembling 6-Panel Cost of Crowding Master Grid...")
            html_p = out_dir / "figure2_cost_of_crowding.html"
            png_p = out_dir / "figure2_cost_of_crowding.png"
            assemble_figure2_cost_of_crowding(df, output_html=html_p, output_png=png_p)
            print(f"  -> Generated: {html_p}")

        elif f == "3":
            print("[figure 3] Assembling 2D Spatial Acoustic Propagation & Worst-Case Emissions Grid...")
            html_p = out_dir / "figure3_spatial_acoustics.html"
            png_p = out_dir / "figure3_spatial_acoustics.png"
            assemble_figure3_spatial_acoustics(df, output_html=html_p, output_png=png_p)
            print(f"  -> Generated: {html_p}")

        elif f == "4":
            print("[figure 4] Assembling Open Office Desk Service & Pareto Frontier...")
            html_p = out_dir / "figure4_office_disturbance.html"
            png_p = out_dir / "figure4_office_disturbance.png"
            assemble_figure4_office_disturbance(df, output_html=html_p, output_png=png_p)
            print(f"  -> Generated: {html_p}")

        elif f == "5":
            print("[figure 5] Assembling Fleet Logistics, Autonomy Scaling & Multi-Axis Radar...")
            html_p = out_dir / "figure5_fleet_autonomy.html"
            png_p = out_dir / "figure5_fleet_autonomy.png"
            assemble_figure5_fleet_autonomy(df, output_html=html_p, output_png=png_p)
            print(f"  -> Generated: {html_p}")

    print(f"\n[success] All requested publication figures generated in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
