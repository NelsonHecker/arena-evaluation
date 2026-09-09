"""Build hero_timeseries.parquet: per-sample time series for the hero figure.

Panel B (stacked decomposed power trace) and Panel C (acoustic emission vs
pedestrian exposure, twin-axis) are declared in the ecobench_hero report
manifest; this module only prepares their shared row-per-sample parquet:

    t_s, p_mech_w, p_static_w, p_heat_w, p_total_w, emitted_dba, exposure_dba

Received exposure per sample is the robot's emitted level attenuated over the
world's wall geometry to the nearest pedestrian, solved with the same
transmission-loss model as the acoustic floor texture (per-pedestrian single-
target solver calls, door-state aware).
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import polars as pl


def _resolve_benchmark_dir(benchmark_arg: pathlib.Path | str) -> pathlib.Path:
    """Resolve a benchmark path or run name against the usual data roots."""
    p = pathlib.Path(benchmark_arg)
    if p.is_dir():
        return p
    roots: list[pathlib.Path] = []
    if os.environ.get("ARENA_DATA_DIR"):
        roots.append(pathlib.Path(os.environ["ARENA_DATA_DIR"]) / "benchmarks")
    if os.environ.get("ARENA_WS_DIR"):
        roots.append(pathlib.Path(os.environ["ARENA_WS_DIR"]) / "data" / "benchmarks")
    roots += [
        pathlib.Path("/opt/arena_ws/data/benchmarks"),
        pathlib.Path.cwd() / "data" / "benchmarks",
    ]
    for root in roots:
        cand = root / str(benchmark_arg)
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Benchmark directory not found: '{benchmark_arg}' (searched {roots}); "
        "pass the full path to the benchmark run."
    )


def _normalize_episode(episode_arg: str) -> str:
    s = str(episode_arg).strip()
    if s.startswith("episode_"):
        return s
    return f"episode_{int(s):03d}"


def _resolve_robot_dir(episode_dir: pathlib.Path) -> pathlib.Path | None:
    topics = episode_dir / "topics"
    if not topics.is_dir():
        return None
    for d in sorted(topics.iterdir()):
        if d.is_dir() and (d / "odom.parquet").is_file():
            return d
    return None


def _nearest_ped_per_sample(peds_df: pl.DataFrame) -> pl.DataFrame:
    """Flatten peds.parquet to one row per sample with the nearest ped (x, y)."""
    if peds_df is None or "peds_positions" not in peds_df.columns:
        return pl.DataFrame()
    rows = []
    for r in peds_df.rows(named=True):
        raw = r.get("peds_positions")
        if not raw:
            rows.append({"time_ns": r["time_ns"], "ped_x": None, "ped_y": None})
            continue
        pts: list[tuple[float, float]] = []
        if isinstance(raw[0], (list, tuple, np.ndarray)):
            pts = [(float(p[0]), float(p[1])) for p in raw if len(p) >= 2]
        elif isinstance(raw[0], dict):
            pts = [(float(p["x"]), float(p["y"])) for p in raw if "x" in p and "y" in p]
        else:
            stride = 3 if len(raw) % 3 == 0 and len(raw) >= 3 else 2
            pts = [(float(raw[j]), float(raw[j + 1])) for j in range(0, len(raw) - 1, stride)]
        if not pts:
            rows.append({"time_ns": r["time_ns"], "ped_x": None, "ped_y": None})
            continue
        nx, ny = pts[0]
        rows.append({"time_ns": r["time_ns"], "ped_x": nx, "ped_y": ny})
    return pl.DataFrame(rows)


def build_hero_timeseries(
    benchmark_dir: pathlib.Path | str,
    episode_arg: str,
    out_parquet: pathlib.Path | str,
    *,
    map_name: str | None = None,
    downsample: int = 2,
) -> pathlib.Path:
    from arena_evaluation.presentation.plot_types.acoustic_field import AcousticFieldRenderer
    from arena_evaluation.processing.acoustics.door_map import door_segments, build_pixel_tl
    from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
    from arena_evaluation.processing.acoustics.impedance_grid import (
        compute_attenuations,
        downsample_occupancy,
    )

    benchmark_dir = _resolve_benchmark_dir(benchmark_dir)
    episode_id = _normalize_episode(episode_arg)
    episode_dir = benchmark_dir / "episodes" / episode_id
    if not episode_dir.is_dir():
        for parent in benchmark_dir.parents:
            cand = parent / "episodes" / episode_id
            if cand.is_dir():
                episode_dir = cand
                break

    robot_dir = _resolve_robot_dir(episode_dir)
    if robot_dir is None:
        raise FileNotFoundError(
            f"No robot topic dir under {episode_dir}/topics. "
            "Run 'evaluation process --benchmark-dir <run>' first to extract topics."
        )

    # Aligned pose + emitted level (same loader the renderers use)
    ep_df = AcousticFieldRenderer._load_episode_data(episode_dir.parent.parent, episode_id)
    if ep_df is None:
        raise FileNotFoundError(f"No episode data for {episode_id}")

    power_df = None
    p_path = robot_dir / "power.parquet"
    if p_path.is_file():
        power_df = pl.read_parquet(p_path)

    peds_path = episode_dir / "topics" / "peds.parquet"
    peds_df = pl.read_parquet(peds_path) if peds_path.is_file() else None
    ped_rows = _nearest_ped_per_sample(peds_df)

    # Map / grid / doors
    if not map_name:
        metrics_path = benchmark_dir / "combined_metrics.parquet"
        if not metrics_path.is_file():
            metrics_path = benchmark_dir / "metrics.parquet"
        if metrics_path.is_file():
            mdf = pl.read_parquet(metrics_path)
            if "map" in mdf.columns and len(mdf):
                map_name = mdf["map"][0]
    if not map_name:
        raise ValueError("Could not determine map name; pass --map explicitly")

    grid, meta = AcousticFieldRenderer._load_grid_and_meta(map_name, run_dir=benchmark_dir)
    if grid is None:
        raise ValueError(f"Map {map_name!r} not found in registry")
    res = meta["resolution"]
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    doors = door_segments(map_name, grid, res, (ox, oy, 0.0), run_dir=benchmark_dir)

    grid_ds = downsample_occupancy(grid, downsample) if downsample > 1 else grid
    eff_res = res * downsample
    doors_ds: dict = {}
    if doors:
        h, w = grid_ds.shape
        for name, (mask, tl_db) in doors.items():
            m_ds = mask[::downsample, ::downsample] if downsample > 1 else mask
            doors_ds[name] = (m_ds[:h, :w], tl_db)

    semantic_path = episode_dir / "topics" / "semantic_snapshot.parquet"
    timeline = None
    if semantic_path.is_file():
        timeline = DoorStateTimeline.from_semantic_frame(pl.read_parquet(semantic_path))

    # Per-sample exposure at the nearest pedestrian (single-target solver calls)
    rows_ep = ep_df.rows(named=True)
    t0 = int(rows_ep[0]["time_ns"])
    tl_cache: dict[frozenset, np.ndarray] = {}
    samples: list[dict] = []

    for r in rows_ep:
        t_ns = int(r["time_ns"])
        t_s = (t_ns - t0) / 1e9
        rx = float(r["pos_x_gt"])
        ry = float(r["pos_y_gt"])
        emitted = float(r.get("source_dba") or 40.0)

        row = {
            "t_s": t_s,
            "p_mech_w": None,
            "p_static_w": None,
            "p_heat_w": None,
            "p_total_w": None,
            "emitted_dba": emitted,
            "exposure_dba": None,
        }
        if power_df is not None and len(power_df):
            p_row = power_df.filter(pl.col("time_ns") <= t_ns)
            if len(p_row):
                p = p_row.row(-1, named=True)
                row["p_mech_w"] = float(p.get("total_mechanical_power_w") or 0.0)
                row["p_static_w"] = float(p.get("static_power_w") or 0.0)
                row["p_heat_w"] = float(p.get("total_thermal_power_w") or 0.0)
                row["p_total_w"] = float(p.get("total_power_w") or 0.0)

        if len(ped_rows):
            p_row = ped_rows.filter(pl.col("time_ns") <= t_ns)
            if len(p_row):
                p = p_row.row(-1, named=True)
                px, py = p.get("ped_x"), p.get("ped_y")
                if px is not None and py is not None and compute_attenuations is not None:
                    open_set = timeline.open_doors_at(t_ns) if timeline is not None else frozenset()
                    key = frozenset(open_set)
                    pixel_tl = tl_cache.get(key)
                    if pixel_tl is None:
                        pixel_tl = build_pixel_tl(grid_ds, doors_ds, open_doors=set(open_set))
                        tl_cache[key] = pixel_tl
                    atten = compute_attenuations(
                        occupancy_grid=grid_ds,
                        resolution=eff_res,
                        start_x_px=(rx - ox) / eff_res,
                        start_y_px=(ry - oy) / eff_res,
                        target_xs_px=np.ascontiguousarray([(px - ox) / eff_res], dtype=np.float32),
                        target_ys_px=np.ascontiguousarray([(py - oy) / eff_res], dtype=np.float32),
                        wall_tl=47.0,
                        mic_distance=1.0,
                        pixel_tl=pixel_tl,
                    )
                    if len(atten):
                        row["exposure_dba"] = float(np.clip(emitted - float(atten[0]), 0.0, None))
        samples.append(row)

    out = pl.DataFrame(samples)
    out_parquet = pathlib.Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_parquet)
    return out_parquet


def main() -> None:
    ap = argparse.ArgumentParser(description="Build hero_timeseries.parquet for the ecobench_hero manifest.")
    ap.add_argument("--benchmark-dir", required=True, type=pathlib.Path, help="Benchmark run directory")
    ap.add_argument("--episode", required=True, help="Episode ID (e.g. 040 or episode_040)")
    ap.add_argument("--output", type=pathlib.Path, default=None, help="Output parquet (default: <benchmark-dir>/hero_timeseries.parquet)")
    ap.add_argument("--map", default=None, help="Override map name (auto from combined_metrics)")
    args = ap.parse_args()

    out = args.output or args.benchmark_dir / "hero_timeseries.parquet"
    p = build_hero_timeseries(args.benchmark_dir, args.episode, out, map_name=args.map)
    n = len(pl.read_parquet(p))
    print(f"[OK] Wrote {p} ({n} samples)")


if __name__ == "__main__":
    main()
