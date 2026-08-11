import argparse
import contextlib
import datetime
import os
import pathlib
import sys

from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.storage.data_root import latest_benchmark
from arena_evaluation.storage.folder_manager import FolderManager


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    search_roots = []

    arena_data_dir_env = os.environ.get("ARENA_DATA_DIR")
    if arena_data_dir_env:
        search_roots.append(pathlib.Path(arena_data_dir_env))

    cwd = pathlib.Path.cwd()
    search_roots.append(cwd / "data")

    for parent in cwd.parents:
        data_candidate = parent / "data"
        if data_candidate.is_dir():
            search_roots.append(data_candidate)

    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_data = pathlib.Path(get_package_share_directory("arena_evaluation")) / "data"
        if pkg_data.is_dir():
            search_roots.append(pkg_data)
    except Exception:
        pass

    unique_search_roots = []
    seen = set()
    for root in search_roots:
        resolved_root = root.resolve()
        if resolved_root not in seen and resolved_root.is_dir():
            seen.add(resolved_root)
            unique_search_roots.append(resolved_root)

    def _resolve_single_path(p: pathlib.Path, subdirs: tuple[str, ...]) -> pathlib.Path:
        if p.exists():
            return p.resolve()
        for root in unique_search_roots:
            for sub in subdirs:
                candidate = root / sub / p
                if candidate.is_dir():
                    return candidate.resolve()
        return p
    if getattr(args, "run_dir", None) is not None:
        if isinstance(args.run_dir, list):
            args.run_dir = [
                _resolve_single_path(p, ("recordings", "recording"))
                for p in args.run_dir
            ]
        else:
            args.run_dir = _resolve_single_path(args.run_dir, ("recordings", "recording"))

    if getattr(args, "benchmark_dir", None) is not None:
        if isinstance(args.benchmark_dir, list):
            args.benchmark_dir = [
                _resolve_single_path(p, ("benchmarks", "benchmark"))
                for p in args.benchmark_dir
            ]
        else:
            args.benchmark_dir = _resolve_single_path(args.benchmark_dir, ("benchmarks", "benchmark"))

    return args

def _handle_acoustic(args: argparse.Namespace) -> None:
    """Handle 'evaluation acoustic' subcommands."""
    import polars as pl
    from arena_evaluation.processing.parquet_store import ParquetStore

    benchmark_dir = args.benchmark_dir
    if not benchmark_dir.is_dir():
        print(f"Error: benchmark directory does not exist: {benchmark_dir}")
        sys.exit(1)

    metrics_path = benchmark_dir / "combined_metrics.parquet"
    if not metrics_path.exists():
        metrics_path = benchmark_dir / "metrics.parquet"
    if not metrics_path.exists():
        print(f"Error: no metrics.parquet or combined_metrics.parquet found in {benchmark_dir}")
        print("Run 'evaluation process --benchmark-dir ...' first.")
        sys.exit(1)

    df, _ = ParquetStore.read(metrics_path)

    if args.acoustic_command == "list":
        _acoustic_list(df)
    elif args.acoustic_command == "animate":
        _acoustic_animate(df, args)
    elif args.acoustic_command == "snapshot":
        _acoustic_snapshot(df, args)


def _acoustic_list(df: "pl.DataFrame") -> None:
    """Print a table of episodes with acoustic metrics."""
    import polars as pl

    cols = []
    if "episode" in df.columns:
        cols.append("episode")
    if "ped_max_exposure_dba" in df.columns:
        cols.append("ped_max_exposure_dba")
    if "ped_leq_exposure_dba" in df.columns:
        cols.append("ped_leq_exposure_dba")

    if not cols:
        print("No acoustic metric columns found in metrics file.")
        return

    work = df.select(cols).sort("ped_max_exposure_dba", descending=True) if "ped_max_exposure_dba" in df.columns else df.select(cols)

    # Compute total exposure if timeseries available
    has_total = "timeseries_acoustic_exposure_dba" in df.columns

    print(f"{'EPISODE':<16} {'MAX_DBA':>10} {'LEQ_DBA':>10}", end="")
    if has_total:
        print(f" {'TOTAL_EXP':>12}", end="")
    print()
    print("-" * (16 + 10 + 10 + (12 if has_total else 0)))

    for row in work.iter_rows(named=True):
        ep = str(row.get("episode", "?"))
        mx = row.get("ped_max_exposure_dba")
        leq = row.get("ped_leq_exposure_dba")
        mx_str = f"{mx:.1f}" if mx is not None else "N/A"
        leq_str = f"{leq:.1f}" if leq is not None else "N/A"
        print(f"{ep:<16} {mx_str:>10} {leq_str:>10}", end="")
        if has_total:
            ts = df.filter(pl.col("episode") == row.get("episode")).select("timeseries_acoustic_exposure_dba").row(0)[0] if "episode" in df.columns else None
            if ts is not None:
                total = sum(sum(f) for f in ts if f)
                print(f" {total:>12.1f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()


def _resolve_episode(df: "pl.DataFrame", episode_spec: str) -> str | None:
    """Resolve an episode specifier ('worst', 'loudest-source', 'max-total', or 'episode_NNN')."""
    import polars as pl

    if episode_spec.startswith("episode_"):
        ep_str = episode_spec
        if "episode" in df.columns:
            ep_int = int(episode_spec.split("_")[-1])
            if df.filter(pl.col("episode") == ep_int).is_empty():
                print(f"Error: {episode_spec} not found in metrics.")
                return None
        return ep_str

    if "ped_max_exposure_dba" not in df.columns:
        print("Error: ped_max_exposure_dba not found in metrics. Cannot resolve episode by metric.")
        return None

    if episode_spec == "worst":
        best = df.filter(pl.col("ped_max_exposure_dba").is_not_null()).sort("ped_max_exposure_dba", descending=True).row(0, named=True)
        ep = int(best.get("episode", 0))
        print(f"Using worst episode: episode_{ep:03d} (max={best['ped_max_exposure_dba']:.1f} dBA)")
        return f"episode_{ep:03d}"

    if episode_spec == "loudest-source":
        if "worst_case_acoustic_frame" not in df.columns:
            print("Error: worst_case_acoustic_frame not in metrics.")
            return None
        best_source = -1.0
        best_ep = 0
        for row in df.iter_rows(named=True):
            wf = row.get("worst_case_acoustic_frame")
            if wf is None:
                continue
            if isinstance(wf, str):
                import json
                try:
                    wf = json.loads(wf)
                except Exception:
                    continue
            src = float(wf.get("source_dba", 0))
            if src > best_source:
                best_source = src
                best_ep = int(row.get("episode", 0))
        print(f"Using loudest-source episode: episode_{best_ep:03d} (source={best_source:.1f} dBA)")
        return f"episode_{best_ep:03d}"

    if episode_spec == "max-total":
        if "timeseries_acoustic_exposure_dba" not in df.columns:
            print("Error: timeseries_acoustic_exposure_dba not in metrics.")
            return None
        best_total = -1.0
        best_ep = 0
        for row in df.iter_rows(named=True):
            ts = row.get("timeseries_acoustic_exposure_dba")
            if ts is None:
                continue
            total = sum(sum(f) for f in ts if f)
            if total > best_total:
                best_total = total
                best_ep = int(row.get("episode", 0))
        print(f"Using max-total episode: episode_{best_ep:03d} (total={best_total:.1f})")
        return f"episode_{best_ep:03d}"

    print(f"Error: unknown episode specifier '{episode_spec}'.")
    return None


def _acoustic_animate(df: "pl.DataFrame", args: argparse.Namespace) -> None:
    """Generate an animated acoustic field visualization."""
    episode_id = _resolve_episode(df, args.episode)
    if episode_id is None:
        sys.exit(1)

    from arena_evaluation.presentation.plot_types.acoustic_field import AcousticFieldRenderer
    from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
    import polars as pl

    renderer = AcousticFieldRenderer(None)
    renderer.run_dir = args.benchmark_dir

    # Load map
    map_name = None
    metrics_path = args.benchmark_dir / "combined_metrics.parquet"
    if not metrics_path.exists():
        metrics_path = args.benchmark_dir / "metrics.parquet"
    if metrics_path.exists():
        from arena_evaluation.processing.parquet_store import ParquetStore
        metrics_df, _ = ParquetStore.read(metrics_path)
        if "map" in metrics_df.columns and len(metrics_df) > 0:
            map_name = metrics_df["map"][0]

    if not map_name:
        print("Error: could not determine map name from metrics.")
        sys.exit(1)

    result = renderer._load_grid_and_meta(map_name, run_dir=args.benchmark_dir)
    if result is None:
        print(f"Error: could not load map '{map_name}'.")
        sys.exit(1)

    grid, meta = result
    resolution = meta["resolution"]
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

    # Load episode data
    episode_df = AcousticFieldRenderer._load_episode_data(args.benchmark_dir, episode_id)
    if episode_df is None:
        print(f"Error: no topic data for {episode_id}. Run 'evaluation extract' first.")
        sys.exit(1)

    # Load doors
    from arena_evaluation.processing.acoustics.door_map import door_segments
    doors = door_segments(map_name, grid, resolution, (ox, oy, 0.0), run_dir=args.benchmark_dir)

    # Load door state timeline
    state_timeline = None
    semantic_path = args.benchmark_dir / "episodes" / episode_id / "topics" / "semantic_snapshot.parquet"
    if semantic_path.exists():
        semantic_df = pl.read_parquet(semantic_path)
        state_timeline = DoorStateTimeline.from_semantic_frame(semantic_df)

    # Output path
    if args.output:
        out_path = args.output
    else:
        plots_dir = args.benchmark_dir / "plots"
        ext = "gif" if args.format != "frames" else ""
        out_path = plots_dir / f"{episode_id}_acoustic.{ext}" if ext else plots_dir / f"{episode_id}_acoustic_frames"

    print(f"Rendering animation for {episode_id} ({len(episode_df)} data frames)...")
    print(f"  fps={args.fps}, max_frames={args.max_frames}, stride={args.stride}")
    print(f"  downsample={args.downsample}, format={args.format}")
    print(f"  doors: {len(doors)} found" + (f", timeline: {'present' if state_timeline else 'ABSENT'}" if doors else ""))

    result_path = renderer.render_animation(
        episode_df, grid, resolution, ox, oy, doors,
        state_timeline=state_timeline,
        out_path=out_path,
        downsample=args.downsample,
        stride=args.stride,
        max_frames=args.max_frames,
        fps=args.fps,
        dpi=args.dpi,
        vmin=args.vmin,
        vmax=args.vmax,
        robot_trail=args.robot_trail,
        show_doors=not args.no_door_overlay,
        fmt=args.format,
    )

    if result_path:
        print(f"Animation saved to: {result_path}")
    else:
        print("Animation generation failed.")
        sys.exit(1)


def _acoustic_snapshot(df: "pl.DataFrame", args: argparse.Namespace) -> None:
    """Render a single acoustic field frame."""
    episode_id = _resolve_episode(df, args.episode)
    if episode_id is None:
        sys.exit(1)

    from arena_evaluation.presentation.plot_types.acoustic_field import AcousticFieldRenderer
    from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
    import polars as pl

    renderer = AcousticFieldRenderer(None)
    renderer.run_dir = args.benchmark_dir

    map_name = None
    metrics_path = args.benchmark_dir / "combined_metrics.parquet"
    if not metrics_path.exists():
        metrics_path = args.benchmark_dir / "metrics.parquet"
    if metrics_path.exists():
        from arena_evaluation.processing.parquet_store import ParquetStore
        metrics_df, _ = ParquetStore.read(metrics_path)
        if "map" in metrics_df.columns and len(metrics_df) > 0:
            map_name = metrics_df["map"][0]

    if not map_name:
        print("Error: could not determine map name.")
        sys.exit(1)

    result = renderer._load_grid_and_meta(map_name, run_dir=args.benchmark_dir)
    if result is None:
        print(f"Error: could not load map '{map_name}'.")
        sys.exit(1)

    grid, meta = result
    resolution = meta["resolution"]
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

    episode_df = AcousticFieldRenderer._load_episode_data(args.benchmark_dir, episode_id)
    if episode_df is None:
        print(f"Error: no topic data for {episode_id}.")
        sys.exit(1)

    from arena_evaluation.processing.acoustics.door_map import door_segments
    doors = door_segments(map_name, grid, resolution, (ox, oy, 0.0), run_dir=args.benchmark_dir)

    state_timeline = None
    semantic_path = args.benchmark_dir / "episodes" / episode_id / "topics" / "semantic_snapshot.parquet"
    if semantic_path.exists():
        semantic_df = pl.read_parquet(semantic_path)
        state_timeline = DoorStateTimeline.from_semantic_frame(semantic_df)

    # Determine which frame
    rows = episode_df.rows(named=True)
    if args.frame is not None and 0 <= args.frame < len(rows):
        frame_idx = args.frame
        row = rows[frame_idx]
    else:
        # Use the frame with highest source_dba
        best_idx = 0
        best_src = -1
        for i, r in enumerate(rows):
            s = r.get("source_dba", 0) or 0
            if s > best_src:
                best_src = s
                best_idx = i
        frame_idx = best_idx
        row = rows[best_idx]
        print(f"Using frame {frame_idx} (source={best_src:.1f} dBA). Use --frame N to pick a specific frame.")

    rx_m = float(row.get("pos_x_gt", 0) or 0)
    ry_m = float(row.get("pos_y_gt", 0) or 0)
    source_dba = float(row.get("source_dba", 42.0) or 42.0)
    time_ns = int(row.get("time_ns", 0))

    peds_pos = row.get("peds_positions", [])
    peds = AcousticFieldRenderer._parse_pedestrian_positions(peds_pos)

    open_set = frozenset()
    if state_timeline is not None:
        open_set = state_timeline.open_doors_at(time_ns)

    from arena_evaluation.processing.acoustics.door_map import build_pixel_tl
    pixel_tl = build_pixel_tl(grid, doors, open_doors=set(open_set)) if doors else None

    out_path = args.output or (args.benchmark_dir / "plots" / f"{episode_id}_acoustic_snapshot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering snapshot: frame {frame_idx}, t={time_ns/1e9:.1f}s, source={source_dba:.0f} dBA, {len(open_set)} doors open")

    ok = renderer._render_cell_png(
        grid, resolution, ox, oy,
        rx_m, ry_m, source_dba,
        peds,
        title=f"{episode_id}  frame {frame_idx}  t={time_ns/1e9:.1f}s  {source_dba:.0f} dBA",
        out_path=out_path,
        downsample=args.downsample if hasattr(args, 'downsample') else 2,
        vmin=42.0,
        vmax=None,
        pixel_tl=pixel_tl,
        doors=doors if pixel_tl is not None else None,
    )

    if ok:
        print(f"Snapshot saved to: {out_path}")
    else:
        print("Snapshot generation failed.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Arena Evaluation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single ad-hoc recording directory:
  evaluation process --run-dir /opt/arena_ws/data/recordings/20260528-215316

  # Process all runs in a benchmark:
  evaluation process --benchmark-dir /opt/arena_ws/data/my_benchmark

  # Full pipeline (process + report) for a benchmark:
  evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark

  # Generate report from multiple already-processed benchmarks:
  evaluation report --benchmark-dir /opt/arena_ws/data/bench1 /opt/arena_ws/data/bench2 --output-dir ./merged_report

  # With no input dir, operate on the most recent benchmark under $ARENA_DATA_DIR/benchmarks:
  evaluation run
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parent = argparse.ArgumentParser(add_help=False)
    run_group = run_parent.add_mutually_exclusive_group(required=False)
    run_group.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        nargs="+",
        metavar="DIR",
        help="Path to one or more benchmark root directories (contains multiple planner/stage runs)",
    )
    run_group.add_argument(
        "--run-dir",
        type=pathlib.Path,
        nargs="+",
        metavar="DIR",
        help="Path to one or more single recording directories (contains metadata.yaml + recording/)",
    )
    run_parent.add_argument(
        "--output-dir",
        type=pathlib.Path,
        metavar="DIR",
        help="Optional path to output directory for reports and plots. Defaults to the first input directory.",
    )
    run_parent.add_argument(
        "--generate-gifs",
        action="store_true",
        help="Generate animated GIFs for trajectories (computationally intensive).",
    )
    run_parent.add_argument(
        "--profile",
        action="store_true",
        help="Enable resource profiling. Writes pipeline_profile.yaml with per-phase CPU, GPU, RAM, and duration stats.",
    )
    run_parent.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of worker processes for parallel extraction and processing (-1 = auto-detect CPU count).",
    )
    run_parent.add_argument(
        "--report-manifest",
        type=str,
        default=None,
        metavar="NAME|PATH|{...}",
        help="Report manifest: a name from configs/benchmark/manifests/, a path to a "
        "YAML file, or inline {...} YAML. Used by run/report/plot; ignored otherwise.",
    )
    run_parent.add_argument(
        "--list-manifests",
        action="store_true",
        help="List the available named report manifests and exit.",
    )

    subparsers.add_parser(
        "extract",
        parents=[run_parent],
        help="Layer 3: Extract topics from MCAP into fast Parquet files (cache).",
    )
    subparsers.add_parser(
        "run",
        parents=[run_parent],
        help="Full pipeline: Extract MCAP (overwrite) → Process → Parquet → HTML report.",
    )
    process_parser = subparsers.add_parser(
        "process",
        parents=[run_parent],
        help="Layer 3: Compute metrics and write metrics.parquet (uses cached extraction by default).",
    )
    process_parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of MCAP files, overwriting the topic cache.",
    )
    subparsers.add_parser(
        "report",
        parents=[run_parent],
        help="Layer 5: Generate an interactive HTML report from existing metrics.parquet.",
    )
    subparsers.add_parser(
        "plot",
        parents=[run_parent],
        help="Layer 5: Generate static PNG plots only (no HTML report).",
    )

    # ── acoustic subcommand group ──────────────────────────────────────────
    acoustic_parser = subparsers.add_parser(
        "acoustic",
        help="Acoustic field visualization: list episodes, animate fields, render snapshots.",
    )
    acoustic_sub = acoustic_parser.add_subparsers(dest="acoustic_command", required=True)

    # acoustic list
    acoustic_list = acoustic_sub.add_parser("list", help="List episodes with acoustic metrics.")
    acoustic_list.add_argument("--benchmark-dir", type=pathlib.Path, required=True,
                               metavar="DIR", help="Path to benchmark directory.")

    # acoustic animate
    acoustic_anim = acoustic_sub.add_parser("animate", help="Generate animated GIF/MP4 of acoustic field.")
    acoustic_anim.add_argument("--benchmark-dir", type=pathlib.Path, required=True,
                               metavar="DIR", help="Path to benchmark directory.")
    acoustic_anim.add_argument("--episode", type=str, default="worst", metavar="EP",
                               help="Episode ID, 'worst', 'loudest-source', or 'max-total' (default: worst).")
    acoustic_anim.add_argument("--fps", type=int, default=10, help="Output frame rate (default: 10).")
    acoustic_anim.add_argument("--max-frames", type=int, default=120, help="Cap rendered frames (default: 120).")
    acoustic_anim.add_argument("--stride", type=int, default=1, help="Render every Nth data frame (default: 1).")
    acoustic_anim.add_argument("--downsample", type=int, default=2, help="Solver grid downsample (default: 2).")
    acoustic_anim.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "frames"],
                               help="Output format (default: gif).")
    acoustic_anim.add_argument("--dpi", type=int, default=150, help="Output resolution (default: 150).")
    acoustic_anim.add_argument("--vmin", type=float, default=42.0, help="Color-scale floor in dBA (default: 42).")
    acoustic_anim.add_argument("--vmax", type=float, default=None, help="Color-scale ceiling in dBA (default: auto).")
    acoustic_anim.add_argument("--no-door-overlay", action="store_true", help="Hide door contours.")
    acoustic_anim.add_argument("--robot-trail", type=int, default=0, help="Show past N robot positions as trail (default: 0).")
    acoustic_anim.add_argument("--output", type=pathlib.Path, default=None, metavar="PATH",
                               help="Override output path (default: plots/{episode}_acoustic.{format}).")

    # acoustic snapshot
    acoustic_snap = acoustic_sub.add_parser("snapshot", help="Render a single acoustic field frame.")
    acoustic_snap.add_argument("--benchmark-dir", type=pathlib.Path, required=True,
                               metavar="DIR", help="Path to benchmark directory.")
    acoustic_snap.add_argument("--episode", type=str, default="worst", metavar="EP",
                               help="Episode ID or keyword (default: worst).")
    acoustic_snap.add_argument("--frame", type=int, default=None, help="Frame index (default: worst-case frame).")
    acoustic_snap.add_argument("--output", type=pathlib.Path, default=None, metavar="PATH",
                               help="Override output path (default: plots/{episode}_acoustic_snapshot.png).")

    args = parser.parse_args()

    if getattr(args, "list_manifests", False):
        from arena_evaluation.presentation.manifest_registry import available_manifests

        print("Available report manifests:")
        for name in available_manifests():
            print(f"  - {name}")
        return 0

    args = resolve_paths(args)

    if args.benchmark_dir is None and args.run_dir is None:
        latest = latest_benchmark()
        if latest is None:
            print("Error: No input directory given and no benchmark runs found.")
            sys.exit(1)
        print(f"Using latest benchmark: {latest}")
        args.benchmark_dir = [latest]

    target_dirs = args.benchmark_dir or args.run_dir

    for d in target_dirs:
        if not d.exists() or not d.is_dir():
            print(f"Error: directory does not exist: {d}")
            sys.exit(1)

    # ── acoustic subcommand handler ────────────────────────────────────────
    if args.command == "acoustic":
        _handle_acoustic(args)
        return 0

    profiler = None
    if getattr(args, "profile", False):
        from arena_evaluation.benchmark.profiler import PipelineProfiler as _PP

        output_dir = getattr(args, "output_dir", None) or target_dirs[0]
        profiler = _PP(output_dir=output_dir, sample_hz=2.0)

    if args.command in ("extract", "run", "process"):
        force_extract = getattr(args, "force_extract", False)
        if args.command == "run":
            force_extract = True 

        if getattr(args, "run_dir", None):
            for run_dir in args.run_dir:
                fm = FolderManager(data_root=run_dir.parent)
                pipeline = ProcessingPipeline(fm, profiler=profiler, workers=getattr(args, "workers", None))
                
                if args.command == "extract":
                    print(f"Extracting single run: {run_dir}")
                    pipeline.extract_run_dir(run_dir)
                else:
                    print(f"Processing single run: {run_dir}")
                    out = pipeline.process_run_dir(run_dir, force_extract=force_extract)
                    if out:
                        print(f"Metrics written to: {out}")
                    else:
                        print(f"Processing failed for {run_dir} — see errors above.")

        else:
            for benchmark_dir in args.benchmark_dir:
                fm = FolderManager(data_root=benchmark_dir.parent)
                pipeline = ProcessingPipeline(fm, profiler=profiler, workers=getattr(args, "workers", None))
                
                if args.command == "extract":
                    print(f"Extracting benchmark: {benchmark_dir.name}")
                    pipeline.extract_benchmark(benchmark_dir.name)
                else:
                    print(f"Processing benchmark: {benchmark_dir.name}")
                    pipeline.process_benchmark(benchmark_dir.name, force_extract=force_extract)

    if args.command in ("run", "report", "plot"):
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            output_dir = target_dirs[0]

        from arena_evaluation.presentation.manifest_registry import resolve_manifest

        manifest_obj = resolve_manifest(getattr(args, "report_manifest", None), benchmark_dir=target_dirs[0])

        print(f"Building report/plots from {len(target_dirs)} sources into: {output_dir}")
        generate_gifs = getattr(args, "generate_gifs", False)
        _ctx = profiler.phase("report") if profiler else contextlib.nullcontext()
        with _ctx:
            builder = ReportBuilder.from_dirs(
                target_dirs,
                output_dir=output_dir,
                manifest=manifest_obj,
                generate_gifs=generate_gifs,
            )
            builder.build()
        print("Report generation complete.")

    if profiler is not None:
        profiler.write_summary()

if __name__ == "__main__":
    main()
