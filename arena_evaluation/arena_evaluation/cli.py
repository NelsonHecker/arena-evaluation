import argparse
import os
import pathlib
import sys

from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.presentation.report_builder import ReportBuilder
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
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parent = argparse.ArgumentParser(add_help=False)
    run_group = run_parent.add_mutually_exclusive_group(required=True)
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

    args = parser.parse_args()
    args = resolve_paths(args)

    target_dirs = getattr(args, "benchmark_dir", None) or getattr(args, "run_dir", None)
    if not target_dirs:
        print("Error: No input directories provided.")
        sys.exit(1)

    for d in target_dirs:
        if not d.exists() or not d.is_dir():
            print(f"Error: directory does not exist: {d}")
            sys.exit(1)

    if args.command in ("extract", "run", "process"):
        force_extract = getattr(args, "force_extract", False)
        if args.command == "run":
            force_extract = True 

        if getattr(args, "run_dir", None):
            for run_dir in args.run_dir:
                fm = FolderManager(data_root=run_dir.parent)
                pipeline = ProcessingPipeline(fm)
                
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
                pipeline = ProcessingPipeline(fm)
                
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
            
        print(f"Building report/plots from {len(target_dirs)} sources into: {output_dir}")
        generate_gifs = getattr(args, "generate_gifs", False)
        builder = ReportBuilder.from_dirs(target_dirs, output_dir=output_dir, generate_gifs=generate_gifs)
        builder.build()
        print("Report generation complete.")

if __name__ == "__main__":
    main()
