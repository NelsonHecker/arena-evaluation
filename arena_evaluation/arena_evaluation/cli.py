import argparse
import pathlib
import sys

from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.storage.folder_manager import FolderManager

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

  # Generate report from already-processed benchmark:
  evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared parent parsers ──────────────────────────────────────────────────
    # For commands that accept either a benchmark dir or a single run dir
    run_parent = argparse.ArgumentParser(add_help=False)
    run_group = run_parent.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        metavar="DIR",
        help="Path to the benchmark root directory (contains multiple planner/stage runs)",
    )
    run_group.add_argument(
        "--run-dir",
        type=pathlib.Path,
        metavar="DIR",
        help="Path to a single recording directory (contains metadata.yaml + recording/)",
    )

    # For commands that only work at benchmark level (report, plot)
    benchmark_parent = argparse.ArgumentParser(add_help=False)
    benchmark_parent.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        required=True,
        metavar="DIR",
        help="Path to the benchmark root directory",
    )

    # ── Subcommands ────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "run",
        parents=[run_parent],
        help="Full pipeline: Process MCAP → Parquet, then generate HTML report.",
    )
    subparsers.add_parser(
        "process",
        parents=[run_parent],
        help="Layer 3: Read MCAP(s), compute metrics, write metrics.parquet (no plots).",
    )
    subparsers.add_parser(
        "report",
        parents=[benchmark_parent],
        help="Layer 5: Generate an interactive HTML report from existing metrics.parquet.",
    )
    subparsers.add_parser(
        "plot",
        parents=[benchmark_parent],
        help="Layer 5: Generate static PNG plots only (no HTML report).",
    )

    args = parser.parse_args()

    # ── Validate path ──────────────────────────────────────────────────────────
    target_dir = getattr(args, "benchmark_dir", None) or getattr(args, "run_dir", None)
    if target_dir is not None and (not target_dir.exists() or not target_dir.is_dir()):
        print(f"Error: directory does not exist: {target_dir}")
        sys.exit(1)

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if args.command in ("run", "process"):
        if getattr(args, "run_dir", None):
            # Single recording directory — no benchmark structure required
            run_dir: pathlib.Path = args.run_dir
            print(f"Processing single run: {run_dir}")
            fm = FolderManager(data_root=run_dir.parent)
            pipeline = ProcessingPipeline(fm)
            out = pipeline.process_run_dir(run_dir)
            if out:
                print(f"Metrics written to: {out}")
            else:
                print("Processing failed — see errors above.")
                sys.exit(1)

        else:
            # Benchmark directory — discover and process all runs
            benchmark_dir: pathlib.Path = args.benchmark_dir
            print(f"Processing benchmark: {benchmark_dir.name}")
            fm = FolderManager(data_root=benchmark_dir.parent)
            pipeline = ProcessingPipeline(fm)
            pipeline.process_benchmark(benchmark_dir.name)

    if args.command in ("run", "report", "plot"):
        target_dir = getattr(args, "benchmark_dir", None) or getattr(args, "run_dir", None)
        if target_dir is not None:
            print(f"Building report/plots for: {target_dir.name}")
            builder = ReportBuilder(target_dir)
            builder.build()
            print("Report generated successfully.")


if __name__ == "__main__":
    main()
