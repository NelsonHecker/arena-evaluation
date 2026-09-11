#!/usr/bin/env python3
"""Merge distributed Arena Evaluation benchmark runs into a single unified directory.

Supports:
1. Full physical merge (episodes, MCAPs, progress.csv, state, manifest)
2. Parquet-only merge (concatenating combined_metrics.parquet from pre-processed runs)

Usage:
  python3 merge_benchmarks.py /path/to/bench1 /path/to/bench2 --target /path/to/merged_bench
  python3 merge_benchmarks.py /path/to/bench1 /path/to/bench2 --parquet-only --target /path/to/merged_bench
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import shutil
import sys
from datetime import datetime
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("merge_benchmarks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge distributed Arena Evaluation benchmark runs into a single unified directory."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=pathlib.Path,
        help="Source benchmark directories to merge.",
    )
    parser.add_argument(
        "--target",
        type=pathlib.Path,
        required=True,
        help="Target directory for the merged benchmark.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them (saves disk space).",
    )
    parser.add_argument(
        "--parquet-only",
        action="store_true",
        help="Only merge combined_metrics.parquet files from pre-processed runs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run_id for merged benchmark. Defaults to auto-generated timestamped name.",
    )
    return parser.parse_args()


def merge_parquets(sources: list[pathlib.Path], target_dir: pathlib.Path, run_id: str) -> bool:
    """Concatenate combined_metrics.parquet files across sources."""
    try:
        import polars as pl
    except ImportError:
        try:
            import pandas as pd
            dfs = []
            for src in sources:
                p = src / "combined_metrics.parquet"
                if not p.exists():
                    logger.warning("No combined_metrics.parquet in %s, skipping", src)
                    continue
                df = pd.read_parquet(p)
                if "benchmark_id" in df.columns:
                    df["benchmark_id"] = run_id
                dfs.append(df)
            if not dfs:
                logger.error("No parquet files found to merge")
                return False
            merged = pd.concat(dfs, ignore_index=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(target_dir / "combined_metrics.parquet", index=False)
            logger.info("Merged %d rows into %s", len(merged), target_dir / "combined_metrics.parquet")
            return True
        except ImportError:
            logger.error("Neither polars nor pandas is installed. Cannot merge parquets.")
            return False

    dfs = []
    for src in sources:
        p = src / "combined_metrics.parquet"
        if not p.exists():
            logger.warning("No combined_metrics.parquet in %s, skipping", src)
            continue
        df = pl.read_parquet(p)
        if "benchmark_id" in df.columns:
            df = df.with_columns(pl.lit(run_id).alias("benchmark_id"))
        dfs.append(df)

    if not dfs:
        logger.error("No parquet files found to merge")
        return False

    merged = pl.concat(dfs, how="diagonal_relaxed")
    target_dir.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(target_dir / "combined_metrics.parquet")
    logger.info("Merged %d rows into %s", len(merged), target_dir / "combined_metrics.parquet")
    return True


def merge_full(sources: list[pathlib.Path], target_dir: pathlib.Path, run_id: str, move: bool = False) -> None:
    """Perform a physical merge of episodes, progress.csv, state, and manifest."""
    transfer = shutil.move if move else shutil.copy2
    transfer_tree = shutil.move if move else shutil.copytree

    target_dir.mkdir(parents=True, exist_ok=True)
    target_episodes_dir = target_dir / "episodes"
    target_episodes_dir.mkdir(parents=True, exist_ok=True)

    global_ep_idx = 0
    merged_progress_rows: list[dict[str, str]] = []
    progress_fieldnames: list[str] = []
    merged_state: dict[str, Any] = {}
    merged_manifest: dict[str, Any] = {}
    contestants_seen: set[str] = set()
    merged_contestants: list[dict[str, Any]] = []

    for src_idx, src in enumerate(sources):
        logger.info("Processing source [%d/%d]: %s", src_idx + 1, len(sources), src)
        if not src.exists():
            logger.warning("Source directory does not exist: %s", src)
            continue

        # 1. Merge Manifest
        manifest_path = src / "manifest.yaml"
        if manifest_path.exists():
            try:
                data = yaml.safe_load(manifest_path.read_text())
                if not merged_manifest:
                    merged_manifest = dict(data)
                    merged_manifest["run_id"] = run_id
                    merged_manifest["created_at"] = datetime.now().astimezone().isoformat()
                
                # Extract and deduplicate contestants
                contest = data.get("contest") or {}
                for contestant in contest.get("contestants") or []:
                    c_name = contestant.get("name")
                    if c_name and c_name not in contestants_seen:
                        contestants_seen.add(c_name)
                        merged_contestants.append(contestant)
            except Exception as e:
                logger.warning("Could not parse %s: %s", manifest_path, e)

        # 2. Merge .benchmark_state.json
        state_path = src / ".benchmark_state.json"
        if state_path.exists():
            try:
                state_data = json.loads(state_path.read_text())
                merged_state.update(state_data)
            except Exception as e:
                logger.warning("Could not parse %s: %s", state_path, e)

        # 3. Read progress.csv rows
        progress_path = src / "progress.csv"
        if progress_path.exists():
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        for fn in reader.fieldnames:
                            if fn not in progress_fieldnames:
                                progress_fieldnames.append(fn)
                        for row in reader:
                            row["run_id"] = run_id
                            merged_progress_rows.append(row)
            except Exception as e:
                logger.warning("Could not read %s: %s", progress_path, e)

        # 4. Process Episodes
        episodes_dir = src / "episodes"
        if not episodes_dir.exists():
            logger.warning("No episodes directory in %s", src)
            continue

        src_ep_dirs = sorted([d for d in episodes_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        for ep_dir in src_ep_dirs:
            target_ep_name = f"episode_{global_ep_idx:03d}"
            target_ep_dir = target_episodes_dir / target_ep_name
            target_ep_dir.mkdir(parents=True, exist_ok=True)

            # Locate source yaml & mcap
            src_yaml = ep_dir / f"{ep_dir.name}.yaml"
            if not src_yaml.exists():
                src_yaml = ep_dir / "metadata.yaml"
            
            src_mcap = ep_dir / f"{ep_dir.name}.mcap"
            if not src_mcap.exists():
                candidates = sorted(p for p in ep_dir.glob("*.mcap") if p.stat().st_size > 0)
                if candidates:
                    src_mcap = candidates[0]

            # Copy or move MCAP
            if src_mcap.exists():
                dest_mcap = target_ep_dir / f"{target_ep_name}.mcap"
                transfer(src_mcap, dest_mcap)

            # Copy or move other artifacts (.mcap.idx, etc.)
            for item in ep_dir.iterdir():
                if item.is_file() and item != src_yaml and item != src_mcap:
                    transfer(item, target_ep_dir / item.name)
                elif item.is_dir():
                    transfer_tree(item, target_ep_dir / item.name)

            # Rewrite episode YAML with unified benchmark_id and new episode_id
            if src_yaml.exists():
                try:
                    meta = yaml.safe_load(src_yaml.read_text()) or {}
                    meta["benchmark_id"] = run_id
                    meta["episode_id"] = global_ep_idx
                    dest_yaml = target_ep_dir / f"{target_ep_name}.yaml"
                    with open(dest_yaml, "w", encoding="utf-8") as f:
                        yaml.safe_dump(meta, f, default_flow_style=False, sort_keys=False)
                except Exception as e:
                    logger.error("Failed to update yaml for %s: %s", ep_dir.name, e)

            global_ep_idx += 1

    # Finalize merged manifest
    if merged_manifest:
        if "contest" in merged_manifest and isinstance(merged_manifest["contest"], dict):
            merged_manifest["contest"]["contestants"] = merged_contestants
        with open(target_dir / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(merged_manifest, f, default_flow_style=False, sort_keys=False)
        logger.info("Wrote unified manifest.yaml with %d contestants", len(merged_contestants))

    # Finalize merged .benchmark_state.json
    if merged_state:
        with open(target_dir / ".benchmark_state.json", "w", encoding="utf-8") as f:
            json.dump(merged_state, f, indent=2)
        logger.info("Wrote unified .benchmark_state.json with %d steps", len(merged_state))

    # Finalize merged progress.csv
    if progress_fieldnames and merged_progress_rows:
        # Renumber episode_ids in progress.csv sequentially to match episodes
        for idx, row in enumerate(merged_progress_rows):
            row["episode_id"] = str(idx + 1)
        with open(target_dir / "progress.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=progress_fieldnames)
            writer.writeheader()
            writer.writerows(merged_progress_rows)
        logger.info("Wrote unified progress.csv with %d rows", len(merged_progress_rows))

    # Also merge combined_metrics.parquet if available in sources
    merge_parquets(sources, target_dir, run_id)

    logger.info("Successfully merged %d total episodes into: %s", global_ep_idx, target_dir)
    logger.info("Next step: Run `arena evaluation process --benchmark-dir %s`", target_dir)
    logger.info("           Run `arena evaluation report --benchmark-dir %s`", target_dir)


def main() -> None:
    args = parse_args()
    sources = [p.resolve() for p in args.sources]
    target = args.target.resolve()

    suite_name = "fig6_tier1_planner_diversity"
    contest_name = "fig6_tier1_planners"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or f"{ts}-{suite_name}-{contest_name}"

    if args.parquet_only:
        logger.info("Running parquet-only merge into %s...", target)
        merge_parquets(sources, target, run_id)
    else:
        logger.info("Running full physical benchmark merge into %s...", target)
        merge_full(sources, target, run_id, move=args.move)


if __name__ == "__main__":
    main()
