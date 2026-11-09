"""Canonical Star Schema access shared by the study analyses."""

from __future__ import annotations

import pathlib
import typing

import polars as pl

if typing.TYPE_CHECKING:
    from arena_evaluation.analytics.catalog import BenchmarkCatalog

__all__ = ["episode_summary", "robot_tracks"]


def _catalog(benchmark_dir: pathlib.Path) -> BenchmarkCatalog:
    from arena_evaluation.analytics.catalog import BenchmarkCatalog

    if not (benchmark_dir / "dim_episodes.parquet").is_file():
        raise FileNotFoundError(f"no Star Schema tables under {benchmark_dir}; run `arena evaluation process --benchmark-dir {benchmark_dir}` first")
    return BenchmarkCatalog.from_directory(benchmark_dir)


def episode_summary(benchmark_dir: pathlib.Path) -> pl.DataFrame:
    """The run's canonical ``v_episode_summary`` view, one row per robot episode."""
    return _catalog(benchmark_dir).get_table("v_episode_summary").collect()


def robot_tracks(benchmark_dir: pathlib.Path) -> pl.DataFrame:
    """The run's canonical ``robot_tracks`` telemetry table, one row per pose sample."""
    return _catalog(benchmark_dir).get_table("robot_tracks").collect()
