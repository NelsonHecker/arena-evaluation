"""Canonical Star Schema fixtures for the consumer tests: write exactly what the readers scan."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.declarations import EPISODE_FACT_SCHEMA
from arena_evaluation.storage.star_schema import DIM_EPISODES_SCHEMA

__all__ = ["read_summary", "write_benchmark"]


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame([{name: row.get(name) for name in schema.names()} for row in rows], schema=schema)


def write_benchmark(directory: pathlib.Path, rows: list[dict[str, object]], *, robot_tracks: list[dict[str, object]] | None = None) -> pathlib.Path:
    """Write ``dim_episodes`` and ``metrics/fact_episodes`` for ``rows`` (dimension keys and fact measures in one mapping).

    ``robot_tracks`` rows carry ``episode_id`` plus pose samples; the catalog stamps the episode
    id itself, so it is stripped from each per-episode telemetry file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _frame(rows, DIM_EPISODES_SCHEMA).write_parquet(directory / "dim_episodes.parquet")
    metrics = directory / "metrics"
    metrics.mkdir(exist_ok=True)
    _frame(rows, EPISODE_FACT_SCHEMA).write_parquet(metrics / "fact_episodes.parquet")

    if robot_tracks is not None:
        for episode_id in sorted({int(row["episode_id"]) for row in rows}):
            samples = [sample for sample in robot_tracks if int(sample["episode_id"]) == episode_id]
            if not samples:
                continue
            path = directory / "episodes" / f"episode_{episode_id:03d}" / "telemetry" / "robot_tracks.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame([{name: value for name, value in sample.items() if name != "episode_id"} for sample in samples]).write_parquet(path)
    return directory


def read_summary(directory: pathlib.Path) -> pl.DataFrame:
    """The canonical ``v_episode_summary`` the readers consume."""
    return BenchmarkCatalog.from_directory(directory).get_table("v_episode_summary").collect()