"""Benchmark facts and series are lazy views over the one per-episode stored copy (06 D6/D7).

The fixture writes real Parquet files in the canonical layout and never uses the writer under
test, so the view assertions compare the scan against an independent per-episode concat.
"""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.declarations import EPISODE_FACT_SCHEMA
from arena_evaluation.storage.star_schema import (
    DIM_EPISODES_SCHEMA,
    DIM_PEDESTRIANS_SCHEMA,
    DIM_ROBOTS_SCHEMA,
    DIM_WORLD_ENTITIES_SCHEMA,
    scan_star_schema,
)

RUN_ID = "bench-views"
ROBOT_ID = "env_0_jackal_0"

SERIES_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "x_m": pl.Float64,
        "y_m": pl.Float64,
    }
)


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame([{name: row.get(name) for name in schema.names()} for row in rows], schema=schema)


def _write_episode(root: pathlib.Path, episode_id: int) -> tuple[pathlib.Path, pathlib.Path]:
    """Write one episode's stored facts and series, returning both file paths."""
    metrics = root / "episodes" / f"episode_{episode_id:03d}" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    fact_path = metrics / "fact_episodes.parquet"
    _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": episode_id,
                "robot_id": ROBOT_ID,
                "success": True,
                "time_to_goal_s": 5.0 + episode_id,
                "null_reason": "fixture",
            }
        ],
        EPISODE_FACT_SCHEMA,
    ).write_parquet(fact_path)
    series_path = metrics / "series_path_metrics.parquet"
    _frame(
        [
            {"run_id": RUN_ID, "episode_id": episode_id, "robot_id": ROBOT_ID, "t_ns": 0, "x_m": 0.0, "y_m": 0.0},
            {"run_id": RUN_ID, "episode_id": episode_id, "robot_id": ROBOT_ID, "t_ns": 1_000_000_000, "x_m": 1.0, "y_m": 0.0},
        ],
        SERIES_SCHEMA,
    ).write_parquet(series_path)
    return fact_path, series_path


@pytest.fixture
def two_episode_benchmark(tmp_path: pathlib.Path) -> tuple[pathlib.Path, tuple[pathlib.Path, pathlib.Path], tuple[pathlib.Path, pathlib.Path]]:
    root = tmp_path / "benchmarks" / RUN_ID
    root.mkdir(parents=True)
    _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": episode_id,
                "env_id": "env_0",
                "robot_id": ROBOT_ID,
                "benchmark_id": RUN_ID,
                "suite_name": "fixture",
                "contest_name": "fixture",
                "stage": "s0",
                "world": "map_empty",
                "scenario": "s0",
                "planner": "dwb",
                "inter_planner": "none",
                "dispatcher": "none",
                "robot_model": "jackal",
                "crowd_condition": None,
                "ped_policy_default": None,
                "crowd_size": None,
                "seed": 1,
                "is_reference": False,
                "duration_s": 5.0 + episode_id,
                "outcome_state": 2,
                "outcome_info": "finished",
            }
            for episode_id in (0, 1)
        ],
        DIM_EPISODES_SCHEMA,
    ).write_parquet(root / "dim_episodes.parquet")
    _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": episode_id,
                "env_id": "env_0",
                "robot_id": ROBOT_ID,
                "robot_model": "jackal",
                "planner": "dwb",
                "inter_planner": "none",
                "dispatcher": "none",
                "robot_radius": 0.267,
                "robot_mass_kg": 17.5,
            }
            for episode_id in (0, 1)
        ],
        DIM_ROBOTS_SCHEMA,
    ).write_parquet(root / "dim_robots.parquet")
    _frame([], DIM_PEDESTRIANS_SCHEMA).write_parquet(root / "dim_pedestrians.parquet")
    _frame([], DIM_WORLD_ENTITIES_SCHEMA).write_parquet(root / "dim_world_entities.parquet")

    written = [_write_episode(root, episode_id) for episode_id in (0, 1)]
    facts = tuple(fact_path for fact_path, _ in written)
    series = tuple(series_path for _, series_path in written)
    return root, facts, series


def test_fact_view_equals_the_per_episode_concat(
    two_episode_benchmark: tuple[pathlib.Path, tuple[pathlib.Path, pathlib.Path], tuple[pathlib.Path, pathlib.Path]],
) -> None:
    root, (first, second), _ = two_episode_benchmark
    view = scan_star_schema(root)["fact_episodes"]
    assert isinstance(view, pl.LazyFrame)
    expected = pl.concat([pl.read_parquet(first), pl.read_parquet(second)], how="vertical")
    assert_frame_equal(view.collect(), expected, check_dtypes=True)


def test_the_view_keeps_projection_and_predicate_pushdown(
    two_episode_benchmark: tuple[pathlib.Path, tuple[pathlib.Path, pathlib.Path], tuple[pathlib.Path, pathlib.Path]],
) -> None:
    root, _, _ = two_episode_benchmark
    view = scan_star_schema(root)["fact_episodes"]
    filtered = view.filter(pl.col("episode_id") == 1).select("success", "time_to_goal_s").collect()
    assert filtered.height == 1
    assert filtered.item(0, "success") is True
    assert filtered.item(0, "time_to_goal_s") == pytest.approx(6.0)


def test_no_fact_is_stored_twice_and_no_dimension_is_copied_into_an_episode(
    two_episode_benchmark: tuple[pathlib.Path, tuple[pathlib.Path, pathlib.Path], tuple[pathlib.Path, pathlib.Path]],
) -> None:
    root, _, _ = two_episode_benchmark
    assert not (root / "metrics" / "fact_episodes.parquet").exists()
    assert not list(root.glob("episodes/*/metrics/dim_*.parquet"))
    for dimension in ("dim_episodes", "dim_robots", "dim_pedestrians", "dim_world_entities"):
        assert (root / f"{dimension}.parquet").is_file()


def test_canonical_views_read_through_the_fact_view(
    two_episode_benchmark: tuple[pathlib.Path, tuple[pathlib.Path, pathlib.Path], tuple[pathlib.Path, pathlib.Path]],
) -> None:
    root, _, _ = two_episode_benchmark
    catalog = BenchmarkCatalog.from_directory(root)
    summary = catalog.get_table("v_episode_summary").collect()
    assert summary.height == 2
    assert set(summary.get_column("planner").to_list()) == {"dwb"}


def test_series_view_equals_the_per_episode_concat(
    two_episode_benchmark: tuple[pathlib.Path, tuple[pathlib.Path, pathlib.Path], tuple[pathlib.Path, pathlib.Path]],
) -> None:
    root, _, (first, second) = two_episode_benchmark
    catalog = BenchmarkCatalog.from_directory(root)
    expected = pl.concat([pl.read_parquet(first), pl.read_parquet(second)], how="vertical").sort("episode_id", "t_ns")
    assert_frame_equal(catalog.get_table("series_path_metrics").collect().sort("episode_id", "t_ns"), expected, check_dtypes=True)
    joined = catalog.get_table("v_series_path_metrics").collect()
    assert joined.height == expected.height
