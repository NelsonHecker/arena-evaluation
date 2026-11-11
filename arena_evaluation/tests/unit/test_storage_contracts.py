"""Publish-time contract checks (08 H1.6): planted failures name the table and column.

Every fixture writes real Parquet files in the canonical layout: facts once per episode,
dimensions once at the benchmark root.
"""

from __future__ import annotations

import dataclasses
import pathlib

import polars as pl
import pytest
import yaml

from arena_evaluation.storage.contracts import (
    ContractReport,
    ContractViolation,
    check_benchmark,
    enforce_benchmark_contract,
    write_contract_report,
)
from arena_evaluation.storage.declarations import (
    ENCOUNTER_FACT_SCHEMA,
    EPISODE_FACT_SCHEMA,
    PHASE_OUTCOME_FACT_SCHEMA,
    ROBOT_FACT_SCHEMA,
    NullReason,
    is_known_null_reason,
)
from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.star_schema import (
    DIM_EPISODES_SCHEMA,
    DIM_PEDESTRIANS_SCHEMA,
    DIM_ROBOTS_SCHEMA,
    DIM_WORLD_ENTITIES_SCHEMA,
    StarSchemaTables,
)

RUN_ID = "bench-contract"
ROBOT_ID = "env_0_jackal_0"


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame([{name: row.get(name) for name in schema.names()} for row in rows], schema=schema)


def _episode_row(episode_id: int = 0, **overrides: object) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "episode_id": episode_id,
        "robot_id": ROBOT_ID,
        "success": True,
        "time_to_goal_s": 5.0,
        "null_reason": "fixture:no_metric",
        **overrides,
    }


def _encounter_row(**overrides: object) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "episode_id": 0,
        "robot_id": ROBOT_ID,
        "encounter_id": "enc-000-001",
        "actor_1": ROBOT_ID,
        "actor_2": "Pedestrian_1",
        "t_start_ns": 0,
        "t_end_ns": 1_000_000_000,
        "null_reason": "fixture:no_metric",
        **overrides,
    }


def _dimension_frames() -> dict[str, pl.DataFrame]:
    return {
        "dim_episodes": _frame([{"run_id": RUN_ID, "episode_id": 0, "env_id": "env_0", "robot_id": ROBOT_ID, "benchmark_id": RUN_ID, "planner": "dwb", "stage": "s0", "world": "map_empty"}], DIM_EPISODES_SCHEMA),
        "dim_robots": _frame([{"run_id": RUN_ID, "episode_id": 0, "env_id": "env_0", "robot_id": ROBOT_ID, "planner": "dwb"}], DIM_ROBOTS_SCHEMA),
        "dim_pedestrians": _frame([], DIM_PEDESTRIANS_SCHEMA),
        "dim_world_entities": _frame([], DIM_WORLD_ENTITIES_SCHEMA),
    }


def _dimensions() -> StarSchemaTables:
    return dataclasses.replace(StarSchemaTables.empty(), **_dimension_frames())


def _write_benchmark(
    root: pathlib.Path,
    *,
    fact_episodes: pl.DataFrame,
    fact_encounters: pl.DataFrame | None = None,
    fact_robots: pl.DataFrame | None = None,
    fact_phase_outcomes: pl.DataFrame | None = None,
    write_dimensions: bool = True,
) -> pathlib.Path:
    """Write the canonical dimension root and the per-episode fact files."""
    root.mkdir(parents=True, exist_ok=True)
    if write_dimensions:
        for name, frame in _dimension_frames().items():
            frame.write_parquet(root / f"{name}.parquet")

    metrics = root / "episodes" / "episode_000" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    fact_episodes.write_parquet(metrics / "fact_episodes.parquet")
    for name, frame in (("fact_encounters", fact_encounters), ("fact_robots", fact_robots), ("fact_phase_outcomes", fact_phase_outcomes)):
        if frame is not None:
            frame.write_parquet(metrics / f"{name}.parquet")
    return root


def _clean_benchmark(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], EPISODE_FACT_SCHEMA),
        fact_encounters=_frame([_encounter_row()], ENCOUNTER_FACT_SCHEMA),
    )


def _messages(report: ContractReport) -> list[str]:
    return [violation.message() for violation in report.violations]


def test_a_clean_benchmark_passes(tmp_path: pathlib.Path) -> None:
    report = check_benchmark(_clean_benchmark(tmp_path))
    assert report.ok
    assert report.violations == ()
    assert "fact_episodes" in report.checked_tables


def test_a_duplicate_key_fails_and_names_the_table_and_column(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(), _episode_row()], EPISODE_FACT_SCHEMA),
    )
    report = check_benchmark(root)
    assert not report.ok
    duplicates = [violation for violation in report.violations if violation.check == "key_duplicate"]
    assert duplicates and duplicates[0].table == "fact_episodes"
    assert "episode_id" in duplicates[0].column
    assert "fact_episodes" in duplicates[0].message() and "episode_id" in duplicates[0].message()


def test_a_nested_column_fails_the_flat_schema_contract(tmp_path: pathlib.Path) -> None:
    nested = _frame([_episode_row()], EPISODE_FACT_SCHEMA).with_columns(pl.Series("tags", [["tag"]], dtype=pl.List(pl.String)))
    root = _write_benchmark(tmp_path / "benchmarks" / RUN_ID, fact_episodes=nested)
    report = check_benchmark(root)
    nested_violations = [violation for violation in report.violations if violation.check == "nested_dtype"]
    assert nested_violations
    assert nested_violations[0].table == "fact_episodes"
    assert nested_violations[0].column == "tags"
    assert "tags" in nested_violations[0].message()


def test_an_orphan_fact_row_fails(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], EPISODE_FACT_SCHEMA),
        fact_robots=_frame([{"run_id": RUN_ID, "episode_id": 0, "env_id": "env_0", "robot_id": "ghost", "null_reason": "fixture:no_metric"}], ROBOT_FACT_SCHEMA),
    )
    report = check_benchmark(root)
    orphans = [violation for violation in report.violations if violation.check == "orphan_fact"]
    assert orphans and orphans[0].table == "fact_robots"
    assert "robot_id" in orphans[0].column


def test_an_unexplained_null_fails(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(time_to_goal_s=None, null_reason=None)], EPISODE_FACT_SCHEMA),
    )
    report = check_benchmark(root)
    unexplained = [violation for violation in report.violations if violation.check == "null_reason_missing"]
    assert unexplained and unexplained[0].table == "fact_episodes"
    assert "time_to_goal_s" in unexplained[0].column


def test_a_null_metric_with_a_vocabulary_reason_passes(tmp_path: pathlib.Path) -> None:
    reason = f"time_to_goal_s:{NullReason.NO_SAMPLES.value}"
    assert is_known_null_reason(reason)
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(time_to_goal_s=None, null_reason=reason)], EPISODE_FACT_SCHEMA),
    )
    report = check_benchmark(root)
    assert not [violation for violation in report.violations if violation.check == "null_reason_missing"]


def test_a_window_that_does_not_move_forward_fails(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], EPISODE_FACT_SCHEMA),
        fact_encounters=_frame([_encounter_row(t_start_ns=1_000_000_000, t_end_ns=1_000_000_000)], ENCOUNTER_FACT_SCHEMA),
    )
    report = check_benchmark(root)
    assert any(violation.check == "window_order" and violation.table == "fact_encounters" for violation in report.violations)


def test_an_encounter_actor_paired_with_itself_fails(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], EPISODE_FACT_SCHEMA),
        fact_encounters=_frame([_encounter_row(actor_2=ROBOT_ID)], ENCOUNTER_FACT_SCHEMA),
    )
    report = check_benchmark(root)
    assert any(violation.check == "actor_self_pair" and violation.table == "fact_encounters" for violation in report.violations)


def test_a_child_count_that_disagrees_with_the_parent_counter_fails(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(phases_total=2)], EPISODE_FACT_SCHEMA),
        fact_phase_outcomes=_frame([{"run_id": RUN_ID, "episode_id": 0, "robot_id": ROBOT_ID, "phase_idx": 0, "met": True, "null_reason": "fixture:no_metric"}], PHASE_OUTCOME_FACT_SCHEMA),
    )
    report = check_benchmark(root)
    assert any(violation.check == "child_count" and violation.table == "fact_episodes" for violation in report.violations)


def test_enforce_writes_the_report_and_raises(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(), _episode_row()], EPISODE_FACT_SCHEMA),
    )
    with pytest.raises(ContractViolationError) as error:
        enforce_benchmark_contract(root)
    assert "fact_episodes" in str(error.value)
    assert "episode_id" in str(error.value)
    report_path = root / "contract_report.yaml"
    assert report_path.is_file()
    payload = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["violations"]
    assert any("fact_episodes" in violation["table"] for violation in payload["violations"])


def test_write_contract_report_round_trips_the_report(tmp_path: pathlib.Path) -> None:
    report = ContractReport(benchmark_dir=tmp_path, violations=(ContractViolation(table="fact_episodes", column="episode_id", check="key_duplicate", detail="2 rows"),), checked_tables=("fact_episodes",))
    path = write_contract_report(report, tmp_path / "report.yaml")
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == report.as_dict()

def test_in_memory_dimensions_are_checked_before_a_file_exists(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row()], EPISODE_FACT_SCHEMA),
        fact_encounters=_frame([_encounter_row()], ENCOUNTER_FACT_SCHEMA),
        write_dimensions=False,
    )
    assert not (root / "dim_episodes.parquet").exists()
    report = check_benchmark(root, dimensions=_dimensions())
    assert report.ok, report.summary()


def test_a_failed_check_publishes_no_benchmark_dimension(tmp_path: pathlib.Path) -> None:
    root = _write_benchmark(
        tmp_path / "benchmarks" / RUN_ID,
        fact_episodes=_frame([_episode_row(), _episode_row()], EPISODE_FACT_SCHEMA),
        write_dimensions=False,
    )
    with pytest.raises(ContractViolationError):
        enforce_benchmark_contract(root, dimensions=_dimensions())
    assert (root / "contract_report.yaml").is_file()
    assert not (root / "dim_episodes.parquet").exists()
