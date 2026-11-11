"""Synthetic tests for the parity certificate: no live run and no copied baseline."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from certify import (
    Certificate,
    DiffKind,
    ExpectedDiff,
    MetricMapping,
    ParityInputError,
    ParityStatus,
    certify,
    load_baseline,
    load_expected_manifest,
    load_mappings,
    load_run,
)
from certify import main as certify_main

from arena_evaluation.storage.declarations import EPISODE_FACT_SCHEMA
from arena_evaluation.storage.star_schema import DIM_EPISODES_SCHEMA, DIM_ROBOTS_SCHEMA

RUN_ID = "bench-synthetic"
ROBOT_ID = "env_0_jackal_0"
BASELINE_ROBOT = "env_0_jackal"


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


def _write_baseline(root: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    baseline_dir = root / "baseline"
    baseline_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(baseline_dir / f"{RUN_ID}__episode_000.parquet")
    return baseline_dir


def _write_run(root: pathlib.Path, fact_row: dict[str, object]) -> pathlib.Path:
    run_dir = root / "benchmarks" / RUN_ID
    run_dir.mkdir(parents=True)
    dimension_rows = [
        {
            "run_id": RUN_ID,
            "episode_id": 0,
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
            "seed": None,
            "is_reference": False,
            "reference_type": None,
            "duration_s": 10.0,
            "outcome_state": 2,
            "outcome_info": "finished",
        }
    ]
    _frame(dimension_rows, DIM_EPISODES_SCHEMA).write_parquet(run_dir / "dim_episodes.parquet")
    _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": ROBOT_ID,
                "planner": "dwb",
            }
        ],
        DIM_ROBOTS_SCHEMA,
    ).write_parquet(run_dir / "dim_robots.parquet")
    metrics = run_dir / "episodes" / "episode_000" / "metrics"
    metrics.mkdir(parents=True)
    _frame([fact_row], EPISODE_FACT_SCHEMA).write_parquet(metrics / "fact_episodes.parquet")
    return run_dir


def _baseline_row(**overrides: object) -> dict[str, object]:
    return {
        "benchmark_id": RUN_ID,
        "episode": 0,
        "robot": BASELINE_ROBOT,
        "time_to_goal": 10.0,
        "path_efficiency": 0.9,
        "success": True,
        "result": "GOAL_REACHED",
        **overrides,
    }


def _fact_row(**overrides: object) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "episode_id": 0,
        "robot_id": ROBOT_ID,
        "time_to_goal_s": 10.0,
        "path_efficiency": 0.9,
        "success": True,
        "result": "GOAL_REACHED",
        "is_valid_for_ranking": True,
        "null_reason": None,
        **overrides,
    }


def _mapping(metric_key: str, star_column: str, legacy_column: str) -> MetricMapping:
    return MetricMapping(
        metric_key=metric_key,
        star_table="fact_episodes",
        star_column=star_column,
        legacy_columns=(legacy_column,),
    )


def _run_certificate(
    root: pathlib.Path,
    *,
    baseline_row: dict[str, object],
    fact_row: dict[str, object],
    mapping: MetricMapping,
    expected: tuple[ExpectedDiff, ...] = (),
) -> Certificate:
    baseline_dir = _write_baseline(root, [baseline_row])
    run_dir = _write_run(root, fact_row)
    catalog, summary = load_run(run_dir)
    baseline = load_baseline(baseline_dir, benchmark=RUN_ID)
    return certify(
        catalog=catalog,
        baseline=baseline,
        run_summary=summary,
        mappings=(mapping,),
        expected_diffs=expected,
    )


def test_exact_comparison_matches_the_suffixed_runtime_robot_name(
    tmp_path: pathlib.Path,
) -> None:
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(),
        fact_row=_fact_row(),
        mapping=_mapping("time_to_goal", "time_to_goal_s", "time_to_goal"),
    )

    row = certificate.rows[0]
    assert row.status is ParityStatus.EXACT
    assert row.episodes_compared == 1
    assert row.nonnull_compared == 1


def test_a_difference_inside_the_tolerance_is_within_tolerance(
    tmp_path: pathlib.Path,
) -> None:
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(),
        fact_row=_fact_row(time_to_goal_s=10.00001),
        mapping=_mapping("time_to_goal", "time_to_goal_s", "time_to_goal"),
    )

    assert certificate.rows[0].status is ParityStatus.WITHIN_TOLERANCE


def test_a_string_result_is_compared_exactly(tmp_path: pathlib.Path) -> None:
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(),
        fact_row=_fact_row(),
        mapping=_mapping("result", "result", "result"),
    )

    assert certificate.rows[0].status is ParityStatus.EXACT


def test_a_reviewed_string_change_is_expected_diff(tmp_path: pathlib.Path) -> None:
    expected = (ExpectedDiff(metric_key="result", kind=DiffKind.GROUND_TRUTH, reason="the fault policy maps the recorded verdict once"),)
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(result="GOAL_REACHED"),
        fact_row=_fact_row(result="UNRESOLVED"),
        mapping=_mapping("result", "result", "result"),
        expected=expected,
    )

    assert certificate.rows[0].status is ParityStatus.EXPECTED_DIFF


def test_a_covered_null_fill_is_expected_diff(tmp_path: pathlib.Path) -> None:
    expected = (
        ExpectedDiff(
            metric_key="path_efficiency",
            kind=DiffKind.NULL_FILL,
            reason="missing measurement is null, not zero",
        ),
    )
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(path_efficiency=0.0),
        fact_row=_fact_row(path_efficiency=None, null_reason="path_efficiency:not_recorded"),
        mapping=_mapping("path_efficiency", "path_efficiency", "path_efficiency"),
        expected=expected,
    )

    assert certificate.rows[0].status is ParityStatus.EXPECTED_DIFF


def test_an_uncovered_difference_is_missing_and_never_passes(
    tmp_path: pathlib.Path,
) -> None:
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(),
        fact_row=_fact_row(time_to_goal_s=20.0),
        mapping=_mapping("time_to_goal", "time_to_goal_s", "time_to_goal"),
    )

    row = certificate.rows[0]
    assert row.status is ParityStatus.MISSING
    assert "UNEXPLAINED" in row.detail


def test_all_null_on_both_sides_is_missing_evidence(tmp_path: pathlib.Path) -> None:
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(path_efficiency=None),
        fact_row=_fact_row(path_efficiency=None, null_reason="path_efficiency:not_recorded"),
        mapping=_mapping("path_efficiency", "path_efficiency", "path_efficiency"),
    )

    assert certificate.rows[0].status is ParityStatus.MISSING
    assert "both sides are null" in certificate.rows[0].detail


def test_an_absent_legacy_column_is_missing(tmp_path: pathlib.Path) -> None:
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(),
        fact_row=_fact_row(),
        mapping=_mapping("absent", "time_to_goal_s", "absent"),
    )

    assert certificate.rows[0].status is ParityStatus.MISSING
    assert "legacy column absent" in certificate.rows[0].detail


def test_an_absent_catalog_column_is_no_producer(tmp_path: pathlib.Path) -> None:
    certificate = _run_certificate(
        tmp_path,
        baseline_row=_baseline_row(),
        fact_row=_fact_row(),
        mapping=_mapping("time_to_goal", "absent_column", "time_to_goal"),
    )

    assert certificate.rows[0].status is ParityStatus.NO_PRODUCER


def test_a_baseline_without_a_matching_benchmark_is_an_input_error(
    tmp_path: pathlib.Path,
) -> None:
    baseline_dir = _write_baseline(tmp_path, [_baseline_row(benchmark_id="other")])

    with pytest.raises(ParityInputError):
        load_baseline(baseline_dir, benchmark=RUN_ID)


def test_the_shipped_manifests_parse_and_record_the_overhaul_agreements() -> None:
    here = pathlib.Path(__file__).resolve().parent
    mappings = load_mappings(here / "metric_map.yaml")
    manifest = load_expected_manifest(here / "expected_diffs.yaml")

    assert len(mappings) > 0
    topics = {str(note["topic"]) for note in manifest.notes}
    assert {"edge_detector", "fault_success_policy", "null_not_zero", "seed_crowd_unknown", "identity_names"} <= topics


def test_self_check_entry_point_passes() -> None:
    assert certify_main(["--self-check"]) == 0
