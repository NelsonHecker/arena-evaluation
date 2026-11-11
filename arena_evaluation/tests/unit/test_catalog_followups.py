"""Catalog follow-ups: other-robot blame, ranking-valid scorecard and episode provenance."""

from __future__ import annotations

import pathlib

import polars as pl
from polars.testing import assert_frame_equal

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.declarations import (
    CONTACT_FACT_SCHEMA,
    EPISODE_FACT_SCHEMA,
)
from arena_evaluation.storage.star_schema import (
    DIM_EPISODES_SCHEMA,
    DIM_PEDESTRIANS_SCHEMA,
    DIM_ROBOTS_SCHEMA,
    DIM_WORLD_ENTITIES_SCHEMA,
)

RUN_ID = "bench-catalog-followups"
ROBOT_ID = "env_0_jackal_0"


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


def _episode_row(
    episode_id: int,
    *,
    valid: bool,
    success: bool,
    no_ped_contact: bool,
    time_to_goal_s: float,
) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "episode_id": episode_id,
        "robot_id": ROBOT_ID,
        "success": success,
        "success_no_ped_contact": no_ped_contact,
        "success_no_robot_fault": success,
        "is_valid_for_ranking": valid,
        "invalid_reason": None if valid else "pose_teleport",
        "time_to_goal_s": time_to_goal_s,
        "path_efficiency": 0.9,
        "total_energy_wh": 2.0,
        "pose_frame": "map",
        "null_reason": None,
    }


def _write_benchmark(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True)
    dim_episodes = [
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
            "crowd_condition": "visible",
            "ped_policy_default": "sfm",
            "crowd_size": 2,
            "seed": 7,
            "is_reference": False,
            "reference_type": None,
            "duration_s": 10.0,
            "outcome_state": 2,
            "outcome_info": "finished",
        }
        for episode_id in (0, 1, 2)
    ]
    _frame(dim_episodes, DIM_EPISODES_SCHEMA).write_parquet(root / "dim_episodes.parquet")
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
            }
            for episode_id in (0, 1, 2)
        ],
        DIM_ROBOTS_SCHEMA,
    ).write_parquet(root / "dim_robots.parquet")
    _frame([], DIM_PEDESTRIANS_SCHEMA).write_parquet(root / "dim_pedestrians.parquet")
    _frame([], DIM_WORLD_ENTITIES_SCHEMA).write_parquet(root / "dim_world_entities.parquet")

    metrics = root / "episodes" / "episode_000" / "metrics"
    metrics.mkdir(parents=True)
    _frame(
        [
            _episode_row(0, valid=True, success=True, no_ped_contact=True, time_to_goal_s=10.0),
            _episode_row(1, valid=True, success=False, no_ped_contact=False, time_to_goal_s=20.0),
            _episode_row(2, valid=False, success=True, no_ped_contact=True, time_to_goal_s=30.0),
        ],
        EPISODE_FACT_SCHEMA,
    ).write_parquet(metrics / "fact_episodes.parquet")
    _frame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": 0,
                "robot_id": ROBOT_ID,
                "contact_id": "c0",
                "closer_label": "robot",
                "fault_attributed_to": "robot",
            },
            {
                "run_id": RUN_ID,
                "episode_id": 0,
                "robot_id": ROBOT_ID,
                "contact_id": "c1",
                "closer_label": "other_robot",
                "fault_attributed_to": "other_robot",
            },
        ],
        CONTACT_FACT_SCHEMA,
    ).write_parquet(metrics / "fact_contacts.parquet")
    return root


def test_episode_summary_exposes_validity_and_pose_provenance(
    tmp_path: pathlib.Path,
) -> None:
    catalog = BenchmarkCatalog.from_directory(_write_benchmark(tmp_path / RUN_ID))

    summary = catalog.get_table("v_episode_summary").select("episode_id", "is_valid_for_ranking", "invalid_reason", "pose_frame").sort("episode_id").collect()

    expected = pl.DataFrame(
        {
            "episode_id": [0, 1, 2],
            "is_valid_for_ranking": [True, True, False],
            "invalid_reason": [None, None, "pose_teleport"],
            "pose_frame": ["map", "map", "map"],
        }
    )
    assert_frame_equal(summary, expected, check_dtypes=True)


def test_planner_scorecard_ranks_only_valid_episodes_and_counts_the_rest(
    tmp_path: pathlib.Path,
) -> None:
    catalog = BenchmarkCatalog.from_directory(_write_benchmark(tmp_path / RUN_ID))

    scorecard = catalog.get_table("v_planner_scorecard").collect()

    assert scorecard.height == 1
    assert scorecard.item(0, "episodes_count") == 3
    assert scorecard.item(0, "valid_episodes") == 2
    assert scorecard.item(0, "excluded_episodes") == 1
    assert scorecard.item(0, "success_rate") == 0.5
    assert scorecard.item(0, "success_no_ped_contact_rate") == 0.5
    assert scorecard.item(0, "success_no_robot_fault_rate") == 0.5
    assert scorecard.item(0, "mean_time_to_goal_s") == 15.0
    assert scorecard.item(0, "other_robot_contacts") == 1


def test_contact_blame_summary_publishes_the_other_robot_row(
    tmp_path: pathlib.Path,
) -> None:
    catalog = BenchmarkCatalog.from_directory(_write_benchmark(tmp_path / RUN_ID))

    blame = catalog.get_table("v_contact_blame_summary").collect()
    other_robot = blame.filter(pl.col("closer_label") == "other_robot")

    assert other_robot.height == 1
    assert other_robot.item(0, "fault_attributed_to") == "other_robot"
    assert other_robot.item(0, "total_contacts") == 1
    assert other_robot.item(0, "closer_label_share") == 0.5
