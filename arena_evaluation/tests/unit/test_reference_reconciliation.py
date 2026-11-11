"""Reference-twin reconciliation: lookup keys, typed missingness and the publish ordering."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.processing.reference_reconciliation import (
    apply_plan_to_disk,
    patch_fact_episodes,
    patch_fact_pedestrians,
    pedestrian_paths,
    plan_reconciliation,
    robot_path,
)
from arena_evaluation.processing.reference_twins import (
    UNHINDERED_PEDS,
    UNOBSTRUCTED_ROBOT,
    build_twin_index,
)
from arena_evaluation.storage.star_schema import (
    DIM_EPISODES_SCHEMA,
    EPISODE_FACT_SCHEMA,
    PEDESTRIAN_FACT_SCHEMA,
    scan_star_schema,
)
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata

STAGE = "stage_direct_book_01"
PLANNER = "dwb-standard"
RUN_ID = "parity-twin-test"
PED_TWIN_PLANNER = "unhindered_peds"

SERIES_PATH_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "x_m": pl.Float64,
        "y_m": pl.Float64,
    }
)
SERIES_PEDESTRIAN_PATH_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "ped_id": pl.String,
        "x_m": pl.Float64,
        "y_m": pl.Float64,
    }
)


def _one_row(schema: pl.Schema, **values: object) -> pl.DataFrame:
    """One typed row of a Star Schema table: every column present, unspecified ones null."""
    return pl.DataFrame(
        {name: [values.get(name)] for name in schema.names()},
        schema=schema,
    )


def _descriptor(episode_dir: pathlib.Path, episode_id: int, *, planner: str = PLANNER, reference_type: str | None = None) -> EpisodeDescriptor:
    return EpisodeDescriptor(
        episode_dir=str(episode_dir),
        benchmark_id=RUN_ID,
        episode_id=episode_id,
        planner=PED_TWIN_PLANNER if reference_type == UNHINDERED_PEDS else planner,
        stage=STAGE,
        is_reference=reference_type is not None,
        reference_type=reference_type,
    )


def _write_episode(
    root: pathlib.Path,
    episode_id: int,
    *,
    reference_type: str | None = None,
    planner: str = PLANNER,
    robot_path_points: list[tuple[float, float]] | None = None,
    pedestrian_paths_points: list[list[tuple[float, float]]] | None = None,
    energy_wh: float = 5.0,
    time_to_goal_s: float = 20.0,
    path_length_m: float = 12.0,
) -> EpisodeDescriptor:
    """Write one episode directory: metadata, dimension/fact rows and the path series tables."""
    episode_dir = root / "episodes" / f"episode_{episode_id:03d}"
    (episode_dir / "metrics").mkdir(parents=True, exist_ok=True)
    MetadataWriter.write(
        RunMetadata(
            benchmark_id=RUN_ID,
            planner=PED_TWIN_PLANNER if reference_type == UNHINDERED_PEDS else planner,
            robot_model=["jackal"],
            map="map_empty",
            stage=STAGE,
            episode_id=episode_id,
            is_reference=reference_type is not None,
            reference_type=reference_type,
            recording_started_at="2026-11-10T00:00:00",
            python_version="3.12",
            ros_distro="jazzy",
            modes={},
            params={},
        ),
        episode_dir / f"episode_{episode_id:03d}.yaml",
    )
    dim = _one_row(
        DIM_EPISODES_SCHEMA,
        run_id=RUN_ID,
        episode_id=episode_id,
        stage=STAGE,
        planner=PED_TWIN_PLANNER if reference_type == UNHINDERED_PEDS else planner,
        is_reference=reference_type is not None,
        reference_type=reference_type,
    )
    dim.write_parquet(episode_dir / "metrics" / "dim_episodes.parquet")
    fact = _one_row(
        EPISODE_FACT_SCHEMA,
        run_id=RUN_ID,
        episode_id=episode_id,
        robot_id="env_0_jackal",
        path_length_m=path_length_m,
        time_to_goal_s=time_to_goal_s,
        total_energy_wh=energy_wh,
        is_valid_for_ranking=True,
        invalid_reason=None,
        null_reason=("result:not_applicable;mar:no_samples;pfi:no_samples;robot_path_dev_m:no_samples;kcsc_wh:no_samples;relative_throughput:no_samples"),
    )
    fact.write_parquet(episode_dir / "metrics" / "fact_episodes.parquet")
    _one_row(
        PEDESTRIAN_FACT_SCHEMA,
        run_id=RUN_ID,
        episode_id=episode_id,
        env_id="env_0",
        ped_id="Pedestrian_1",
        null_reason="ped_name:not_applicable;ped_path_deflection_m:no_samples",
    ).write_parquet(episode_dir / "metrics" / "fact_pedestrians.parquet")
    if robot_path_points is not None:
        rows = [
            {
                "run_id": RUN_ID,
                "episode_id": episode_id,
                "robot_id": "env_0_jackal",
                "t_ns": index * 100_000_000,
                "x_m": point[0],
                "y_m": point[1],
            }
            for index, point in enumerate(robot_path_points)
        ]
        pl.DataFrame(rows, schema=SERIES_PATH_SCHEMA).write_parquet(episode_dir / "metrics" / "series_path_metrics.parquet")
    if pedestrian_paths_points is not None:
        rows = []
        for ped_id, path in enumerate(pedestrian_paths_points):
            for index, point in enumerate(path):
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "episode_id": episode_id,
                        "robot_id": "env_0_jackal",
                        "t_ns": index * 100_000_000,
                        "ped_id": f"Pedestrian_{ped_id + 1}",
                        "x_m": point[0],
                        "y_m": point[1],
                    }
                )
        pl.DataFrame(rows, schema=SERIES_PEDESTRIAN_PATH_SCHEMA).write_parquet(episode_dir / "metrics" / "series_pedestrian_path_metrics.parquet")
    return _descriptor(episode_dir, episode_id, planner=planner, reference_type=reference_type)


def _twin_benchmark(tmp_path: pathlib.Path) -> tuple[pathlib.Path, list[EpisodeDescriptor]]:
    """A one-stage benchmark: contestant, obstacle-free robot twin and robot-free pedestrian twin."""
    root = tmp_path / "benchmarks" / RUN_ID
    contestant = _write_episode(
        root,
        1,
        robot_path_points=[(0.0, 0.0), (5.0, 1.0), (10.0, 0.0)],
        pedestrian_paths_points=[[(0.0, 1.0), (1.0, 1.0)]],
        energy_wh=5.0,
        time_to_goal_s=20.0,
        path_length_m=12.0,
    )
    robot_twin = _write_episode(
        root,
        2,
        reference_type=UNOBSTRUCTED_ROBOT,
        robot_path_points=[(0.0, 0.0), (10.0, 0.0)],
        energy_wh=3.0,
        time_to_goal_s=10.0,
        path_length_m=10.0,
    )
    peds_twin = _write_episode(
        root,
        3,
        reference_type=UNHINDERED_PEDS,
        pedestrian_paths_points=[[(0.0, 0.0), (1.0, 0.0)]],
        energy_wh=0.0,
        time_to_goal_s=60.0,
        path_length_m=0.0,
    )
    return root, [contestant, robot_twin, peds_twin]


def _facts(descriptors: list[EpisodeDescriptor]) -> dict[int, pl.DataFrame]:
    return {descriptor.episode_id: pl.read_parquet(pathlib.Path(descriptor.episode_dir) / "metrics" / "fact_episodes.parquet") for descriptor in descriptors}


def test_twin_index_uses_the_runner_keys(tmp_path: pathlib.Path):
    _, descriptors = _twin_benchmark(tmp_path)
    index = build_twin_index(descriptors)
    contestant, robot_twin, peds_twin = descriptors

    assert index.robot_twin(contestant) is robot_twin
    assert index.pedestrian_twin(contestant) is peds_twin
    assert index.robot_twin(robot_twin) is None
    assert index.pedestrian_twin(peds_twin) is None
    assert index.has_twin(contestant, UNOBSTRUCTED_ROBOT)
    assert index.has_twin(contestant, UNHINDERED_PEDS)


def test_plan_reconciles_the_five_twin_columns(tmp_path: pathlib.Path):
    _, descriptors = _twin_benchmark(tmp_path)
    plan = plan_reconciliation(descriptors, _facts(descriptors), build_twin_index(descriptors))

    assert len(plan.episodes) == 1
    reconciled = plan.episodes[0]
    assert reconciled.pfi == pytest.approx(1.0)
    assert reconciled.robot_path_dev_m == pytest.approx(1.0)
    assert reconciled.mar == pytest.approx(1.0)
    assert reconciled.kcsc_wh == pytest.approx(2.0)
    assert reconciled.relative_throughput == pytest.approx(0.5)


def test_plan_types_a_missing_pedestrian_twin(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks" / RUN_ID
    contestant = _write_episode(root, 1, robot_path_points=[(0.0, 0.0), (10.0, 0.0)])
    robot_twin = _write_episode(root, 2, reference_type=UNOBSTRUCTED_ROBOT, robot_path_points=[(0.0, 0.0), (10.0, 0.0)])
    descriptors = [contestant, robot_twin]
    plan = plan_reconciliation(descriptors, _facts(descriptors), build_twin_index(descriptors))

    reconciled = plan.episodes[0]
    assert reconciled.mar is None
    assert reconciled.pfi is None
    assert reconciled.missing_reference("mar") == UNHINDERED_PEDS
    assert reconciled.missing_reference("pfi") == UNHINDERED_PEDS
    assert reconciled.robot_path_dev_m == pytest.approx(0.0)
    patched = patch_fact_episodes(_facts(descriptors)[1], plan)
    row = patched.to_dicts()[0]
    assert "mar:reference_missing:unhindered_peds" in row["null_reason"]
    assert "pfi:reference_missing:unhindered_peds" in row["null_reason"]
    assert "mar:no_samples" not in row["null_reason"]


def test_plan_types_a_missing_robot_twin(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks" / RUN_ID
    contestant = _write_episode(root, 1, pedestrian_paths_points=[[(0.0, 1.0), (1.0, 1.0)]])
    peds_twin = _write_episode(root, 3, reference_type=UNHINDERED_PEDS, pedestrian_paths_points=[[(0.0, 0.0), (1.0, 0.0)]])
    descriptors = [contestant, peds_twin]
    plan = plan_reconciliation(descriptors, _facts(descriptors), build_twin_index(descriptors))

    reconciled = plan.episodes[0]
    assert reconciled.pfi == pytest.approx(1.0)
    assert reconciled.mar is None
    assert reconciled.robot_path_dev_m is None
    assert reconciled.missing_reference("mar") == UNOBSTRUCTED_ROBOT
    assert reconciled.missing_reference("robot_path_dev_m") == UNOBSTRUCTED_ROBOT


def test_patch_fact_pedestrians_records_the_missing_reference(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks" / RUN_ID
    contestant = _write_episode(root, 1)
    plan = plan_reconciliation([contestant], _facts([contestant]), build_twin_index([contestant]))

    frame = pl.read_parquet(pathlib.Path(contestant.episode_dir) / "metrics" / "fact_pedestrians.parquet")
    patched = patch_fact_pedestrians(frame, plan)
    reason = patched.to_dicts()[0]["null_reason"]
    assert "ped_name:not_applicable" in reason
    assert "ped_path_deflection_m:reference_missing:unhindered_peds" in reason


def test_apply_plan_to_disk_round_trips_the_values(tmp_path: pathlib.Path):
    _, descriptors = _twin_benchmark(tmp_path)
    plan = plan_reconciliation(descriptors, _facts(descriptors), build_twin_index(descriptors))

    assert apply_plan_to_disk(pathlib.Path(descriptors[0].episode_dir), plan)
    patched = pl.read_parquet(pathlib.Path(descriptors[0].episode_dir) / "metrics" / "fact_episodes.parquet").to_dicts()[0]
    assert patched["mar"] == pytest.approx(1.0)
    assert patched["pfi"] == pytest.approx(1.0)
    assert patched["robot_path_dev_m"] == pytest.approx(1.0)
    assert patched["kcsc_wh"] == pytest.approx(2.0)
    assert patched["relative_throughput"] == pytest.approx(0.5)
    assert "reference_missing" not in (patched["null_reason"] or "")


def test_robot_and_pedestrian_paths_are_read_from_the_series_tables(tmp_path: pathlib.Path):
    _, descriptors = _twin_benchmark(tmp_path)
    contestant_dir = pathlib.Path(descriptors[0].episode_dir)

    assert robot_path(contestant_dir) == [(0.0, 0.0), (5.0, 1.0), (10.0, 0.0)]
    assert pedestrian_paths(contestant_dir) == [[(0.0, 1.0), (1.0, 1.0)]]
    assert robot_path(contestant_dir, "another_robot") == [(0.0, 0.0), (5.0, 1.0), (10.0, 0.0)]


def test_process_benchmark_reconciles_already_evaluated_episodes(tmp_path: pathlib.Path):
    """Regression for the ordering bug: a fully cached run still gets its reconciled values into the facts."""
    root, descriptors = _twin_benchmark(tmp_path)
    pipeline = ProcessingPipeline(FolderManager(data_root=root.parent), workers=1)

    combined = pipeline.process_benchmark(RUN_ID)

    assert combined is not None
    facts = scan_star_schema(combined)["fact_episodes"].collect()
    contestant = facts.filter(pl.col("episode_id") == descriptors[0].episode_id).to_dicts()[0]
    assert contestant["mar"] == pytest.approx(1.0)
    assert contestant["pfi"] == pytest.approx(1.0)
    assert contestant["robot_path_dev_m"] == pytest.approx(1.0)
    assert contestant["kcsc_wh"] == pytest.approx(2.0)
    assert contestant["relative_throughput"] == pytest.approx(0.5)
    reference = facts.filter(pl.col("episode_id") == descriptors[1].episode_id).to_dicts()[0]
    assert reference["mar"] is None
    assert reference["pfi"] is None
