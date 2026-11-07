"""Unit tests for the Benchmark Star Schema Lakehouse & Polars OLAP Core (WP3).

Verifies:
1. StarSchemaBuilder dimension/fact construction, 1NF flatness (zero List/Struct),
   and Parquet round-tripping via write_star_schema / scan_star_schema.
2. ProcessingPipeline.process_episode and process_benchmark emission of per-episode
   scoped tables and benchmark-level Star Schema lakehouse tables.
3. PolarsSQLEngine execution of multi-table OLAP SQL queries (including the exact
   Blame Attribution SQL query from 00 Master Architecture, planner comparison under
   visible vs. blind crowds, and Jackal vs. Go1 energy/acoustics) in < 50 ms, plus
   SQLGuardError enforcement on destructive / multi-statement queries.
4. Statistical procedures: cluster_bootstrap_ci, paired_treatment_diff,
   rank_swap_matrix, kendalls_w (W = 1.0 identical, W = 0.0 inverted), and
   split_discovery_verification_seeds.
5. In-memory sensitivity analysis sweeps (what_if) and EvidenceStore content-addressed
   determinism (ev-<sha256[:12]>).
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import time
from typing import Final

import numpy as np
import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal

from arena_evaluation.analytics import (
    CANONICAL_VIEW_NAMES,
    BenchmarkCatalog,
    EvidenceStore,
    PolarsSQLEngine,
    SQLGuardError,
    cluster_bootstrap_ci,
    kendalls_w,
    paired_treatment_diff,
    rank_swap_matrix,
    split_discovery_verification_seeds,
    sweep_closer_share_thresholds,
    sweep_encounter_clearance_thresholds,
)
from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry
from arena_evaluation.processing.metrics.ecological.condition_metrics import _world_cache
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.processing.scoped_evaluator import ScopedEvaluator
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import (
    AlignedEpisodeBundle,
    RobotParams,
    RunMetadata,
)
from arena_evaluation.storage.star_schema import (
    DIM_EPISODES_SCHEMA,
    DIM_WORLD_ENTITIES_SCHEMA,
    STAR_SCHEMA_SCHEMAS,
    STAR_TABLE_NAMES,
    StarSchemaBuilder,
    StarSchemaTables,
    scan_star_schema,
    write_star_schema,
)

NS_PER_S: Final[int] = 1_000_000_000
RUN_ID: Final[str] = "bench-wp3"


def _make_synthetic_episode(
    *,
    episode_id: int,
    planner: str,
    robot_model: str,
    crowd_condition: str,
    seed: int,
    power_w: float,
    acoustic_dba: float,
    ped_closer: bool,
) -> tuple[AlignedEpisodeBundle, RobotParams, RunMetadata]:
    """Build a synthetic 4 s (10 Hz) episode with one crossing pedestrian, one contact, and world semantics."""
    n_samples = 41
    t_ns = np.arange(n_samples, dtype=np.int64) * (NS_PER_S // 10)
    times_s = t_ns.astype(np.float64) / NS_PER_S

    robot_id = f"env_0_{robot_model}_0"
    if ped_closer:
        # Robot stationary at x = 2.0, pedestrian walks into robot along y -> closer_label == "pedestrian"
        rx = np.full(n_samples, 2.0, dtype=np.float64)
        rvx = np.zeros(n_samples, dtype=np.float64)
        px = np.full(n_samples, 2.0, dtype=np.float64)
        py = 1.2 * (times_s - 2.0)
        pvx = np.zeros(n_samples, dtype=np.float64)
        pvy = np.full(n_samples, 1.2, dtype=np.float64)
    else:
        # Robot drives along +x at 1.0 m/s into stationary pedestrian at (2.0, 0.0) -> closer_label == "robot"
        rx = times_s.copy()
        rvx = np.ones(n_samples, dtype=np.float64)
        px = np.full(n_samples, 2.0, dtype=np.float64)
        py = np.zeros(n_samples, dtype=np.float64)
        pvx = np.zeros(n_samples, dtype=np.float64)
        pvy = np.zeros(n_samples, dtype=np.float64)

    robot_tracks = pl.DataFrame(
        {
            "robot_id": [robot_id] * n_samples,
            "env_id": ["env_0"] * n_samples,
            "t_ns": t_ns,
            "x": rx,
            "y": np.zeros(n_samples, dtype=np.float64),
            "yaw": np.zeros(n_samples, dtype=np.float64),
            "vx": rvx,
            "vy": np.zeros(n_samples, dtype=np.float64),
            "pose_source": ["tf_gt"] * n_samples,
        },
        schema={
            "robot_id": pl.String,
            "env_id": pl.String,
            "t_ns": pl.Int64,
            "x": pl.Float64,
            "y": pl.Float64,
            "yaw": pl.Float64,
            "vx": pl.Float64,
            "vy": pl.Float64,
            "pose_source": pl.String,
        },
    )

    pedestrian_tracks = pl.DataFrame(
        {
            "t_ns": t_ns,
            "env_id": ["env_0"] * n_samples,
            "ped_id": ["Pedestrian_1"] * n_samples,
            "ped_name": ["nurse_1"] * n_samples,
            "x": px,
            "y": py,
            "yaw": np.full(n_samples, math.pi / 2.0, dtype=np.float64),
            "vx": pvx,
            "vy": pvy,
            "speed_mps": np.hypot(pvx, pvy),
            "radius_m": np.full(n_samples, 0.3, dtype=np.float64),
            "desired_speed": np.full(n_samples, 1.2, dtype=np.float64),
            "policy": ["sfm"] * n_samples,
            "agent_type": ["adult"] * n_samples,
        },
        schema={
            "t_ns": pl.Int64,
            "env_id": pl.String,
            "ped_id": pl.String,
            "ped_name": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "yaw": pl.Float64,
            "vx": pl.Float64,
            "vy": pl.Float64,
            "speed_mps": pl.Float64,
            "radius_m": pl.Float64,
            "desired_speed": pl.Float64,
            "policy": pl.String,
            "agent_type": pl.String,
        },
    )

    robot_power = pl.DataFrame(
        {
            "t_ns": t_ns,
            "total_power_w": np.full(n_samples, power_w, dtype=np.float64),
        },
        schema={"t_ns": pl.Int64, "total_power_w": pl.Float64},
    )

    robot_acoustics = pl.DataFrame(
        {
            "t_ns": t_ns,
            "total_level_af_dba": np.full(n_samples, acoustic_dba, dtype=np.float64),
        },
        schema={"t_ns": pl.Int64, "total_level_af_dba": pl.Float64},
    )

    collision_contacts = pl.DataFrame(
        [
            {
                "t_ns": 2 * NS_PER_S,
                "obstacle_id": "env_0/Pedestrian_1",
                "kind": "pedestrian",
                "obstacle_x": 2.0,
                "obstacle_y": 0.0,
                "obstacle_z": 0.0,
                "distance": 0.0,
            }
        ],
        schema={
            "t_ns": pl.Int64,
            "obstacle_id": pl.String,
            "kind": pl.String,
            "obstacle_x": pl.Float64,
            "obstacle_y": pl.Float64,
            "obstacle_z": pl.Float64,
            "distance": pl.Float64,
        },
    )

    semantic_snapshot = pl.DataFrame(
        [
            {
                "t_ns": 0,
                "env_id": "env_0",
                "world": "hospital_1",
                "entity_name": "SM_DoorSliding_01",
                "kind": "door",
                "field_name": "state",
                "field_kind": "discrete",
                "value_str": "closed",
                "value_num": None,
                "value_bool": None,
            },
            {
                "t_ns": 0,
                "env_id": "env_0",
                "world": "hospital_1",
                "entity_name": "SM_DoorSliding_01",
                "kind": "door",
                "field_name": "activation_dist",
                "field_kind": "continuous",
                "value_str": None,
                "value_num": 1.2,
                "value_bool": None,
            },
            {
                "t_ns": 0,
                "env_id": "env_0",
                "world": "hospital_1",
                "entity_name": "SM_DoorSliding_01",
                "kind": "door",
                "field_name": "acoustic_tl_db",
                "field_kind": "continuous",
                "value_str": None,
                "value_num": 24.0,
                "value_bool": None,
            },
            {
                "t_ns": 0,
                "env_id": "env_0",
                "world": "hospital_1",
                "entity_name": "SM_DoorSliding_01",
                "kind": "door",
                "field_name": "zone",
                "field_kind": "discrete",
                "value_str": "hallway_west",
                "value_num": None,
                "value_bool": None,
            },
        ],
        schema={
            "t_ns": pl.Int64,
            "env_id": pl.String,
            "world": pl.String,
            "entity_name": pl.String,
            "kind": pl.String,
            "field_name": pl.String,
            "field_kind": pl.String,
            "value_str": pl.String,
            "value_num": pl.Float64,
            "value_bool": pl.Boolean,
        },
    )

    meta = RunMetadata(
        benchmark_id=RUN_ID,
        planner=planner,
        robot_model=[f"{robot_model}[lidar=sick_microscan3]"],
        map="hospital_1",
        stage="opposing_flow",
        episode_id=episode_id,
        suite_name="socio_acoustic_suite",
        contest_name="nav2_full_suite",
        local_planner=planner,
        inter_planner="none",
        outcome_state=2,
        outcome_info="finished",
        recording_started_at="2026-11-07T00:00:00+00:00",
        python_version="3.12",
        ros_distro="jazzy",
        params={
            "scenario": "opposing_flow",
            "dispatcher": "cbs",
            "seed": seed,
            "robot.mobile.max_linear_vel": 0.75,
            "static_power_w": 7.2,
        },
    )

    robot_params = RobotParams(
        model=robot_model,
        robot_radius=0.267 if robot_model == "jackal" else 0.30,
        base_mass=17.5 if robot_model == "jackal" else 12.0,
        max_linear_velocity=0.75,
    )

    bundle = AlignedEpisodeBundle(
        episode_id=episode_id,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[4.0, 0.0, 0.0],
        num_pedestrians=1,
        robot_name=f"{robot_model}_0",
        semantic_snapshot=semantic_snapshot,
        human_params={
            "humansim.robots_visible": crowd_condition == "visible",
            "seed": seed,
            "ped_policy": "sfm",
        },
        outcome_state=2,
        outcome_info="finished",
        run=meta,
        map="hospital_1",
        topics={
            "robot_tracks": robot_tracks,
            "pedestrian_tracks": pedestrian_tracks,
            "robot_power": robot_power,
            "robot_acoustics": robot_acoustics,
            "collision_contacts": collision_contacts,
        },
    )
    return bundle, robot_params, meta


@pytest.fixture(scope="module")
def star_tables() -> StarSchemaTables:
    """Build a multi-episode StarSchemaTables fixture covering planners, crowds, robots, and seeds."""
    specs = [
        # (episode_id, planner, robot_model, crowd_condition, seed, power_w, acoustic_dba, ped_closer, time_to_goal)
        (1, "dwb-standard", "jackal", "visible", 101, 80.0, 58.0, False, 5.0),
        (2, "dwb-standard", "jackal", "blind", 101, 85.0, 59.0, True, 6.2),
        (3, "teb", "jackal", "visible", 101, 95.0, 61.0, False, 4.5),
        (4, "teb", "jackal", "blind", 101, 105.0, 62.0, False, 7.8),
        (5, "mppi", "go1", "visible", 102, 140.0, 68.0, False, 4.8),
        (6, "mppi", "go1", "blind", 102, 150.0, 70.0, True, 5.4),
        (7, "dwb-standard", "jackal", "visible", 102, 82.0, 58.5, False, 5.2),
        (8, "teb", "jackal", "visible", 102, 92.0, 60.5, False, 4.7),
    ]

    ep_star_list: list[StarSchemaTables] = []
    builder = StarSchemaBuilder()

    for ep_id, planner, robot_model, crowd, seed, power_w, dba, ped_closer, t_goal in specs:
        bundle, robot_params, meta = _make_synthetic_episode(
            episode_id=ep_id,
            planner=planner,
            robot_model=robot_model,
            crowd_condition=crowd,
            seed=seed,
            power_w=power_w,
            acoustic_dba=dba,
            ped_closer=ped_closer,
        )
        ep_metrics = {
            "result": "SUCCESS",
            "success": True,
            "time_to_goal": t_goal,
            "path_length": 4.0,
            "path_efficiency": 0.92,
            "energy_total_wh": (power_w * t_goal) / 3600.0,
            "energy_mechanical_wh": (0.6 * power_w * t_goal) / 3600.0,
            "energy_thermal_wh": (0.2 * power_w * t_goal) / 3600.0,
            "energy_per_meter": (power_w * t_goal) / (3600.0 * 4.0),
            "velocity_mean": 4.0 / t_goal,
            "velocity_max": 1.0,
            "acceleration_max": 1.5,
            "jerk_max": 2.5,
            "min_obstacle_clearance": 0.45,
            "min_pedestrian_clearance": 0.0,
            "collision_amount_pedestrian": 1,
            "collision_amount_wall": 0,
            "collision_amount_static": 0,
            "mar": 0.5,
            "pfi": 0.2,
            "robot_path_deviation_m": 0.3,
            "power_peak_w": power_w * 1.2,
            "friction_dissipation_j": 4.0,
        }
        evaluator = ScopedEvaluator(robot_params)
        facts = evaluator.evaluate(
            bundle,
            run_id=RUN_ID,
            episode_id=ep_id,
            episode_metrics=ep_metrics,
        )
        ep_star_list.append(
            builder.build_episode_tables(
                bundle,
                facts,
                run_id=RUN_ID,
                episode_id=ep_id,
                robot_params=robot_params,
                metadata=meta,
            )
        )

    return StarSchemaBuilder.concat_tables(ep_star_list)


@pytest.mark.parametrize("table_name", STAR_TABLE_NAMES)
def test_star_schema_tables_match_authoritative_schemas_and_1nf(
    table_name: str,
    star_tables: StarSchemaTables,
) -> None:
    frame = star_tables.by_name()[table_name]
    expected_schema = STAR_SCHEMA_SCHEMAS[table_name]
    assert frame.schema == expected_schema
    assert frame.height > 0
    # Invariant 3: 100% Flat Tables (1NF) - zero nested List, Struct, or Array columns
    for col_name, dtype in frame.schema.items():
        assert not isinstance(dtype, (pl.List, pl.Struct, pl.Array)), f"Table {table_name} column {col_name} has nested dtype {dtype}"


def test_star_schema_empty_builder_preserves_schemas() -> None:
    empty = StarSchemaBuilder.concat_tables([])
    for name, schema in STAR_SCHEMA_SCHEMAS.items():
        frame = empty.by_name()[name]
        assert frame.height == 0
        assert frame.schema == schema


def test_write_and_scan_star_schema_round_trip(
    star_tables: StarSchemaTables,
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = tmp_path / "benchmarks" / RUN_ID
    written = write_star_schema(star_tables, bench_dir)
    assert set(written) == set(STAR_TABLE_NAMES)

    for dim_name in ("dim_episodes", "dim_robots", "dim_pedestrians", "dim_world_entities"):
        assert written[dim_name] == bench_dir / f"{dim_name}.parquet"
        assert written[dim_name].is_file()

    for fact_name in ("fact_episodes", "fact_encounters", "fact_contacts", "fact_pedestrians", "fact_robots"):
        assert written[fact_name] == bench_dir / "metrics" / f"{fact_name}.parquet"
        assert written[fact_name].is_file()

    scanned = scan_star_schema(bench_dir)
    assert set(scanned) == set(STAR_TABLE_NAMES)
    for name in STAR_TABLE_NAMES:
        reloaded = scanned[name].collect()
        original = star_tables.by_name()[name]
        assert_frame_equal(reloaded, original, check_dtypes=True, check_exact=False, rel_tol=1e-6)


def test_dim_tables_extract_expected_metadata_and_world_entities(star_tables: StarSchemaTables) -> None:
    ep1 = star_tables.dim_episodes.filter(pl.col("episode_id") == 1)
    expected_ep1 = pl.DataFrame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": 1,
                "env_id": "env_0",
                "robot_id": "env_0_jackal_0",
                "benchmark_id": RUN_ID,
                "suite_name": "socio_acoustic_suite",
                "contest_name": "nav2_full_suite",
                "stage": "opposing_flow",
                "world": "hospital_1",
                "scenario": "opposing_flow",
                "planner": "dwb-standard",
                "inter_planner": "none",
                "dispatcher": "cbs",
                "robot_model": "jackal",
                "crowd_condition": "visible",
                "ped_policy_default": "sfm",
                "crowd_size": 1,
                "seed": 101,
                "is_reference": False,
                "reference_type": None,
                "duration_s": 5.0,
                "outcome_state": 2,
                "outcome_info": "finished",
            }
        ],
        schema=DIM_EPISODES_SCHEMA,
    )
    assert_frame_equal(ep1, expected_ep1, check_dtypes=True, check_exact=False)

    rob1 = star_tables.dim_robots.filter(pl.col("episode_id") == 1)
    assert rob1.item(0, "lidar_variant") == "sick_microscan3"
    assert rob1.item(0, "max_linear_vel") == pytest.approx(0.75)
    assert rob1.item(0, "robot_mass_kg") == pytest.approx(17.5)

    world1 = star_tables.dim_world_entities.filter(pl.col("episode_id") == 1)
    expected_world1 = pl.DataFrame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": 1,
                "env_id": "env_0",
                "entity_name": "SM_DoorSliding_01",
                "entity_kind": "door",
                "zone": "hallway_west",
                "activation_dist": 1.2,
                "acoustic_tl_db": 24.0,
                "initial_state": "closed",
            }
        ],
        schema=DIM_WORLD_ENTITIES_SCHEMA,
    )
    assert_frame_equal(world1, expected_world1, check_dtypes=True, check_exact=False)


def test_polars_sql_engine_executes_master_architecture_blame_query_under_50ms(
    star_tables: StarSchemaTables,
) -> None:
    engine = PolarsSQLEngine(star_tables)

    # Exact Blame Attribution SQL query from 00 Master Architecture (§Layer 3)
    sql = """
    SELECT
        d.crowd_condition,
        d.planner,
        c.closer_label,
        COUNT(*) AS total_contacts,
        SUM(CASE WHEN c.counterfactual_freeze_persists THEN 1 ELSE 0 END) AS unavoidable_contacts
    FROM fact_contacts c
    JOIN dim_episodes d USING (run_id, episode_id, robot_id)
    GROUP BY d.crowd_condition, d.planner, c.closer_label
    ORDER BY d.crowd_condition, d.planner, c.closer_label;
    """
    # Warm up SQLContext once, then time execution
    _ = engine.query(sql)
    t0 = time.perf_counter()
    res = engine.query(sql, save_as="derived_blame_matrix")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50.0, f"Expected OLAP SQL query in < 50 ms, took {elapsed_ms:.2f} ms"
    assert "derived_blame_matrix" in engine.catalog.table_names()

    # Verify blind dwb-standard and mppi attributed closer_label to "pedestrian"
    blind_ped = res.filter((pl.col("crowd_condition") == "blind") & (pl.col("closer_label") == "pedestrian"))
    assert blind_ped.height == 2
    assert set(blind_ped.get_column("planner").to_numpy()) == {"dwb-standard", "mppi"}


def test_polars_sql_engine_planner_comparison_and_robot_platform_queries_under_50ms(
    star_tables: StarSchemaTables,
) -> None:
    engine = PolarsSQLEngine(star_tables)

    # 1. Planner comparison under visible vs. blind crowds
    sql_planners = """
    SELECT
        d.crowd_condition,
        d.planner,
        COUNT(*) AS episodes,
        AVG(f.time_to_goal_s) AS mean_time_s,
        AVG(f.total_energy_wh) AS mean_energy_wh,
        SUM(CASE WHEN f.success_no_robot_fault THEN 1 ELSE 0 END) AS fault_free_successes
    FROM fact_episodes f
    JOIN dim_episodes d USING (run_id, episode_id, robot_id)
    GROUP BY d.crowd_condition, d.planner
    ORDER BY d.crowd_condition, d.planner
    """
    t0 = time.perf_counter()
    df_planners = engine.query(sql_planners)
    elapsed_planners_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_planners_ms < 50.0
    assert df_planners.height == 6

    # 2. Jackal vs. Go1 energy and acoustic exposure comparison
    sql_platforms = """
    SELECT
        d.robot_model,
        AVG(r.mean_power_w) AS avg_power_w,
        AVG(r.electrical_energy_wh) AS avg_energy_wh,
        AVG(r.acoustic_emission_pa2s) AS avg_acoustic_dose_pa2s,
        AVG(e.sound_exposure_dose_pa2s) AS avg_encounter_sound_pa2s
    FROM fact_robots r
    JOIN dim_episodes d USING (run_id, episode_id, robot_id)
    JOIN fact_encounters e USING (run_id, episode_id, robot_id)
    GROUP BY d.robot_model
    ORDER BY d.robot_model
    """
    t1 = time.perf_counter()
    df_platforms = engine.query(sql_platforms)
    elapsed_platforms_ms = (time.perf_counter() - t1) * 1000.0
    assert elapsed_platforms_ms < 50.0
    assert df_platforms.height == 2

    go1_row = df_platforms.filter(pl.col("robot_model") == "go1")
    jackal_row = df_platforms.filter(pl.col("robot_model") == "jackal")
    assert go1_row.item(0, "avg_power_w") > jackal_row.item(0, "avg_power_w")
    assert go1_row.item(0, "avg_acoustic_dose_pa2s") > jackal_row.item(0, "avg_acoustic_dose_pa2s")


@pytest.mark.parametrize("view_name", CANONICAL_VIEW_NAMES)
def test_catalog_canonical_views_materialize_cleanly(
    view_name: str,
    star_tables: StarSchemaTables,
) -> None:
    catalog = BenchmarkCatalog(star_tables)
    view_df = catalog.get_table(view_name).collect()
    assert view_df.height > 0


@pytest.mark.parametrize(
    "bad_sql",
    [
        "",
        "   -- just a comment",
        "DROP TABLE fact_episodes",
        "DELETE FROM dim_episodes WHERE episode_id = 1",
        "INSERT INTO dim_episodes VALUES ('a', 1)",
        "UPDATE fact_episodes SET success = false",
        "CREATE TABLE hacked AS SELECT * FROM dim_episodes",
        "ALTER TABLE dim_episodes DROP COLUMN seed",
        "TRUNCATE TABLE fact_contacts",
        "SELECT * FROM dim_episodes; DROP TABLE dim_episodes",
        "SELECT * FROM dim_episodes; SELECT * FROM fact_episodes",
        "PRAGMA table_info(dim_episodes)",
    ],
)
def test_polars_sql_engine_blocks_destructive_and_multistatement_sql(
    bad_sql: str,
    star_tables: StarSchemaTables,
) -> None:
    engine = PolarsSQLEngine(star_tables)
    with pytest.raises(SQLGuardError):
        engine.query(bad_sql)


def test_cluster_bootstrap_ci_and_paired_treatment_diff(star_tables: StarSchemaTables) -> None:
    catalog = BenchmarkCatalog(star_tables)
    summary_lf = catalog.get_table("v_episode_summary")

    boot_df = cluster_bootstrap_ci(
        summary_lf,
        "time_to_goal_s",
        group_cols=("planner",),
        cluster_col="seed",
        n_boot=500,
        ci_level=0.95,
        random_seed=123,
    )
    assert boot_df.height == 3
    for col in ("mean", "std_error", "ci_low", "ci_high", "basic_low", "basic_high"):
        assert boot_df.schema[col] == pl.Float64
    assert (boot_df.get_column("ci_low") <= boot_df.get_column("mean")).all()
    assert (boot_df.get_column("mean") <= boot_df.get_column("ci_high")).all()

    diff_res = paired_treatment_diff(
        summary_lf,
        "time_to_goal_s",
        treatment_col="planner",
        treatment_a="dwb-standard",
        treatment_b="teb",
        pair_cols=("world", "scenario", "crowd_condition", "seed"),
        cluster_col="seed",
        n_boot=500,
        random_seed=123,
    )
    assert diff_res.n_pairs == 3
    assert diff_res.mean_diff is not None
    assert diff_res.ci_low is not None and diff_res.ci_high is not None
    assert diff_res.p_value is not None and 0.0 <= diff_res.p_value <= 1.0
    assert diff_res.summary_frame.height == 1


def test_kendalls_w_exact_concordance_and_inversion_boundaries() -> None:
    # 1. Identical rankings across 3 judges and 4 planners -> W == 1.0
    identical_lf = pl.LazyFrame(
        {
            "crowd_condition": ["j1"] * 4 + ["j2"] * 4 + ["j3"] * 4,
            "planner": ["p1", "p2", "p3", "p4"] * 3,
            "time_to_goal_s": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 2.0, 4.0, 6.0, 8.0],
        },
        schema={"crowd_condition": pl.String, "planner": pl.String, "time_to_goal_s": pl.Float64},
    )
    res_ident = kendalls_w(identical_lf, "time_to_goal_s", entity_col="planner", judge_col="crowd_condition")
    assert res_ident.n_judges == 3
    assert res_ident.n_entities == 4
    assert res_ident.w == pytest.approx(1.0, abs=1e-12)
    assert res_ident.friedman_chi2 == pytest.approx(3.0 * 3.0 * 1.0, abs=1e-12)

    swap_ident = rank_swap_matrix(identical_lf, "time_to_goal_s", entity_col="planner", regime_col="crowd_condition")
    assert (swap_ident.get_column("swap_count") == 0).all()
    assert (swap_ident.get_column("kendall_tau") == 1.0).all()

    # 2. Opposite rankings across 2 judges and 4 planners -> W == 0.0 and 100% rank swaps
    inverted_lf = pl.LazyFrame(
        {
            "crowd_condition": ["visible"] * 4 + ["blind"] * 4,
            "planner": ["p1", "p2", "p3", "p4"] * 2,
            "time_to_goal_s": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        },
        schema={"crowd_condition": pl.String, "planner": pl.String, "time_to_goal_s": pl.Float64},
    )
    res_inv = kendalls_w(inverted_lf, "time_to_goal_s", entity_col="planner", judge_col="crowd_condition")
    assert res_inv.w == pytest.approx(0.0, abs=1e-12)
    assert res_inv.friedman_chi2 == pytest.approx(0.0, abs=1e-12)

    swap_inv = rank_swap_matrix(inverted_lf, "time_to_goal_s", entity_col="planner", regime_col="crowd_condition")
    assert swap_inv.height == 1
    assert swap_inv.item(0, "swap_count") == 6  # 4 * 3 / 2 = 6 inverted pairs
    assert swap_inv.item(0, "swap_rate") == pytest.approx(1.0)
    assert swap_inv.item(0, "kendall_tau") == pytest.approx(-1.0)


def test_split_discovery_verification_seeds_partitions_deterministically() -> None:
    input_lf = pl.LazyFrame(
        {
            "episode_id": list(range(10)),
            "seed": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        },
        schema={"episode_id": pl.Int64, "seed": pl.Int64},
    )
    partitioned_df = split_discovery_verification_seeds(input_lf, seed_col="seed", discovery_fraction=0.5).collect()
    discovery_seeds = partitioned_df.filter(pl.col("cohort") == "discovery").get_column("seed")
    verification_seeds = partitioned_df.filter(pl.col("cohort") == "verification").get_column("seed")
    assert len(discovery_seeds) == 5
    assert len(verification_seeds) == 5
    assert discovery_seeds.max() < verification_seeds.min()


def test_what_if_threshold_sweeps(star_tables: StarSchemaTables) -> None:
    closer_sweep = sweep_closer_share_thresholds(
        star_tables.fact_contacts.lazy(),
        star_tables.dim_episodes.lazy(),
        dominant_share_thresholds=(0.60, 0.70, 0.80),
    ).collect()
    assert closer_sweep.height > 0
    assert set(closer_sweep.get_column("dominant_share_threshold").unique().to_numpy()) == {0.60, 0.70, 0.80}

    enc_sweep = sweep_encounter_clearance_thresholds(
        star_tables.fact_encounters.lazy(),
        star_tables.dim_episodes.lazy(),
        max_min_clearance_m=(-1.0, 0.0, 2.0),
    ).collect()
    assert enc_sweep.height > 0
    # Monotonic non-decreasing encounter counts with increasing clearance threshold
    by_group = enc_sweep.sort("crowd_condition", "planner", "max_min_clearance_m")
    diffs = by_group.with_columns(cnt_diff=pl.col("encounter_count").diff().over("crowd_condition", "planner").fill_null(0))
    assert (diffs.get_column("cnt_diff") >= 0).all()


def test_evidence_store_content_addressed_determinism_and_persistence(
    star_tables: StarSchemaTables,
    tmp_path: pathlib.Path,
) -> None:
    ev_dir = tmp_path / "evidence"
    store = EvidenceStore(evidence_dir=ev_dir)
    engine = PolarsSQLEngine(star_tables)

    sql = "SELECT planner, COUNT(*) AS n FROM dim_episodes GROUP BY planner ORDER BY planner"
    df = engine.query(sql)

    rec1 = store.record(df, kind="sql", operation=sql, params={"limit": None})
    rec2 = store.record(df, kind="sql", operation=sql, params={"limit": None})

    assert rec1.evidence_id == rec2.evidence_id
    assert re.fullmatch(r"^ev-[0-9a-f]{12}$", rec1.evidence_id) is not None
    assert rec1.row_count == 3
    assert rec1.columns == ("planner", "n")
    assert rec1.artifact_path is not None and rec1.artifact_path.is_file()
    assert (ev_dir / f"{rec1.evidence_id}.json").is_file()

    # Reload from disk in a fresh EvidenceStore
    reopened = EvidenceStore(evidence_dir=ev_dir)
    loaded_rec = reopened.get(rec1.evidence_id)
    loaded_df = reopened.read_frame(rec1.evidence_id)
    assert loaded_rec.evidence_id == rec1.evidence_id
    assert_frame_equal(loaded_df, df, check_dtypes=True)


def test_processing_pipeline_writes_episode_and_benchmark_star_schema(tmp_path: pathlib.Path) -> None:
    world_name = "synthetic_wp3_world"
    _world_cache[world_name] = [
        _ZoneGeometry(
            name="hallway",
            polygon=shapely.Polygon([(0, -2), (0, 2), (10, 2), (10, -2)]),
            max_speed=None,
            quiet=False,
            restricted=False,
        )
    ]

    bench_id = "wp3_pipeline_bench"
    episode_dir = tmp_path / bench_id / "episodes" / "episode_000"
    topics_dir = episode_dir / "topics"
    topics_dir.mkdir(parents=True)
    (episode_dir / "episode_000.mcap").touch()

    meta = RunMetadata(
        benchmark_id=bench_id,
        planner="dwb-standard",
        map=world_name,
        stage="s0",
        robot_model=["jackal[lidar=sick_microscan3]"],
        outcome_state=2,
        outcome_info="finished",
        recording_started_at="2026-11-07T00:00:00+00:00",
        python_version="3.12",
        ros_distro="jazzy",
    )
    MetadataWriter.write(meta, episode_dir / "episode_000.yaml")

    samples = 31
    stamps = [int(i * NS_PER_S / 10) for i in range(samples)]
    xs = [0.1 * i for i in range(samples)]
    robot_id = "env_0_jackal_0"

    pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": ["env_0"] * samples,
            "robot_id": [robot_id] * samples,
            "odom_x": xs,
            "odom_y": [0.0] * samples,
            "odom_yaw": [0.0] * samples,
            "odom_vx": [1.0] * samples,
            "odom_wz": [0.0] * samples,
            "drift_error_m": [0.0] * samples,
        }
    ).write_parquet(topics_dir / "robot_odom.parquet")

    pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": ["env_0"] * samples,
            "robot_id": [robot_id] * samples,
            "x": xs,
            "y": [0.0] * samples,
            "z": [0.0] * samples,
            "yaw": [0.0] * samples,
            "vx": [1.0] * samples,
            "vy": [0.0] * samples,
            "vz": [0.0] * samples,
            "omega_z": [0.0] * samples,
            "speed_mps": [1.0] * samples,
            "pose_source": ["tf_gt"] * samples,
        }
    ).write_parquet(topics_dir / "robot_tracks.parquet")

    phases = {
        "jackal_0": {
            "phases": [{"goto": [3.0, 0.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}],
            "conditions": [],
            "map_poses": [[3.0, 0.0, 0.0]],
        }
    }
    pl.DataFrame(
        {
            "t_ns": [0, 3 * NS_PER_S],
            "env_id": ["env_0", "env_0"],
            "episode_id": [0, 0],
            "outcome_state": [1, 2],
            "outcome_info": ["", "finished"],
            "phases_yaml": [json.dumps(phases), json.dumps(phases)],
            "conditions_yaml": ["", ""],
            "human_params": ["", ""],
        }
    ).write_parquet(topics_dir / "episode_lifecycle.parquet")
    TopicParquetStore.mark_complete(topics_dir)

    fm = FolderManager(data_root=tmp_path)
    pipeline = ProcessingPipeline(fm, workers=1)
    combined_path = pipeline.process_benchmark(bench_id)
    assert combined_path is not None and combined_path.is_file()

    # Check per-episode metrics dir has fact_*.parquet
    ep_metrics_dir = episode_dir / "metrics"
    for name in STAR_TABLE_NAMES:
        assert (ep_metrics_dir / f"{name}.parquet").is_file()

    # Check benchmark-level Star Schema tables
    bench_dir = combined_path.parent
    scanned = scan_star_schema(bench_dir)
    dim_ep = scanned["dim_episodes"].collect()
    fact_ep = scanned["fact_episodes"].collect()
    assert dim_ep.height == 1
    assert fact_ep.height == 1
    assert dim_ep.item(0, "planner") == "dwb-standard"
    assert fact_ep.item(0, "robot_id") == robot_id
