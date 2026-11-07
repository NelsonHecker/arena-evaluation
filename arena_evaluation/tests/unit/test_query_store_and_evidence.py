"""Unit tests for Evidence Store Index, SQL Query Store & MCP Query Gateway (Layer 4).

Verifies:
1. Authoritative Parquet evidence index (``evidence_index.parquet``) schema and upsert invariants.
2. Direct SQL querying of ``evidence_index`` via ``BenchmarkCatalog`` and ``PolarsSQLEngine``.
3. Transient output vs. pinned evidence decision mechanism in ``evaluation_query``.
4. Parameterized query declarations catalog, safe literal formatting, and < 50 ms query executions.
5. MCP Query Gateway tools (``evaluation_query``, ``evaluation_queries``, ``evaluation_query_save``, ``evaluation_evidence``).
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_mcp.core import ToolError

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import (
    EVIDENCE_INDEX_SCHEMA,
    EvidenceRecord,
    EvidenceStore,
    compute_frame_hash,
    compute_schema_signature,
)
from arena_evaluation.analytics.queries import (
    QueryCatalog,
    QueryDeclaration,
    format_sql_literal,
)
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine, SQLGuardError
from arena_evaluation.mcp.query import (
    evaluation_evidence,
    evaluation_queries,
    evaluation_query,
    evaluation_query_save,
)
from arena_evaluation.storage.star_schema import (
    DIM_EPISODES_SCHEMA,
    DIM_PEDESTRIANS_SCHEMA,
    DIM_ROBOTS_SCHEMA,
    DIM_WORLD_ENTITIES_SCHEMA,
    FACT_CONTACTS_SCHEMA,
    FACT_ENCOUNTERS_SCHEMA,
    FACT_EPISODES_SCHEMA,
    FACT_PEDESTRIANS_SCHEMA,
    FACT_ROBOTS_SCHEMA,
    StarSchemaTables,
)


@pytest.fixture
def sample_star_tables() -> StarSchemaTables:
    """Build a synthetic StarSchemaTables fixture with episodes, encounters, contacts, and telemetry."""
    run_id = "test-run-wp5"

    dim_episodes = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "benchmark_id": run_id,
                "suite_name": "socio_acoustic",
                "contest_name": "nav2_full",
                "stage": "stage_1",
                "world": "hospital_1",
                "scenario": "hallway_encounter",
                "planner": "dwb-standard",
                "inter_planner": "none",
                "dispatcher": "cbs",
                "robot_model": "jackal",
                "crowd_condition": "visible",
                "ped_policy_default": "sfm",
                "crowd_size": 2,
                "seed": 42,
                "is_reference": False,
                "reference_type": None,
                "duration_s": 10.0,
                "outcome_state": 2,
                "outcome_info": "finished",
            },
            {
                "run_id": run_id,
                "episode_id": 1,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "benchmark_id": run_id,
                "suite_name": "socio_acoustic",
                "contest_name": "nav2_full",
                "stage": "stage_1",
                "world": "hospital_1",
                "scenario": "hallway_encounter",
                "planner": "dwb-standard",
                "inter_planner": "none",
                "dispatcher": "cbs",
                "robot_model": "jackal",
                "crowd_condition": "blind",
                "ped_policy_default": "sfm",
                "crowd_size": 2,
                "seed": 43,
                "is_reference": False,
                "reference_type": None,
                "duration_s": 12.0,
                "outcome_state": 2,
                "outcome_info": "finished",
            },
            {
                "run_id": run_id,
                "episode_id": 2,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "benchmark_id": run_id,
                "suite_name": "socio_acoustic",
                "contest_name": "nav2_full",
                "stage": "stage_1",
                "world": "hospital_1",
                "scenario": "hallway_encounter",
                "planner": "mppi",
                "inter_planner": "none",
                "dispatcher": "cbs",
                "robot_model": "go1",
                "crowd_condition": "visible",
                "ped_policy_default": "sfm",
                "crowd_size": 2,
                "seed": 44,
                "is_reference": False,
                "reference_type": None,
                "duration_s": 9.5,
                "outcome_state": 2,
                "outcome_info": "finished",
            },
        ],
        schema=DIM_EPISODES_SCHEMA,
    )

    dim_robots = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": i,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "robot_model": "jackal" if i < 2 else "go1",
                "drive_type": "differential" if i < 2 else "quadruped",
                "max_linear_vel": 0.75,
                "max_angular_vel": 1.5,
                "linear_acc": 2.0,
                "linear_decel": 2.0,
                "robot_radius": 0.35,
                "robot_mass_kg": 17.5 if i < 2 else 12.0,
                "static_power_w": 15.0,
                "lidar_variant": "sick_microscan3",
                "start_x": 0.0,
                "start_y": 0.0,
                "goal_x": 10.0,
                "goal_y": 0.0,
            }
            for i in range(3)
        ],
        schema=DIM_ROBOTS_SCHEMA,
    )

    fact_episodes = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "success": True,
                "success_no_robot_fault": True,
                "time_to_goal_s": 10.0,
                "path_length_m": 10.5,
                "path_efficiency": 0.95,
                "mean_velocity_mps": 0.72,
                "max_velocity_mps": 0.75,
                "total_energy_wh": 12.5,
                "total_collisions_ped": 0,
                "total_collisions_static": 0,
                "min_ped_distance_m": 0.85,
            },
            {
                "run_id": run_id,
                "episode_id": 1,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "success": True,
                "success_no_robot_fault": True,
                "time_to_goal_s": 12.0,
                "path_length_m": 11.2,
                "path_efficiency": 0.89,
                "mean_velocity_mps": 0.65,
                "max_velocity_mps": 0.75,
                "total_energy_wh": 15.2,
                "total_collisions_ped": 1,
                "total_collisions_static": 0,
                "min_ped_distance_m": 0.0,
            },
            {
                "run_id": run_id,
                "episode_id": 2,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "success": True,
                "success_no_robot_fault": True,
                "time_to_goal_s": 9.5,
                "path_length_m": 10.2,
                "path_efficiency": 0.98,
                "mean_velocity_mps": 0.74,
                "max_velocity_mps": 0.75,
                "total_energy_wh": 18.0,
                "total_collisions_ped": 0,
                "total_collisions_static": 0,
                "min_ped_distance_m": 0.92,
            },
        ],
        schema=FACT_EPISODES_SCHEMA,
    )

    fact_encounters = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "encounter_id": "enc_0_0",
                "pedestrian_id": "ped_1",
                "start_stamp_ns": 2_000_000_000,
                "end_stamp_ns": 4_000_000_000,
                "duration_s": 2.0,
                "min_clearance_m": 0.85,
                "initial_ttc_s": 3.0,
                "min_ttc_s": 1.5,
                "approach_type": "head_on",
                "detour_time_s": 0.5,
                "path_deviation_m": 0.2,
                "encounter_energy_j": 120.0,
                "energy_surge_ratio": 1.8,
                "sound_exposure_dose_pa2s": 0.04,
                "probst_R_robot": 0.25,
                "probst_R_actor2": 0.75,
                "probst_R_time": 0.5,
                "probst_E_robot": 0.3,
                "probst_E_actor2": 0.7,
                "had_contact": False,
            },
            {
                "run_id": run_id,
                "episode_id": 1,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "encounter_id": "enc_1_0",
                "pedestrian_id": "ped_1",
                "start_stamp_ns": 2_500_000_000,
                "end_stamp_ns": 5_000_000_000,
                "duration_s": 2.5,
                "min_clearance_m": 0.0,
                "initial_ttc_s": 2.0,
                "min_ttc_s": 0.0,
                "approach_type": "crossing",
                "detour_time_s": 1.2,
                "path_deviation_m": 0.6,
                "encounter_energy_j": 180.0,
                "energy_surge_ratio": 2.4,
                "sound_exposure_dose_pa2s": 0.08,
                "probst_R_robot": 0.15,
                "probst_R_actor2": 0.85,
                "probst_R_time": 0.8,
                "probst_E_robot": 0.2,
                "probst_E_actor2": 0.8,
                "had_contact": True,
            },
        ],
        schema=FACT_ENCOUNTERS_SCHEMA,
    )

    fact_contacts = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 1,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "contact_id": "con_1_0",
                "stamp_ns": 3_500_000_000,
                "other_id": "ped_1",
                "contact_type": "pedestrian",
                "impact_speed_mps": 0.65,
                "robot_closing_share": 0.25,
                "closer_label": "pedestrian",
                "counterfactual_freeze_persists": True,
                "fault_attributed_to": "pedestrian",
            }
        ],
        schema=FACT_CONTACTS_SCHEMA,
    )

    return StarSchemaTables(
        dim_episodes=dim_episodes,
        dim_robots=dim_robots,
        dim_pedestrians=pl.DataFrame(schema=DIM_PEDESTRIANS_SCHEMA),
        dim_world_entities=pl.DataFrame(schema=DIM_WORLD_ENTITIES_SCHEMA),
        fact_episodes=fact_episodes,
        fact_encounters=fact_encounters,
        fact_contacts=fact_contacts,
        fact_pedestrians=pl.DataFrame(schema=FACT_PEDESTRIANS_SCHEMA),
        fact_robots=pl.DataFrame(schema=FACT_ROBOTS_SCHEMA),
    )


def test_evidence_store_index_schema_and_upsert_invariants(tmp_path: pathlib.Path) -> None:
    """Verify authoritative Parquet index schema, content addressing, and upsert uniqueness."""
    ev_dir = tmp_path / "evidence"
    store = EvidenceStore(evidence_dir=ev_dir)

    df1 = pl.DataFrame({"planner": ["dwb", "teb"], "mean_ttg": [15.2, 14.8]})
    rec1 = store.record(
        df1,
        kind="sql",
        operation="SELECT planner, mean_ttg FROM scores",
        purpose="Baseline travel times",
        params={"limit": 10},
    )

    # 1. Parquet index created and matches schema
    index_path = ev_dir / "evidence_index.parquet"
    assert index_path.is_file()
    index_df = store.load_index()
    assert index_df.schema == EVIDENCE_INDEX_SCHEMA
    assert index_df.height == 1
    assert index_df.item(0, "evidence_id") == rec1.evidence_id
    assert index_df.item(0, "purpose") == "Baseline travel times"
    assert index_df.item(0, "kind") == "sql"

    # 2. Duplicate record with identical data and purpose generates identical evidence_id and upserts
    rec2 = store.record(
        df1,
        kind="sql",
        operation="SELECT planner, mean_ttg FROM scores",
        purpose="Baseline travel times",
        params={"limit": 10},
    )
    assert rec1.evidence_id == rec2.evidence_id
    reloaded_index = store.load_index()
    assert reloaded_index.height == 1

    # 3. Second distinct record with different purpose
    df2 = pl.DataFrame({"planner": ["mppi"], "mean_ttg": [12.1]})
    rec3 = store.record(
        df2,
        kind="what_if",
        operation="SELECT planner, mean_ttg FROM whatif",
        purpose="What-if evaluation with 1.2x surge cap",
    )
    assert rec3.evidence_id != rec1.evidence_id
    index_after = store.load_index()
    assert index_after.height == 2
    assert set(index_after.get_column("evidence_id").to_list()) == {rec1.evidence_id, rec3.evidence_id}

    # 4. read_frame matches written frame
    read_df1 = store.read_frame(rec1.evidence_id)
    assert_frame_equal(read_df1, df1)


def test_evidence_index_queryable_via_sql_in_benchmark_catalog(
    sample_star_tables: StarSchemaTables,
    tmp_path: pathlib.Path,
) -> None:
    """Verify evidence_index is registered in BenchmarkCatalog and queryable via Polars SQL."""
    ev_dir = tmp_path / "evidence"
    store = EvidenceStore(evidence_dir=ev_dir)

    df = pl.DataFrame({"metric": ["time", "energy"], "value": [10.5, 25.0]})
    rec = store.record(
        df,
        kind="derived_table",
        operation="derive_table_summary",
        purpose="Attribution hypothesis test",
        params={"scope": "contact"},
    )

    catalog = BenchmarkCatalog(sample_star_tables)
    catalog.register_table("evidence_index", store.scan_index())

    engine = PolarsSQLEngine(catalog)
    sql = "SELECT evidence_id, kind, purpose, row_count, created_at_utc FROM evidence_index WHERE purpose LIKE '%Attribution%'"
    res = engine.query(sql)

    assert res.height == 1
    assert res.item(0, "evidence_id") == rec.evidence_id
    assert res.item(0, "kind") == "derived_table"
    assert "Attribution" in res.item(0, "purpose")
    assert res.item(0, "row_count") == 2


def test_query_declarations_load_and_parameter_substitution() -> None:
    """Verify QueryCatalog loads built-in declarations and performs typed parameter substitution."""
    catalog = QueryCatalog()
    builtins = catalog.list_queries()
    names = {q.name for q in builtins}

    expected_names = {
        "blame_attribution_breakdown",
        "planner_scorecard_overview",
        "near_miss_encounter_spikes",
        "blind_vs_aware_standoff_delta",
        "closer_threshold_sensitivity",
        "telemetry_slice",
    }
    assert expected_names <= names

    # Test parameter substitution and literal formatting
    blame_decl = catalog.get_query("blame_attribution_breakdown")
    rendered_default = blame_decl.render_sql()
    assert "'all' = 'all'" in rendered_default

    rendered_custom = blame_decl.render_sql({"crowd_condition": "blind", "planner": "dwb-standard"})
    assert "'blind' = 'all'" in rendered_custom or "d.crowd_condition = 'blind'" in rendered_custom
    assert "'dwb-standard' = 'all'" in rendered_custom or "d.planner = 'dwb-standard'" in rendered_custom

    # Test format_sql_literal
    assert format_sql_literal("jackal's lidar") == "'jackal''s lidar'"
    assert format_sql_literal(42) == "42"
    assert format_sql_literal(3.14) == "3.14"
    assert format_sql_literal(True) == "TRUE"
    assert format_sql_literal(False) == "FALSE"
    assert format_sql_literal(None) == "NULL"
    assert format_sql_literal(["a", "b"]) == "('a', 'b')"


def test_execute_all_six_stored_queries_under_50ms(sample_star_tables: StarSchemaTables) -> None:
    """Verify all 6 stored query declarations execute cleanly on star tables within < 50 ms."""
    catalog = BenchmarkCatalog(sample_star_tables)
    engine = PolarsSQLEngine(catalog)
    q_catalog = QueryCatalog()

    stored_names = (
        "blame_attribution_breakdown",
        "planner_scorecard_overview",
        "near_miss_encounter_spikes",
        "blind_vs_aware_standoff_delta",
        "closer_threshold_sensitivity",
        "telemetry_slice",
    )

    for name in stored_names:
        # Warmup
        _ = q_catalog.execute(name, engine)

        t0 = time.perf_counter()
        res = q_catalog.execute(name, engine)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 50.0, f"Query '{name}' took {elapsed_ms:.2f} ms (budget: 50 ms)"
        assert isinstance(res, pl.DataFrame)


def test_save_new_query_declaration(tmp_path: pathlib.Path) -> None:
    """Verify saving a new query declaration writes YAML and can be loaded and executed."""
    q_dir = tmp_path / "queries"
    catalog = QueryCatalog(search_dirs=[q_dir])

    new_sql = """
    SELECT
        planner,
        AVG(time_to_goal_s) AS mean_time_s
    FROM fact_episodes
    JOIN dim_episodes USING (run_id, episode_id, robot_id)
    WHERE (:planner = 'all' OR planner = :planner)
    GROUP BY planner
    """

    decl = catalog.save_query(
        name="custom_planner_speed",
        description="Average traversal time per planner",
        sql=new_sql,
        parameters={"planner": {"type": "string", "default": "all"}},
        title="Custom Planner Speed Analysis",
        directory=q_dir,
    )

    assert (q_dir / "custom_planner_speed.yaml").is_file()
    assert catalog.has_query("custom_planner_speed")

    reloaded = catalog.get_query("custom_planner_speed")
    assert reloaded.title == "Custom Planner Speed Analysis"
    rendered = reloaded.render_sql({"planner": "dwb-standard"})
    assert "'dwb-standard'" in rendered


def test_mcp_query_decision_mechanism_transient_vs_pinned(
    tmp_path: pathlib.Path,
    sample_star_tables: StarSchemaTables,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify evaluation_query returns transient output when unpinned, and persists pinned evidence when requested."""
    run_dir = tmp_path / "data" / "benchmarks" / "run-wp5-001"
    run_dir.mkdir(parents=True, exist_ok=True)
    from arena_evaluation.storage.star_schema import write_star_schema

    write_star_schema(sample_star_tables, run_dir)

    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    # Case 1: Transient query (pin_evidence=False, purpose="")
    res_transient = evaluation_query(
        run_id="run-wp5-001",
        query="SELECT planner, COUNT(*) AS n FROM dim_episodes GROUP BY planner ORDER BY planner",
        pin_evidence=False,
        purpose="",
    )
    assert res_transient["pin_status"] == "transient"
    assert res_transient["evidence_id"] is None
    assert res_transient["artifact_path"] is None
    assert res_transient["row_count"] == 2
    assert not (run_dir / "evidence").exists()

    # Case 2: Pinned query via pin_evidence=True
    res_pinned = evaluation_query(
        run_id="run-wp5-001",
        query="SELECT planner, COUNT(*) AS n FROM dim_episodes GROUP BY planner ORDER BY planner",
        pin_evidence=True,
        purpose="Pinning planner breakdown",
    )
    assert res_pinned["pin_status"] == "pinned"
    assert res_pinned["evidence_id"] is not None
    assert res_pinned["evidence_id"].startswith("ev-")
    assert res_pinned["artifact_path"] is not None
    assert pathlib.Path(res_pinned["artifact_path"]).is_file()

    ev_index_path = run_dir / "evidence" / "evidence_index.parquet"
    assert ev_index_path.is_file()
    index_df = pl.read_parquet(ev_index_path)
    assert res_pinned["evidence_id"] in index_df.get_column("evidence_id").to_list()

    # Case 3: Pinned query via purpose string provided even with pin_evidence=False
    res_pinned_purpose = evaluation_query(
        run_id="run-wp5-001",
        query="SELECT crowd_condition, COUNT(*) AS cnt FROM dim_episodes GROUP BY crowd_condition",
        pin_evidence=False,
        purpose="Crowd distribution check",
    )
    assert res_pinned_purpose["pin_status"] == "pinned"
    assert res_pinned_purpose["evidence_id"] is not None

    # Case 4: Stored query execution via MCP
    res_stored = evaluation_query(
        run_id="run-wp5-001",
        stored_query="blame_attribution_breakdown",
        params={"crowd_condition": "all", "planner": "all"},
        pin_evidence=False,
    )
    assert res_stored["pin_status"] == "transient"
    assert res_stored["row_count"] > 0


def test_mcp_queries_and_evidence_gateway_tools(
    tmp_path: pathlib.Path,
    sample_star_tables: StarSchemaTables,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify evaluation_queries, evaluation_query_save, and evaluation_evidence MCP tools."""
    run_dir = tmp_path / "data" / "benchmarks" / "run-wp5-002"
    run_dir.mkdir(parents=True, exist_ok=True)
    from arena_evaluation.storage.star_schema import write_star_schema

    write_star_schema(sample_star_tables, run_dir)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    # 1. evaluation_queries
    queries_list = evaluation_queries()
    assert len(queries_list) >= 6
    query_names = {q["name"] for q in queries_list}
    assert "blame_attribution_breakdown" in query_names
    assert "closer_threshold_sensitivity" in query_names

    # 2. Pin an evidence record first
    pin_res = evaluation_query(
        run_id="run-wp5-002",
        query="SELECT world, scenario, COUNT(*) AS count FROM dim_episodes GROUP BY world, scenario",
        pin_evidence=True,
        purpose="World scenario distribution evidence",
    )
    ev_id = pin_res["evidence_id"]
    assert ev_id is not None

    # 3. evaluation_evidence - index list with filter
    ev_list = evaluation_evidence(run_id="run-wp5-002", purpose_filter="distribution")
    assert ev_list["total_records"] >= 1
    assert any(r["evidence_id"] == ev_id for r in ev_list["records"])

    # 4. evaluation_evidence - fetch specific record data
    ev_detail = evaluation_evidence(run_id="run-wp5-002", evidence_id=ev_id)
    assert ev_detail["evidence_id"] == ev_id
    assert ev_detail["purpose"] == "World scenario distribution evidence"
    assert len(ev_detail["data"]) == 1

    # 5. evaluation_query_save
    save_res = evaluation_query_save(
        name="mcp_test_query",
        description="Test query saved via MCP",
        sql="SELECT world, COUNT(*) AS n FROM dim_episodes GROUP BY world",
        title="MCP Test Query",
    )
    assert save_res["status"] == "saved"
    assert save_res["name"] == "mcp_test_query"

    # Verify saved query is executable via evaluation_query
    exec_saved = evaluation_query(run_id="run-wp5-002", stored_query="mcp_test_query")
    assert exec_saved["row_count"] == 1

    for s_dir in QueryCatalog().search_dirs:
        test_file = s_dir / "mcp_test_query.yaml"
        if test_file.is_file():
            test_file.unlink()


def test_mcp_query_error_handling(
    tmp_path: pathlib.Path,
    sample_star_tables: StarSchemaTables,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify error handling on invalid queries, forbidden keywords, and conflicting options."""
    run_dir = tmp_path / "data" / "benchmarks" / "run-wp5-003"
    run_dir.mkdir(parents=True, exist_ok=True)
    from arena_evaluation.storage.star_schema import write_star_schema

    write_star_schema(sample_star_tables, run_dir)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    # Error: neither query nor stored_query
    with pytest.raises(ToolError, match="Must provide either 'query' or 'stored_query'"):
        evaluation_query(run_id="run-wp5-003")

    # Error: both query and stored_query
    with pytest.raises(ToolError, match="Cannot provide both 'query' and 'stored_query'"):
        evaluation_query(run_id="run-wp5-003", query="SELECT 1", stored_query="blame_attribution_breakdown")

    # Error: non-existent stored query
    with pytest.raises(ToolError, match="Stored query 'non_existent_query' not found"):
        evaluation_query(run_id="run-wp5-003", stored_query="non_existent_query")

    # Error: forbidden SQL keywords (read-only guard)
    with pytest.raises(ToolError, match="Invalid SQL query"):
        evaluation_query(run_id="run-wp5-003", query="DROP TABLE dim_episodes")

    # Error: evidence_id not found
    with pytest.raises(ToolError, match="Evidence 'ev-nonexistent' not found"):
        evaluation_evidence(run_id="run-wp5-003", evidence_id="ev-nonexistent")
