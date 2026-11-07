"""Unit tests for Presentation Engine Simplification & Query Store Binding (WP 5.2).

Verifies:
1. PlotDataSpec and PlotSpec schema coercion with `stored_query` and `params`.
2. DataResolver execution of stored SQL queries via QueryCatalog and PolarsSQLEngine.
3. Direct, unambiguous channel bindings (x, y, color, facet_col) across plot renderers
   without heuristic guessing.
4. Publication-quality rendering of both interactive Plotly HTML and static Seaborn/Matplotlib
   outputs (PNG, SVG, PDF).
"""

from __future__ import annotations

import pathlib
from typing import Final

import polars as pl
import pytest
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.presentation.plot_types.bar import BarRenderer
from arena_evaluation.presentation.plot_types.box import BoxRenderer
from arena_evaluation.presentation.plot_types.heatmap import HeatmapRenderer
from arena_evaluation.presentation.plot_types.histogram import HistogramRenderer
from arena_evaluation.presentation.plot_types.line import LineRenderer
from arena_evaluation.presentation.plot_types.radar import RadarRenderer
from arena_evaluation.presentation.plot_types.scatter import ScatterRenderer
from arena_evaluation.presentation.plot_types.table import TableRenderer
from arena_evaluation.presentation.plot_types.timeseries import TimeseriesRenderer
from arena_evaluation.presentation.plot_types.trajectory import TrajectoryRenderer
from arena_evaluation.presentation.plot_types.violin import ViolinRenderer
from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.storage.schemas import PlotChannels, PlotDataSpec, PlotSpec
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

_PNG_MAGIC: Final[bytes] = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def sample_star_tables() -> StarSchemaTables:
    """Build a synthetic StarSchemaTables fixture with episodes, encounters, and contacts."""
    run_id = "test-run-task2"

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
                "robot_model": "jackal",
                "crowd_condition": "visible",
                "ped_policy_default": "sfm",
                "crowd_size": 2,
                "seed": 44,
                "is_reference": False,
                "reference_type": None,
                "duration_s": 8.5,
                "outcome_state": 2,
                "outcome_info": "finished",
            },
        ],
        schema=DIM_EPISODES_SCHEMA,
    )

    fact_episodes = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 0,
                "robot_id": "robot_0",
                "success": True,
                "collision": False,
                "timeout": False,
                "time_to_goal_s": 10.0,
                "path_length_m": 8.2,
                "path_efficiency": 0.85,
                "mean_speed_mps": 0.82,
                "max_speed_mps": 1.1,
                "mean_accel_mps2": 0.3,
                "mean_jerk_mps3": 0.9,
                "energy_total_j": 450.0,
                "energy_wh": 0.125,
                "acoustic_energy_proxy_db": 62.0,
                "near_miss_count": 1,
                "contact_count": 0,
                "min_pedestrian_clearance_m": 0.9,
                "mean_pedestrian_clearance_m": 1.5,
                "time_under_0_5m_s": 0.0,
                "time_under_1_0m_s": 1.2,
                "time_under_2_0m_s": 4.5,
                "probst_R_robot": 0.1,
                "probst_R_actor2": 0.1,
                "probst_E_robot": 0.1,
                "success_no_robot_fault": True,
            },
            {
                "run_id": run_id,
                "episode_id": 1,
                "robot_id": "robot_0",
                "success": False,
                "collision": True,
                "timeout": False,
                "time_to_goal_s": 12.0,
                "path_length_m": 7.5,
                "path_efficiency": 0.72,
                "mean_speed_mps": 0.65,
                "max_speed_mps": 0.9,
                "mean_accel_mps2": 0.4,
                "mean_jerk_mps3": 1.2,
                "energy_total_j": 520.0,
                "energy_wh": 0.144,
                "acoustic_energy_proxy_db": 68.0,
                "near_miss_count": 2,
                "contact_count": 1,
                "min_pedestrian_clearance_m": 0.1,
                "mean_pedestrian_clearance_m": 0.8,
                "time_under_0_5m_s": 2.1,
                "time_under_1_0m_s": 3.8,
                "time_under_2_0m_s": 6.2,
                "probst_R_robot": 0.8,
                "probst_R_actor2": 0.2,
                "probst_E_robot": 0.7,
                "success_no_robot_fault": False,
            },
            {
                "run_id": run_id,
                "episode_id": 2,
                "robot_id": "robot_0",
                "success": True,
                "collision": False,
                "timeout": False,
                "time_to_goal_s": 8.5,
                "path_length_m": 8.0,
                "path_efficiency": 0.92,
                "mean_speed_mps": 0.94,
                "max_speed_mps": 1.2,
                "mean_accel_mps2": 0.25,
                "mean_jerk_mps3": 0.7,
                "energy_total_j": 380.0,
                "energy_wh": 0.105,
                "acoustic_energy_proxy_db": 58.0,
                "near_miss_count": 0,
                "contact_count": 0,
                "min_pedestrian_clearance_m": 1.4,
                "mean_pedestrian_clearance_m": 2.1,
                "time_under_0_5m_s": 0.0,
                "time_under_1_0m_s": 0.0,
                "time_under_2_0m_s": 2.8,
                "probst_R_robot": 0.05,
                "probst_R_actor2": 0.05,
                "probst_E_robot": 0.05,
                "success_no_robot_fault": True,
            },
        ],
        schema=FACT_EPISODES_SCHEMA,
    )

    fact_encounters = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 0,
                "robot_id": "robot_0",
                "encounter_id": "enc_0_0",
                "actor2_id": "ped_1",
                "approach_type": "head_on",
                "start_time_ns": 1_000_000_000,
                "end_time_ns": 4_000_000_000,
                "duration_s": 3.0,
                "min_clearance_m": 0.9,
                "mean_clearance_m": 1.3,
                "relative_speed_at_min_mps": 1.4,
                "closing_speed_max_mps": 1.6,
                "encounter_energy_j": 120.0,
                "baseline_energy_j": 80.0,
                "energy_surge_ratio": 1.5,
                "path_deviation_m": 0.3,
                "detour_time_s": 0.5,
                "had_contact": False,
                "probst_R_robot": 0.1,
                "probst_R_actor2": 0.1,
                "probst_E_robot": 0.1,
            },
            {
                "run_id": run_id,
                "episode_id": 1,
                "robot_id": "robot_0",
                "encounter_id": "enc_1_0",
                "actor2_id": "ped_2",
                "approach_type": "crossing",
                "start_time_ns": 2_000_000_000,
                "end_time_ns": 6_000_000_000,
                "duration_s": 4.0,
                "min_clearance_m": 0.1,
                "mean_clearance_m": 0.6,
                "relative_speed_at_min_mps": 1.2,
                "closing_speed_max_mps": 1.8,
                "encounter_energy_j": 210.0,
                "baseline_energy_j": 90.0,
                "energy_surge_ratio": 2.33,
                "path_deviation_m": 0.8,
                "detour_time_s": 1.8,
                "had_contact": True,
                "probst_R_robot": 0.8,
                "probst_R_actor2": 0.2,
                "probst_E_robot": 0.7,
            },
        ],
        schema=FACT_ENCOUNTERS_SCHEMA,
    )

    fact_contacts = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 1,
                "robot_id": "robot_0",
                "contact_id": "cnt_1_0",
                "encounter_id": "enc_1_0",
                "actor2_id": "ped_2",
                "timestamp_ns": 4_500_000_000,
                "relative_speed_mps": 1.1,
                "robot_speed_mps": 0.8,
                "actor2_speed_mps": 0.4,
                "robot_closing_share": 0.75,
                "closer_label": "robot_closer",
                "counterfactual_freeze_persists": False,
                "fault_attributed_to": "robot",
                "blame_confidence": 0.85,
                "impact_speed_mps": 1.1,
            }
        ],
        schema=FACT_CONTACTS_SCHEMA,
    )

    empty_robots = pl.DataFrame([], schema=DIM_ROBOTS_SCHEMA)
    empty_peds = pl.DataFrame([], schema=DIM_PEDESTRIANS_SCHEMA)
    empty_world = pl.DataFrame([], schema=DIM_WORLD_ENTITIES_SCHEMA)
    empty_fact_robots = pl.DataFrame([], schema=FACT_ROBOTS_SCHEMA)
    empty_fact_peds = pl.DataFrame([], schema=FACT_PEDESTRIANS_SCHEMA)

    return StarSchemaTables(
        dim_episodes=dim_episodes,
        dim_robots=empty_robots,
        dim_pedestrians=empty_peds,
        dim_world_entities=empty_world,
        fact_episodes=fact_episodes,
        fact_robots=empty_fact_robots,
        fact_pedestrians=empty_fact_peds,
        fact_encounters=fact_encounters,
        fact_contacts=fact_contacts,
    )


def test_plot_spec_stored_query_coercion() -> None:
    # 1. Direct PlotDataSpec construction
    data_spec = PlotDataSpec(stored_query="blame_attribution_breakdown", params={"min_speed": 0.2})
    assert data_spec.stored_query == "blame_attribution_breakdown"
    assert data_spec.params == {"min_speed": 0.2}

    # 2. Coercion from nested dictionary
    spec1 = PlotSpec.model_validate(
        {
            "id": "closer_test",
            "type": "bar",
            "data": {
                "stored_query": "blame_attribution_breakdown",
                "params": {"crowd_condition": "blind"},
            },
            "channels": {"x": "planner", "y": "total_contacts"},
        }
    )
    assert spec1.data is not None
    assert spec1.data.stored_query == "blame_attribution_breakdown"
    assert spec1.data.params == {"crowd_condition": "blind"}

    # 3. Coercion from top-level stored_query & params (ergonomic authoring)
    spec2 = PlotSpec.model_validate(
        {
            "id": "closer_top_level",
            "type": "bar",
            "stored_query": "blame_attribution_breakdown",
            "params": {"planner": "dwb-standard"},
            "channels": {"x": "planner", "y": "total_contacts"},
        }
    )
    assert spec2.data is not None
    assert spec2.data.stored_query == "blame_attribution_breakdown"
    assert spec2.data.params == {"planner": "dwb-standard"}

    # 4. YAML round-trip
    yaml_text = """
    id: yaml_test
    type: table
    stored_query: planner_scorecard_overview
    params:
      world: hospital_1
    channels:
      x: planner
      y: [success_rate, mean_energy_wh]
    """
    spec_yaml = PlotSpec.model_validate(yaml.safe_load(yaml_text))
    assert spec_yaml.data is not None
    assert spec_yaml.data.stored_query == "planner_scorecard_overview"
    assert spec_yaml.data.params == {"world": "hospital_1"}


def test_data_resolver_stored_query_execution(tmp_path: pathlib.Path, sample_star_tables: StarSchemaTables) -> None:
    catalog = BenchmarkCatalog(sample_star_tables)
    sql_engine = PolarsSQLEngine(catalog)
    resolver = DataResolver(catalog=catalog, sql_engine=sql_engine, benchmark_dir=tmp_path)

    # Execute stored query: blame_attribution_breakdown
    spec = PlotSpec(
        id="blame_matrix",
        type="bar",
        data=PlotDataSpec(stored_query="blame_attribution_breakdown", params={"crowd_condition": "all"}),
        channels=PlotChannels(x="planner", y="total_contacts", color="closer_label"),
    )

    df_resolved, diagnostics = resolver.resolve(spec)
    assert len(df_resolved) == 1
    assert "planner" in df_resolved.columns
    assert "closer_label" in df_resolved.columns
    assert "total_contacts" in df_resolved.columns
    assert df_resolved["closer_label"][0] == "robot_closer"
    assert df_resolved["total_contacts"][0] == 1
    assert diagnostics.row_count == 1

    # Execute stored query with filter parameter: crowd_condition='visible' (should yield 0 rows for contact)
    spec_filtered = PlotSpec(
        id="blame_matrix_vis",
        type="bar",
        data=PlotDataSpec(stored_query="blame_attribution_breakdown", params={"crowd_condition": "visible"}),
        channels=PlotChannels(x="planner", y="total_contacts"),
    )
    df_filtered, _ = resolver.resolve(spec_filtered)
    assert len(df_filtered) == 0


def test_bar_renderer_with_query_store_data(tmp_path: pathlib.Path, sample_star_tables: StarSchemaTables) -> None:
    catalog = BenchmarkCatalog(sample_star_tables)
    sql_engine = PolarsSQLEngine(catalog)
    resolver = DataResolver(catalog=catalog, sql_engine=sql_engine, benchmark_dir=tmp_path)

    spec = PlotSpec(
        id="closer_attribution_matrix",
        type="bar",
        data=PlotDataSpec(stored_query="blame_attribution_breakdown"),
        channels=PlotChannels(
            x="planner",
            y="total_contacts",
            color="closer_label",
            facet_col="crowd_condition",
        ),
        options={"barmode": "stack"},
    )

    df, diagnostics = resolver.resolve(spec)
    renderer = BarRenderer(spec)

    # Plotly interactive HTML
    html = renderer.render_plotly(df)
    assert html is not None
    assert html.startswith("<div")
    assert "dwb-standard" in html

    # Seaborn static outputs (PNG, SVG, PDF)
    out_png = tmp_path / "bar.png"
    out_svg = tmp_path / "bar.svg"
    out_pdf = tmp_path / "bar.pdf"
    renderer.render_seaborn(df, out_png)
    renderer.render_seaborn(df, out_svg)
    renderer.render_seaborn(df, out_pdf)

    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_png.read_bytes()[:8] == _PNG_MAGIC
    assert out_svg.exists() and out_svg.stat().st_size > 0
    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_pdf.read_bytes()[:4] == b"%PDF"


def test_table_renderer_with_query_store_data(tmp_path: pathlib.Path, sample_star_tables: StarSchemaTables) -> None:
    catalog = BenchmarkCatalog(sample_star_tables)
    sql_engine = PolarsSQLEngine(catalog)
    resolver = DataResolver(catalog=catalog, sql_engine=sql_engine, benchmark_dir=tmp_path)

    spec = PlotSpec(
        id="scorecard_table",
        type="table",
        data=PlotDataSpec(stored_query="planner_scorecard_overview"),
        channels=PlotChannels(
            x="planner",
            y=["success_rate", "mean_time_to_goal_s", "mean_path_efficiency", "mean_energy_wh"],
        ),
        options={"format": "{:.2f}"},
    )

    df, diagnostics = resolver.resolve(spec)
    renderer = TableRenderer(spec)

    html = renderer.render_plotly(df)
    assert html is not None
    assert "<table" in html
    assert "dwb-standard" in html
    assert "mppi" in html

    out_png = tmp_path / "scorecard.png"
    renderer.render_seaborn(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_png.read_bytes()[:8] == _PNG_MAGIC


def test_radar_renderer_with_star_schema_view(tmp_path: pathlib.Path, sample_star_tables: StarSchemaTables) -> None:
    catalog = BenchmarkCatalog(sample_star_tables)
    sql_engine = PolarsSQLEngine(catalog)
    resolver = DataResolver(catalog=catalog, sql_engine=sql_engine, benchmark_dir=tmp_path)

    spec = PlotSpec(
        id="multi_planner_radar",
        type="radar",
        data=PlotDataSpec(source="v_planner_scorecard"),
        channels=PlotChannels(
            color="planner",
            y=["mean_path_efficiency", "mean_time_to_goal_s", "success_rate", "mean_energy_wh"],
        ),
    )

    df, diagnostics = resolver.resolve(spec)
    renderer = RadarRenderer(spec)

    html = renderer.render_plotly(df)
    assert html is not None
    assert html.startswith("<div")

    out_png = tmp_path / "radar.png"
    renderer.render_seaborn(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_heatmap_2d_pivot_and_correlation(tmp_path: pathlib.Path) -> None:
    df = pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "mppi", "mppi"],
            "crowd_condition": ["sparse", "dense", "sparse", "dense"],
            "composite_rank": [1.0, 2.0, 2.0, 1.0],
        }
    )

    spec_pivot = PlotSpec(
        id="planner_ranking_stability",
        type="heatmap",
        channels=PlotChannels(x="planner", y="crowd_condition", color="composite_rank"),
    )
    renderer_pivot = HeatmapRenderer(spec_pivot)
    html_pivot = renderer_pivot.render_plotly(df)
    assert html_pivot is not None
    assert html_pivot.startswith("<div")

    out_png = tmp_path / "heatmap_pivot.png"
    renderer_pivot.render_seaborn(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0

    # Correlation matrix test
    df_corr = pl.DataFrame(
        {
            "speed": [0.5, 0.8, 1.0, 1.2],
            "accel": [0.2, 0.4, 0.5, 0.7],
            "energy": [50.0, 90.0, 120.0, 150.0],
        }
    )
    spec_corr = PlotSpec(
        id="corr_matrix",
        type="heatmap",
        options={"kind": "correlation", "metrics": ["speed", "accel", "energy"]},
    )
    renderer_corr = HeatmapRenderer(spec_corr)
    html_corr = renderer_corr.render_plotly(df_corr)
    assert html_corr is not None


def test_timeseries_1nf_flat_rows(tmp_path: pathlib.Path) -> None:
    times = [0.0, 1.0, 2.0, 3.0]
    df = pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "dwb", "dwb", "mppi", "mppi", "mppi", "mppi"],
            "time_s": times + times,
            "speed_mps": [0.0, 0.5, 0.8, 0.7, 0.0, 0.6, 1.0, 0.9],
        }
    )

    spec = PlotSpec(
        id="speed_over_time",
        type="timeseries",
        channels=PlotChannels(x="time_s", y="speed_mps", color="planner"),
    )
    renderer = TimeseriesRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None
    assert html.startswith("<div")

    out_png = tmp_path / "timeseries.png"
    renderer.render_seaborn(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_trajectory_rendering_and_formats(tmp_path: pathlib.Path) -> None:
    df = pl.DataFrame(
        {
            "planner": ["dwb", "mppi"],
            "episode_id": [1, 2],
            "path": [
                [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]],
                [[0.0, 0.0], [0.8, 0.8], [2.0, 1.0]],
            ],
            "start": [[0.0, 0.0], [0.0, 0.0]],
            "goal": [[2.0, 1.0], [2.0, 1.0]],
        },
        schema={
            "planner": pl.Utf8,
            "episode_id": pl.Int64,
            "path": pl.List(pl.List(pl.Float64)),
            "start": pl.List(pl.Float64),
            "goal": pl.List(pl.Float64),
        },
    )

    spec = PlotSpec(
        id="traj_test",
        type="trajectory",
        channels=PlotChannels(x="path", color="planner"),
        options={"overlay_markers": True},
    )
    renderer = TrajectoryRenderer(spec)
    html = renderer.render_plotly(df)
    assert html is not None
    assert html.startswith("<div")

    out_png = tmp_path / "traj.png"
    renderer.render_seaborn(df, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_unambiguous_channels_no_heuristic_guessing() -> None:
    """Verify that when channels.color is explicitly declared, resolve_diff_col is bypassed."""
    df = pl.DataFrame(
        {
            "my_color": ["alpha", "beta"],
            "value": [10.0, 20.0],
            "local_planner": ["lp1", "lp2"],
            "inter_planner": ["ip1", "ip2"],
        }
    )

    spec = PlotSpec(
        id="channel_test",
        type="violin",
        channels=PlotChannels(y="value", color="my_color"),
    )
    renderer = ViolinRenderer(spec)
    # The renderer should use 'my_color' directly without generating '__label__'
    html = renderer.render_plotly(df)
    assert html is not None
    assert "__label__" not in html
