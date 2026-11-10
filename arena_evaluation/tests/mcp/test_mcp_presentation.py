"""Integration and unit tests for MCP Presentation Gateway (Layer 5).

Verifies:
1. `evaluation_plot_declarations`: Recipe discovery and catalog indexing.
2. `evaluation_read_plot_declaration`: Reading YAML text and parsed PlotSpec.
3. `evaluation_save_plot_declaration`: Schema validation and declaration persistence.
4. `evaluation_render_plot`: Standalone visual artifact rendering (Plotly HTML,
   Seaborn PNG/SVG/PDF) from recipes and inline spec YAML with quality diagnostics.
5. Strict contract invariants and robust error handling.
"""

from __future__ import annotations

import pathlib
from typing import Final

import polars as pl
import pytest

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError

from arena_evaluation.mcp.presentation import (
    evaluation_plot_declarations,
    evaluation_read_plot_declaration,
    evaluation_render_plot,
    evaluation_save_plot_declaration,
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
    write_star_schema,
)

_PNG_MAGIC: Final[bytes] = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def mcp_star_tables() -> StarSchemaTables:
    """Build a synthetic StarSchemaTables fixture with episodes, encounters, and contacts."""
    run_id = "run-mcp-pres"

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
                "planner": "mppi",
                "inter_planner": "none",
                "dispatcher": "cbs",
                "robot_model": "jackal",
                "crowd_condition": "blind",
                "ped_policy_default": "sfm",
                "crowd_size": 2,
                "seed": 43,
                "is_reference": False,
                "reference_type": None,
                "duration_s": 11.5,
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
                "path_length_m": 8.0,
                "path_efficiency": 0.88,
                "mean_speed_mps": 0.8,
                "max_speed_mps": 1.1,
                "mean_accel_mps2": 0.3,
                "mean_jerk_mps3": 0.9,
                "energy_total_j": 400.0,
                "energy_wh": 0.111,
                "acoustic_energy_proxy_db": 60.0,
                "near_miss_count": 1,
                "contact_count": 0,
                "min_pedestrian_clearance_m": 1.0,
                "mean_pedestrian_clearance_m": 1.6,
                "time_under_0_5m_s": 0.0,
                "time_under_1_0m_s": 1.0,
                "time_under_2_0m_s": 4.0,
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
                "time_to_goal_s": 11.5,
                "path_length_m": 7.6,
                "path_efficiency": 0.74,
                "mean_speed_mps": 0.66,
                "max_speed_mps": 0.95,
                "mean_accel_mps2": 0.4,
                "mean_jerk_mps3": 1.1,
                "energy_total_j": 500.0,
                "energy_wh": 0.138,
                "acoustic_energy_proxy_db": 66.0,
                "near_miss_count": 2,
                "contact_count": 1,
                "min_pedestrian_clearance_m": 0.15,
                "mean_pedestrian_clearance_m": 0.9,
                "time_under_0_5m_s": 1.8,
                "time_under_1_0m_s": 3.2,
                "time_under_2_0m_s": 5.5,
                "probst_R_robot": 0.75,
                "probst_R_actor2": 0.25,
                "probst_E_robot": 0.65,
                "success_no_robot_fault": False,
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
                "start_t_ns": 1_000_000_000,
                "end_t_ns": 3_500_000_000,
                "duration_s": 2.5,
                "min_clearance_m": 1.0,
                "mean_clearance_m": 1.4,
                "relative_speed_at_min_mps": 1.3,
                "closing_speed_max_mps": 1.5,
                "encounter_energy_j": 100.0,
                "baseline_energy_j": 70.0,
                "energy_surge_ratio": 1.43,
                "path_deviation_m": 0.25,
                "detour_time_s": 0.4,
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
                "start_t_ns": 2_000_000_000,
                "end_t_ns": 5_500_000_000,
                "duration_s": 3.5,
                "min_clearance_m": 0.15,
                "mean_clearance_m": 0.7,
                "relative_speed_at_min_mps": 1.1,
                "closing_speed_max_mps": 1.7,
                "encounter_energy_j": 190.0,
                "baseline_energy_j": 85.0,
                "energy_surge_ratio": 2.23,
                "path_deviation_m": 0.7,
                "detour_time_s": 1.5,
                "had_contact": True,
                "probst_R_robot": 0.75,
                "probst_R_actor2": 0.25,
                "probst_E_robot": 0.65,
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
                "timestamp_ns": 4_000_000_000,
                "relative_speed_mps": 1.05,
                "robot_speed_mps": 0.75,
                "actor2_speed_mps": 0.35,
                "robot_closing_share": 0.72,
                "closer_label": "robot_closer",
                "counterfactual_freeze_persists": False,
                "fault_attributed_to": "robot",
                "blame_confidence": 0.82,
                "impact_speed_mps": 1.05,
            }
        ],
        schema=FACT_CONTACTS_SCHEMA,
    )

    return StarSchemaTables(
        dim_episodes=dim_episodes,
        dim_robots=pl.DataFrame([], schema=DIM_ROBOTS_SCHEMA),
        dim_pedestrians=pl.DataFrame([], schema=DIM_PEDESTRIANS_SCHEMA),
        dim_world_entities=pl.DataFrame([], schema=DIM_WORLD_ENTITIES_SCHEMA),
        fact_episodes=fact_episodes,
        fact_robots=pl.DataFrame([], schema=FACT_ROBOTS_SCHEMA),
        fact_pedestrians=pl.DataFrame([], schema=FACT_PEDESTRIANS_SCHEMA),
        fact_encounters=fact_encounters,
        fact_contacts=fact_contacts,
    )


def test_mcp_evaluation_plot_declarations() -> None:
    declarations = evaluation_plot_declarations()
    assert isinstance(declarations, list)
    assert len(declarations) >= 6

    ids = {d["id"] for d in declarations}
    assert "closer_attribution_matrix" in ids
    assert "scorecard_table" in ids
    assert "multi_planner_radar" in ids
    assert "planner_ranking_stability" in ids

    for item in declarations:
        assert "title" in item
        assert "type" in item
        assert "encodings" in item
        assert "options" in item


def test_mcp_evaluation_read_plot_declaration() -> None:
    res = evaluation_read_plot_declaration(name="closer_attribution_matrix")
    assert res["name"] == "closer_attribution_matrix"
    assert isinstance(res["yaml"], str)
    assert isinstance(res["spec"], dict)
    assert res["spec"]["type"] == "bar"

    with pytest.raises(ToolError, match="not found"):
        evaluation_read_plot_declaration(name="unknown_non_existent_recipe")


def test_mcp_evaluation_save_plot_declaration(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point default search dir to tmp_path
    monkeypatch.setattr(
        "arena_evaluation.presentation.resolver._DEFAULT_RECIPE_SEARCH_DIRS",
        (tmp_path,),
    )
    monkeypatch.setattr(
        "arena_evaluation.mcp.presentation._DEFAULT_RECIPE_SEARCH_DIRS",
        (tmp_path,),
    )

    new_recipe_yaml = "id: mcp_custom_density_bar\ntitle: \"Encounter Density Profile Across Planners\"\ntype: bar\ndata:\n  stored_query: near_miss_encounter_spikes\nchannels:\n  x: approach_type\n  y: mean_energy_surge_ratio\n  color: planner\noptions:\n  barmode: group\n"

    save_res = evaluation_save_plot_declaration(
        name="mcp_custom_density_bar",
        yaml_text=new_recipe_yaml,
    )
    assert save_res["status"] == "saved"
    assert save_res["name"] == "mcp_custom_density_bar"
    assert save_res["id"] == "mcp_custom_density_bar"
    assert save_res["type"] == "bar"

    # Verify readable via read tool
    read_res = evaluation_read_plot_declaration("mcp_custom_density_bar")
    assert read_res["spec"]["id"] == "mcp_custom_density_bar"
    assert read_res["spec"]["type"] == "bar"

    # Error case: Invalid YAML
    with pytest.raises(ToolError, match="not valid YAML"):
        evaluation_save_plot_declaration(name="invalid_yaml", yaml_text="foo: [bar: 1")


def test_mcp_evaluation_render_plot_recipe(
    tmp_path: pathlib.Path,
    mcp_star_tables: StarSchemaTables,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmarks_dir = tmp_path / "data" / "benchmarks"
    run_dir = benchmarks_dir / "run-pres-001"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_star_schema(mcp_star_tables, run_dir)

    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    # Render recipe to PNG
    res_png = evaluation_render_plot(
        run_id="run-pres-001",
        declaration="closer_attribution_matrix",
        output_format="png",
    )
    assert res_png["format"] == "png"
    assert res_png["spec_id"] == "closer_attribution_matrix"
    assert "diagnostics" in res_png
    png_path = pathlib.Path(res_png["plot_path"])
    assert png_path.exists() and png_path.stat().st_size > 0
    assert png_path.read_bytes()[:8] == _PNG_MAGIC

    # Render recipe to HTML (Plotly standalone)
    res_html = evaluation_render_plot(
        run_id="run-pres-001",
        declaration="closer_attribution_matrix",
        output_format="html",
    )
    assert res_html["format"] == "html"
    html_path = pathlib.Path(res_html["plot_path"])
    assert html_path.exists() and html_path.stat().st_size > 0
    html_text = html_path.read_text(encoding="utf-8")
    assert "plotly" in html_text.lower()
    assert "Collision Fault Attribution" in html_text or "closer_label" in html_text

    # Render recipe to SVG (Vector)
    res_svg = evaluation_render_plot(
        run_id="run-pres-001",
        declaration="scorecard_table",
        output_format="svg",
    )
    assert res_svg["format"] == "svg"
    svg_path = pathlib.Path(res_svg["plot_path"])
    assert svg_path.exists() and svg_path.stat().st_size > 0
    assert "<svg" in svg_path.read_text(encoding="utf-8")


def test_mcp_evaluation_render_plot_inline_yaml_and_overrides(
    tmp_path: pathlib.Path,
    mcp_star_tables: StarSchemaTables,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmarks_dir = tmp_path / "data" / "benchmarks"
    run_dir = benchmarks_dir / "run-pres-002"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_star_schema(mcp_star_tables, run_dir)

    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    inline_spec = """
    id: inline_scorecard_bar
    title: "Scorecard Mean Time to Goal"
    type: bar
    data:
      stored_query: planner_scorecard_overview
    channels:
      x: planner
      y: mean_time_to_goal_s
    """

    res = evaluation_render_plot(
        run_id="run-pres-002",
        spec_yaml=inline_spec,
        overrides={"title": "Custom Overridden Scorecard Title"},
        output_format="png",
    )
    assert res["spec_id"] == "inline_scorecard_bar"
    out_file = pathlib.Path(res["plot_path"])
    assert out_file.exists() and out_file.stat().st_size > 0


def test_mcp_evaluation_render_plot_error_handling(
    tmp_path: pathlib.Path,
    mcp_star_tables: StarSchemaTables,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmarks_dir = tmp_path / "data" / "benchmarks"
    run_dir = benchmarks_dir / "run-pres-003"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_star_schema(mcp_star_tables, run_dir)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    # Error: Neither declaration nor spec_yaml
    with pytest.raises(ToolError, match="Must provide either 'declaration' or 'spec_yaml'"):
        evaluation_render_plot(run_id="run-pres-003")

    # Error: Both declaration and spec_yaml
    with pytest.raises(ToolError, match="Cannot provide both 'declaration' and 'spec_yaml'"):
        evaluation_render_plot(run_id="run-pres-003", declaration="foo", spec_yaml="bar")

    # Error: Non-existent declaration
    with pytest.raises(ToolError, match="Plot declaration 'non_existent_rec' not found"):
        evaluation_render_plot(run_id="run-pres-003", declaration="non_existent_rec")


def test_mcp_evaluation_render_plot_zero_row_diagnostic(
    tmp_path: pathlib.Path,
    mcp_star_tables: StarSchemaTables,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendering a plot whose query returns 0 rows raises a clear ToolError diagnostic."""
    benchmarks_dir = tmp_path / "data" / "benchmarks"
    run_dir = benchmarks_dir / "run-pres-empty"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_star_schema(mcp_star_tables, run_dir)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    spec_with_empty_filter = """
    id: empty_filter_plot
    title: "Empty Filter Plot"
    type: bar
    data:
      source: v_episode_summary
      where: "planner = 'non_existent_planner'"
    channels:
      x: planner
      y: time_to_goal_s
    """

    with pytest.raises(ToolError) as exc_info:
        evaluation_render_plot(
            run_id="run-pres-empty",
            spec_yaml=spec_with_empty_filter,
            output_format="png",
        )

    err = exc_info.value
    assert "returned 0 rows; cannot render plot with an empty dataset" in str(err)
    assert "Verify filter conditions or run 'evaluation_query' to check row counts." in (err.hint or "")
