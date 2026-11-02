from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.encounter_costs import (
    EncounterCostsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def test_encounter_costs_matches_reference_twin_by_pedestrian_name() -> None:
    dt = 0.1
    t = np.arange(0.0, 5.0 + 1e-9, dt)
    time_ns = (t * 1e9).astype(np.int64)

    rx = 1.0 * t
    odom = pl.DataFrame(
        {
            "time_ns": time_ns,
            "pos_x": rx,
            "pos_y": np.zeros_like(t),
            "yaw": np.zeros_like(t),
        }
    )

    ped_x = 5.0 - 1.0 * t
    ped_y = 1.0 * np.sin(np.pi * t / 5.0)
    peds_dyn = pl.DataFrame(
        {
            "time_ns": time_ns,
            "peds_ids": [[7] for _ in t],
            "peds_names": [["alice"] for _ in t],
            "peds_positions": [[float(px), float(py), 0.0] for px, py in zip(ped_x, ped_y, strict=True)],
            "peds_twists": [[-1.0, 0.0, 0.0] for _ in t],
            "peds_radii": [[0.3] for _ in t],
        }
    )

    peds_ref = pl.DataFrame(
        {
            "time_ns": time_ns,
            "peds_ids": [[99, 42] for _ in t],
            "peds_names": [["bob", "alice"] for _ in t],
            "peds_positions": [[10.0, 10.0, 0.0, float(px), 0.0, 0.0] for px in ped_x],
            "peds_twists": [[0.0, 0.0, 0.0, -1.0, 0.0, 0.0] for _ in t],
            "peds_radii": [[0.3, 0.3] for _ in t],
        }
    )
    ref_ep = AlignedEpisodeBundle(
        episode_id="ep_ref",
        robot_name="jackal",
        data=odom,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0, 0.0],
        topics={"odom": odom, "peds": peds_ref},
    )
    dyn_ep = AlignedEpisodeBundle(
        episode_id="ep_dyn",
        robot_name="jackal",
        data=odom,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0, 0.0],
        topics={"odom": odom, "peds": peds_dyn},
    )

    calc = EncounterCostsCalculator(RobotParams(robot_radius=0.3))
    calc.metrics_config = {"reference_episode": ref_ep, "encounter_distance_m": 10.0}
    out = calc.calculate(dyn_ep, {})

    assert out["encounter_count"] == 1
    assert out["cost_time_s"] == pytest.approx(5.0, abs=1e-6)
    assert out["cost_path_m"] == pytest.approx(5.0, abs=1e-6)
    assert out["cost_energy_j"] > 0.0
    assert out["cost_sound_pa2s"] > 0.0
    assert out["cost_ped_detour_m"] is not None and out["cost_ped_detour_m"] > 0.3
    assert out["cost_ped_deflection_m"] is not None and out["cost_ped_deflection_m"] > 0.4


def test_overlapping_encounters_use_interval_union_for_robot_costs_and_sum_for_ped_costs() -> None:
    dt = 0.1
    t = np.arange(0.0, 4.0 + 1e-9, dt)
    time_ns = (t * 1e9).astype(np.int64)

    rx = 1.0 * t
    odom = pl.DataFrame(
        {
            "time_ns": time_ns,
            "pos_x": rx,
            "pos_y": np.zeros_like(t),
            "yaw": np.zeros_like(t),
        }
    )

    peds = pl.DataFrame(
        {
            "time_ns": time_ns,
            "peds_ids": [[1, 2] for _ in t],
            "peds_names": [["ped_a", "ped_b"] for _ in t],
            "peds_positions": [[float(x_val), 1.0, 0.0, float(x_val), -1.0, 0.0] for x_val in rx],
            "peds_twists": [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0] for _ in t],
            "peds_radii": [[0.3, 0.3] for _ in t],
        }
    )
    ep = AlignedEpisodeBundle(
        episode_id="ep_overlap",
        robot_name="jackal",
        data=odom,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[4.0, 0.0, 0.0],
        topics={"odom": odom, "peds": peds},
    )

    calc = EncounterCostsCalculator(RobotParams(robot_radius=0.3))
    records = calc.calculate_encounters(ep)
    assert len(records) == 2

    rolled = calc.calculate(ep, {})
    assert rolled["encounter_count"] == 2
    assert rolled["cost_time_s"] == pytest.approx(records[0].cost_time_s, abs=1e-6)
    assert rolled["cost_path_m"] == pytest.approx(records[0].cost_path_m, abs=1e-6)
    assert rolled["cost_energy_j"] == pytest.approx(records[0].cost_energy_j, abs=1e-6)
    assert rolled["cost_sound_pa2s"] == pytest.approx(records[0].cost_sound_pa2s + records[1].cost_sound_pa2s, abs=1e-9)
