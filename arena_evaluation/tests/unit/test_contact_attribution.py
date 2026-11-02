from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.contact_attribution import (
    ContactAttributionCalculator,
    classify_closer_label,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def test_classify_closer_label_thresholds() -> None:
    label_static, s_static = classify_closer_label(0.8, 0.0, obstacle_kind="wall")
    assert label_static == "static"
    assert s_static == pytest.approx(1.0)

    label_ped, s_ped = classify_closer_label(0.02, 1.0, obstacle_kind="pedestrian")
    assert label_ped == "pedestrian"
    assert s_ped < 0.05

    label_rob, s_rob = classify_closer_label(1.2, 0.01, obstacle_kind="pedestrian")
    assert label_rob == "robot"
    assert s_rob > 0.95

    label_mut, s_mut = classify_closer_label(0.6, 0.6, obstacle_kind="pedestrian")
    assert label_mut == "mutual"
    assert s_mut == pytest.approx(0.5)

    label_dom_r, _ = classify_closer_label(0.75, 0.25, obstacle_kind="pedestrian")
    assert label_dom_r == "robot"

    label_dom_p, _ = classify_closer_label(0.25, 0.75, obstacle_kind="pedestrian")
    assert label_dom_p == "pedestrian"


def test_stationary_robot_struck_by_pedestrian_in_blind_regime_is_not_robot_fault() -> None:
    dt = 0.05
    t = np.arange(0.0, 3.0, dt)
    time_ns = (t * 1e9).astype(np.int64)
    odom = pl.DataFrame(
        {
            "time_ns": time_ns,
            "pos_x": np.zeros_like(t),
            "pos_y": np.zeros_like(t),
            "yaw": np.zeros_like(t),
            "crowd_condition": ["blind"] * len(t),
        }
    )
    ped_x = 2.0 - 1.0 * t
    peds = pl.DataFrame(
        {
            "time_ns": time_ns,
            "peds_ids": [[10] for _ in t],
            "peds_names": [["Pedestrian_10"] for _ in t],
            "peds_positions": [[float(px), 0.0, 0.0] for px in ped_x],
            "peds_twists": [[-1.0, 0.0, 0.0] for _ in t],
            "peds_radii": [[0.3] for _ in t],
        }
    )
    ep = AlignedEpisodeBundle(
        episode_id="ep_passive_hit",
        robot_name="jackal",
        data=odom,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[0.0, 0.0, 0.0],
        topics={"odom": odom, "peds": peds},
    )

    calc = ContactAttributionCalculator(RobotParams(robot_radius=0.3))
    out = calc.calculate(ep, {})

    assert out["contact_count"] == 1
    assert out["closer_label"] == "pedestrian"
    assert out["counterfactual_regime"] == "blind_exact"
    assert out["counterfactual_freeze_persists"] is True
    assert out["is_robot_fault"] is False
    assert out["is_robot_fault_lower_bound"] is None
    assert out["contact_robot_fault_count"] == 0
    assert out["contact_passive_ped_impact_count"] == 1
    assert out["contact_impact_speed_mps"] == pytest.approx(1.0, abs=1e-2)


def test_moving_robot_avoidable_by_freezing_is_robot_fault_and_respects_reactive_regime() -> None:
    dt = 0.05
    t = np.arange(0.0, 3.0, dt)
    time_ns = (t * 1e9).astype(np.int64)
    rx = 1.2 * t
    odom = pl.DataFrame(
        {
            "time_ns": time_ns,
            "pos_x": rx,
            "pos_y": np.zeros_like(t),
            "yaw": np.zeros_like(t),
            "crowd_condition": ["visible"] * len(t),
        }
    )
    peds = pl.DataFrame(
        {
            "time_ns": time_ns,
            "peds_ids": [[5] for _ in t],
            "peds_names": [["Pedestrian_5"] for _ in t],
            "peds_positions": [[2.0, 0.0, 0.0] for _ in t],
            "peds_twists": [[0.0, 0.0, 0.0] for _ in t],
            "peds_radii": [[0.3] for _ in t],
        }
    )
    ep = AlignedEpisodeBundle(
        episode_id="ep_robot_ram",
        robot_name="jackal",
        data=odom,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[3.0, 0.0, 0.0],
        topics={"odom": odom, "peds": peds},
    )

    calc = ContactAttributionCalculator(RobotParams(robot_radius=0.3))
    out = calc.calculate(ep, {})

    assert out["contact_count"] == 1
    assert out["closer_label"] == "robot"
    assert out["counterfactual_regime"] == "reactive_approx"
    assert out["counterfactual_freeze_persists"] is False
    assert out["is_robot_fault"] is None
    assert out["is_robot_fault_lower_bound"] is True
    assert out["contact_impact_speed_mps"] == pytest.approx(1.2, abs=1e-2)


def test_multi_contact_episode_attributes_every_contact_including_static() -> None:
    dt = 0.05
    t = np.arange(0.0, 6.0, dt)
    time_ns = (t * 1e9).astype(np.int64)
    rx = 1.0 * t
    odom = pl.DataFrame(
        {
            "time_ns": time_ns,
            "pos_x": rx,
            "pos_y": np.zeros_like(t),
            "yaw": np.zeros_like(t),
            "crowd_condition": ["blind"] * len(t),
        }
    )
    peds = pl.DataFrame(
        {
            "time_ns": time_ns,
            "peds_ids": [[1, 2] for _ in t],
            "peds_names": [["Pedestrian_1", "Pedestrian_2"] for _ in t],
            "peds_positions": [[1.5, 0.0, 0.0, 3.5, 0.0, 0.0] for _ in t],
            "peds_twists": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in t],
            "peds_radii": [[0.3, 0.3] for _ in t],
        }
    )
    contacts = pl.DataFrame(
        {
            "time_ns": np.array([int(1.0 * 1e9), int(3.0 * 1e9), int(5.0 * 1e9)], dtype=np.int64),
            "stamp_ns": np.array([int(1.0 * 1e9), int(3.0 * 1e9), int(5.0 * 1e9)], dtype=np.int64),
            "kind": ["pedestrian", "pedestrian", "wall"],
            "obstacle_id": ["env_0/Pedestrian_1", "env_0/Pedestrian_2", "wall_north"],
            "obstacle_x": [1.5, 3.5, 5.5],
            "obstacle_y": [0.0, 0.0, 0.0],
            "obstacle_z": [0.0, 0.0, 0.0],
            "distance": [0.0, 0.0, 0.0],
        }
    )
    ep = AlignedEpisodeBundle(
        episode_id="ep_multi_contact",
        robot_name="jackal",
        data=odom,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[6.0, 0.0, 0.0],
        topics={"odom": odom, "peds": peds, "collision_contacts": contacts},
    )

    calc = ContactAttributionCalculator(RobotParams(robot_radius=0.3))
    out = calc.calculate(ep, {})

    assert out["contact_count"] == 3
    assert out["contact_closer_robot_count"] == 2
    assert out["contact_closer_static_count"] == 1
    records = out["contact_records"]
    assert len(records) == 3
    assert records[0]["target_agent_name"] == "Pedestrian_1"
    assert records[1]["target_agent_name"] == "Pedestrian_2"
    assert records[2]["closer_label"] == "static"
    assert records[2]["counterfactual_regime"] == "unavailable"
    assert records[2]["counterfactual_freeze_min_dist_m"] is None
    assert records[2]["is_robot_fault"] is None
