from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.interaction_common import Trajectory2D
from arena_evaluation.processing.metrics.social.probst_responsibility import (
    ProbstPair,
    ProbstParts,
    ProbstResponsibilityCalculator,
    derive_probst_shares,
    distance_to_closest_encounter,
    evaluate_probst,
    rollup_probst_results,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def _make_pair(
    t: np.ndarray,
    pos_r: np.ndarray,
    vel_r: np.ndarray,
    pos_t: np.ndarray,
    vel_t: np.ndarray,
    r_robot: float = 0.5,
    r_target: float = 0.5,
) -> ProbstPair:
    """Construct a ProbstPair from planar position and velocity arrays."""
    time_ns = (t * 1e9).astype(np.int64)
    robot = Trajectory2D(time_ns, pos_r[:, 0], pos_r[:, 1], vel_r[:, 0], vel_r[:, 1], r_robot)
    target = Trajectory2D(time_ns, pos_t[:, 0], pos_t[:, 1], vel_t[:, 0], vel_t[:, 1], r_target)
    return ProbstPair(robot=robot, target=target)


def _make_episode(
    times_s: np.ndarray,
    robot_xy: np.ndarray,
    ped_tracks: list[tuple[int, str, np.ndarray, np.ndarray, float]],
) -> AlignedEpisodeBundle:
    """Build a real AlignedEpisodeBundle from robot and pedestrian trajectories."""
    time_ns = (times_s * 1e9).astype(np.int64)
    odom = pl.DataFrame(
        {
            "time_ns": time_ns,
            "pos_x": robot_xy[:, 0].astype(np.float64),
            "pos_y": robot_xy[:, 1].astype(np.float64),
            "yaw": np.zeros(len(times_s), dtype=np.float64),
        }
    )

    ped_ids_rows: list[list[int]] = []
    ped_names_rows: list[list[str]] = []
    ped_pos_rows: list[list[float]] = []
    ped_twist_rows: list[list[float]] = []
    ped_radii_rows: list[list[float]] = []

    for idx in range(len(times_s)):
        ids_step: list[int] = []
        names_step: list[str] = []
        pos_step: list[float] = []
        twist_step: list[float] = []
        rad_step: list[float] = []
        for aid, name, p_xy, p_vel, rad in ped_tracks:
            ids_step.append(aid)
            names_step.append(name)
            pos_step.extend([float(p_xy[idx, 0]), float(p_xy[idx, 1]), 0.0])
            twist_step.extend([float(p_vel[idx, 0]), float(p_vel[idx, 1]), 0.0])
            rad_step.append(float(rad))
        ped_ids_rows.append(ids_step)
        ped_names_rows.append(names_step)
        ped_pos_rows.append(pos_step)
        ped_twist_rows.append(twist_step)
        ped_radii_rows.append(rad_step)

    peds = pl.DataFrame(
        {
            "time_ns": time_ns,
            "peds_ids": ped_ids_rows,
            "peds_names": ped_names_rows,
            "peds_positions": ped_pos_rows,
            "peds_twists": ped_twist_rows,
            "peds_radii": ped_radii_rows,
        }
    )
    return AlignedEpisodeBundle(
        episode_id="ep_probst",
        robot_name="jackal",
        data=odom,
        start_pos=[float(robot_xy[0, 0]), float(robot_xy[0, 1]), 0.0],
        goal_pos=[float(robot_xy[-1, 0]), float(robot_xy[-1, 1]), 0.0],
        topics={"odom": odom, "peds": peds},
    )


def test_pdce_head_on_and_diverging() -> None:
    d_head_on = distance_to_closest_encounter(
        (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([10.0, 0.4]), np.array([-1.0, 0.0])),
    )
    assert d_head_on == pytest.approx(0.4, abs=1e-9)

    d_diverging = distance_to_closest_encounter(
        (np.array([0.0, 0.0]), np.array([-1.0, 0.0])),
        (np.array([3.0, 4.0]), np.array([1.0, 0.0])),
    )
    assert d_diverging == pytest.approx(5.0, abs=1e-9)


def test_probst_unresponsive_head_on_assigns_full_responsibility_to_time() -> None:
    dt = 0.1
    t = np.arange(0.0, 6.0, dt)
    pos_robot = np.column_stack((1.0 * t, np.zeros_like(t)))
    vel_robot = np.column_stack((np.ones_like(t), np.zeros_like(t)))
    pos_target = np.column_stack((10.0 - 1.0 * t, np.zeros_like(t)))
    vel_target = np.column_stack((-np.ones_like(t), np.zeros_like(t)))

    res = evaluate_probst(_make_pair(t, pos_robot, vel_robot, pos_target, vel_target))
    assert res.c_total is not None and res.c_total > 0.0
    assert res.r_robot == pytest.approx(0.0, abs=1e-9)
    assert res.r_target == pytest.approx(0.0, abs=1e-9)
    assert res.r_time == pytest.approx(1.0, abs=1e-9)
    assert res.e_robot == pytest.approx(0.0, abs=1e-9)
    assert res.e_target == pytest.approx(0.0, abs=1e-9)
    assert res.e_time == pytest.approx(1.0, abs=1e-9)


def test_probst_unilateral_robot_avoidance_credits_robot_responsibility() -> None:
    dt = 0.1
    t = np.arange(0.0, 6.0, dt)
    vy_robot = np.where((t >= 1.5) & (t <= 3.0), 0.8, 0.0)
    y_robot = np.cumsum(vy_robot * dt)
    pos_robot = np.column_stack((1.0 * t, y_robot))
    vel_robot = np.column_stack((np.ones_like(t), vy_robot))
    pos_target = np.column_stack((8.0 - 1.0 * t, np.zeros_like(t)))
    vel_target = np.column_stack((-np.ones_like(t), np.zeros_like(t)))

    res = evaluate_probst(_make_pair(t, pos_robot, vel_robot, pos_target, vel_target))
    assert res.c_total is not None and res.c_total > 0.0
    assert res.r_robot is not None and res.r_robot > 0.70
    assert res.r_target == pytest.approx(0.0, abs=1e-9)
    assert (res.r_robot + (res.r_target or 0.0) + (res.r_time or 0.0)) == pytest.approx(1.0, abs=1e-9)
    assert ((res.e_robot or 0.0) + (res.e_target or 0.0) + (res.e_time or 0.0)) == pytest.approx(1.0, abs=1e-9)


def test_probst_symmetric_mutual_avoidance_splits_responsibility_equally() -> None:
    dt = 0.1
    t = np.arange(0.0, 6.0, dt)
    vy_r = np.clip((t - 1.5) * 0.3, 0.0, 0.45)
    vy_t = -vy_r
    pos_robot = np.column_stack((1.0 * t, np.cumsum(vy_r * dt)))
    vel_robot = np.column_stack((np.ones_like(t), vy_r))
    pos_target = np.column_stack((8.0 - 1.0 * t, np.cumsum(vy_t * dt)))
    vel_target = np.column_stack((-np.ones_like(t), vy_t))

    res = evaluate_probst(_make_pair(t, pos_robot, vel_robot, pos_target, vel_target))
    assert res.r_robot is not None and res.r_target is not None
    assert res.r_robot == pytest.approx(res.r_target, abs=1e-9)
    assert res.r_robot > 0.35
    assert (res.r_robot + res.r_target + (res.r_time or 0.0)) == pytest.approx(1.0, abs=1e-9)


def test_probst_additive_rollup_rebuilds_shares_from_summed_parts() -> None:
    dt = 0.1
    t = np.arange(0.0, 6.0, dt)
    vy_r = np.where((t >= 1.5) & (t <= 3.0), 0.8, 0.0)
    pos_r1 = np.column_stack((1.0 * t, np.cumsum(vy_r * dt)))
    vel_r1 = np.column_stack((np.ones_like(t), vy_r))
    pos_t1 = np.column_stack((8.0 - 1.0 * t, np.zeros_like(t)))
    vel_t1 = np.column_stack((-np.ones_like(t), np.zeros_like(t)))
    r1 = evaluate_probst(_make_pair(t, pos_r1, vel_r1, pos_t1, vel_t1))

    pos_r2 = np.column_stack((1.0 * t, np.zeros_like(t)))
    vel_r2 = np.column_stack((np.ones_like(t), np.zeros_like(t)))
    pos_t2 = np.column_stack((8.0 - 1.0 * t, -np.cumsum(vy_r * dt)))
    vel_t2 = np.column_stack((-np.ones_like(t), -vy_r))
    r2 = evaluate_probst(_make_pair(t, pos_r2, vel_r2, pos_t2, vel_t2))

    rolled = rollup_probst_results([r1, r2])
    assert rolled.c_total == pytest.approx((r1.c_total or 0.0) + (r2.c_total or 0.0), abs=1e-9)
    assert rolled.r_robot == pytest.approx(rolled.r_target, abs=1e-9)
    rebuilt = derive_probst_shares(
        ProbstParts(
            c_total=rolled.c_total or 0.0,
            cc_minus_robot=rolled.cc_minus_robot or 0.0,
            cc_minus_target=rolled.cc_minus_target or 0.0,
            cc_minus_time=rolled.cc_minus_time or 0.0,
            cc_plus_robot=rolled.cc_plus_robot or 0.0,
            cc_plus_target=rolled.cc_plus_target or 0.0,
            cc_plus_time=rolled.cc_plus_time or 0.0,
        )
    )
    assert rolled.r_robot == pytest.approx(rebuilt["r_robot"], abs=1e-12)


def test_probst_calculator_encounter_and_pre_contact_scopes() -> None:
    dt = 0.1
    t = np.arange(0.0, 5.0, dt)
    vy_r = np.where((t >= 1.0) & (t <= 2.5), 0.7, 0.0)
    robot_xy = np.column_stack((1.0 * t, np.cumsum(vy_r * dt)))
    ped_xy = np.column_stack((6.0 - 1.0 * t, np.zeros_like(t)))
    ped_vel = np.column_stack((-np.ones_like(t), np.zeros_like(t)))

    ep = _make_episode(t, robot_xy, [(1, "ped_1", ped_xy, ped_vel, 0.4)])
    calc = ProbstResponsibilityCalculator(RobotParams(robot_radius=0.4))

    out_enc = calc.calculate(ep, {})
    assert set(out_enc.keys()) == set(calc.output_keys())
    assert out_enc["probst_c_total"] is not None and out_enc["probst_c_total"] > 0.0
    assert out_enc["probst_r_robot"] > 0.5
    assert out_enc["probst_ps_c_total"] is not None and out_enc["probst_ps_c_total"] > 0.0

    out_pre = calc.calculate_pre_contact(ep, int(3.0 * 1e9), 1)
    assert out_pre["probst_c_total"] is not None and out_pre["probst_c_total"] > 0.0
    assert (out_pre["probst_r_robot"] + out_pre["probst_r_target"] + out_pre["probst_r_time"]) == pytest.approx(1.0, abs=1e-9)
