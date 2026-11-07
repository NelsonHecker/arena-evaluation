from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest
from flat_peds import flat_peds

from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_ENCOUNTER_SPEC,
    Trajectory2D,
    detect_encounters,
    extract_episode_ped_tracks,
    resolve_robot_trajectory,
)
from arena_evaluation.processing.metrics.social.probst_responsibility import (
    ProbstPair,
    ProbstParts,
    ProbstResponsibilityCalculator,
    ProbstResult,
    compute_normalization,
    derive_probst_shares,
    distance_to_closest_encounter,
    evaluate_probst,
    rollup_probst_results,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

# One pedestrian trajectory: canonical id, name, (N, 2) positions, (N, 2) velocities, collision radius in metres.
_PedSpec = tuple[str, str, np.ndarray, np.ndarray, float]
SHARE_KEYS = frozenset({"r_robot", "r_target", "r_time", "e_robot", "e_target", "e_time"})


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


def _make_zero_pair(times_s: np.ndarray, *, window_size_s: float = 12.0) -> ProbstPair:
    """ProbstPair at the origin with the given time grid, for normalization boundary tests."""
    time_ns = np.rint(times_s * 1e9).astype(np.int64)
    zeros = np.zeros(len(times_s), dtype=np.float64)
    trajectory = Trajectory2D(time_ns, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), 0.5)
    return ProbstPair(robot=trajectory, target=trajectory, window_size_s=window_size_s)


def _pedestrian_tracks(time_ns: np.ndarray, ped_tracks: Sequence[_PedSpec]) -> pl.DataFrame:
    """Flat pedestrian_tracks table, one row per pedestrian per frame, built from _PedSpec trajectories."""
    if not ped_tracks:
        return flat_peds([])
    frames = [
        (
            int(t_ns),
            [
                {
                    "ped_id": ped_id,
                    "ped_name": ped_name,
                    "x": float(xy[index, 0]),
                    "y": float(xy[index, 1]),
                    "vx": float(vel[index, 0]),
                    "vy": float(vel[index, 1]),
                }
                for ped_id, ped_name, xy, vel, _radius_m in ped_tracks
            ],
        )
        for index, t_ns in enumerate(time_ns)
    ]
    radii = {ped_id: float(radius_m) for ped_id, _name, _xy, _vel, radius_m in ped_tracks}
    return flat_peds(frames).with_columns(pl.col("ped_id").replace_strict(radii, return_dtype=pl.Float64).alias("radius_m"))


def _make_episode(
    times_s: np.ndarray,
    robot_xy: np.ndarray,
    robot_vel: np.ndarray,
    *,
    ped_tracks: Sequence[_PedSpec] | None = None,
) -> AlignedEpisodeBundle:
    """Build a flat AlignedEpisodeBundle from robot and pedestrian trajectories."""
    time_ns = (times_s * 1e9).astype(np.int64)
    num_samples = len(times_s)
    robot_tracks = pl.DataFrame(
        {
            "t_ns": time_ns,
            "x": robot_xy[:, 0],
            "y": robot_xy[:, 1],
            "z": np.zeros(num_samples, dtype=np.float64),
            "yaw": np.zeros(num_samples, dtype=np.float64),
            "vx": robot_vel[:, 0],
            "vy": robot_vel[:, 1],
        }
    )
    topics: dict[str, pl.DataFrame] = {"robot_tracks": robot_tracks}
    if ped_tracks is not None:
        topics["pedestrian_tracks"] = _pedestrian_tracks(time_ns, ped_tracks)
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[float(robot_xy[0, 0]), float(robot_xy[0, 1]), 0.0],
        goal_pos=[float(robot_xy[-1, 0]), float(robot_xy[-1, 1]), 0.0],
        topics=topics,
    )


def _straight_robot(times_s: np.ndarray, speed_mps: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Straight +x robot motion at constant speed."""
    robot_xy = np.column_stack((speed_mps * times_s, np.zeros_like(times_s)))
    robot_vel = np.column_stack((np.full_like(times_s, speed_mps), np.zeros_like(times_s)))
    return robot_xy, robot_vel


def _stationary_pedestrian(x: float, y: float, num_samples: int, radius_m: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Stationary pedestrian positions and velocities."""
    positions = np.tile(np.array([x, y], dtype=np.float64), (num_samples, 1))
    return positions, np.zeros((num_samples, 2), dtype=np.float64)


@pytest.mark.parametrize(
    ("robot_pos", "robot_vel", "target_pos", "target_vel", "expected_m"),
    [
        ([0.0, 0.0], [1.0, 0.0], [10.0, 0.4], [-1.0, 0.0], 0.4),  # head-on
        ([0.0, 0.0], [-1.0, 0.0], [3.0, 4.0], [1.0, 0.0], 5.0),  # diverging
        ([0.0, 0.0], [1.0, 0.0], [10.0, 0.4], [0.0, 0.0], 0.4),  # static target
        ([0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [1.0, 0.0], 5.0),  # zero relative velocity
    ],
    ids=["head-on", "diverging", "static-target", "zero-relative-velocity"],
)
def test_distance_to_closest_encounter(
    robot_pos: Sequence[float],
    robot_vel: Sequence[float],
    target_pos: Sequence[float],
    target_vel: Sequence[float],
    expected_m: float,
) -> None:
    distance_m = distance_to_closest_encounter(
        (np.asarray(robot_pos, dtype=np.float64), np.asarray(robot_vel, dtype=np.float64)),
        (np.asarray(target_pos, dtype=np.float64), np.asarray(target_vel, dtype=np.float64)),
    )
    assert distance_m == pytest.approx(expected_m, abs=1e-9)


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
    vy_r = np.where((t >= 1.0) & (t <= 2.5), 0.4, 0.0)
    robot_xy = np.column_stack((1.0 * t, np.cumsum(vy_r * dt)))
    robot_vel = np.column_stack((np.ones_like(t), vy_r))
    ped_xy = np.column_stack((6.0 - 1.0 * t, np.zeros_like(t)))
    ped_vel = np.column_stack((-np.ones_like(t), np.zeros_like(t)))

    ep = _make_episode(t, robot_xy, robot_vel, ped_tracks=[("Pedestrian_1", "ped_1", ped_xy, ped_vel, 0.4)])
    calc = ProbstResponsibilityCalculator(RobotParams(robot_radius=0.4))

    out_enc = calc.calculate(ep, {})
    assert set(out_enc) == set(calc.output_keys())
    assert out_enc["probst_c_total"] is not None and out_enc["probst_c_total"] > 0.0
    assert out_enc["probst_r_robot"] is not None and out_enc["probst_r_robot"] > 0.5
    assert out_enc["probst_ps_c_total"] is not None and out_enc["probst_ps_c_total"] > 0.0

    out_pre = calc.calculate_pre_contact(ep, int(3.0 * 1e9), "Pedestrian_1")
    assert out_pre["probst_c_total"] is not None and out_pre["probst_c_total"] > 0.0
    assert ((out_pre["probst_r_robot"] or 0.0) + (out_pre["probst_r_target"] or 0.0) + (out_pre["probst_r_time"] or 0.0)) == pytest.approx(1.0, abs=1e-9)


def test_compute_normalization_uniform_grid() -> None:
    times_s = np.linspace(0.0, 2.0, 21)
    pair = _make_zero_pair(times_s)
    norm = compute_normalization(pair, len(times_s) - 1)
    assert norm is not None
    assert norm.shape == times_s.shape
    assert norm[-1] == pytest.approx(0.0, abs=1e-12)
    assert norm[0] == pytest.approx(1.0 - 2.0 / 12.0, abs=1e-12)
    assert norm[-2] == pytest.approx(1.0, abs=1e-12)
    assert bool(np.all(np.diff(norm[:-1]) > 0.0))


def test_compute_normalization_non_uniform_grid() -> None:
    times_s = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0])
    pair = _make_zero_pair(times_s, window_size_s=1.0)
    norm = compute_normalization(pair, len(times_s) - 1)
    assert norm is not None
    np.testing.assert_allclose(norm, [0.0, 0.125, 0.25, 0.4375, 0.625, 1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    ("times_s", "idx_tce"),
    [
        (np.array([0.0]), 0),
        (np.linspace(0.0, 1.0, 11), 0),
        (np.linspace(0.0, 1.0, 11), 11),
    ],
    ids=["single-sample", "tce-at-first", "tce-past-last"],
)
def test_compute_normalization_returns_none_off_grid(times_s: np.ndarray, idx_tce: int) -> None:
    assert compute_normalization(_make_zero_pair(times_s), idx_tce) is None


@pytest.mark.parametrize(
    "parts",
    [
        None,
        ProbstParts(
            c_total=0.0,
            cc_minus_robot=0.0,
            cc_minus_target=0.0,
            cc_minus_time=0.0,
            cc_plus_robot=0.0,
            cc_plus_target=0.0,
            cc_plus_time=0.0,
        ),
    ],
    ids=["no-parts", "zero-total"],
)
def test_derive_probst_shares_returns_none_without_total(parts: ProbstParts | None) -> None:
    shares = derive_probst_shares(parts)
    assert set(shares) == SHARE_KEYS
    assert all(value is None for value in shares.values())


def test_derive_probst_shares_from_parts() -> None:
    parts = ProbstParts(
        c_total=2.0,
        cc_minus_robot=1.0,
        cc_minus_target=0.5,
        cc_minus_time=0.5,
        cc_plus_robot=0.8,
        cc_plus_target=0.6,
        cc_plus_time=0.6,
    )
    shares = derive_probst_shares(parts)
    assert shares["r_robot"] == pytest.approx(0.5, abs=1e-12)
    assert shares["r_target"] == pytest.approx(0.25, abs=1e-12)
    assert shares["r_time"] == pytest.approx(0.25, abs=1e-12)
    assert shares["e_robot"] == pytest.approx(0.4, abs=1e-12)
    assert shares["e_target"] == pytest.approx(0.3, abs=1e-12)
    assert shares["e_time"] == pytest.approx(0.3, abs=1e-12)


@pytest.mark.parametrize("num_samples", [0, 1])
def test_evaluate_probst_requires_two_samples(num_samples: int) -> None:
    t = np.arange(num_samples, dtype=np.float64) * 0.1
    empty = np.zeros((num_samples, 2), dtype=np.float64)
    assert evaluate_probst(_make_pair(t, empty, empty, empty, empty)) == ProbstResult()


def test_calculate_without_pedestrian_tracks_returns_none_outputs() -> None:
    t = np.arange(0.0, 3.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    episode = _make_episode(t, robot_xy, robot_vel)
    calc = ProbstResponsibilityCalculator(RobotParams(robot_radius=0.4))

    out = calc.calculate(episode, {})
    assert set(out) == set(calc.output_keys())
    assert all(value is None for value in out.values())


def test_calculate_with_empty_pedestrian_tracks_returns_none_outputs() -> None:
    t = np.arange(0.0, 3.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    episode = _make_episode(t, robot_xy, robot_vel, ped_tracks=[])
    calc = ProbstResponsibilityCalculator(RobotParams(robot_radius=0.4))

    out = calc.calculate(episode, {})
    assert set(out) == set(calc.output_keys())
    assert all(value is None for value in out.values())


def test_calculate_with_two_pedestrians_rolls_up_additive_parts() -> None:
    t = np.arange(0.0, 10.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    ped_1_xy, zero_vel = _stationary_pedestrian(2.0, 0.1, len(t))
    ped_2_xy, _ = _stationary_pedestrian(8.0, 0.1, len(t))
    episode = _make_episode(
        t,
        robot_xy,
        robot_vel,
        ped_tracks=[
            ("Pedestrian_1", "alpha", ped_1_xy, zero_vel, 0.4),
            ("Pedestrian_2", "beta", ped_2_xy, zero_vel, 0.4),
        ],
    )
    calc = ProbstResponsibilityCalculator(RobotParams(robot_radius=0.4))

    robot, _ = resolve_robot_trajectory(calc, episode)
    tracks = extract_episode_ped_tracks(episode)
    encounters = detect_encounters(robot, tracks, DEFAULT_ENCOUNTER_SPEC)
    assert len(encounters) == 2
    rolled = rollup_probst_results([calc.evaluate_encounter_track(robot, tracks[encounter.ped_id], (encounter.t_start_ns, encounter.t_end_ns))[0] for encounter in encounters])

    out = calc.calculate(episode, {})
    assert set(out) == set(calc.output_keys())
    assert out["probst_c_total"] == pytest.approx(rolled.c_total, abs=1e-9)
    assert out["probst_r_robot"] == pytest.approx(rolled.r_robot, abs=1e-9)
    assert out["probst_e_target"] == pytest.approx(rolled.e_target, abs=1e-9)
    assert ((out["probst_r_robot"] or 0.0) + (out["probst_r_target"] or 0.0) + (out["probst_r_time"] or 0.0)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("target_agent_id", [1, "1", "Pedestrian_1", "ped_1", "alpha"])
def test_calculate_resolves_target_identifiers(target_agent_id: int | str) -> None:
    t = np.arange(0.0, 5.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    ped_xy, ped_vel = _stationary_pedestrian(2.5, 0.1, len(t))
    episode = _make_episode(t, robot_xy, robot_vel, ped_tracks=[("Pedestrian_1", "alpha", ped_xy, ped_vel, 0.4)])
    calc = ProbstResponsibilityCalculator(RobotParams(robot_radius=0.4))
    calc.metrics_config = {"target_agent_id": target_agent_id}

    out = calc.calculate(episode, {})
    assert out["probst_c_total"] is not None and out["probst_c_total"] > 0.0
    assert ((out["probst_r_robot"] or 0.0) + (out["probst_r_target"] or 0.0) + (out["probst_r_time"] or 0.0)) == pytest.approx(1.0, abs=1e-9)


def test_calculate_pre_contact_unknown_pedestrian_returns_none_outputs() -> None:
    t = np.arange(0.0, 5.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    ped_xy, ped_vel = _stationary_pedestrian(2.5, 0.1, len(t))
    episode = _make_episode(t, robot_xy, robot_vel, ped_tracks=[("Pedestrian_1", "alpha", ped_xy, ped_vel, 0.4)])
    calc = ProbstResponsibilityCalculator(RobotParams(robot_radius=0.4))

    out = calc.calculate_pre_contact(episode, int(3.0 * 1e9), "Pedestrian_99")
    assert set(out) == set(calc.output_keys())
    assert all(value is None for value in out.values())
