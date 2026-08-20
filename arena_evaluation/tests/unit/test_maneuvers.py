"""Unit tests for the open-loop characterization maneuver schedule.

Pure Python, no ROS, no Gazebo required. The schedule lives in
task_generator (owned by the TM_Characterization robot task mode); the offline
calculator imports the same module so labels never drift.
"""

import pathlib

import pytest

from task_generator.tasks.robots.characterization.schedule import (
    ANGULAR_DWELL_S,
    IDLE_DURATION_S,
    LINEAR_DWELL_S,
    VX_MAX,
    PhaseKind,
    build_schedule,
    classify_cmd_point,
    resolve_envelope,
    schedule_duration,
)


def test_schedule_structure():
    phases = build_schedule()
    assert phases[0].kind == PhaseKind.IDLE
    assert phases[-1].kind == PhaseKind.IDLE
    # Idle blocks at start, middle(s), and end.
    idle_blocks = [p for p in phases if p.kind == PhaseKind.IDLE and p.duration_s == IDLE_DURATION_S]
    assert len(idle_blocks) == 4
    assert all(p.duration_s == IDLE_DURATION_S for p in idle_blocks)


def test_linear_sweep_coverage_up_to_2mps():
    phases = build_schedule()
    # The ramp-apex settles are also kind=LINEAR - match the sweep steps only.
    linear = [p for p in phases if p.name.startswith("linear_vx_")]
    targets = sorted({p.vx_target for p in linear if p.vx_target > 0.0})
    assert targets == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    assert max(targets) == VX_MAX
    assert all(p.duration_s == LINEAR_DWELL_S for p in linear)
    # Out-and-back: every forward step has a matching backward return at the
    # same speed so the robot never runs out of map.
    forward = {p.vx_target for p in linear if p.vx_target > 0.0}
    backward = {p.vx_target for p in linear if p.vx_target < 0.0}
    assert forward == {-v for v in backward}
    assert all(p.duration_s == LINEAR_DWELL_S for p in linear)


def test_ramp_tests():
    phases = build_schedule()
    ramps_up = [p for p in phases if p.kind == PhaseKind.RAMP_UP and p.vx_target > 0.0]
    ramps_down = [p for p in phases if p.kind == PhaseKind.RAMP_DOWN and p.vx_target > 0.0]
    ramps_up_neg = [p for p in phases if p.kind == PhaseKind.RAMP_UP and p.vx_target < 0.0]
    ramps_down_neg = [p for p in phases if p.kind == PhaseKind.RAMP_DOWN and p.vx_target < 0.0]

    # Covers the linear envelope steps
    targets = sorted(p.vx_target for p in ramps_up)
    assert targets == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    assert sorted(p.vx_target for p in ramps_down) == targets
    assert sorted(p.vx_target for p in ramps_up_neg) == [-v for v in reversed(targets)]
    assert sorted(p.vx_target for p in ramps_down_neg) == [-v for v in reversed(targets)]

    # Settle at apex for each positive and negative step
    apexes = [p for p in phases if p.name.startswith("ramp_apex")]
    assert len(apexes) == len(targets) * 2
    assert all(p.duration_s == 1.0 for p in apexes)


def test_angular_sweep():
    phases = build_schedule()
    angular = [p for p in phases if p.kind == PhaseKind.ANGULAR]
    assert [p.wz_target for p in angular] == [
        -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
    ]
    assert all(p.duration_s == ANGULAR_DWELL_S for p in angular)
    assert all(p.vx_target == 0.0 for p in angular)


def test_names_unique():
    phases = build_schedule()
    names = [p.name for p in phases]
    assert len(names) == len(set(names))


def test_schedule_duration():
    phases = build_schedule()
    assert schedule_duration(phases) == pytest.approx(sum(p.duration_s for p in phases))


def test_classify_cmd_point():
    assert classify_cmd_point(0.0, 0.0) == (PhaseKind.IDLE, 0.0, 0.0)
    assert classify_cmd_point(1.5, 0.0) == (PhaseKind.LINEAR, 1.5, 0.0)
    assert classify_cmd_point(0.0, -1.0) == (PhaseKind.ANGULAR, 0.0, -1.0)
    # Angular dominates linear
    assert classify_cmd_point(0.5, 2.0)[0] == PhaseKind.ANGULAR


def test_resolve_envelope_from_caps(tmp_path: pathlib.Path):
    # Simulate an arena_robots caps file for an arbitrary robot model.
    caps = tmp_path / "some_robot" / "caps"
    caps.mkdir(parents=True)
    (caps / "mobile.yaml").write_text(
        "actions:\n"
        "  continuous:\n"
        "    linear:\n"
        "      min: -1.5\n"
        "      max: 3.0\n"
        "    angular:\n"
        "      min: -4.0\n"
        "      max: 4.0\n"
    )
    env = resolve_envelope("some_robot", caps_dir=tmp_path)
    assert env == {"vx_max": 3.0, "wz_max": 4.0}


def test_resolve_envelope_fallback(tmp_path: pathlib.Path):
    # Unknown robot with no caps file -> generic defaults.
    env = resolve_envelope("ghost_robot", caps_dir=tmp_path)
    assert env == {"vx_max": VX_MAX, "wz_max": 2.5}


def test_schedule_uses_resolved_envelope(tmp_path: pathlib.Path):
    caps = tmp_path / "big_robot" / "caps"
    caps.mkdir(parents=True)
    (caps / "mobile.yaml").write_text(
        "actions:\n  continuous:\n    linear:\n      max: 3.0\n    angular:\n      max: 4.0\n"
    )
    env = resolve_envelope("big_robot", caps_dir=tmp_path)
    phases = build_schedule(vx_max=env["vx_max"], wz_max=env["wz_max"])
    linear = [p for p in phases if p.kind == PhaseKind.LINEAR and p.wz_target == 0.0]
    assert max(p.vx_target for p in linear) == 3.0
    angular = [p for p in phases if p.kind == PhaseKind.ANGULAR]
    assert max(p.wz_target for p in angular) == 4.0
