"""Unit tests for the per-episode resume planner."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
from arena_evaluation.benchmark.resume import (
    decide_step,
    episode_index_for_seed,
    load_episode_dirs,
    plan_resume,
    retire_episode_dir,
    seed_for_episode,
)
from arena_evaluation.benchmark.step import Step
from arena_evaluation.benchmark.config import Contest, Suite
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import RunMetadata
from task_generator.constants import Constants


def _stage(name: str = "s1", *, episodes: int = 5, seed: int | None = 0) -> Suite.Stage:
    return Suite.Stage(
        name=name,
        episodes=episodes,
        robot="turtlebot3_burger",
        map="map1",
        tm_robots=Constants.TaskMode.TM_Robots.RANDOM,
        tm_obstacles=Constants.TaskMode.TM_Obstacles.RANDOM,
        config={},
        seed=seed,
        timeout=120.0,
    )


def _contestant(name: str = "pa") -> Contest.Contestant:
    return Contest.Contestant(name=name, args={})


def _step(contestant: Contest.Contestant, stage: Suite.Stage, *, is_reference: bool = False, reference_type: str | None = None) -> Step:
    return Step(
        contestant=contestant,
        stage=stage,
        episodes=stage.episodes,
        record_dir=None,
        is_reference=is_reference,
        reference_type=reference_type,
    )


def _mk_episode(
    root: pathlib.Path,
    number: int,
    *,
    planner: str,
    stage: str,
    sim_id: int,
    seed: int,
    outcome: int | None,
    is_reference: bool = False,
    reference_type: str | None = None,
    mcap: bool = True,
) -> pathlib.Path:
    ep_dir = root / "episodes" / ("episode_%03d" % number)
    ep_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "planner": planner,
        "stage": stage,
        "is_reference": is_reference,
        "reference_type": reference_type,
        "task_generator_episode_id": sim_id,
        "seed": seed,
        "outcome_state": outcome,
    }
    (ep_dir / ("episode_%03d.yaml" % number)).write_text(json.dumps(meta))
    if mcap:
        (ep_dir / ("episode_%03d.mcap" % number)).write_bytes(b"x" * 128)
    return ep_dir


def _write_episode_meta(
    root: pathlib.Path,
    number: int,
    *,
    planner: str,
    stage: str,
    sim_id: int,
    seed: int,
    outcome: int,
    is_reference: bool = False,
    reference_type: str | None = None,
) -> pathlib.Path:
    """Create an episode sidecar through the real MetadataWriter path."""
    ep_dir = root / "episodes" / ("episode_%03d" % number)
    ep_dir.mkdir(parents=True, exist_ok=True)
    meta = RunMetadata(
        benchmark_id="bench",
        planner=planner,
        map="map1",
        stage=stage,
        task_generator_episode_id=sim_id,
        seed=seed,
        outcome_state=outcome,
        is_reference=is_reference,
        reference_type=reference_type,
        recording_started_at="2026-01-01T00:00:00+00:00",
        python_version="3.12",
        ros_distro="jazzy",
    )
    MetadataWriter.write(meta, ep_dir / ("episode_%03d.yaml" % number))
    (ep_dir / ("episode_%03d.mcap" % number)).write_bytes(b"x" * 128)
    return ep_dir


def _row(step_key: str, seed: int, sim_id: int, outcome: int, *, ts: str = "2024-01-01T00:00:00+00:00") -> dict:
    contestant, stage = step_key.split("/", 1)
    return {
        "ts_iso": ts,
        "step_key": step_key,
        "contestant": contestant,
        "stage": stage,
        "episode_id": sim_id,
        "seed": seed,
        "outcome_state": outcome,
    }


# --- pure helpers ----------------------------------------------------------


def test_seed_for_episode_offsets_by_stage_seed() -> None:
    assert seed_for_episode(100, 3) == 103
    assert seed_for_episode(0, 3) == 3
    assert seed_for_episode(None, 3) == 3


def test_episode_index_for_seed_inverts_offset() -> None:
    assert episode_index_for_seed(100, 103, 5) == 3
    assert episode_index_for_seed(0, 3, 5) == 3
    assert episode_index_for_seed(None, 3, 5) == 3
    assert episode_index_for_seed(100, 90, 5) is None
    assert episode_index_for_seed(100, 105, 5) is None


def test_load_episode_dirs_reads_sidecar_and_mcap(tmp_path: pathlib.Path) -> None:
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=2, mcap=True)
    _mk_episode(tmp_path, 1, planner="pa", stage="s1", sim_id=101, seed=1, outcome=None, mcap=False)
    superseded = tmp_path / "episodes" / ".superseded"
    retired_dir = superseded / "episode_002"
    retired_dir.mkdir(parents=True)
    (retired_dir / "episode_002.yaml").write_text(json.dumps({"planner": "pa", "stage": "s1", "task_generator_episode_id": 99, "seed": 9, "outcome_state": 3}))
    (retired_dir / "episode_002.mcap").write_bytes(b"x" * 128)

    live, retired = load_episode_dirs(tmp_path / "episodes")
    assert [d.number for d in live] == [0, 1]
    assert [d.number for d in retired] == [2]

    good = next(d for d in live if d.number == 0)
    assert good.planner == "pa"
    assert good.stage == "s1"
    assert good.sim_episode_id == 100
    assert good.seed == 0
    assert good.outcome_state == 2
    assert good.is_terminal is True
    assert good.has_mcap is True

    partial = next(d for d in live if d.number == 1)
    assert partial.outcome_state is None
    assert partial.is_terminal is False
    assert partial.has_mcap is False


def test_retire_episode_dir_moves_and_uniquifies(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "episodes"
    a = root / "episode_000"
    a.mkdir(parents=True)
    superseded = root / ".superseded"

    dest = retire_episode_dir(a, superseded)
    assert dest == superseded / "episode_000"
    assert dest.is_dir()
    assert not a.exists()

    # A second dir with the same name must not clobber the first retirement.
    b = root / "episode_000"
    b.mkdir(parents=True)
    dest2 = retire_episode_dir(b, superseded)
    assert dest2 == superseded / "episode_000_1"
    assert (superseded / "episode_000").is_dir()


# --- decide_step: GOOD / RECORDED / DEAD -----------------------------------


def test_decide_step_good_success_is_kept(tmp_path: pathlib.Path) -> None:
    stage = _stage(episodes=3, seed=0)
    step = _step(_contestant("pa"), stage)
    dirs = []
    rows = []
    for i in range(3):
        dirs.append(_mk_episode(tmp_path, i, planner="pa", stage="s1", sim_id=100 + i, seed=i, outcome=2))
        rows.append(_row("pa/s1", i, 100 + i, 2))
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, rows, live, retired, retry_failed=False)

    assert decision.run_indices == []
    assert decision.delete_dirs == []
    assert decision.retire_dirs == []
    assert decision.kept_sim_ids == {0: 100, 1: 101, 2: 102}


def test_decide_step_success_without_row_kept_from_yaml(tmp_path: pathlib.Path) -> None:
    # Recorder wrote the terminal outcome into the YAML but the runner died
    # before appending the progress row: the dir alone keeps the episode.
    stage = _stage(episodes=3, seed=0)
    step = _step(_contestant("pa"), stage)
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=2)
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, [], live, retired, retry_failed=False)

    assert decision.run_indices == [1, 2]
    assert decision.kept_sim_ids == {0: 100}


def test_decide_step_recorded_failure_preserved_by_default(tmp_path: pathlib.Path) -> None:
    stage = _stage(episodes=2, seed=0)
    step = _step(_contestant("pa"), stage)
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=3)
    rows = [_row("pa/s1", 0, 100, 3)]
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, rows, live, retired, retry_failed=False)

    # The recorded failure is real evidence -> preserved, never re-run in place.
    assert decision.run_indices == [1]
    assert decision.delete_dirs == []
    assert decision.retire_dirs == []
    assert decision.kept_sim_ids == {0: 100}


def test_decide_step_recorded_failure_retired_when_forced(tmp_path: pathlib.Path) -> None:
    stage = _stage(episodes=2, seed=0)
    step = _step(_contestant("pa"), stage)
    ep0 = _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=3)
    rows = [_row("pa/s1", 0, 100, 3)]
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, rows, live, retired, retry_failed=True)

    assert decision.run_indices == [0, 1]
    assert decision.retire_dirs == [ep0]
    assert decision.delete_dirs == []


def test_decide_step_dead_episode_rerun_and_garbage_deleted(tmp_path: pathlib.Path) -> None:
    stage = _stage(episodes=3, seed=0)
    step = _step(_contestant("pa"), stage)
    # ep0 finished SUCCESS, ep1 crashed mid-recording (dir, no terminal outcome).
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=2)
    garbage = _mk_episode(tmp_path, 1, planner="pa", stage="s1", sim_id=101, seed=1, outcome=None)
    rows = [_row("pa/s1", 0, 100, 2)]
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, rows, live, retired, retry_failed=False)

    assert decision.run_indices == [1, 2]
    assert decision.delete_dirs == [garbage]
    assert decision.retire_dirs == []


def test_decide_step_never_started_episode_is_pending(tmp_path: pathlib.Path) -> None:
    stage = _stage(episodes=2, seed=0)
    step = _step(_contestant("pa"), stage)
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=2)
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, [_row("pa/s1", 0, 100, 2)], live, retired, retry_failed=False)

    assert decision.run_indices == [1]
    assert decision.delete_dirs == [] and decision.retire_dirs == []


def test_decide_step_stale_duplicate_dir_retired(tmp_path: pathlib.Path) -> None:
    # Two surviving dirs attributed to the same episode seed; the newest (the
    # row's sim id) is kept, the older duplicate is retired, not deleted.
    stage = _stage(episodes=1, seed=0)
    step = _step(_contestant("pa"), stage)
    keep = _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=2)
    dup = _mk_episode(tmp_path, 1, planner="pa", stage="s1", sim_id=200, seed=0, outcome=2)
    rows = [_row("pa/s1", 0, 100, 2)]
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, rows, live, retired, retry_failed=False)

    assert decision.run_indices == []
    assert decision.retire_dirs == [dup]
    assert decision.delete_dirs == []
    # decide_step is pure: the kept dir stays on disk and is never scheduled for
    # retirement or deletion; only the stale duplicate is retired.
    assert keep.exists()
    assert keep not in decision.retire_dirs and keep not in decision.delete_dirs
    assert decision.kept_sim_ids == {0: 100}


def test_decide_step_retires_seeded_duplicate_written_by_real_path(tmp_path: pathlib.Path) -> None:
    """A duplicate of an already-SUCCESS episode is retired even when nothing re-runs."""
    stage = _stage(episodes=1, seed=0)
    step = _step(_contestant("pa"), stage)
    _write_episode_meta(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=2)
    dup = _write_episode_meta(tmp_path, 1, planner="pa", stage="s1", sim_id=200, seed=0, outcome=2)
    rows = [_row("pa/s1", 0, 100, 2)]
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, rows, live, retired, retry_failed=False)

    assert decision.run_indices == []
    assert decision.retire_dirs == [dup]
    assert decision.delete_dirs == []


def test_decide_step_unhindered_peds_kept_with_surviving_mcap(tmp_path: pathlib.Path) -> None:
    stage = _stage(episodes=2, seed=0)
    step = _step(_contestant("unhindered_peds"), stage, is_reference=True, reference_type="unhindered_peds")
    # unhindered_peds legitimately emits no terminal outcome; a surviving mcap
    # means the reference run completed.
    _mk_episode(tmp_path, 0, planner="unhindered_peds", stage="s1", sim_id=100, seed=0, outcome=None, is_reference=True, reference_type="unhindered_peds")
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, [], live, retired, retry_failed=False)

    assert decision.run_indices == [1]
    assert decision.delete_dirs == []
    assert decision.kept_sim_ids == {0: 100}


# --- plan_resume -----------------------------------------------------------


def test_plan_resume_plans_each_step_from_its_own_rows(tmp_path: pathlib.Path) -> None:
    stage_a = _stage("a", episodes=2, seed=0)
    stage_b = _stage("b", episodes=2, seed=0)
    pa = _contestant("pa")
    pb = _contestant("pb")
    step_a = _step(pa, stage_a)
    step_b = _step(pb, stage_b)

    # step_a completely recorded; step_b untouched.
    _mk_episode(tmp_path, 0, planner="pa", stage="a", sim_id=100, seed=0, outcome=2)
    _mk_episode(tmp_path, 1, planner="pa", stage="a", sim_id=101, seed=1, outcome=2)
    live, retired = load_episode_dirs(tmp_path / "episodes")
    rows_by_key = {
        "pa/a": [_row("pa/a", 0, 100, 2), _row("pa/a", 1, 101, 2)],
    }

    decisions = plan_resume([step_a, step_b], rows_by_key, live, retired, retry_failed=False)
    by_key = {d.step.key: d for d in decisions}

    assert by_key["pa/a"].run_indices == []
    assert by_key["pb/b"].run_indices == [0, 1]


def test_plan_resume_returns_one_decision_per_step(tmp_path: pathlib.Path) -> None:
    stages = [_stage("a", episodes=1), _stage("b", episodes=1)]
    contestant = _contestant("pa")
    steps = [_step(contestant, s) for s in stages]

    decisions = plan_resume(steps, {}, [], [], retry_failed=False)

    assert [d.step.key for d in decisions] == ["pa/a", "pa/b"]
    assert all(d.run_indices == [0] for d in decisions)


# --- regressions: ROS-free import, legacy rows, retired names, newest dir --


def test_resume_module_imports_without_rclpy() -> None:
    """The planner imports and runs in a ROS-free interpreter."""
    import os
    import subprocess

    package_root = pathlib.Path(__file__).parents[2]
    code = "import sys; sys.modules['rclpy'] = None; import arena_evaluation.benchmark.resume as r; assert r.seed_for_episode(100, 3) == 103"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_rows_without_seed_do_not_raise(tmp_path: pathlib.Path) -> None:
    """Legacy progress.csv rows with an empty seed column do not crash the sweep."""
    step = _step(_contestant("pa"), _stage(episodes=2, seed=0))
    rows = [{"episode_id": 1}, {"episode_id": 2, "seed": None}]

    decision = decide_step(step, rows, [], [], retry_failed=False)

    assert decision.run_indices == [0, 1]


def _mk_named_episode(
    root: pathlib.Path,
    dirname: str,
    *,
    sidecar: str,
    planner: str,
    stage: str,
    sim_id: int,
    seed: int,
    outcome: int | None,
) -> pathlib.Path:
    ep_dir = root / dirname
    ep_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "planner": planner,
        "stage": stage,
        "is_reference": False,
        "reference_type": None,
        "task_generator_episode_id": sim_id,
        "seed": seed,
        "outcome_state": outcome,
    }
    (ep_dir / sidecar).write_text(json.dumps(meta))
    (ep_dir / sidecar.replace(".yaml", ".mcap")).write_bytes(b"x" * 128)
    return ep_dir


def test_retired_numbered_dirs_are_scanned(tmp_path: pathlib.Path) -> None:
    """A retired `episode_003_1` must stay visible, sidecar included."""
    _mk_named_episode(
        tmp_path / "episodes" / ".superseded",
        "episode_003_1",
        sidecar="episode_003.yaml",
        planner="pa",
        stage="s1",
        sim_id=99,
        seed=3,
        outcome=3,
    )

    live, retired = load_episode_dirs(tmp_path / "episodes")

    assert live == []
    assert [d.path.name for d in retired] == ["episode_003_1"]
    assert retired[0].number == 3
    # The sidecar kept its original name inside the moved dir.
    assert retired[0].planner == "pa"
    assert retired[0].sim_episode_id == 99


def test_recorded_without_row_prefers_newest(tmp_path: pathlib.Path) -> None:
    """A newer SUCCESS dir must not be shadowed by an older duplicate."""
    step = _step(_contestant("pa"), _stage(episodes=2, seed=0))
    older = _mk_named_episode(
        tmp_path / "episodes",
        "episode_001",
        sidecar="episode_001.yaml",
        planner="pa",
        stage="s1",
        sim_id=100,
        seed=1,
        outcome=2,
    )
    newer = _mk_named_episode(
        tmp_path / "episodes",
        "episode_001_1",
        sidecar="episode_001.yaml",
        planner="pa",
        stage="s1",
        sim_id=200,
        seed=1,
        outcome=2,
    )
    live, retired = load_episode_dirs(tmp_path / "episodes")

    decision = decide_step(step, [], live, retired, retry_failed=False)

    assert decision.kept_sim_ids == {1: 200}
    assert decision.retire_dirs == [older]
    assert newer.exists()
    assert decision.run_indices == [0]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
