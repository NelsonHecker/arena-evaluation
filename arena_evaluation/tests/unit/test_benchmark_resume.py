"""Unit tests for the per-episode resume planner."""

from __future__ import annotations

import json
import pathlib

from arena_evaluation.benchmark.config import Contestant, Suite
from arena_evaluation.benchmark.resume import (
    OnDiskEpisode,
    StepResumeDecision,
    decide_step,
    dirs_for_step,
    episode_index_for_seed,
    load_episode_dirs,
    plan_resume,
    retire_episode_dir,
    seed_for_episode,
)
from arena_evaluation.benchmark.step import Step

SUCCESS = 2
FAILED = 3
SKIPPED = 4


def _stage(name: str = "s1", *, episodes: int = 5, seed: int | None = 0) -> Suite.Stage:
    return Suite.Stage(
        name=name,
        episodes=episodes,
        robot="turtlebot3_burger",
        map="map1",
        modes={"robots": "random", "obstacles": "random"},
        config={},
        seed=seed,
        timeout=120.0,
    )


def _contestant(name: str = "pa") -> Contestant:
    return Contestant(name=name, args={})


def _step(contestant: Contestant, stage: Suite.Stage, *, is_reference: bool = False, reference_type: str | None = None) -> Step:
    return Step(contestant=contestant, stage=stage, episodes=stage.episodes, record_dir=None, is_reference=is_reference, reference_type=reference_type)


def _mk_episode(
    root: pathlib.Path,
    number: int,
    *,
    planner: str,
    stage: str,
    sim_id: int,
    seed: int | None,
    outcome: int | None,
    is_reference: bool = False,
    reference_type: str | None = None,
    mcap: bool = True,
    dirname: str | None = None,
) -> pathlib.Path:
    name = dirname or f"episode_{number:03d}"
    episode_dir = root / "episodes" / name
    episode_dir.mkdir(parents=True)
    meta = {
        "planner": planner,
        "stage": stage,
        "is_reference": is_reference,
        "reference_type": reference_type,
        "task_generator_episode_id": sim_id,
        "seed": seed,
        "outcome_state": outcome,
    }
    (episode_dir / f"{name}.yaml").write_text(json.dumps(meta))
    if mcap:
        (episode_dir / f"{name}.mcap").write_bytes(b"x" * 128)
    return episode_dir


def _row(step_key: str, seed: int, sim_id: int, outcome: int, *, ts: str = "2024-01-01T00:00:00+00:00", integrity: str = "[]") -> dict[str, object]:
    return {
        "ts_iso": ts,
        "step_key": step_key,
        "seed": seed,
        "episode_id": sim_id,
        "outcome_state": outcome,
        "integrity": integrity,
    }


def _decide(root: pathlib.Path, step: Step, rows: list[dict[str, object]] | None = None, *, force: bool = False) -> StepResumeDecision:
    live = load_episode_dirs(root / "episodes")
    return decide_step(step, rows or [], dirs_for_step(live, step), force_retry=force)


# pure helpers


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


def test_load_episode_dirs_reads_sidecar_and_recording(tmp_path: pathlib.Path) -> None:
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=SUCCESS)
    _mk_episode(tmp_path, 1, planner="pa", stage="s1", sim_id=101, seed=1, outcome=None, mcap=False)

    live = load_episode_dirs(tmp_path / "episodes")

    assert [episode.number for episode in live] == [0, 1]
    good = next(episode for episode in live if episode.number == 0)
    assert (good.planner, good.stage, good.sim_episode_id, good.seed, good.outcome_state) == ("pa", "s1", 100, 0, SUCCESS)
    assert good.is_terminal is True and good.has_mcap is True
    partial = next(episode for episode in live if episode.number == 1)
    assert partial.outcome_state is None and partial.is_terminal is False and partial.has_mcap is False


def test_retired_directories_are_not_live_work(tmp_path: pathlib.Path) -> None:
    _mk_episode(tmp_path / "episodes" / ".superseded", 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=SUCCESS, dirname="episode_000")

    assert load_episode_dirs(tmp_path / "episodes") == []


def test_retire_episode_dir_moves_and_uniquifies(tmp_path: pathlib.Path) -> None:
    episodes = tmp_path / "episodes"
    first = episodes / "episode_000"
    first.mkdir(parents=True)
    superseded = episodes / ".superseded"

    destination = retire_episode_dir(first, superseded)
    assert destination == superseded / "episode_000"
    assert destination.is_dir() and not first.exists()

    second = episodes / "episode_000"
    second.mkdir(parents=True)
    destination = retire_episode_dir(second, superseded)
    assert destination == superseded / "episode_000_1"
    assert (superseded / "episode_000").is_dir()


# decide_step


def test_success_is_kept(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=3, seed=0))
    for index in range(3):
        _mk_episode(tmp_path, index, planner="pa", stage="s1", sim_id=100 + index, seed=index, outcome=SUCCESS)
    rows = [_row("pa/s1", index, 100 + index, SUCCESS) for index in range(3)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == []
    assert decision.kept_sim_ids == {0: 100, 1: 101, 2: 102}
    assert decision.retire_dirs == [] and decision.delete_dirs == []


def test_kept_compromised_episode_stays_counted(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=2, seed=0))
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=SUCCESS)
    rows = [_row("pa/s1", 0, 100, SUCCESS, integrity='["robot moved by hand"]')]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == [1]
    assert (decision.kept_run, decision.kept_compromised) == (1, 1)


def test_success_without_a_progress_row_is_kept_from_the_recorded_seed(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=3, seed=0))
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=SUCCESS)

    decision = _decide(tmp_path, step)

    assert decision.run_indices == [1, 2]
    assert decision.kept_sim_ids == {0: 100}


def test_recorded_failure_is_kept_by_default(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=2, seed=0))
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=FAILED)
    rows = [_row("pa/s1", 0, 100, FAILED)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == [1]
    assert decision.kept_sim_ids == {0: 100}
    assert decision.kept_failed == 1
    assert decision.retire_dirs == [] and decision.delete_dirs == []


def test_recorded_failure_is_retired_and_rerun_when_forced(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=2, seed=0))
    failed = _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=FAILED)
    rows = [_row("pa/s1", 0, 100, FAILED)]

    decision = _decide(tmp_path, step, rows, force=True)

    assert decision.run_indices == [0, 1]
    assert decision.retire_dirs == [failed]
    assert decision.kept_run == 0


def test_unclosed_recording_is_deleted_and_rerun(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=2, seed=0))
    unclosed = _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=None, mcap=False)
    rows = [_row("pa/s1", 0, 100, SUCCESS)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == [0, 1]
    assert decision.delete_dirs == [unclosed]
    assert decision.retire_dirs == []


def test_recording_without_a_row_is_retired_when_the_episode_is_rerun(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=1, seed=0))
    partial = _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=None, mcap=True)

    decision = _decide(tmp_path, step)

    assert decision.run_indices == [0]
    assert decision.retire_dirs == [partial]


def test_episode_without_a_recording_is_rerun(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=2, seed=0))
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=SUCCESS, mcap=False)
    rows = [_row("pa/s1", 0, 100, SUCCESS)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == [0, 1]
    assert decision.kept_run == 0


def test_missing_episode_is_rerun(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=3, seed=0))
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=SUCCESS)
    rows = [_row("pa/s1", 0, 100, SUCCESS)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == [1, 2]
    assert decision.delete_dirs == [] and decision.retire_dirs == []


def test_stale_duplicate_directory_is_retired(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=1, seed=0))
    keep = _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=100, seed=0, outcome=SUCCESS)
    duplicate = _mk_episode(tmp_path, 1, planner="pa", stage="s1", sim_id=200, seed=0, outcome=SUCCESS)
    rows = [_row("pa/s1", 0, 100, SUCCESS)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == []
    assert decision.retire_dirs == [duplicate]
    assert keep.exists()
    assert decision.kept_sim_ids == {0: 100}


def test_recorded_failure_only_in_the_superseded_tree_is_rerun(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=2, seed=0))
    _mk_episode(tmp_path / "episodes" / ".superseded", 0, planner="pa", stage="s1", sim_id=0, seed=0, outcome=FAILED, dirname="episode_000")
    _mk_episode(tmp_path, 1, planner="pa", stage="s1", sim_id=1, seed=1, outcome=FAILED)
    rows = [_row("pa/s1", 0, 0, FAILED), _row("pa/s1", 1, 1, FAILED)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == [0]
    assert decision.kept_sim_ids == {1: 1}
    assert decision.kept_failed == 1


def test_unhindered_peds_recording_is_kept_without_an_outcome(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant("unhindered_peds"), _stage(episodes=2, seed=0), is_reference=True, reference_type="unhindered_peds")
    _mk_episode(
        tmp_path,
        0,
        planner="unhindered_peds",
        stage="s1",
        sim_id=100,
        seed=0,
        outcome=None,
        is_reference=True,
        reference_type="unhindered_peds",
    )

    decision = _decide(tmp_path, step)

    assert decision.run_indices == [1]
    assert decision.kept_sim_ids == {0: 100}


def test_reused_sim_episode_id_is_disambiguated_by_the_recorded_seed(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=2, seed=0))
    _mk_episode(tmp_path, 0, planner="pa", stage="s1", sim_id=0, seed=0, outcome=SUCCESS)
    _mk_episode(tmp_path, 1, planner="pa", stage="s1", sim_id=0, seed=1, outcome=SUCCESS)
    rows = [
        _row("pa/s1", 0, 0, SUCCESS, ts="2026-01-01T00:00:01+00:00"),
        _row("pa/s1", 1, 0, SUCCESS, ts="2026-01-01T00:00:02+00:00"),
    ]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == []
    assert decision.kept_sim_ids == {0: 0, 1: 0}
    assert decision.retire_dirs == [] and decision.delete_dirs == []


def test_legacy_random_seeds_map_by_episode_id_when_the_stage_has_no_seed(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=3, seed=None))
    for index, sim_id in enumerate((1, 2, 3)):
        _mk_episode(tmp_path, index, planner="pa", stage="s1", sim_id=sim_id, seed=None, outcome=SUCCESS)
    rows = [_row("pa/s1", 9000 + sim_id, sim_id, SUCCESS) for sim_id in (1, 2, 3)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == []
    assert decision.kept_sim_ids == {0: 1, 1: 2, 2: 3}


def test_legacy_missing_episode_is_the_only_one_rerun(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=3, seed=None))
    for index, sim_id in ((0, 1), (2, 3)):
        _mk_episode(tmp_path, index, planner="pa", stage="s1", sim_id=sim_id, seed=None, outcome=SUCCESS)
    rows = [_row("pa/s1", 9000 + sim_id, sim_id, SUCCESS) for sim_id in (1, 3)]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == [1]
    assert decision.kept_sim_ids == {0: 1, 2: 3}


def test_seeded_stage_maps_by_seed_even_when_episode_ids_are_offset(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=3, seed=7))
    for index, sim_id in enumerate((100, 101, 102)):
        _mk_episode(tmp_path, index, planner="pa", stage="s1", sim_id=sim_id, seed=7 + index, outcome=SUCCESS)
    rows = [_row("pa/s1", 7 + index, sim_id, SUCCESS) for index, sim_id in enumerate((100, 101, 102))]

    decision = _decide(tmp_path, step, rows)

    assert decision.run_indices == []
    assert decision.kept_sim_ids == {0: 100, 1: 101, 2: 102}


def test_legacy_rows_without_a_seed_are_rerun(tmp_path: pathlib.Path) -> None:
    step = _step(_contestant(), _stage(episodes=2, seed=0))

    decision = _decide(tmp_path, step, [{"episode_id": 1}, {"episode_id": 2, "seed": None}])

    assert decision.run_indices == [0, 1]


def test_episode_identity_without_a_seed_uses_the_environment_id() -> None:
    episode = OnDiskEpisode(path=pathlib.Path("episode_000"), number=0, planner="pa", stage="s1", sim_episode_id=5)

    assert episode.matches_step(_step(_contestant(), _stage())) is True


# plan_resume


def test_plan_resume_plans_each_step_from_its_own_rows(tmp_path: pathlib.Path) -> None:
    step_a = _step(_contestant("pa"), _stage("a", episodes=2, seed=0))
    step_b = _step(_contestant("pb"), _stage("b", episodes=2, seed=0))
    _mk_episode(tmp_path, 0, planner="pa", stage="a", sim_id=100, seed=0, outcome=SUCCESS)
    _mk_episode(tmp_path, 1, planner="pa", stage="a", sim_id=101, seed=1, outcome=SUCCESS)
    live = load_episode_dirs(tmp_path / "episodes")
    rows_by_key = {"pa/a": [_row("pa/a", 0, 100, SUCCESS), _row("pa/a", 1, 101, SUCCESS)]}

    decisions = plan_resume([step_a, step_b], rows_by_key, live, retry_failed=False)
    by_key = {decision.step.key: decision for decision in decisions}

    assert by_key["pa/a"].run_indices == []
    assert by_key["pb/b"].run_indices == [0, 1]


def test_plan_resume_returns_one_decision_per_step(tmp_path: pathlib.Path) -> None:
    steps = [_step(_contestant(), _stage("a", episodes=1)), _step(_contestant(), _stage("b", episodes=1))]

    decisions = plan_resume(steps, {}, [], retry_failed=False)

    assert [decision.step.key for decision in decisions] == ["pa/a", "pa/b"]
    assert all(decision.run_indices == [0] for decision in decisions)
