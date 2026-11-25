from __future__ import annotations

import pathlib

import pytest

from arena_evaluation.benchmark.config import Contest, Contestant, Suite
from arena_evaluation.benchmark.resume import OnDiskEpisode, decide_step, seed_for_episode
from arena_evaluation.benchmark.runner import (
    _default_run_id,
    build_launch_args,
    group_pending,
    resolve_planner_identity,
    robot_has_depth_camera,
    runtime_launch_args,
)
from arena_evaluation.benchmark.step import Step

RANDOM_ROBOTS = "random"
RANDOM_OBSTACLES = "random"


def _stage(
    name: str = "s1",
    *,
    episodes: int = 4,
    robot: str = "jackal",
    world: str = "map1",
    optim: dict | None = None,
) -> Suite.Stage:
    return Suite.Stage(
        name=name,
        episodes=episodes,
        robot=robot,
        map=world,
        modes={"robots": RANDOM_ROBOTS, "obstacles": RANDOM_OBSTACLES},
        config={},
        seed=7,
        timeout=60.0,
        optim=optim,
    )


def _suite(*stages: Suite.Stage, references: bool = False) -> Suite:
    return Suite(name="suite", stages=list(stages), references=references)


def _contest(*names: str, args: dict | None = None) -> Contest:
    return Contest(
        name="contest",
        description=None,
        contestants=[Contestant(name=n, args=args or {}) for n in names],
    )


def _step(
    contestant: str = "dwb",
    stage: Suite.Stage | None = None,
    args: dict | None = None,
    **kwargs: object,
) -> Step:
    return Step(
        contestant=Contestant(name=contestant, args=args or {}),
        stage=stage or _stage(),
        **kwargs,
    )


def test_launch_args_carry_stage_optim_overrides():
    args = build_launch_args(_step(stage=_stage(optim={"lr": 0.1, "beta": 2})), None)

    assert "optim.lr:=0.1" in args
    assert "optim.beta:=2" in args
    assert not any(a.startswith("sim:=") for a in args)


def test_launch_args_drop_stage_owned_keys_from_cap_dicts(runner_log: list[str]):
    step = _step(
        args={
            "robot": {"driver": "other"},
            "task": {"robots": "scenario", "speed": 2, "unset": ""},
            "mobile": {"driver": "nav2", "local_planner": "dwb"},
        }
    )

    args = build_launch_args(step, "gazebo")

    assert "robot:=other" not in args
    assert "task.robots:=scenario" not in args
    assert "task.speed:=2" in args
    assert "task.unset:=" not in args
    assert "robot.mobile:=nav2" in args
    assert "robot.mobile.local_planner:=dwb" in args
    assert sum("controlled by stage" in m for m in runner_log) == 2


def test_launch_args_forward_passthrough_except_runner_owned_and_stage_keys():
    args = build_launch_args(
        _step(),
        "gazebo",
        passthrough={
            "headless": "true",
            "env.n": "4",
            "env_n": "4",
            "world": "other",
            "viz.view": "robot",
        },
    )

    assert "viz.view:=robot" in args
    assert "world:=map1" in args
    assert not any(a.startswith(("headless:=", "env.n:=", "env_n:=", "world:=other")) for a in args)


def test_runtime_launch_args_keep_an_explicit_paused_lockstep():
    assert runtime_launch_args({"lockstep": "True", "lockstep.paused": "True"}, {})["lockstep.paused"] == "True"
    assert runtime_launch_args({"lockstep": "True"}, {"lockstep.paused": "true"})["lockstep.paused"] == "true"


def test_resume_plan_runs_missing_episodes_and_optionally_retries_failures(
    tmp_path: pathlib.Path,
):
    step = _step(stage=_stage("s0", episodes=4), episodes=4)
    live = [
        OnDiskEpisode(path=tmp_path / "episode_000", number=0, planner="dwb", stage="s0", sim_episode_id=100, seed=7, outcome_state=2, mcap_bytes=128, has_recording=True),
        OnDiskEpisode(path=tmp_path / "episode_001", number=1, planner="dwb", stage="s0", sim_episode_id=101, seed=8, outcome_state=3, mcap_bytes=128, has_recording=True),
        OnDiskEpisode(path=tmp_path / "episode_003", number=3, planner="dwb", stage="s0", sim_episode_id=103, seed=10, outcome_state=2, mcap_bytes=128, has_recording=True),
    ]
    rows = [
        {"ts_iso": "2026-01-01T00:00:01+00:00", "step_key": step.key, "seed": seed_for_episode(step.stage.seed, 0), "episode_id": 100, "outcome_state": 2},
        {"ts_iso": "2026-01-01T00:00:02+00:00", "step_key": step.key, "seed": seed_for_episode(step.stage.seed, 1), "episode_id": 101, "outcome_state": 3},
        {"ts_iso": "2026-01-01T00:00:03+00:00", "step_key": step.key, "seed": seed_for_episode(step.stage.seed, 3), "episode_id": 103, "outcome_state": 2},
    ]

    plain = decide_step(step, rows, live)
    forced = decide_step(step, rows, live, force_retry=True)

    assert plain.run_indices == [2]
    assert (plain.kept_run, plain.kept_failed) == (3, 1)
    assert forced.run_indices == [1, 2]
    assert (forced.kept_run, forced.kept_failed) == (2, 0)
    assert forced.retire_dirs == [tmp_path / "episode_001"]


def test_planner_identity_prefers_the_mobile_cap_planners():
    assert resolve_planner_identity(
        Contestant(
            name="x",
            args={"mobile": {"local_planner": "dwb", "inter_planner": "spline"}},
        )
    ) == ("dwb", "spline")
    assert resolve_planner_identity(Contestant(name="x", args={"mobile": {"local_planner": "mppi"}})) == ("mppi", "none")


def test_planner_identity_falls_back_to_the_contestant_name():
    from arena_evaluation.storage.planner_names import split_planner_name

    contestant = Contestant(name="teb-spline", args={"mobile": {"driver": "nav2"}})

    assert resolve_planner_identity(contestant) == split_planner_name("teb-spline")
    assert resolve_planner_identity(Contestant(name="teb-spline", args={})) == split_planner_name("teb-spline")


def test_unhindered_peds_steps_join_the_block_of_their_robot_and_world():
    jackal = _step("a", _stage("s1", robot="jackal", world="w1"))
    burger = _step("b", _stage("s2", robot="burger", world="w1"))
    peds_jackal = _step(
        "unhindered_peds",
        _stage("s1", robot="jackal", world="w1"),
        is_reference=True,
        reference_type="unhindered_peds",
    )
    peds_orphan = _step(
        "unhindered_peds",
        _stage("s3", robot="jackal", world="w9"),
        is_reference=True,
        reference_type="unhindered_peds",
    )

    groups = group_pending([jackal, burger, peds_jackal, peds_orphan], "gazebo")

    assert [[s.key for s in g] for g in groups] == [
        ["a/s1", "unhindered_peds/s1"],
        ["b/s2"],
        ["unhindered_peds/s3"],
    ]


def test_robot_without_an_assembly_is_judged_by_its_model_sensors():
    pytest.importorskip("arena_robots")

    assert robot_has_depth_camera("ridgeback_plus") is False


def test_default_run_id_names_lane_and_config_stems():
    run_id = _default_run_id("dir/basic.yaml", "[{name: a}]", "p1")

    assert run_id.endswith("-p1-basic-inline")
    assert len(run_id.split("-")[0]) == 8


def test_default_run_id_without_lane():
    assert _default_run_id("{stages: []}", "allplanners", None).endswith("-inline-allplanners")
