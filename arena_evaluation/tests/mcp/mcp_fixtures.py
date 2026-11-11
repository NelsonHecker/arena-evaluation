from __future__ import annotations

import asyncio
import datetime
import json
import os
import pathlib
import sys
import time

import polars as pl
import pytest

sourced = pytest.mark.skipif("BASH_FUNC__arena_cli%%" not in os.environ, reason="needs a shell that sourced arena")

RUN_ID = "20261019-120000-basic-basic"
EMPTY_RUN_ID = "20261019-110000-empty-empty"
STEPS = ("dwb/s1", "dwb_social/s1", "unhindered_peds/s1", "dwb/s2")


def _record(world: str, outcome: int) -> object:
    from arena_runtime.env.task_layer import encode_modes
    from arena_runtime_msgs.msg import EpisodeRecord

    return EpisodeRecord(world=world, seed=7, modes=encode_modes({"robots": ["random"], "obstacles": ["random"]}), robots=["jackal"], outcome_state=outcome, outcome_info="")


def _manifest(run_id: str, steps: tuple[str, ...]) -> object:
    from arena_evaluation.benchmark.state import Manifest

    return Manifest(
        run_id=run_id,
        created_at="2026-10-19T12:00:00+00:00",
        arena_git_sha=None,
        arena_git_dirty=False,
        cli_args=["--suite", "basic", "--contest", "basic"],
        env_n=1,
        headless=True,
        config_hash="fixture",
        simulator="dummy",
        scale_episodes=1.0,
        suite_name="basic",
        contest_name="basic",
        suite={"stages": []},
        contest=[],
        steps=[{"key": key, "episodes_planned": 2} for key in steps],
    )


def metric_rows() -> list[dict]:
    """Canonical episode-summary rows: two contestants over two stages plus two reference rows."""
    rows = []
    for planner, success, ttg, collisions, path_efficiency, energy in (
        ("dwb", [True, True, True, False], [40.0, 42.0, 44.0, 90.0], [0, 0, 1, 2], [0.9, 0.8, 0.7, 0.5], [10.0, 11.0, 12.0, 20.0]),
        ("dwb_social", [True, False, False, False], [60.0, 90.0, 90.0, 90.0], [1, 2, 3, 3], [0.6, 0.5, 0.4, 0.3], [15.0, 18.0, 20.0, 25.0]),
    ):
        for index in range(4):
            rows.append(
                {
                    "episode_id": len(rows),
                    "planner": planner,
                    "stage": "s1" if index < 2 else "s2",
                    "world": "map_empty",
                    "is_reference": False,
                    "success": success[index],
                    "time_to_goal_s": ttg[index],
                    "total_collisions_ped": collisions[index],
                    "total_collisions_static": 0,
                    "path_efficiency": path_efficiency[index],
                    "total_energy_wh": energy[index],
                    "min_ped_clearance_m": 0.5,
                    "mar": 0.4,
                    "pfi": 0.2,
                }
            )
    for _ in range(2):
        rows.append(
            {
                "episode_id": len(rows),
                "planner": "unhindered_peds",
                "stage": "s1",
                "world": "map_empty",
                "is_reference": True,
                "success": None,
                "time_to_goal_s": None,
                "total_collisions_ped": None,
                "total_collisions_static": None,
                "path_efficiency": None,
                "total_energy_wh": None,
                "min_ped_clearance_m": None,
                "mar": None,
                "pfi": None,
            }
        )
    return rows


def write_star_schema(benchmark_dir: pathlib.Path, rows: list[dict]) -> None:
    """Write the canonical benchmark dimension and per-episode fact tables the catalog scans."""
    from arena_evaluation.storage.declarations import EPISODE_FACT_SCHEMA
    from arena_evaluation.storage.star_schema import DIM_EPISODES_SCHEMA

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    resolved = [{"run_id": "run", "robot_id": "robot_0", "env_id": "env_0", **row} for row in rows]
    pl.DataFrame([{name: row.get(name) for name in DIM_EPISODES_SCHEMA.names()} for row in resolved], schema=DIM_EPISODES_SCHEMA).write_parquet(benchmark_dir / "dim_episodes.parquet")
    for episode_id in sorted({int(row["episode_id"]) for row in resolved}):
        metrics = benchmark_dir / "episodes" / f"episode_{episode_id:03d}" / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        episode_rows = [row for row in resolved if int(row["episode_id"]) == episode_id]
        pl.DataFrame([{name: row.get(name) for name in EPISODE_FACT_SCHEMA.names()} for row in episode_rows], schema=EPISODE_FACT_SCHEMA).write_parquet(metrics / "fact_episodes.parquet")


def _iso(timestamp: float) -> str:
    return datetime.datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def build_runs(root: pathlib.Path) -> pathlib.Path:
    """A benchmark run written by the package's own writers: manifest, state and progress plus a second lane, an episode, logs and the Star Schema."""
    from arena_evaluation.benchmark.state import RunDir
    from arena_evaluation.benchmark.step import StepErrorKind, StepResult
    from arena_evaluation.storage.manifest import MetadataWriter
    from arena_evaluation.storage.schemas import RunMetadata

    root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    manifest = _manifest(RUN_ID, STEPS)
    run = RunDir.create(root, RUN_ID, manifest)
    run.state.write({"dwb/s1": StepResult("dwb/s1", "ok", 0, now - 60, now - 30, None, None, episodes_run=2), "unhindered_peds/s1": StepResult("unhindered_peds/s1", "in_progress", 0, now - 10, None, None, None)})
    for episode in range(2):
        run.progress.append(ts_iso=_iso(now - 55 + episode), run_id=RUN_ID, step_key="dwb/s1", contestant="dwb", stage="s1", env_id=0, episode_id=episode, episode_record=_record("map_empty", 2), started_at=now - 60, ended_at=now - 45)
    run.progress.close()
    lane = RunDir.create(root, RUN_ID, manifest, lane="p2")
    lane.state.write({"dwb_social/s1": StepResult("dwb_social/s1", "failed", 1, now - 50, now - 20, StepErrorKind.SIM_DEAD, "gz died")})
    lane.progress.append(ts_iso=_iso(now - 40), run_id=RUN_ID, step_key="dwb_social/s1", contestant="dwb_social", stage="s1", env_id=1, episode_id=2, episode_record=_record("map_empty", 3), started_at=now - 50, ended_at=now - 20)
    lane.progress.close()
    (run.path / "runner.log").write_text("".join(f"runner line {index}\n" for index in range(300)))
    (run.path / "runner.p2.log").write_text("lane p2 started\nlane p2 sim died\n")
    episode_dir = run.path / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True, exist_ok=True)
    MetadataWriter.write(RunMetadata(benchmark_id=RUN_ID, planner="dwb", map="map_empty", stage="s1", recording_started_at="2026-10-19T12:00:00", python_version="3.12", ros_distro="jazzy"), episode_dir / "episode_000.yaml")
    write_star_schema(run.path, metric_rows())
    empty = RunDir.create(root, EMPTY_RUN_ID, _manifest(EMPTY_RUN_ID, ("dwb/s1",)))
    empty.progress.close()
    return run.path


def server_env(tmp_path: pathlib.Path, **extra: str) -> dict[str, str]:
    return {**os.environ, "ARENA_DATA_DIR": str(tmp_path / "data"), **extra}


async def session(body: object, env: dict[str, str], domains: tuple[str, ...] = ("evaluation", "jobs")) -> object:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "arena_mcp", "serve", *domains], env=env)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        return await body(client)


def run_session(body: object, env: dict[str, str], domains: tuple[str, ...] = ("evaluation", "jobs")) -> object:
    return asyncio.run(session(body, env, domains))


def payload(result: object) -> dict:
    """The JSON object a tool returned."""
    assert not result.is_error, result.content[0].text if result.content else result
    assert len(result.content) == 1, result.content
    return json.loads(result.content[0].text)


def error_text(result: object) -> str:
    assert result.is_error, result
    return result.content[0].text
