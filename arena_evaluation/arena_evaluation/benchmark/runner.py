from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import datetime
import logging
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import typing

_T = typing.TypeVar("_T")

import attrs
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_evaluation_msgs.msg import BenchmarkState
from arena_evaluation_msgs.srv import RecordEpisode
from arena_rclpy_mixins import ActionClientWrapper, ArenaMixinNode, ClientWrapper
from arena_runtime_msgs.msg import EnvRecord, EnvRegistry
from arena_runtime_msgs.srv import DespawnEnv, SpawnEnv

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from task_generator.constants import Constants
from task_generator_msgs.action import RunEpisode
from task_generator_msgs.msg import EpisodeRecord
from task_generator_msgs.srv import QueueEpisode

STATE_TOPIC = "/arena/benchmark/state"

_CANCEL_SETTLE_S = 30.0
_HEARTBEAT_S = 30.0

# EpisodeRecord.outcome_state labels (task_generator_msgs/msg/EpisodeRecord.msg)
_EPISODE_OUTCOME_LABELS = {0: "QUEUED", 1: "RUNNING", 2: "SUCCESS", 3: "FAILED", 4: "SKIPPED", 5: "FATAL"}

from ..storage.planner_names import split_planner_name
from .config import Contest, Suite
from .state import (
    Manifest,
    RunDir,
    capture_git_sha,
    compute_config_hash,
    find_most_recent_resumable,
)
from .step import Step, StepErrorKind, StepResult


class _WithSteps(typing.Protocol):
    steps: dict[str, StepResult]


class _HasStateSteps(typing.Protocol):
    """Structural interface required by build_pending: any object with .state.steps."""

    @property
    def state(self) -> _WithSteps: ...


_log = logging.getLogger(__name__)


def build_launch_args(step: Step, simulator: str | None, passthrough: dict[str, str] | None = None) -> list[str]:
    """Return the arena launch argument list for a step, given the simulator name."""
    s = step.stage
    args = [
        *([f"sim:={simulator}"] if simulator is not None else []),
        f"robot:={s.robot}",
        f"world:={s.map}",
        f"tm_robots:={s.tm_robots.value}",
        f"tm_obstacles:={s.tm_obstacles.value}",
        f"run_seed:={s.seed}",
        "auto_reset:=false",
        "tm_modules:=",
    ]
    if s.optim:
        for k, v in s.optim.items():
            args.append(f"optim.{k}:={v}")
    if step.record_dir is not None:
        args.append(f"record_data_dir:={step.record_dir}")
        args.append("disable_auto_recorder:=true")
    own_keys = {a.split(":=", 1)[0] for a in args}
    for k, v in step.contestant.args.items():
        if isinstance(v, dict):
            driver = v.get("driver")
            if driver:
                if k in own_keys:
                    _log.warning(
                        "contestant %r: arg %r=%r ignored, controlled by stage",
                        step.contestant.name,
                        k,
                        driver,
                    )
                else:
                    args.append(f"{k}:={driver}")
            for ik, iv in v.items():
                if ik == "driver" or not iv:
                    continue
                fk = f"{k}.{ik}"
                if fk in own_keys:
                    _log.warning(
                        "contestant %r: arg %r=%r ignored, controlled by stage",
                        step.contestant.name,
                        fk,
                        iv,
                    )
                    continue
                args.append(f"{fk}:={iv}")
        else:
            if not v:
                continue
            if k in own_keys:
                _log.warning(
                    "contestant %r: arg %r=%r ignored, controlled by stage",
                    step.contestant.name,
                    k,
                    v,
                )
                continue
            args.append(f"{k}:={v}")
            
    if passthrough:
        for k, v in passthrough.items():
            if k in ("headless", "env_n"):
                continue
            if k not in own_keys:
                args.append(f"{k}:={v}")
    return args


def _all_steps_grid(
    suite: Suite,
    contest: Contest,
    scale_episodes: float,
    record_root: pathlib.Path | None = None,
) -> list[Step]:
    """Generate all steps for a benchmark suite and contest."""
    steps: list[Step] = []
    seen: set[str] = set()

    episodes_dir = (record_root / "episodes") if record_root is not None else None

    for contestant in contest.contestants:
        for stage in suite.stages:
            main_step = Step(
                contestant=contestant,
                stage=stage,
                episodes=int(round(stage.episodes * scale_episodes)),
                record_dir=episodes_dir,
            )
            if not suite.references:
                if main_step.key in seen:
                    raise ValueError(f"duplicate step key: {main_step.key!r}")
                seen.add(main_step.key)
                steps.append(main_step)
                continue
            ref_robot_step = Step(
                contestant=contestant,
                stage=stage,
                episodes=int(round(stage.episodes * scale_episodes)),
                record_dir=episodes_dir,
                is_reference=True,
                reference_type="unobstructed_robot",
            )
            for step in (main_step, ref_robot_step):
                if step.key in seen:
                    raise ValueError(f"duplicate step key: {step.key!r}")
                seen.add(step.key)
                steps.append(step)

    if not suite.references:
        return steps

    dummy_peds_contestant = Contest.Contestant(name="unhindered_peds", args={})
    for stage in suite.stages:
        ref_peds_step = Step(
            contestant=dummy_peds_contestant,
            stage=stage,
            episodes=int(round(stage.episodes * scale_episodes)),
            record_dir=episodes_dir,
            is_reference=True,
            reference_type="unhindered_peds",
        )
        if ref_peds_step.key in seen:
            raise ValueError(f"duplicate step key: {ref_peds_step.key!r}")
        seen.add(ref_peds_step.key)
        steps.append(ref_peds_step)

    return steps


def build_pending(
    suite: Suite,
    contest: Contest,
    scale_episodes: float,
    run_dir: _HasStateSteps,
    retry_failed: bool,
    record_root: pathlib.Path,
) -> list[Step]:
    """Return the list of steps that still need to be run."""
    state_steps = run_dir.state.steps
    pending: list[Step] = []
    for step in _all_steps_grid(suite, contest, scale_episodes, record_root):
        existing = state_steps.get(step.key)
        if existing is None:
            pending.append(step)
            continue
        if existing.status == "ok":
            continue
        if existing.status == "failed" and not retry_failed:
            continue
        pending.append(step)
    return pending


def resolve_planner_identity(contestant: Contest.Contestant) -> tuple[str, str]:
    """Return (local_planner, inter_planner) for a contestant."""
    mobile = (contestant.args or {}).get("mobile")
    if isinstance(mobile, dict):
        lp = mobile.get("local_planner")
        if lp:
            ip = mobile.get("inter_planner")
            return str(lp), str(ip) if ip else "none"
    return split_planner_name(contestant.name)


def env_key(step: Step, simulator: str | None) -> tuple:
    """Steps with the same env_key reuse one env. Contestants always force a new env."""
    return (step.contestant.name, step.stage.robot, simulator)


def group_pending(steps: list[Step], simulator: str | None) -> list[list[Step]]:
    groups = collections.defaultdict(list)
    peds_steps = []
    
    for step in steps:
        if step.is_reference and step.reference_type == "unhindered_peds":
            peds_steps.append(step)
        else:
            groups[env_key(step, simulator)].append(step)
    
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    
    for step in peds_steps:
        placed = False
        for g in sorted_groups:
            if g[0].stage.robot == step.stage.robot:
                g.append(step)
                placed = True
                break
        if not placed:
            sorted_groups.append([step])
            
    return sorted_groups


def _walk_dict(d: dict, prefix: str = "") -> list[Parameter]:
    """Flatten a nested dict to rcl_interfaces Parameter[] with dot-joined leaf names."""
    out: list[Parameter] = []
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            if "min" in v and "max" in v:
                n_name = f"{name}.n"
                pv = ParameterValue()
                val_min = v["min"]
                val_max = v["max"]
                if isinstance(val_min, int) and isinstance(val_max, int):
                    pv.type = ParameterType.PARAMETER_INTEGER_ARRAY
                    pv.integer_array_value = [val_min, val_max]
                else:
                    pv.type = ParameterType.PARAMETER_DOUBLE_ARRAY
                    pv.double_array_value = [float(val_min), float(val_max)]
                p = Parameter()
                p.name = n_name
                p.value = pv
                out.append(p)
                
                other = {ik: iv for ik, iv in v.items() if ik not in ("min", "max")}
                if other:
                    out.extend(_walk_dict(other, name))
                continue
            out.extend(_walk_dict(v, name))
            continue
        pv = ParameterValue()
        if isinstance(v, bool):
            pv.type = ParameterType.PARAMETER_BOOL
            pv.bool_value = v
        elif isinstance(v, int):
            pv.type = ParameterType.PARAMETER_INTEGER
            pv.integer_value = v
        elif isinstance(v, float):
            pv.type = ParameterType.PARAMETER_DOUBLE
            pv.double_value = v
        elif isinstance(v, str):
            pv.type = ParameterType.PARAMETER_STRING
            pv.string_value = v
        elif isinstance(v, list):
            if not v:
                pv.type = ParameterType.PARAMETER_STRING_ARRAY
                pv.string_array_value = []
            elif all(isinstance(x, bool) for x in v):
                pv.type = ParameterType.PARAMETER_BOOL_ARRAY
                pv.bool_array_value = v
            elif all(isinstance(x, int) for x in v):
                pv.type = ParameterType.PARAMETER_INTEGER_ARRAY
                pv.integer_array_value = v
            elif all(isinstance(x, float) for x in v):
                pv.type = ParameterType.PARAMETER_DOUBLE_ARRAY
                pv.double_array_value = v
            elif all(isinstance(x, str) for x in v):
                pv.type = ParameterType.PARAMETER_STRING_ARRAY
                pv.string_array_value = v
            else:
                raise TypeError(f"unsupported mixed list type for {name!r}")
        else:
            raise TypeError(f"unsupported param type for {name!r}: {type(v).__name__}")
        p = Parameter()
        p.name = name
        p.value = pv
        out.append(p)
    return out


def _flatten_per_mode_params(
    stage_config: dict,
    *,
    tm_obstacles: str,
    tm_robots: str,
) -> tuple[list[Parameter], list[Parameter]]:
    """Route stage.config blocks to (obstacles_params, robots_params) as leaf-keyed Parameter[]."""
    obs: list[Parameter] = []
    rob: list[Parameter] = []
    for mode, mode_dict in (stage_config or {}).items():
        if not isinstance(mode_dict, dict):
            continue
        is_scenario = mode == "scenario"
        patched: dict = {k: (pathlib.Path(val).stem if is_scenario and k == "file" and isinstance(val, str) else val) for k, val in mode_dict.items()}
        params = _walk_dict(patched)
        if mode == tm_obstacles:
            obs.extend(params)
        if mode == tm_robots:
            rob.extend(params)
    return obs, rob


_LATCHED = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

_SYSTEMIC = (StepErrorKind.ENV_SETUP, StepErrorKind.ROBOT_SETUP)


class _EnvDied(Exception):
    """Raised when an env disappears from /arena/state/envs while the runner was waiting on it."""


class BenchmarkRunner(ArenaMixinNode):
    exit_code: typing.ClassVar[int] = 0

    def __init__(
        self,
        suite: Suite,
        contest: Contest,
        *,
        simulator: str | None,
        scale_episodes: float,
        env_n: int,
        run_id: str,
        headless: bool,
        run_dir: RunDir,
        retry_failed: bool = False,
        arena_passthrough: dict[str, str] | None = None,
        noexit: bool = False,
    ) -> None:
        super().__init__("arena_benchmark_runner")
        self._suite = suite
        self._contest = contest
        self._simulator = simulator
        self._scale_episodes = scale_episodes
        self._env_n = env_n
        self._run_id = run_id
        self._headless = headless
        self._run_dir = run_dir
        self._retry_failed = retry_failed
        self._arena_passthrough: dict[str, str] = dict(arena_passthrough or {})
        self._noexit = noexit
        self._total_groups = 0
        self._completed_groups = 0

        self._spawn = self.create_client_wrapper(SpawnEnv, "/arena/spawn_env")
        self._despawn = self.create_client_wrapper(DespawnEnv, "/arena/despawn_env")
        self._env_records: dict[int, EnvRecord] = {}
        self._env_visible_events: dict[int, asyncio.Event] = {}
        self._env_gone_events: dict[int, asyncio.Event] = {}

        self._arena_proc: subprocess.Popen | None = None
        self._arena_log_file = None
        self._total_groups = 0
        self._episode_action_clients: dict[int, ActionClientWrapper] = {}
        self._queue_clients: dict[int, ClientWrapper] = {}

        self._episode_records: dict[int, dict[int, EpisodeRecord]] = {}
        self._env_subs: dict[int, list] = {}
        self._recorder_clients: dict[int, ClientWrapper] = {}
        self._triggered_episodes: dict[int, set[int]] = {}

        self.create_subscription(EnvRegistry, "/arena/state/envs", self._on_envs, _LATCHED)
        self._state_pub = self.create_publisher(BenchmarkState, STATE_TOPIC, _LATCHED)

        self._arena_proc: subprocess.Popen | None = None
        self._parent_episode_map: dict[tuple[str, str, int], int] = {}

    def _build_pending(self) -> list[Step]:
        return build_pending(
            suite=self._suite,
            contest=self._contest,
            scale_episodes=self._scale_episodes,
            run_dir=self._run_dir,
            retry_failed=self._retry_failed,
            record_root=self._run_dir.path,
        )

    def _on_envs(self, msg: EnvRegistry) -> None:
        new_ids = {e.env_id for e in msg.envs}
        for env_id in new_ids:
            self._env_visible_events.setdefault(env_id, asyncio.Event()).set()
        for env_id in list(self._env_gone_events):
            if env_id not in new_ids:
                self._env_gone_events[env_id].set()
        self._env_records = {e.env_id: e for e in msg.envs}

    def _build_launch_args(self, step: Step) -> list[str]:
        return build_launch_args(step, self._simulator, passthrough=self._arena_passthrough)

    async def _await_env_visible(self, env_id: int) -> None:
        """Wait for env_id to appear on /arena/state/envs."""
        if env_id in self._env_records:
            return
        await self._await_hb(self._env_visible_events.setdefault(env_id, asyncio.Event()).wait(), f"env {env_id} to appear on /arena/state/envs")

    async def _await_hb(self, awaitable: typing.Awaitable[_T], what: str) -> _T:
        """Await with no timeout, warning every _HEARTBEAT_S while `what` stays pending."""
        task = asyncio.ensure_future(awaitable)
        waited = 0.0
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=_HEARTBEAT_S)
                if task in done:
                    return task.result()
                waited += _HEARTBEAT_S
                _log.warning(f"still waiting for {what} ({waited:.0f}s elapsed)")
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _await_or_env_died(self, env_id: int, awaitable: typing.Awaitable[_T]) -> _T:
        """Race awaitable against env death. Raises _EnvDied if env_id disappears first."""
        death = self._env_gone_events.setdefault(env_id, asyncio.Event())
        op_task = asyncio.ensure_future(awaitable)
        death_task = asyncio.ensure_future(death.wait())
        try:
            done, pending = await asyncio.wait({op_task, death_task}, return_when=asyncio.FIRST_COMPLETED)
            if death_task in done and op_task not in done:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise _EnvDied(f"env {env_id} disappeared from /arena/state/envs")
            if not death_task.done():
                death_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await death_task
            return op_task.result()
        except asyncio.CancelledError:
            op_task.cancel()
            death_task.cancel()
            raise

    async def _wait_env_gone(self, env_id: int, *, timeout: float | None) -> bool:
        ev = asyncio.Event()
        self._env_gone_events[env_id] = ev
        try:
            if env_id not in self._env_records:
                return True
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False
        finally:
            self._env_gone_events.pop(env_id, None)

    async def _setup_env_clients(self, env_id: int, env_ns_root: str, robot_name: str) -> None:
        """Create per-env action client, queue_episode client, and subscriptions. Idempotent."""
        if env_id in self._episode_action_clients:
            return

        action_name = f"{env_ns_root}/lifecycle/run_episode"
        self._episode_action_clients[env_id] = self.create_action_client_wrapper(RunEpisode, action_name)
        self._episode_records[env_id] = {}

        queue_client = self.create_client_wrapper(QueueEpisode, f"{env_ns_root}/config/queue_episode")
        await self._await_hb(queue_client.ensure(timeout_sec=None), f"queue_episode service on env {env_id}")
        self._queue_clients[env_id] = queue_client

        self._recorder_clients[env_id] = self.create_client_wrapper(
            RecordEpisode, f"{env_ns_root}/start_episode"
        )
        self._triggered_episodes[env_id] = set()

        def _on_episode_record(msg: EpisodeRecord) -> None:
            recs = self._episode_records.get(env_id)
            if recs is not None:
                recs[msg.episode_id] = msg
            label = _EPISODE_OUTCOME_LABELS.get(msg.outcome_state, f"UNKNOWN({msg.outcome_state})")
            _log.info(
                f"[env {env_id}] episode record: id={msg.episode_id} "
                f"outcome={msg.outcome_state} ({label}) info={msg.outcome_info!r}"
            )
            if msg.outcome_state in (EpisodeRecord.QUEUED, EpisodeRecord.RUNNING):
                triggered = self._triggered_episodes.get(env_id)
                if triggered is not None and msg.episode_id not in triggered:
                    triggered.add(msg.episode_id)
                    rc = self._recorder_clients.get(env_id)
                    if rc is not None:
                        _log.info(f"[env {env_id}] signalling recorder to start episode {msg.episode_id}")
                        req = RecordEpisode.Request()
                        req.command = RecordEpisode.Request.COMMAND_START
                        req.episode_id = msg.episode_id
                        fut = rc.client.call_async(req)
                        fut.add_done_callback(
                            lambda f, eid=msg.episode_id: _log.warning(
                                f"[env {env_id}] recorder start_episode({eid}) failed: {f.exception()}"
                            ) if f.exception() else _log.info(
                                f"[env {env_id}] recorder confirmed start of episode {eid}"
                            )
                        )

        sub_ep = self.create_subscription(
            EpisodeRecord,
            f"{env_ns_root}/state/episode",
            _on_episode_record,
            10,
        )
        self._env_subs[env_id] = [sub_ep]

    def _teardown_env_clients(self, env_id: int) -> None:
        """Destroy per-env subscriptions, action client, and queue_episode client."""
        for sub in self._env_subs.pop(env_id, []):
            self.destroy_subscription(sub)
        ac = self._episode_action_clients.pop(env_id, None)
        if ac is not None:
            ac.client.destroy()
        qc = self._queue_clients.pop(env_id, None)
        if qc is not None:
            qc.client.destroy()
        rc = self._recorder_clients.pop(env_id, None)
        if rc is not None:
            rc.client.destroy()
        self._triggered_episodes.pop(env_id, None)
        self._episode_records.pop(env_id, None)
        self._env_visible_events.pop(env_id, None)

    async def _push_stage_config(self, env_id: int, step: Step) -> None:
        queue = self._queue_clients[env_id]
        req = QueueEpisode.Request()
        req.action = QueueEpisode.Request.MERGE
        req.world = step.stage.map
        req.tm_robots = step.stage.tm_robots.value
        req.tm_obstacles = step.stage.tm_obstacles.value
        req.tm_modules = []
        req.keep_modules = False
        stage_config = step.stage.config or {}
        if step.is_reference and step.reference_type == "unobstructed_robot":
            import copy
            stage_config = copy.deepcopy(stage_config)
            mode_block = stage_config.setdefault(step.stage.tm_obstacles.value, {})
            if isinstance(mode_block, dict):
                mode_block["dynamic"] = {"min": 0, "max": 0}

        obs_params, rob_params = _flatten_per_mode_params(
            stage_config,
            tm_obstacles=step.stage.tm_obstacles.value,
            tm_robots=step.stage.tm_robots.value,
        )
        if step.is_reference:
            if step.reference_type == "unobstructed_robot":
                pass
            elif step.reference_type == "unhindered_peds":
                req.tm_robots = "stationary"
                rob_params = [
                    Parameter(
                        name="pos_x",
                        value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=1000.0)
                    ),
                    Parameter(
                        name="pos_y",
                        value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=1000.0)
                    )
                ]
        req.obstacles_params = obs_params
        req.robots_params = rob_params
        resp = await self.await_ros(queue.client.call_async(req))
        if not resp.success:
            raise RuntimeError(f"queue_episode failed for {step.key}: {resp.error_msg}")

    async def _run_episodes(
        self,
        step: Step,
        env_id: int,
    ) -> StepResult:
        """Drive all episodes for one step. Env is already up and clients are set up."""
        started = time.time()
        episodes_run = 0
        episodes_failed = 0
        ac = self._episode_action_clients[env_id]

        try:
            for ep_idx in range(step.episodes):
                goal = RunEpisode.Goal()
                goal.world = step.stage.map
                goal.seed = (step.stage.seed + ep_idx) if step.stage.seed is not None else ep_idx

                ep_started_sim = self.sim_time.to_seconds()
                ep_started_wall = time.time()

                try:
                    goal_handle = await self._await_or_env_died(env_id, ac.send_goal(goal))
                except (_EnvDied, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    episodes_failed += 1
                    _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} goal rejected ({exc}); env action server is wedged, abandoning env")
                    return StepResult(
                        step.key,
                        "failed",
                        env_id,
                        started,
                        time.time(),
                        StepErrorKind.ENV_SETUP,
                        f"run_episode goal rejected: {exc}",
                        episodes_run=episodes_run,
                        episodes_failed=episodes_failed,
                        episodes_total=step.episodes,
                    )

                timeout_s = step.stage.timeout
                if step.is_reference and step.reference_type == "unhindered_peds" and step.stage.timeout_peds is not None:
                    timeout_s = step.stage.timeout_peds

                try:
                    result_obj = await asyncio.wait_for(
                        self._await_or_env_died(env_id, ac.await_result(goal_handle)),
                        timeout=timeout_s,
                    )
                except TimeoutError:
                    episodes_failed += 1
                    _log.warning(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} TIMEOUT after {timeout_s}s; cancelling and advancing")
                    try:
                        await self.await_ros(goal_handle.cancel_goal_async())
                        await asyncio.wait_for(
                            self._await_or_env_died(env_id, ac.await_result(goal_handle)),
                            timeout=_CANCEL_SETTLE_S,
                        )
                    except _EnvDied:
                        raise
                    except TimeoutError:
                        _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} cancel did not settle in {_CANCEL_SETTLE_S}s; env is wedged, abandoning env")
                        return StepResult(
                            step.key,
                            "failed",
                            env_id,
                            started,
                            time.time(),
                            StepErrorKind.ENV_SETUP,
                            "run_episode cancel did not settle; env wedged",
                            episodes_run=episodes_run,
                            episodes_failed=episodes_failed,
                            episodes_total=step.episodes,
                        )
                    except Exception as exc:
                        _log.warning(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} error awaiting cancel ({exc}); advancing")
                    continue
                ep_ended_sim = self.sim_time.to_seconds()
                ep_ended_wall = time.time()

                result: RunEpisode.Result = result_obj.result
                episode_id = result.episode_id

                if result.state == RunEpisode.Result.FATAL:
                    _log.error(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} FATAL: {result.info} -- aborting step")
                    return StepResult(
                        step.key,
                        "failed",
                        env_id,
                        started,
                        time.time(),
                        StepErrorKind.ROBOT_SETUP,
                        f"env reported FATAL: {result.info}",
                        episodes_run=episodes_run,
                        episodes_failed=episodes_failed,
                        episodes_total=step.episodes,
                    )

                recs = self._episode_records.get(env_id, {})
                rec = recs.get(episode_id)
                if rec is None:
                    episodes_failed += 1
                    _log.warning(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} no EpisodeRecord for episode_id={episode_id}; counted as failed")
                    continue

                episodes_run += 1
                if rec.outcome_state == EpisodeRecord.FAILED:
                    episodes_failed += 1

                state_label = {
                    EpisodeRecord.SUCCESS: "SUCCESS",
                    EpisodeRecord.FAILED: "FAILED",
                    EpisodeRecord.SKIPPED: "SKIPPED",
                }.get(rec.outcome_state, str(rec.outcome_state))
                _log.info(f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} {state_label} info={rec.outcome_info!r} sim={ep_ended_sim - ep_started_sim:.1f}s wall={ep_ended_wall - ep_started_wall:.1f}s")

                if rec.outcome_state in (
                    EpisodeRecord.SUCCESS,
                    EpisodeRecord.FAILED,
                    EpisodeRecord.SKIPPED,
                    EpisodeRecord.FATAL,
                ):
                    rc = self._recorder_clients.get(env_id)
                    if rc is not None:
                        req = RecordEpisode.Request()
                        req.command = RecordEpisode.Request.COMMAND_STOP
                        req.episode_id = episode_id
                        req.outcome_state = rec.outcome_state
                        req.outcome_info = rec.outcome_info
                        try:
                            resp = await rc.call_timeout(req, timeout_sec=5.0)
                            if resp is None:
                                _log.warning(f"[env {env_id}] recorder stop_episode({episode_id}) timed out")
                            else:
                                _log.info(f"[env {env_id}] recorder stopped episode {episode_id} ({state_label})")
                        except Exception as exc:
                            _log.warning(f"[env {env_id}] recorder stop_episode({episode_id}) failed: {exc}")

                parent_ep_id = None
                if not step.is_reference:
                    self._parent_episode_map[(step.contestant.name, step.stage.name, ep_idx)] = episode_id
                    self._parent_episode_map[("__stage__", step.stage.name, ep_idx)] = episode_id
                else:
                    if step.reference_type == "unobstructed_robot":
                        parent_ep_id = self._parent_episode_map.get((step.contestant.name, step.stage.name, ep_idx))
                    else:
                        parent_ep_id = self._parent_episode_map.get(("__stage__", step.stage.name, ep_idx))

                ts_iso = datetime.datetime.now(tz=datetime.UTC).isoformat()
                self._run_dir.progress.append(
                    ts_iso=ts_iso,
                    run_id=self._run_id,
                    step_key=step.key,
                    contestant=step.contestant.name,
                    stage=step.stage.name,
                    env_id=env_id,
                    episode_id=episode_id,
                    episode_record=rec,
                    started_at=ep_started_sim,
                    ended_at=ep_ended_sim,
                    parent_episode_id=parent_ep_id,
                    is_reference=step.is_reference,
                    reference_type=step.reference_type,
                )
        except _EnvDied as exc:
            _log.warning(f"{step.key} env={env_id} env died mid-step after run={episodes_run}, failed={episodes_failed}: {exc}")
            return StepResult(
                step.key,
                "failed",
                env_id,
                started,
                time.time(),
                StepErrorKind.ENV_SETUP,
                repr(exc),
                episodes_run=episodes_run,
                episodes_failed=episodes_failed,
                episodes_total=step.episodes,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.exception(f"{step.key} env={env_id} unexpected error mid-step after run={episodes_run}, failed={episodes_failed}")
            return StepResult(
                step.key,
                "failed",
                env_id,
                started,
                time.time(),
                StepErrorKind.INTERNAL,
                repr(exc),
                episodes_run=episodes_run,
                episodes_failed=episodes_failed,
                episodes_total=step.episodes,
            )

        if episodes_run == 0:
            status = "failed"
        elif episodes_failed == 0:
            status = "ok"
        elif episodes_failed < episodes_run:
            status = "partial"
        else:
            status = "failed"

        return StepResult(
            step.key,
            status,
            env_id,
            started,
            time.time(),
            None,
            None,
            episodes_run=episodes_run,
            episodes_failed=episodes_failed,
            episodes_total=step.episodes,
        )

    async def _spawn_and_setup_env(self, launch_step: Step) -> tuple[int, str] | None:
        """Spawn one env booting launch_step's world and set up its clients. None on failure."""
        req = SpawnEnv.Request()
        req.ns = ""
        req.headless = self._headless
        req.launch_args = self._build_launch_args(launch_step)
        resp = await self._await_hb(self.await_ros(self._spawn.client.call_async(req)), f"spawn_env for {launch_step.key}")
        if resp is None or not resp.success:
            msg = resp.error_msg if resp is not None else "no response"
            _log.error(f"spawn_env failed for {launch_step.key}: {msg}")
            return None
        env_id = resp.env_id
        await self._await_env_visible(env_id)
        env_ns_root = self._env_records[env_id].fqn
        await self._setup_env_clients(env_id, env_ns_root, launch_step.stage.robot)
        return env_id, env_ns_root

    async def _despawn_env(self, env_id: int) -> None:
        """Tear down clients and despawn an env, waiting for it to disappear from the registry."""
        self._teardown_env_clients(env_id)
        if env_id in self._env_records:
            with contextlib.suppress(Exception):
                dreq = DespawnEnv.Request()
                dreq.env_id = env_id
                await self._await_hb(self._despawn.call_forever(dreq), f"despawn of env {env_id}")
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await self._await_hb(self._wait_env_gone(env_id, timeout=None), f"env {env_id} to disappear")
        await asyncio.sleep(2.0)

    async def _run_group_queue(self, rep_step: Step, q: asyncio.Queue[Step], slot_index: int, flush_cb: typing.Callable[[StepResult], bool]) -> bool:
        env_id: int | None = None
        try:
            spawned = await self._spawn_and_setup_env(rep_step)
            if spawned is None:
                abort = False
                while not q.empty():
                    try:
                        step = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    res = StepResult(
                        step.key,
                        "failed",
                        None,
                        time.time(),
                        time.time(),
                        StepErrorKind.ENV_SETUP,
                        "spawn_env failed",
                        episodes_total=step.episodes,
                    )
                    if flush_cb(res):
                        abort = True
                return abort
            env_id, env_ns_root = spawned

            while not q.empty():
                try:
                    step = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                
                recorder_proc = None
                if step.record_dir is not None:
                    episode_id_offset = self._global_episode_id_offset
                    self._global_episode_id_offset += step.episodes

                    lp, ip = resolve_planner_identity(step.contestant)
                    recorder_args = [
                        "run",
                        "arena_evaluation",
                        "record",
                        "--ros-args",
                        "-p", "use_sim_time:=true",
                        "-p", f"record_data_dir:={step.record_dir}",
                        "-p", f"benchmark_id:={self._run_id}",
                        "-p", f"contestant:={step.contestant.name}",
                        "-p", f"stage:={step.stage.name}",
                        "-p", f"map:={step.stage.map}",
                        "-p", f"suite_name:={self._suite.name}",
                        "-p", f"contest_name:={self._contest.name}",
                        "-p", f"local_planner:={lp}",
                        "-p", f"inter_planner:={ip}",
                        "-p", f"robot:={step.stage.robot}",
                        "-p", f"episodes_requested:={step.episodes}",
                        "-p", f"is_reference:={str(step.is_reference).lower()}",
                    ]
                    if step.reference_type:
                        recorder_args.extend(["-p", f"reference_type:={step.reference_type}"])

                    recorder_args.extend([
                        "-p", f"episode_id_offset:={episode_id_offset}",
                        "-r", f"__ns:={env_ns_root}",
                    ])

                    recorder_proc = await asyncio.create_subprocess_exec(
                        "ros2",
                        *recorder_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    await asyncio.sleep(2.0)

                await self._push_stage_config(env_id, step)

                try:
                    step_result = await self._run_episodes(step, env_id)
                except asyncio.CancelledError:
                    step_result = StepResult(
                        step.key,
                        "skipped",
                        env_id,
                        time.time(),
                        time.time(),
                        StepErrorKind.CANCELLED,
                        "cancelled",
                        episodes_total=step.episodes,
                    )
                    flush_cb(step_result)
                    # if cancelled, drain the queue and mark skipped
                    while not q.empty():
                        try:
                            rem_step = q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        flush_cb(StepResult(
                            rem_step.key,
                            "skipped",
                            env_id,
                            time.time(),
                            time.time(),
                            StepErrorKind.CANCELLED,
                            "cancelled",
                            episodes_total=rem_step.episodes,
                        ))
                    raise
                except _EnvDied as exc:
                    step_result = StepResult(
                        step.key,
                        "failed",
                        env_id,
                        time.time(),
                        time.time(),
                        StepErrorKind.ENV_SETUP,
                        repr(exc),
                        episodes_total=step.episodes,
                    )
                except Exception as exc:
                    step_result = StepResult(
                        step.key,
                        "failed",
                        env_id,
                        time.time(),
                        time.time(),
                        StepErrorKind.INTERNAL,
                        repr(exc),
                        episodes_total=step.episodes,
                    )
                finally:
                    if recorder_proc is not None:
                        try:
                            os.killpg(os.getpgid(recorder_proc.pid), signal.SIGINT)
                            await asyncio.wait_for(recorder_proc.wait(), timeout=10.0)
                        except asyncio.TimeoutError:
                            _log.warning(f"Recorder process timed out during shutdown, killing it.")
                            try:
                                os.killpg(os.getpgid(recorder_proc.pid), signal.SIGKILL)
                            except Exception:
                                pass
                        except Exception as e:
                            _log.warning(f"Failed to cleanly terminate recorder process: {e}")
                            try:
                                os.killpg(os.getpgid(recorder_proc.pid), signal.SIGKILL)
                            except Exception:
                                pass

                abort = flush_cb(step_result)
                if abort:
                    return True

        finally:
            self._completed_groups += 1
            keep_alive = self._noexit and self._completed_groups == self._total_groups and env_id is not None
            if env_id is not None:
                self._teardown_env_clients(env_id)
                if env_id in self._env_records and not keep_alive:
                    with contextlib.suppress(Exception):
                        dreq = DespawnEnv.Request()
                        dreq.env_id = env_id
                        await self._await_hb(self._despawn.call_forever(dreq), f"despawn of env {env_id}")
                    with contextlib.suppress(asyncio.TimeoutError, Exception):
                        await self._await_hb(self._wait_env_gone(env_id, timeout=None), f"env {env_id} to disappear")
                if keep_alive:
                    _log.info(f"--noexit: keeping env {env_id} alive after last group {rep_step.key}")
        
        return False

    def _publish_state(
        self,
        results: typing.Mapping[str, StepResult],
        steps_total: int,
    ) -> None:
        if not rclpy.ok():
            return
        msg = BenchmarkState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.run_id = self._run_id
        msg.suite = self._suite.name
        msg.contest = self._contest.name
        msg.simulator = self._simulator or ""
        msg.env_n = self._env_n
        msg.headless = self._headless
        msg.steps_total = steps_total
        msg.steps_done = sum(1 for r in results.values() if r.status == "ok")
        msg.steps_partial = sum(1 for r in results.values() if r.status == "partial")
        msg.steps_failed = sum(1 for r in results.values() if r.status == "failed")
        msg.steps_skipped = sum(1 for r in results.values() if r.status == "skipped")
        msg.steps_in_flight = sum(1 for r in results.values() if r.status == "in_progress")
        msg.active_keys = [k for k, r in results.items() if r.status == "in_progress"]
        self._state_pub.publish(msg)

    async def setup(self) -> None:
        try:
            BenchmarkRunner.exit_code = await self._run_steps()
        except Exception as exc:
            _log.error(f"benchmark crashed: {exc!r}")
            BenchmarkRunner.exit_code = 2
        finally:
            await self._shutdown_arena()
            rclpy.try_shutdown()

    async def teardown(self) -> None:
        await self._shutdown_arena()

    async def _shutdown_arena(self) -> None:
        if self._noexit:
            _log.info("--noexit: leaving arena_runtime.launch.py running; Ctrl+C its terminal to stop")
            return
        p = self._arena_proc
        if p is None or p.poll() is not None:
            return
        loop = asyncio.get_running_loop()
        for sig, grace in ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0)):
            try:
                os.killpg(os.getpgid(p.pid), sig)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(loop.run_in_executor(None, p.wait), timeout=grace)
                return
            except TimeoutError:
                continue
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

        if self._arena_log_file is not None:
            with contextlib.suppress(Exception):
                self._arena_log_file.close()
            self._arena_log_file = None

    async def _run_steps(self) -> int:
        pending = self._build_pending()
        results: dict[str, StepResult] = dict(self._run_dir.state.steps)
        steps_total = len(results) + len(pending)
        aborted_systemic = False

        self._publish_state(results, steps_total)
        _log.info(f"benchmark: signalled READY on {STATE_TOPIC}")

        passthrough = dict(self._arena_passthrough)
        cmd = [
            "ros2",
            "launch",
            "arena_bringup",
            "arena_runtime.launch.py",
            *(f"{k}:={v}" for k, v in passthrough.items()),
        ]

        log_path = self._run_dir.path / "runner.log"
        self._arena_log_file = log_path.open("a")

        self._arena_proc = subprocess.Popen(
            cmd,
            stdout=self._arena_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        await self._spawn.ensure(timeout_sec=300.0)
        await self._despawn.ensure(timeout_sec=300.0)

        self._global_episode_id_offset = 0
        episodes_dir = self._run_dir.path / "episodes"
        if episodes_dir.exists():
            for d in episodes_dir.glob("episode_*"):
                try:
                    idx = int(d.name.split("_")[1])
                    self._global_episode_id_offset = max(self._global_episode_id_offset, idx + 1)
                except Exception:
                    pass

        blocks = group_pending(pending, self._simulator)
        self._total_groups = len(blocks)
        
        block_queues: list[tuple[Step, asyncio.Queue[Step]]] = []
        for block in blocks:
            q = asyncio.Queue()
            for step in block:
                q.put_nowait(step)
            block_queues.append((block[0], q))

        cap = max(1, min(self._env_n, len(pending) or 1))
        in_flight: set[asyncio.Task[bool]] = set()

        def _mark_step_in_progress(step: Step) -> None:
            results[step.key] = StepResult(
                step.key,
                "in_progress",
                None,
                time.time(),
                None,
                None,
                None,
                episodes_total=step.episodes,
            )
            self._run_dir.state.write(results)
            self._publish_state(results, steps_total)

        def _flush_step_result(res: StepResult) -> bool:
            results[res.key] = res
            _log.info(f"[{res.status}] {res.key} env={res.env_id} episodes={res.episodes_run}/{res.episodes_total} (failed={res.episodes_failed}) t={((res.ended_at or 0.0) - res.started_at):.1f}s")
            self._run_dir.state.write(results)
            self._publish_state(results, steps_total)
            
            total_episodes_run = sum(r.episodes_run for r in results.values())
            if total_episodes_run > 0:
                return False
            return res.status == "failed" and res.error_kind in _SYSTEMIC

        async def _worker(slot_index: int) -> bool:
            while True:
                target_q = None
                rep_step = None
                for r_step, q in block_queues:
                    if not q.empty():
                        target_q = q
                        rep_step = r_step
                        break
                
                if target_q is None:
                    break
                
                abort = await self._run_group_queue(rep_step, target_q, slot_index, _flush_step_result)
                if abort:
                    return True
            return False

        try:
            for slot in range(cap):
                in_flight.add(asyncio.create_task(_worker(slot), name=f"worker_{slot}"))
            
            while in_flight:
                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    abort = t.result()
                    if abort:
                        aborted_systemic = True
                        _log.error(f"benchmark: worker hit a systemic setup failure; aborting run")
                        for t2 in in_flight:
                            t2.cancel()
                        with contextlib.suppress(Exception):
                            await asyncio.gather(*in_flight, return_exceptions=True)
                        in_flight.clear()
                        block_queues.clear()
                        break
        except asyncio.CancelledError:
            for t in in_flight:
                t.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)
            raise
        finally:
            self._run_dir.progress.dedupe_in_place()
            self._publish_state(results, steps_total)

        return 1 if aborted_systemic else 0


def _all_steps(contest: Contest, suite: Suite, scale_episodes: float) -> list[Step]:
    return _all_steps_grid(suite, contest, scale_episodes, record_root=None)


def _is_inline_contest(contest_name: str) -> bool:
    stripped = contest_name.strip()
    return stripped.startswith("[") or stripped.startswith("{")


def _is_inline_suite(suite_name: str) -> bool:
    stripped = suite_name.strip()
    return stripped.startswith("[") or stripped.startswith("{")


def _load_suite_contest(suite_name: str, contest_name: str) -> tuple[Suite, Contest, dict, list | dict]:
    share = pathlib.Path(get_package_share_directory("arena_evaluation"))
    bench_dir = share / "configs" / "benchmark"

    if _is_inline_suite(suite_name):
        suite_dict = yaml.safe_load(suite_name)
        suite = Suite.parse("inline", suite_dict)
    else:
        suite_stem = suite_name.removesuffix(".yaml")
        suite_path = bench_dir / "suites" / f"{suite_stem}.yaml"
        suite_dict = yaml.safe_load(suite_path.read_text())
        suite = Suite.parse(suite_stem, suite_dict)

    if _is_inline_contest(contest_name):
        contest_dict = yaml.safe_load(contest_name)
        contest = Contest.parse("inline", contest_dict)
    else:
        contest_stem = contest_name.removesuffix(".yaml")
        contest_path = bench_dir / "contests" / f"{contest_stem}.yaml"
        contest_dict = yaml.safe_load(contest_path.read_text())
        contest = Contest.parse(contest_stem, contest_dict)

    return suite, contest, suite_dict, contest_dict


def _warn_config_drift(manifest: Manifest) -> None:
    """Non-fatal: note when the on-disk suite/contest for this run's names has drifted
    from the config the run was created with. Resume always replays the stored config."""
    try:
        _, _, suite_dict, contest_dict = _load_suite_contest(manifest.suite_name, manifest.contest_name)
    except FileNotFoundError:
        return
    if compute_config_hash(suite_dict, contest_dict) != manifest.config_hash:
        _log.warning(f"on-disk config for suite={manifest.suite_name!r} contest={manifest.contest_name!r} differs from this run's stored config, resuming with the stored config")


def _resolve_resume_config(
    manifest: Manifest,
) -> tuple[Suite, Contest, float, str | None]:
    """Rebuild a resumed run's config from its manifest, not from CLI args or argparse
    defaults. Resume continues the same experiment, so the manifest is authoritative."""
    return (
        Suite.parse(manifest.suite_name, manifest.suite),
        Contest.parse(manifest.contest_name, manifest.contest),
        manifest.scale_episodes,
        manifest.simulator,
    )


def _default_run_id(suite_name: str, contest_name: str) -> str:
    ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    if _is_inline_suite(suite_name):
        suite_stem = "inline"
    else:
        suite_stem = pathlib.Path(suite_name.removesuffix(".yaml")).stem
    if _is_inline_contest(contest_name):
        contest_stem = "inline"
    else:
        contest_stem = pathlib.Path(contest_name.removesuffix(".yaml")).stem
    return f"{ts}-{suite_stem}-{contest_stem}"


_KV_RE = re.compile(r"^[\w\.\-]+:=.*$")


def cli_main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().setLevel(logging.DEBUG)
    _log.setLevel(logging.INFO)
    _log.propagate = True
    p = argparse.ArgumentParser(prog="benchmark")
    p.add_argument("--suite", default="basic")
    p.add_argument("--contest", default="basic")
    p.add_argument("--scale-episodes", type=float, default=1.0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument(
        "--resume",
        nargs="?",
        const="__auto__",
        default=None,
        help="Resume a prior run. Bare --resume picks the most recent resumable; --resume <run_id> opens that run explicitly.",
    )
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable system resource profiling. Writes simulation_profile.yaml with peak/mean CPU, GPU, RAM, and disk I/O stats.",
    )
    p.add_argument(
        "--noexit",
        action="store_true",
        help="On completion, leave arena_runtime.launch.py and the last env running so you can poke at it. Recording stops with the last episode as usual.",
    )
    args, extras = p.parse_known_args(argv)

    for arg in extras:
        if not _KV_RE.match(arg):
            p.error(f"unrecognized argument: {arg!r}")

    arena_passthrough: dict[str, str] = {}
    for arg in extras:
        k, v = arg.split(":=", 1)
        arena_passthrough[k] = v

    env_n = int(arena_passthrough.get("env_n", "1"))
    headless = arena_passthrough.get("headless", "false").lower() in ("true", "1")
    simulator = arena_passthrough.get("sim", None)

    try:
        share = pathlib.Path(get_package_share_directory("arena_evaluation"))

        if args.data_root:
            data_root = pathlib.Path(args.data_root)
            print(f"benchmark: data_root from --data-root: {data_root}", file=sys.stderr)
        elif os.environ.get("ARENA_DATA_DIR"):
            data_root = pathlib.Path(os.environ["ARENA_DATA_DIR"]) / "benchmarks"
            print(f"benchmark: data_root from ARENA_DATA_DIR: {data_root}", file=sys.stderr)
        else:
            data_root = share / "data"
            print(f"benchmark: data_root from default: {data_root}", file=sys.stderr)

        if args.resume:
            resume_id = args.resume
            if resume_id == "__auto__":
                resolved = find_most_recent_resumable(data_root)
                if resolved is None:
                    print(
                        f"benchmark: no resumable runs in {data_root}",
                        file=sys.stderr,
                    )
                    return 2
                print(
                    f"benchmark: auto-resume picked run_id={resolved}",
                    file=sys.stderr,
                )
                resume_id = resolved
            run_dir = RunDir.open(data_root, resume_id)
            man = run_dir.manifest
            suite, contest, scale_episodes, simulator = _resolve_resume_config(man)
            if simulator is not None:
                arena_passthrough["sim"] = simulator
            _warn_config_drift(man)
            run_dir.progress.write_comment(f"resumed at {datetime.datetime.now(tz=datetime.UTC).isoformat()}")
        else:
            suite, contest, suite_dict, contest_dict = _load_suite_contest(args.suite, args.contest)
            scale_episodes = args.scale_episodes
            cfg_hash = compute_config_hash(suite_dict, contest_dict)

        steps = _all_steps(contest, suite, scale_episodes)
        if not steps:
            print(
                f"benchmark: empty grid (suite={suite.name!r} contest={contest.name!r} produced no steps)",
                file=sys.stderr,
            )
            return 2
        seen: set[str] = set()
        for c in steps:
            if c.key in seen:
                print(f"benchmark: duplicate step key {c.key!r}", file=sys.stderr)
                return 2
            seen.add(c.key)

        if not args.resume:
            run_id = args.run_id or _default_run_id(args.suite, args.contest)
            sha, dirty = capture_git_sha(share.parent.parent.parent)
            steps_list = [
                {
                    "key": c.key,
                    "contestant": attrs.asdict(c.contestant),
                    "stage": {k: v.value if isinstance(v, (Constants.TaskMode.TM_Robots, Constants.TaskMode.TM_Obstacles)) else v for k, v in c.stage._asdict().items()},
                    "episodes_planned": c.episodes,
                    "is_reference": getattr(c, "is_reference", False),
                    "reference_type": getattr(c, "reference_type", None),
                }
                for c in steps
            ]
            manifest = Manifest(
                run_id=run_id,
                created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
                arena_git_sha=sha,
                arena_git_dirty=dirty,
                cli_args=sys.argv[1:] if argv is None else list(argv),
                env_n=env_n,
                headless=headless,
                config_hash=cfg_hash,
                simulator=simulator,
                scale_episodes=scale_episodes,
                suite_name=suite.name,
                contest_name=contest.name,
                suite=suite_dict,
                contest=contest_dict,
                steps=steps_list,
            )
            run_dir = RunDir.create(data_root, run_id, manifest)
    except FileNotFoundError as exc:
        print(f"benchmark: config file not found: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2

    print(
        f"benchmark: prepared run_id={run_dir.manifest.run_id} steps={len(steps)} dir={run_dir.path}",
        file=sys.stderr,
    )

    run_dir.attach_log_handler(logging.getLogger())

    profiler = None
    if args.profile:
        from .profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=run_dir.path, sample_hz=2.0)
        profiler.start()

    try:
        BenchmarkRunner.run_main(
            suite=suite,
            contest=contest,
            simulator=simulator,
            scale_episodes=scale_episodes,
            env_n=env_n,
            run_id=run_dir.manifest.run_id,
            headless=headless,
            run_dir=run_dir,
            retry_failed=args.retry_failed,
            arena_passthrough=arena_passthrough,
            noexit=args.noexit,
        )
    except KeyboardInterrupt:
        return 130
    finally:
        if profiler is not None:
            profiler.stop()
    return BenchmarkRunner.exit_code


if __name__ == "__main__":
    raise SystemExit(cli_main())
