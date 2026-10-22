"""Per-episode resume planning for benchmark runs."""

from __future__ import annotations

import collections.abc
import dataclasses
import logging
import pathlib
import typing

import yaml

if typing.TYPE_CHECKING:
    from .step import Step

_log = logging.getLogger("arena_evaluation.resume")

_OUTCOME_SUCCESS = 2
_OUTCOME_FAILED = 3
_OUTCOME_SKIPPED = 4
_OUTCOME_FATAL = 5

_TERMINAL = frozenset({_OUTCOME_SUCCESS, _OUTCOME_FAILED, _OUTCOME_SKIPPED, _OUTCOME_FATAL})
_RECORDED_NONSUCCESS = _TERMINAL - {_OUTCOME_SUCCESS}


@dataclasses.dataclass(frozen=True)
class OnDiskEpisode:
    """One `episode_NNN/` directory as observed on disk."""

    path: pathlib.Path
    number: int
    planner: str | None = None
    stage: str | None = None
    is_reference: bool = False
    reference_type: str | None = None
    sim_episode_id: int | None = None
    seed: int | None = None
    outcome_state: int | None = None
    mcap_bytes: int = 0
    has_recording: bool = False
    retired: bool = False

    def matches_step(self, step: Step) -> bool:
        return (
            self.planner == step.contestant.name
            and self.stage == step.stage.name
            and bool(self.is_reference) == bool(step.is_reference)
            and self.reference_type == step.reference_type
        )

    @property
    def is_terminal(self) -> bool:
        return self.outcome_state in _TERMINAL

    @property
    def has_mcap(self) -> bool:
        return self.mcap_bytes > 0


def _episode_number(name: str) -> int | None:
    if not name.startswith("episode_"):
        return None
    try:
        return int(name.split("_", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _scan_episode_dirs(root: pathlib.Path) -> list[OnDiskEpisode]:
    if not root.is_dir():
        return []
    found: list[OnDiskEpisode] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        num = _episode_number(d.name)
        if num is None:
            continue
        ep = OnDiskEpisode(path=d, number=num)
        sidecar = next((p for p in (d / f"{d.name}.yaml", d / "metadata.yaml", *sorted(d.glob("episode_*.yaml"))) if p.is_file()), None)
        if sidecar is not None:
            try:
                meta = yaml.safe_load(sidecar.read_text()) or {}
                if isinstance(meta, dict):
                    ep = dataclasses.replace(
                        ep,
                        planner=meta.get("planner"),
                        stage=meta.get("stage"),
                        is_reference=bool(meta.get("is_reference", False)),
                        reference_type=meta.get("reference_type"),
                        sim_episode_id=_as_int(meta.get("task_generator_episode_id")),
                        seed=_as_int(meta.get("seed")),
                        outcome_state=_as_int(meta.get("outcome_state")),
                    )
            except Exception as exc:
                _log.warning(f"unreadable episode metadata {sidecar}: {exc}")
        canonical = d / f"{d.name}.mcap"
        flat_mcaps = [f for f in d.glob("*.mcap") if f.is_file()]
        size = canonical.stat().st_size if canonical.is_file() else max((f.stat().st_size for f in flat_mcaps), default=0)
        has_rec = bool(flat_mcaps or [f for f in d.glob("*/*.mcap") if f.is_file()])
        found.append(dataclasses.replace(ep, mcap_bytes=size, has_recording=has_rec))
    return found


def load_episode_dirs(episodes_root: pathlib.Path) -> tuple[list[OnDiskEpisode], list[OnDiskEpisode]]:
    live = _scan_episode_dirs(episodes_root)
    retired = [dataclasses.replace(d, retired=True) for d in _scan_episode_dirs(episodes_root / ".superseded")]
    return live, retired


def seed_for_episode(stage_seed: int | None, ep_idx: int) -> int:
    return (stage_seed + ep_idx) if stage_seed is not None else ep_idx


def episode_index_for_seed(stage_seed: int | None, seed: int, episode_count: int) -> int | None:
    ep_idx = seed - stage_seed if stage_seed is not None else seed
    return ep_idx if 0 <= ep_idx < episode_count else None


def dirs_for_step(dirs: list[OnDiskEpisode], step: Step) -> list[OnDiskEpisode]:
    return [d for d in dirs if d.matches_step(step)]


def _step_identity(step: Step) -> tuple[str, str, bool, str | None]:
    return (step.contestant.name, step.stage.name, bool(step.is_reference), step.reference_type)


def _dir_identity(d: OnDiskEpisode) -> tuple[str | None, str | None, bool, str | None]:
    return (d.planner, d.stage, bool(d.is_reference), d.reference_type)


@dataclasses.dataclass
class StepResumeDecision:
    """What a resume should do for one step."""

    step: Step
    run_indices: list[int] = dataclasses.field(default_factory=list)
    delete_dirs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    retire_dirs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    kept_sim_ids: dict[int, int] = dataclasses.field(default_factory=dict)
    kept_outcomes: dict[int, int] = dataclasses.field(default_factory=dict)

    @property
    def kept_run(self) -> int:
        return len(self.kept_outcomes)

    @property
    def kept_failed(self) -> int:
        return sum(1 for outcome in self.kept_outcomes.values() if outcome == _OUTCOME_FAILED)


def decide_step(
    step: Step,
    rows: collections.abc.Sequence[collections.abc.Mapping[str, object]],
    live_dirs: list[OnDiskEpisode],
    retired_dirs: list[OnDiskEpisode] = (),
    *,
    force_retry: bool = False,
) -> StepResumeDecision:
    """Classify every episode of *step* from its ledger rows and on-disk dirs."""
    decision = StepResumeDecision(step=step)
    episode_count, stage_seed = step.episodes, step.stage.seed
    live_by_seed: dict[int, list[OnDiskEpisode]] = {}
    for d in live_dirs:
        if d.seed is not None:
            live_by_seed.setdefault(d.seed, []).append(d)

    claimed: set[pathlib.Path] = set()

    def _discard(d: OnDiskEpisode | None) -> None:
        if d is not None and d.path not in claimed:
            claimed.add(d.path)
            (decision.retire_dirs if (d.is_terminal or d.has_recording) else decision.delete_dirs).append(d.path)

    for ep_idx in range(episode_count):
        seed = seed_for_episode(stage_seed, ep_idx)
        attempts = sorted(
            (r for r in rows if _as_int(r.get("seed")) == seed),
            key=lambda r: str(r.get("ts_iso") or ""),
            reverse=True,
        )

        cur_row: collections.abc.Mapping[str, object] | None = None
        cur_dir: OnDiskEpisode | None = None
        for pool in (live_dirs, retired_dirs):
            for row in attempts:
                sim_id = _as_int(row.get("episode_id"))
                matches = [d for d in pool if d.path not in claimed and d.sim_episode_id == sim_id and d.seed in (None, seed)]
                if matches:
                    cur_row, cur_dir = row, max(matches, key=lambda d: (d.seed == seed, d.number, d.path.name))
                    break
            if cur_row is not None:
                break

        live_cur = cur_dir if (cur_dir is not None and not cur_dir.retired) else None
        yaml_dirs = [d for d in live_by_seed.get(seed, []) if d.path not in claimed]
        recorded_without_row: OnDiskEpisode | None = None
        if cur_row is None and yaml_dirs:
            candidates = [d for d in yaml_dirs if d.outcome_state is not None or (step.is_reference and step.reference_type == "unhindered_peds" and d.has_mcap)]
            if candidates:
                recorded_without_row = max(candidates, key=lambda d: (d.is_terminal, d.outcome_state == _OUTCOME_SUCCESS, d.number, d.path.name))

        outcome = _as_int(cur_row.get("outcome_state")) if cur_row is not None else None
        if outcome is None and cur_dir is not None:
            outcome = cur_dir.outcome_state
        if outcome is None and recorded_without_row is not None:
            outcome = recorded_without_row.outcome_state

        evidence = live_cur or recorded_without_row
        if step.is_reference and step.reference_type == "unhindered_peds" and evidence is not None and evidence.has_mcap:
            outcome = _OUTCOME_SUCCESS

        keep = evidence is not None and evidence.has_mcap and (outcome == _OUTCOME_SUCCESS or (outcome in _RECORDED_NONSUCCESS and not force_retry))
        if keep and evidence is not None and outcome is not None:
            claimed.add(evidence.path)
            for d in live_by_seed.get(seed, []):
                _discard(d)
            decision.kept_outcomes[ep_idx] = outcome
            sim_id = _as_int(cur_row.get("episode_id")) if cur_row is not None else evidence.sim_episode_id
            if sim_id is not None:
                decision.kept_sim_ids[ep_idx] = sim_id
            continue

        decision.run_indices.append(ep_idx)
        _discard(live_cur)
        _discard(recorded_without_row)
        for d in live_by_seed.get(seed, []):
            _discard(d)
        if outcome not in _TERMINAL:
            for row in attempts:
                sim_id = _as_int(row.get("episode_id"))
                for d in live_dirs:
                    if d.sim_episode_id == sim_id and d.seed in (None, seed):
                        _discard(d)

    if decision.run_indices:
        bad_sim_ids = {_as_int(r.get("episode_id")) for r in rows if _as_int(r.get("seed")) is None or episode_index_for_seed(stage_seed, _as_int(r.get("seed")), episode_count) is None} - {None}
        for d in live_dirs:
            if d.path not in claimed and d.sim_episode_id not in bad_sim_ids:
                _discard(d)

    return decision


def plan_step(
    step: Step,
    rows: collections.abc.Sequence[collections.abc.Mapping[str, object]],
    live_dirs: list[OnDiskEpisode],
    retired_dirs: list[OnDiskEpisode] = (),
    *,
    force_retry: bool = False,
) -> StepResumeDecision:
    return decide_step(step, rows, dirs_for_step(live_dirs, step), dirs_for_step(retired_dirs, step), force_retry=force_retry)


def plan_resume(
    steps: list[Step],
    rows_by_key: collections.abc.Mapping[str, collections.abc.Sequence[collections.abc.Mapping[str, object]]],
    live_dirs: list[OnDiskEpisode],
    retired_dirs: list[OnDiskEpisode] = (),
    *,
    retry_failed: bool,
    retry_recorded_keys: collections.abc.Container[str] = frozenset(),
) -> list[StepResumeDecision]:
    live_index: dict[tuple[str | None, str | None, bool, str | None], list[OnDiskEpisode]] = {}
    retired_index: dict[tuple[str | None, str | None, bool, str | None], list[OnDiskEpisode]] = {}
    for d in live_dirs:
        live_index.setdefault(_dir_identity(d), []).append(d)
    for d in retired_dirs:
        retired_index.setdefault(_dir_identity(d), []).append(d)
    return [
        decide_step(
            step,
            rows_by_key.get(step.key, []),
            live_index.get(_step_identity(step), []),
            retired_index.get(_step_identity(step), []),
            force_retry=retry_failed or step.key in retry_recorded_keys,
        )
        for step in steps
    ]


def retire_episode_dir(path: pathlib.Path, superseded_root: pathlib.Path) -> pathlib.Path:
    superseded_root.mkdir(parents=True, exist_ok=True)
    dest = superseded_root / path.name
    counter = 1
    while dest.exists():
        dest = superseded_root / f"{path.name}_{counter}"
        counter += 1
    path.rename(dest)
    return dest
