"""Per-episode resume planning for benchmark runs."""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import logging
import pathlib
import typing

import yaml

if typing.TYPE_CHECKING:
    from .step import Step

_log = logging.getLogger("arena_evaluation.resume")

# EpisodeRecord.outcome_state values of arena_runtime_msgs/msg/EpisodeRecord.msg
_OUTCOME_SUCCESS = 2
_OUTCOME_FAILED = 3
_OUTCOME_SKIPPED = 4
_OUTCOME_FATAL = 5

_TERMINAL = frozenset({_OUTCOME_SUCCESS, _OUTCOME_FAILED, _OUTCOME_SKIPPED, _OUTCOME_FATAL})
_RECORDED_NONSUCCESS = _TERMINAL - {_OUTCOME_SUCCESS}


@dataclasses.dataclass(frozen=True, slots=True)
class OnDiskEpisode:
    """One `episode_NNN/` directory of a run."""

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

    def matches_step(self, step: Step) -> bool:
        return self.planner == step.contestant.name and self.stage == step.stage.name and self.is_reference == step.is_reference and self.reference_type == step.reference_type

    @property
    def is_terminal(self) -> bool:
        return self.outcome_state in _TERMINAL

    @property
    def has_mcap(self) -> bool:
        return self.mcap_bytes > 0


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _episode_number(name: str) -> int | None:
    if not name.startswith("episode_"):
        return None
    try:
        return int(name.split("_", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return None


def _scan_episode_dirs(root: pathlib.Path) -> list[OnDiskEpisode]:
    """Every episode directory under root, with the metadata the recorder wrote and whether a recording exists."""
    if not root.is_dir():
        return []
    found: list[OnDiskEpisode] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        number = _episode_number(directory.name)
        if number is None:
            continue
        episode = OnDiskEpisode(path=directory, number=number)
        sidecar = next((path for path in (directory / f"{directory.name}.yaml", directory / "metadata.yaml", *sorted(directory.glob("episode_*.yaml"))) if path.is_file()), None)
        if sidecar is not None:
            try:
                meta = yaml.safe_load(sidecar.read_text()) or {}
                if isinstance(meta, dict):
                    episode = dataclasses.replace(
                        episode,
                        planner=_as_str(meta.get("planner")),
                        stage=_as_str(meta.get("stage")),
                        is_reference=bool(meta.get("is_reference", False)),
                        reference_type=_as_str(meta.get("reference_type")),
                        sim_episode_id=_as_int(meta.get("task_generator_episode_id")),
                        seed=_as_int(meta.get("seed")),
                        outcome_state=_as_int(meta.get("outcome_state")),
                    )
            except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
                _log.warning(f"unreadable episode metadata {sidecar}: {exc}")
        mcaps = [path for path in (*directory.glob("*.mcap"), *directory.glob("*/*.mcap")) if path.is_file()]
        size = max((path.stat().st_size for path in mcaps), default=0)
        found.append(dataclasses.replace(episode, mcap_bytes=size, has_recording=bool(mcaps)))
    return found


def load_episode_dirs(episodes_root: pathlib.Path) -> list[OnDiskEpisode]:
    """The live episode directories of a run; retired recordings live under `.superseded/` and are not live work."""
    return _scan_episode_dirs(episodes_root)


def seed_for_episode(stage_seed: int | None, ep_idx: int) -> int:
    """The deterministic seed of episode ep_idx, the same one the runner put on the RunEpisode goal."""
    return (stage_seed + ep_idx) if stage_seed is not None else ep_idx


def episode_index_for_seed(stage_seed: int | None, seed: int, episode_count: int) -> int | None:
    """The episode index a seed belongs to, or None when it is outside the step."""
    ep_idx = seed - stage_seed if stage_seed is not None else seed
    return ep_idx if 0 <= ep_idx < episode_count else None


def dirs_for_step(dirs: collections.abc.Iterable[OnDiskEpisode], step: Step) -> list[OnDiskEpisode]:
    return [episode for episode in dirs if episode.matches_step(step)]


def _step_identity(step: Step) -> tuple[str, str, bool, str | None]:
    return (step.contestant.name, step.stage.name, step.is_reference, step.reference_type)


def _dir_identity(episode: OnDiskEpisode) -> tuple[str | None, str | None, bool, str | None]:
    return (episode.planner, episode.stage, episode.is_reference, episode.reference_type)


def _evidence_rank(episode: OnDiskEpisode) -> tuple[bool, bool, bool, int, str]:
    return (episode.has_mcap and episode.outcome_state == _OUTCOME_SUCCESS, episode.has_mcap, episode.is_terminal, episode.number, episode.path.name)


@dataclasses.dataclass(slots=True)
class StepResumeDecision:
    """What a resume should keep and which episode indices it should run for one step."""

    step: Step
    run_indices: list[int] = dataclasses.field(default_factory=list)
    delete_dirs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    retire_dirs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    kept_sim_ids: dict[int, int] = dataclasses.field(default_factory=dict)
    kept_outcomes: dict[int, int] = dataclasses.field(default_factory=dict)
    kept_compromised: int = 0

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
    *,
    force_retry: bool = False,
) -> StepResumeDecision:
    """Classify every episode of step from its progress rows and its live episode directories.

    A recorded episode is kept when its directory holds a recording and its outcome is terminal.
    A recorded failure is also kept unless force_retry asks for another attempt. Everything else
    is scheduled as a run index, and the directory it replaces is retired (recorded) or deleted
    (unclosed or without a recording).
    """
    decision = StepResumeDecision(step=step)
    stage_seed = step.stage.seed
    claimed: set[pathlib.Path] = set()

    def _discard(episode: OnDiskEpisode | None) -> None:
        if episode is None or episode.path in claimed:
            return
        claimed.add(episode.path)
        bucket = decision.retire_dirs if (episode.is_terminal or episode.has_recording) else decision.delete_dirs
        bucket.append(episode.path)

    episode_count = max(0, step.episodes)
    attempts_by_index: dict[int, list[collections.abc.Mapping[str, object]]] = {}
    expected_index = {seed_for_episode(stage_seed, index): index for index in range(episode_count)}
    exact_rows: set[int] = set()
    for row in rows:
        seed = _as_int(row.get("seed"))
        if seed is None:
            continue
        index = expected_index.get(seed)
        if index is not None:
            attempts_by_index.setdefault(index, []).append(row)
            exact_rows.add(id(row))
    if stage_seed is None:
        # Runs recorded before the seed was pinned to the episode index have runtime-derived
        # seeds. Episode ids still increase by one per episode, so they recover the index.
        episode_ids = [episode_id for row in rows if (episode_id := _as_int(row.get("episode_id"))) is not None]
        base = min(episode_ids) if episode_ids else 0
        by_episode_id: dict[int, list[collections.abc.Mapping[str, object]]] = {}
        for row in rows:
            if id(row) in exact_rows:
                continue
            episode_id = _as_int(row.get("episode_id"))
            if episode_id is not None:
                by_episode_id.setdefault(episode_id, []).append(row)
        for episode_id in sorted(by_episode_id):
            index = episode_id - base
            if 0 <= index < episode_count and index not in attempts_by_index:
                attempts_by_index[index] = by_episode_id[episode_id]
    for attempts in attempts_by_index.values():
        attempts.sort(key=lambda row: str(row.get("ts_iso") or ""), reverse=True)

    for ep_idx in range(episode_count):
        seed = seed_for_episode(stage_seed, ep_idx)
        attempts = attempts_by_index.get(ep_idx, [])
        row: collections.abc.Mapping[str, object] | None = attempts[0] if attempts else None
        row_seed = _as_int(row.get("seed")) if row is not None else None
        evidence: OnDiskEpisode | None = None
        for attempt in attempts:
            sim_episode_id = _as_int(attempt.get("episode_id"))
            matches = [episode for episode in live_dirs if episode.path not in claimed and episode.sim_episode_id == sim_episode_id and (row_seed is None or episode.seed in (None, row_seed))]
            if matches:
                row = attempt
                evidence = max(matches, key=_evidence_rank)
                break
        if not attempts:
            seated = step.is_reference and step.reference_type == "unhindered_peds"
            seeded = [episode for episode in live_dirs if episode.path not in claimed and episode.seed == seed and episode.has_mcap and (episode.is_terminal or seated)]
            if seeded:
                evidence = max(seeded, key=_evidence_rank)

        outcome = _as_int(row.get("outcome_state")) if row is not None else None
        if outcome is None and evidence is not None:
            outcome = evidence.outcome_state
        if step.is_reference and step.reference_type == "unhindered_peds" and evidence is not None and evidence.has_mcap:
            outcome = _OUTCOME_SUCCESS

        keep = evidence is not None and evidence.has_mcap and outcome is not None and (outcome == _OUTCOME_SUCCESS or (outcome in _RECORDED_NONSUCCESS and not force_retry))
        if keep:
            assert evidence is not None
            assert outcome is not None
            claimed.add(evidence.path)
            decision.kept_outcomes[ep_idx] = outcome
            sim_episode_id = _as_int(row.get("episode_id")) if row is not None else evidence.sim_episode_id
            if sim_episode_id is not None:
                decision.kept_sim_ids[ep_idx] = sim_episode_id
            integrity = row.get("integrity") if row is not None else None
            if isinstance(integrity, str) and integrity:
                try:
                    if json.loads(integrity):
                        decision.kept_compromised += 1
                except json.JSONDecodeError:
                    pass
            continue

        decision.run_indices.append(ep_idx)
        _discard(evidence)
        for attempt in attempts:
            sim_episode_id = _as_int(attempt.get("episode_id"))
            for episode in live_dirs:
                if episode.sim_episode_id == sim_episode_id:
                    _discard(episode)

    for episode in live_dirs:
        _discard(episode)
    return decision


def plan_step(
    step: Step,
    rows: collections.abc.Sequence[collections.abc.Mapping[str, object]],
    live_dirs: collections.abc.Iterable[OnDiskEpisode],
    *,
    force_retry: bool = False,
) -> StepResumeDecision:
    return decide_step(step, rows, dirs_for_step(live_dirs, step), force_retry=force_retry)


def plan_resume(
    steps: collections.abc.Sequence[Step],
    rows_by_key: collections.abc.Mapping[str, collections.abc.Sequence[collections.abc.Mapping[str, object]]],
    live_dirs: collections.abc.Iterable[OnDiskEpisode],
    *,
    retry_failed: bool,
    retry_recorded_keys: collections.abc.Container[str] = frozenset(),
) -> list[StepResumeDecision]:
    """One per-episode decision per step of the grid."""
    live_index: dict[tuple[str | None, str | None, bool, str | None], list[OnDiskEpisode]] = {}
    for episode in live_dirs:
        live_index.setdefault(_dir_identity(episode), []).append(episode)
    return [
        decide_step(
            step,
            rows_by_key.get(step.key, ()),
            live_index.get(_step_identity(step), []),
            force_retry=retry_failed or step.key in retry_recorded_keys,
        )
        for step in steps
    ]


def retire_episode_dir(path: pathlib.Path, superseded_root: pathlib.Path) -> pathlib.Path:
    """Move an episode directory under the run's superseded tree, never clobbering an earlier retirement."""
    superseded_root.mkdir(parents=True, exist_ok=True)
    destination = superseded_root / path.name
    counter = 1
    while destination.exists():
        destination = superseded_root / f"{path.name}_{counter}"
        counter += 1
    path.rename(destination)
    return destination
