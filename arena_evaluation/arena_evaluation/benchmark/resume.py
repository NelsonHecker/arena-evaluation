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
        """True when this dir's metadata identity equals the step's."""
        return self.planner == step.contestant.name and self.stage == step.stage.name and bool(self.is_reference) == bool(step.is_reference) and self.reference_type == step.reference_type

    @property
    def is_terminal(self) -> bool:
        return self.outcome_state in _TERMINAL

    @property
    def has_mcap(self) -> bool:
        return self.mcap_bytes > 0


def _episode_number(name: str) -> int | None:
    """Parse the episode number out of `episode_NNN` or a retired `episode_NNN_1`."""
    if not name.startswith("episode_"):
        return None
    try:
        return int(name.split("_", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return None


def _sidecar_path(d: pathlib.Path) -> pathlib.Path | None:
    """Locate an episode dir's sidecar YAML, including a retired dir's original name."""
    direct = d / f"{d.name}.yaml"
    if direct.exists():
        return direct
    metadata = d / "metadata.yaml"
    if metadata.exists():
        return metadata
    candidates = sorted(p for p in d.glob("episode_*.yaml") if p.is_file())
    return candidates[0] if candidates else None


def _scan_episode_dirs(root: pathlib.Path) -> list[OnDiskEpisode]:
    """List `episode_*` dirs under *root*, reading each sidecar YAML."""
    found: list[OnDiskEpisode] = []
    if not root.is_dir():
        return found
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        number = _episode_number(d.name)
        if number is None:
            continue
        ep = OnDiskEpisode(path=d, number=number)
        yaml_path = _sidecar_path(d)
        if yaml_path is not None:
            try:
                text = yaml_path.read_text()
                meta = yaml.safe_load(text) or {}
                if not isinstance(meta, dict):
                    raise ValueError(f"episode metadata is not a mapping: {yaml_path}")
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
            except (OSError, ValueError, yaml.YAMLError) as exc:
                _log.warning(f"unreadable episode metadata {yaml_path}: {exc}")
        canonical = d / f"{d.name}.mcap"
        flat_mcaps = [f for f in d.glob("*.mcap") if f.is_file()]
        if canonical.is_file():
            size = canonical.stat().st_size
        else:
            size = max((f.stat().st_size for f in flat_mcaps), default=0)
        has_recording = bool(flat_mcaps or [f for f in d.glob("*/*.mcap") if f.is_file()])
        found.append(dataclasses.replace(ep, mcap_bytes=size, has_recording=has_recording))
    return found


def load_episode_dirs(episodes_root: pathlib.Path) -> tuple[list[OnDiskEpisode], list[OnDiskEpisode]]:
    """Return (live dirs under episodes/, retired dirs under episodes/.superseded/)."""
    live = _scan_episode_dirs(episodes_root)
    retired = [dataclasses.replace(d, retired=True) for d in _scan_episode_dirs(episodes_root / ".superseded")]
    return live, retired


def seed_for_episode(stage_seed: int | None, ep_idx: int) -> int:
    return (stage_seed + ep_idx) if stage_seed is not None else ep_idx


def episode_index_for_seed(stage_seed: int | None, seed: int, episode_count: int) -> int | None:
    """Invert seed_for_episode; None when the seed cannot belong to this step."""
    ep_idx = seed - stage_seed if stage_seed is not None else seed
    if 0 <= ep_idx < episode_count:
        return ep_idx
    return None


def dirs_for_step(dirs: list[OnDiskEpisode], step: Step) -> list[OnDiskEpisode]:
    return [d for d in dirs if d.matches_step(step)]


def _step_identity(step: Step) -> tuple[str, str, bool, str | None]:
    return (step.contestant.name, step.stage.name, bool(step.is_reference), step.reference_type)


def _dir_identity(d: OnDiskEpisode) -> tuple[str | None, str | None, bool, str | None]:
    return (d.planner, d.stage, bool(d.is_reference), d.reference_type)


def plan_step(
    step: Step,
    rows: collections.abc.Sequence[collections.abc.Mapping[str, object]],
    live_dirs: list[OnDiskEpisode],
    retired_dirs: list[OnDiskEpisode],
    *,
    force_retry: bool = False,
) -> StepResumeDecision:
    """Plan one step from evidence already loaded for its run dir."""
    return decide_step(
        step,
        rows,
        dirs_for_step(live_dirs, step),
        dirs_for_step(retired_dirs, step),
        force_retry=force_retry,
    )


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


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def decide_step(
    step: Step,
    rows: collections.abc.Sequence[collections.abc.Mapping[str, object]],
    live_dirs: list[OnDiskEpisode],
    retired_dirs: list[OnDiskEpisode],
    *,
    force_retry: bool = False,
) -> StepResumeDecision:
    """Classify every episode of *step* from its ledger rows and on-disk dirs."""
    decision = StepResumeDecision(step=step)
    episode_count = step.episodes
    stage_seed = step.stage.seed

    live_by_seed: dict[int, list[OnDiskEpisode]] = {}
    for d in live_dirs:
        if d.seed is not None:
            live_by_seed.setdefault(d.seed, []).append(d)

    claimed: set[pathlib.Path] = set()

    def _discard(d: OnDiskEpisode | None) -> None:
        if d is None or d.path in claimed:
            return
        claimed.add(d.path)
        if d.is_terminal or d.has_recording:
            decision.retire_dirs.append(d.path)
        else:
            decision.delete_dirs.append(d.path)

    def _match_sim_dir(pool: list[OnDiskEpisode], sim_id: int | None, seed: int) -> OnDiskEpisode | None:
        if sim_id is None:
            return None
        matches = [d for d in pool if d.path not in claimed and d.sim_episode_id == sim_id and d.seed in (None, seed)]
        if not matches:
            return None
        return max(matches, key=lambda d: (d.seed == seed, d.number, d.path.name))

    def _tidy_duplicates(seed: int) -> None:
        """Retire or delete extra dirs of the same seed besides any already-claimed dir."""
        for d in live_by_seed.get(seed, []):
            _discard(d)

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
                d = _match_sim_dir(pool, _as_int(row.get("episode_id")), seed)
                if d is not None:
                    cur_row, cur_dir = row, d
                    break
            if cur_row is not None:
                break

        live_cur_dir = cur_dir if (cur_dir is not None and not cur_dir.retired) else None
        yaml_seed_dirs = [d for d in live_by_seed.get(seed, []) if d.path not in claimed]
        recorded_without_row: OnDiskEpisode | None = None
        if cur_row is None:
            candidates = [d for d in yaml_seed_dirs if d.outcome_state is not None or (step.is_reference and step.reference_type == "unhindered_peds" and d.has_mcap)]
            if candidates:
                recorded_without_row = max(
                    candidates,
                    key=lambda d: (
                        d.is_terminal,
                        d.outcome_state == _OUTCOME_SUCCESS,
                        d.number,
                        d.path.name,
                    ),
                )

        if cur_row is not None:
            outcome = _as_int(cur_row.get("outcome_state"))
            if outcome is None and cur_dir is not None:
                outcome = cur_dir.outcome_state
        elif recorded_without_row is not None:
            outcome = recorded_without_row.outcome_state
        else:
            outcome = None

        if step.is_reference and step.reference_type == "unhindered_peds":
            candidate = live_cur_dir or recorded_without_row
            if candidate is not None and candidate.has_mcap:
                outcome = _OUTCOME_SUCCESS

        evidence = live_cur_dir if live_cur_dir is not None else recorded_without_row
        keep_episode = evidence is not None and evidence.has_mcap and (outcome == _OUTCOME_SUCCESS or (outcome in _RECORDED_NONSUCCESS and not force_retry))
        if keep_episode and evidence is not None and outcome is not None:
            claimed.add(evidence.path)
            _tidy_duplicates(seed)
            decision.kept_outcomes[ep_idx] = outcome
            sim_id = _as_int(cur_row.get("episode_id")) if cur_row is not None else None
            if sim_id is None:
                sim_id = evidence.sim_episode_id
            if sim_id is not None:
                decision.kept_sim_ids[ep_idx] = sim_id
            continue

        decision.run_indices.append(ep_idx)
        _discard(live_cur_dir)
        _discard(recorded_without_row)
        _tidy_duplicates(seed)
        if outcome != _OUTCOME_SUCCESS and outcome not in _RECORDED_NONSUCCESS:
            for row in attempts:
                _discard(_match_sim_dir(live_dirs, _as_int(row.get("episode_id")), seed))

    if decision.run_indices:
        bad_sim_ids: set[object] = set()
        for r in rows:
            seed = _as_int(r.get("seed"))
            if seed is None or episode_index_for_seed(stage_seed, seed, episode_count) is None:
                episode_id = r.get("episode_id")
                if episode_id is not None:
                    bad_sim_ids.add(episode_id)
        for d in live_dirs:
            if d.path in claimed:
                continue
            if d.sim_episode_id in bad_sim_ids:
                _log.warning(f"resume: {step.key}: leaving {d.path.name} untouched (row seed not attributable to an episode index)")
                continue
            _discard(d)

    return decision


def plan_resume(
    steps: list[Step],
    rows_by_key: collections.abc.Mapping[str, collections.abc.Sequence[collections.abc.Mapping[str, object]]],
    live_dirs: list[OnDiskEpisode],
    retired_dirs: list[OnDiskEpisode],
    *,
    retry_failed: bool,
    retry_recorded_keys: collections.abc.Container[str] = frozenset(),
) -> list[StepResumeDecision]:
    """Plan all *steps*; caller keeps only decisions with run_indices."""
    decisions: list[StepResumeDecision] = []
    live_index: dict[tuple[str | None, str | None, bool, str | None], list[OnDiskEpisode]] = {}
    retired_index: dict[tuple[str | None, str | None, bool, str | None], list[OnDiskEpisode]] = {}
    for d in live_dirs:
        live_index.setdefault(_dir_identity(d), []).append(d)
    for d in retired_dirs:
        retired_index.setdefault(_dir_identity(d), []).append(d)
    for step in steps:
        identity = _step_identity(step)
        decision = decide_step(
            step,
            rows_by_key.get(step.key, []),
            live_index.get(identity, []),
            retired_index.get(identity, []),
            force_retry=retry_failed or step.key in retry_recorded_keys,
        )
        decisions.append(decision)
    return decisions


def retire_episode_dir(path: pathlib.Path, superseded_root: pathlib.Path) -> pathlib.Path:
    """Move an episode dir into superseded_root, uniquifying on name clash."""
    superseded_root.mkdir(parents=True, exist_ok=True)
    dest = superseded_root / path.name
    counter = 1
    while dest.exists():
        dest = superseded_root / f"{path.name}_{counter}"
        counter += 1
    path.rename(dest)
    return dest
