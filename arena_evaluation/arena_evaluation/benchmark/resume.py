"""Per-episode resume planning for benchmark runs.

Problem this solves
-------------------
Resume used to operate at step granularity: a step (contestant x stage) whose
recorded status was "failed"/"partial" was re-queued as a whole, and its re-run
always recorded into brand-new `episode_NNN` dirs appended past the existing
ones. Old (failed / crashed / partial) episode dirs were never touched, so a
run accumulated stale duplicates that downstream processing counted as real
episodes.

What this module provides
-------------------------
A pure, ROS-free planner that decides, per (step, episode_index):

* GOOD      - the episode has a surviving, completed recording with a SUCCESS
              terminal outcome: keep its dir, never re-run it.
* RECORDED  - a terminal outcome exists (FAILED / SKIPPED / FATAL), i.e. the
              sim genuinely finished the attempt (e.g. the planner collided).
              Kept by default; re-run only when `retry_failed` is set. On a
              forced re-run the old evidence dir is *retired* (moved to
              `episodes/.superseded/`) rather than deleted, because that data
              is a real experimental result.
* DEAD      - no terminal outcome: the env crashed / wedged / spawn failed /
              recorder died mid-episode. Re-run automatically (plain
              `--resume`); partial garbage dirs from the crashed attempt are
              deleted so the fresh recording properly replaces them.

Evidence sources
----------------
* progress.csv rows (per finished episode: step_key, seed, episode_id = sim
  episode id, outcome_state, ts_iso). One row per attempt.
* Episode dirs under `episodes/episode_NNN/` with `episode_NNN.yaml`
  (planner/stage/is_reference/reference_type/task_generator_episode_id,
  outcome_state; `seed` on recordings made after this module landed).
  Dirs are matched to rows by task_generator_episode_id == row.episode_id and
  to steps by the metadata identity tuple.
* Episode index attribution: seed = stage.seed + ep_idx (stage.seed None ->
  seed == ep_idx), so an attempt row / dir can be attributed to its ep_idx.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import yaml

from .step import Step

_log = logging.getLogger("arena_evaluation.resume")

_OUTCOME_QUEUED = 0
_OUTCOME_RUNNING = 1
_OUTCOME_SUCCESS = 2
_OUTCOME_FAILED = 3
_OUTCOME_SKIPPED = 4
_OUTCOME_FATAL = 5

_TERMINAL = frozenset(
    {_OUTCOME_SUCCESS, _OUTCOME_FAILED, _OUTCOME_SKIPPED, _OUTCOME_FATAL}
)
_RECORDED_NONSUCCESS = frozenset({_OUTCOME_FAILED, _OUTCOME_SKIPPED, _OUTCOME_FATAL})


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

    def matches_step(self, step: Step) -> bool:
        """True when this dir's metadata identity equals the step's.

        The recorder is spawned per step with planner=contestant.name,
        stage=stage.name and the step's reference flags, so the identity tuple
        is unambiguous even when several steps share a stage (main vs
        unobstructed_robot vs unhindered_peds references).
        """
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


def _scan_episode_dirs(root: pathlib.Path) -> list[OnDiskEpisode]:
    """List `episode_*` dirs under *root*, reading each sidecar YAML."""
    found: list[OnDiskEpisode] = []
    if not root.is_dir():
        return found
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("episode_"):
            continue
        try:
            number = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        ep = OnDiskEpisode(path=d, number=number)
        yaml_path = d / f"{d.name}.yaml"
        if not yaml_path.exists():
            yaml_path = d / "metadata.yaml"
        if yaml_path.exists():
            try:
                meta = yaml.safe_load(yaml_path.read_text()) or {}
                ep = dataclasses.replace(
                    ep,
                    planner=meta.get("planner"),
                    stage=meta.get("stage"),
                    is_reference=bool(meta.get("is_reference", False)),
                    reference_type=meta.get("reference_type"),
                    sim_episode_id=meta.get("task_generator_episode_id"),
                    seed=meta.get("seed"),
                    outcome_state=meta.get("outcome_state"),
                )
            except Exception as exc:
                _log.warning(f"unreadable episode metadata {yaml_path}: {exc}")
        canonical = d / f"{d.name}.mcap"
        if canonical.is_file():
            size = canonical.stat().st_size
        else:
            size = max(
                (f.stat().st_size for f in d.glob("*.mcap") if f.is_file()),
                default=0,
            )
        found.append(dataclasses.replace(ep, mcap_bytes=size))
    return found


def load_episode_dirs(episodes_root: pathlib.Path) -> tuple[list[OnDiskEpisode], list[OnDiskEpisode]]:
    """Return (live dirs under episodes/, retired dirs under episodes/.superseded/)."""
    live = _scan_episode_dirs(episodes_root)
    retired = _scan_episode_dirs(episodes_root / ".superseded")
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


@dataclasses.dataclass
class StepResumeDecision:
    """What a resume should do for one step."""

    step: Step
    #: Planned episode indices (0-based) to actually run this session.
    run_indices: list[int] = dataclasses.field(default_factory=list)
    #: Episode dirs to delete (garbage: no terminal outcome, being replaced).
    delete_dirs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    #: Episode dirs to move into `episodes/.superseded/` (recorded evidence
    #: superseded by a forced re-run, or a stale duplicate).
    retire_dirs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    #: ep_idx -> sim episode id of the surviving recording that is *kept*
    #: (skipped this session), for relinking reference steps to their parents.
    kept_sim_ids: dict[int, int] = dataclasses.field(default_factory=dict)


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def decide_step(
    step: Step,
    rows: list[dict],
    live_dirs: list[OnDiskEpisode],
    retired_dirs: list[OnDiskEpisode],
    *,
    retry_failed: bool,
) -> StepResumeDecision:
    """Classify every episode of *step* from the ledger rows + on-disk dirs.

    *rows* must already be filtered to this step (row["step_key"] == step.key).
    Pure: returns a StepResumeDecision; never touches the filesystem.
    """
    decision = StepResumeDecision(step=step)
    episode_count = step.episodes
    stage_seed = step.stage.seed

    live_by_sim = {d.sim_episode_id: d for d in live_dirs if d.sim_episode_id is not None}
    retired_by_sim = {d.sim_episode_id: d for d in retired_dirs if d.sim_episode_id is not None}
    live_by_seed: dict[int, list[OnDiskEpisode]] = {}
    for d in live_dirs:
        if d.seed is not None:
            live_by_seed.setdefault(d.seed, []).append(d)

    # Track dirs claimed as evidence or already classified so the final sweep
    # below only sees leftovers.
    claimed: set[pathlib.Path] = set()

    def _claim(d: OnDiskEpisode) -> None:
        claimed.add(d.path)

    def _tidy_duplicates(seed: int, keep: OnDiskEpisode | None) -> None:
        """Retire/delete extra dirs of the same seed besides the kept evidence."""
        for d in live_by_seed.get(seed, []):
            if d.path in claimed:
                continue
            _claim(d)
            if d.is_terminal:
                decision.retire_dirs.append(d.path)
            else:
                decision.delete_dirs.append(d.path)

    for ep_idx in range(episode_count):
        seed = seed_for_episode(stage_seed, ep_idx)
        attempts = sorted(
            (r for r in rows if r.get("seed") == seed and _as_int(r.get("seed")) is not None),
            key=lambda r: str(r.get("ts_iso") or ""),
            reverse=True,
        )

        # Newest attempt whose recording still survives (live dir preferred;
        # a dir retired by an earlier forced re-run also counts as evidence).
        cur_row: dict | None = None
        cur_dir: OnDiskEpisode | None = None
        for row in attempts:
            d = live_by_sim.get(row.get("episode_id"))
            if d is not None:
                cur_row, cur_dir = row, d
                break
        if cur_row is None:
            for row in attempts:
                d = retired_by_sim.get(row.get("episode_id"))
                if d is not None:
                    cur_row, cur_dir = row, d
                    break

        # Recorded-without-row evidence (runner died between recorder STOP and
        # the progress.csv append): visible through the YAML seed only.
        yaml_seed_dirs = [d for d in live_by_seed.get(seed, []) if d.path not in claimed]
        recorded_without_row: OnDiskEpisode | None = None
        if cur_row is None:
            for d in yaml_seed_dirs:
                if d.outcome_state is not None or (step.is_reference and step.reference_type == "unhindered_peds" and d.has_mcap):
                    recorded_without_row = d
                    break

        if cur_row is not None:
            outcome = _as_int(cur_row.get("outcome_state"))
            if outcome is None and cur_dir is not None:
                outcome = cur_dir.outcome_state
        elif recorded_without_row is not None:
            outcome = recorded_without_row.outcome_state
        else:
            outcome = None

        if step.is_reference and step.reference_type == "unhindered_peds":
            candidate = cur_dir or recorded_without_row
            if candidate is not None and candidate.has_mcap:
                outcome = _OUTCOME_SUCCESS

        if outcome == _OUTCOME_SUCCESS:
            # GOOD: an intact successful recording exists. Evidence can be the
            # row's dir, or a dir whose YAML carries the terminal outcome when
            # the runner died between recorder STOP and the progress.csv append.
            evidence = cur_dir if cur_dir is not None else recorded_without_row
            if evidence is not None and evidence.has_mcap:
                _claim(evidence)
                _tidy_duplicates(seed, keep=evidence)
                sim_id = _as_int(cur_row.get("episode_id")) if cur_row is not None else None
                if sim_id is None and evidence.sim_episode_id is not None:
                    sim_id = evidence.sim_episode_id
                if sim_id is not None:
                    decision.kept_sim_ids[ep_idx] = sim_id
                continue
            # Success claimed but no surviving recording (data loss) -> re-run.
            decision.run_indices.append(ep_idx)
            continue

        if outcome in _RECORDED_NONSUCCESS:
            if not retry_failed:
                # RECORDED: preserved as a real experimental result.
                if recorded_without_row is not None and cur_row is None:
                    _claim(recorded_without_row)
                if cur_dir is not None and cur_dir.path not in claimed:
                    _claim(cur_dir)
                _tidy_duplicates(seed, keep=cur_dir or recorded_without_row)
                sim_id = _as_int(cur_row.get("episode_id")) if cur_row is not None else None
                if sim_id is None and recorded_without_row is not None:
                    sim_id = recorded_without_row.sim_episode_id
                if sim_id is not None:
                    decision.kept_sim_ids[ep_idx] = sim_id
                continue
            # Forced re-run: keep the old evidence, just move it aside.
            decision.run_indices.append(ep_idx)
            for d in (cur_dir, recorded_without_row):
                if d is None or d.path in claimed or not d.is_terminal:
                    continue
                _claim(d)
                # Only live dirs need moving; retired evidence already sits
                # under episodes/.superseded/.
                if d.path.parent.name != ".superseded":
                    decision.retire_dirs.append(d.path)
            _tidy_duplicates(seed, keep=None)
            continue

        # DEAD: no terminal outcome -> re-run; garbage dirs go away so the
        # fresh recording properly replaces them.
        decision.run_indices.append(ep_idx)
        for d in yaml_seed_dirs:
            _claim(d)
            if d.is_terminal:
                decision.retire_dirs.append(d.path)
            else:
                decision.delete_dirs.append(d.path)
        # A dead attempt may still have a dir linked by its (non-terminal) row.
        for row in attempts:
            d = live_by_sim.get(row.get("episode_id"))
            if d is not None and d.path not in claimed:
                _claim(d)
                if d.is_terminal:
                    decision.retire_dirs.append(d.path)
                else:
                    decision.delete_dirs.append(d.path)

    # ----- leftover sweep: dirs of this step no seed/row attribution could
    # explain (pre-seed recordings, crashed attempts that never produced a row,
    # superseded duplicates of earlier sessions). Only touch them when the step
    # is actually being re-run this session.
    if decision.run_indices:
        rows_with_bad_seed = {
            r for r in rows
            if _as_int(r.get("seed")) is None
            or episode_index_for_seed(stage_seed, r.get("seed"), episode_count) is None
        }
        bad_sim_ids = {r.get("episode_id") for r in rows_with_bad_seed if r.get("episode_id") is not None}
        for d in live_dirs:
            if d.path in claimed:
                continue
            if d.sim_episode_id in bad_sim_ids:
                # Cannot reason about it; never destroy unclassifiable data.
                _log.warning(
                    f"resume: {step.key}: leaving {d.path.name} untouched "
                    f"(row seed not attributable to an episode index)"
                )
                continue
            _claim(d)
            if d.is_terminal:
                decision.retire_dirs.append(d.path)
            else:
                decision.delete_dirs.append(d.path)

    return decision


def plan_resume(
    steps: list[Step],
    rows_by_key: dict[str, list[dict]],
    live_dirs: list[OnDiskEpisode],
    retired_dirs: list[OnDiskEpisode],
    *,
    retry_failed: bool,
) -> list[StepResumeDecision]:
    """Plan all *steps*; caller keeps only decisions with run_indices.

    Pure: no filesystem writes. Caller executes delete_dirs/retire_dirs.
    """
    decisions: list[StepResumeDecision] = []
    for step in steps:
        step_dirs_live = dirs_for_step(live_dirs, step)
        step_dirs_retired = dirs_for_step(retired_dirs, step)
        decision = decide_step(
            step,
            rows_by_key.get(step.key, []),
            step_dirs_live,
            step_dirs_retired,
            retry_failed=retry_failed,
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
