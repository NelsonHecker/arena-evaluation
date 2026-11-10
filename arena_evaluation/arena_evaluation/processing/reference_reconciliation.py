"""Reference-twin reconciliation of the Star Schema facts.

MAR, PFI, the robot detour, KCSC and relative throughput compare an episode with the
reference episodes of its stage, which are separate recordings; the comparison is therefore
benchmark-level. It runs after every episode has its scoped facts (they cannot be computed
inside a single episode's scope) and patches the fact rows *before* the benchmark Star Schema
is consolidated, so the reconciled values are published exactly once, in the facts.

The pedestrian deflection of the pedestrian fact is likewise twin-dependent: when the stage
has no robot-free pedestrian recording, its row says so with a typed
``reference_missing:unhindered_peds`` reason instead of a bare ``no_samples``.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import typing
from collections.abc import Mapping, Sequence
from typing import Final

import polars as pl

from arena_evaluation.processing.metrics.social.mutual_accommodation import (
    MutualAccommodationCalculator,
)
from arena_evaluation.processing.reference_twins import (
    UNHINDERED_PEDS,
    UNOBSTRUCTED_ROBOT,
    TwinIndex,
)
from arena_evaluation.storage.schemas import EpisodeDescriptor

Point = tuple[float, float]

RECONCILED_COLUMNS: Final[tuple[str, ...]] = (
    "mar",
    "pfi",
    "robot_path_dev_m",
    "kcsc_wh",
    "relative_throughput",
)
PEDESTRIAN_DEFLECTION_COLUMN: Final[str] = "ped_path_deflection_m"

# The reference type whose absence explains a missing value of each reconciled column.
_COLUMN_REFERENCE: Final[dict[str, str]] = {
    "mar": UNHINDERED_PEDS,
    "pfi": UNHINDERED_PEDS,
    "robot_path_dev_m": UNOBSTRUCTED_ROBOT,
    "kcsc_wh": UNOBSTRUCTED_ROBOT,
    "relative_throughput": UNOBSTRUCTED_ROBOT,
}


_FACT_EPISODES_FILE: Final[str] = "fact_episodes.parquet"
_FACT_PEDESTRIANS_FILE: Final[str] = "fact_pedestrians.parquet"
_PATH_SERIES_FILE: Final[str] = "series_path_metrics.parquet"
_PEDESTRIAN_PATH_SERIES_FILE: Final[str] = "series_pedestrian_path_metrics.parquet"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ReconciledEpisode:
    """One episode robot's twin comparison, None where the comparison is not computable."""

    episode_id: int
    robot_id: str
    mar: float | None = None
    pfi: float | None = None
    robot_path_dev_m: float | None = None
    kcsc_wh: float | None = None
    relative_throughput: float | None = None
    has_pedestrian_twin: bool = False
    has_robot_twin: bool = False

    def column_value(self, column: str) -> float | None:
        """Reconciled value of one episode fact column."""
        match column:
            case "mar":
                return self.mar
            case "pfi":
                return self.pfi
            case "robot_path_dev_m":
                return self.robot_path_dev_m
            case "kcsc_wh":
                return self.kcsc_wh
            case "relative_throughput":
                return self.relative_throughput
            case _:
                raise KeyError(f"{column!r} is not a reconciled episode column")

    def missing_reference(self, column: str) -> str | None:
        """The reference type whose absence explains a null value, None when every twin it needs is present."""
        if column == "mar":
            # MAR needs both twins: the pedestrian flow interference and the robot detour.
            if not self.has_pedestrian_twin:
                return UNHINDERED_PEDS
            if not self.has_robot_twin:
                return UNOBSTRUCTED_ROBOT
            return None
        reference_type = _COLUMN_REFERENCE[column]
        present = (
            self.has_pedestrian_twin
            if reference_type == UNHINDERED_PEDS
            else self.has_robot_twin
        )
        return None if present else reference_type


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ReconciliationPlan:
    """The twin comparison of every evaluated episode and the pedestrian missingness codes."""

    episodes: tuple[ReconciledEpisode, ...]
    pedestrian_reasons: Mapping[int, str]

    def by_episode(self) -> dict[int, tuple[ReconciledEpisode, ...]]:
        """Reconciled values grouped by episode id (one entry per robot)."""
        grouped: dict[int, list[ReconciledEpisode]] = {}
        for episode in self.episodes:
            grouped.setdefault(episode.episode_id, []).append(episode)
        return {episode_id: tuple(entries) for episode_id, entries in grouped.items()}

    def for_row(self, episode_id: object, robot_id: object) -> ReconciledEpisode | None:
        """The reconciled values of one fact row, matched by robot then by episode alone."""
        entries = self.by_episode().get(int(typing.cast(int, episode_id)), ())
        if not entries:
            return None
        for entry in entries:
            if robot_id is not None and entry.robot_id == str(robot_id):
                return entry
        return entries[0]


def _read_series(episode_dir: pathlib.Path, name: str) -> pl.DataFrame | None:
    """One published series table of an episode, None when it was not published."""
    path = episode_dir / "metrics" / name
    if not path.is_file():
        return None
    return pl.read_parquet(path)


def robot_path(episode_dir: pathlib.Path, robot_id: str | None = None) -> list[Point] | None:
    """Planar ground-truth path of an episode robot, None when the episode published no path series."""
    frame = _read_series(episode_dir, _PATH_SERIES_FILE)
    if frame is None or frame.height == 0:
        return None
    if robot_id is not None and "robot_id" in frame.columns:
        filtered = frame.filter(pl.col("robot_id") == robot_id)
        if filtered.height > 0:
            frame = filtered
    points = frame.sort("t_ns").select("x_m", "y_m").to_numpy().astype("float64")
    return [
        (float(x), float(y))
        for x, y in points
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]


def pedestrian_paths(episode_dir: pathlib.Path) -> list[list[Point]] | None:
    """Per-pedestrian planar paths of an episode, ordered by pedestrian id, None without a path series."""
    frame = _read_series(episode_dir, _PEDESTRIAN_PATH_SERIES_FILE)
    if frame is None or frame.height == 0:
        return None
    paths: list[list[Point]] = []
    for pedestrian in frame.sort(["ped_id", "t_ns"]).partition_by("ped_id", maintain_order=True):
        points = pedestrian.select("x_m", "y_m").to_numpy().astype("float64")
        path = [
            (float(x), float(y))
            for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if path:
            paths.append(path)
    return paths


def _legacy_row(
    episode_dir: pathlib.Path,
    fact_row: Mapping[str, typing.Any],
    *,
    include_robot_path: bool,
    include_pedestrian_path: bool,
) -> dict[str, typing.Any]:
    """The legacy row keys the twin recognizer reads, reconstructed from the published tables."""
    robot_id = None if fact_row.get("robot_id") is None else str(fact_row["robot_id"])
    row: dict[str, typing.Any] = {
        "path": robot_path(episode_dir, robot_id) if include_robot_path else None,
        "pedestrian_path": pedestrian_paths(episode_dir) if include_pedestrian_path else None,
        "path_length": fact_row.get("path_length_m"),
        "time_to_goal": fact_row.get("time_to_goal_s"),
        "energy_total_wh": fact_row.get("total_energy_wh"),
    }
    return row


def _descriptor_by_episode(episodes: Sequence[EpisodeDescriptor]) -> dict[int, EpisodeDescriptor]:
    return {episode.episode_id: episode for episode in episodes}


def plan_reconciliation(
    episodes: Sequence[EpisodeDescriptor],
    facts_by_episode: Mapping[int, pl.DataFrame],
    twin_index: TwinIndex,
) -> ReconciliationPlan:
    """Compare every evaluated episode against its stage's twins and plan the fact-row patches."""
    by_episode = _descriptor_by_episode(episodes)
    reference_rows: dict[tuple[int, bool], dict[str, typing.Any]] = {}

    def reference_row(twin: EpisodeDescriptor, *, include_robot_path: bool) -> dict[str, typing.Any]:
        key = (twin.episode_id, include_robot_path)
        cached = reference_rows.get(key)
        if cached is None:
            frame = facts_by_episode.get(twin.episode_id)
            fact_row = {} if frame is None or frame.height == 0 else frame.to_dicts()[0]
            cached = _legacy_row(
                pathlib.Path(twin.episode_dir),
                fact_row,
                include_robot_path=include_robot_path,
                include_pedestrian_path=not include_robot_path,
            )
            reference_rows[key] = cached
        return cached

    reconciled: list[ReconciledEpisode] = []
    pedestrian_reasons: dict[int, str] = {}
    for episode in episodes:
        if episode.is_reference:
            continue
        frames = facts_by_episode.get(episode.episode_id)
        if frames is None or frames.height == 0:
            continue
        pedestrian_twin = twin_index.pedestrian_twin(episode)
        robot_twin = twin_index.robot_twin(episode)
        robot_reference = (
            None if robot_twin is None else reference_row(robot_twin, include_robot_path=True)
        )
        pedestrian_reference = (
            None
            if pedestrian_twin is None
            else reference_row(pedestrian_twin, include_robot_path=False)
        )
        pedestrian_reasons[episode.episode_id] = (
            f"reference_missing:{UNHINDERED_PEDS}" if pedestrian_twin is None else "no_samples"
        )
        episode_dir = pathlib.Path(episode.episode_dir)
        for fact_row in frames.to_dicts():
            dynamic = _legacy_row(
                episode_dir,
                fact_row,
                include_robot_path=True,
                include_pedestrian_path=True,
            )
            values = MutualAccommodationCalculator.reconcile_stage_references(
                dynamic_row=dynamic,
                ref_robot_row=robot_reference,
                ref_ped_row=pedestrian_reference,
            )
            reconciled.append(
                ReconciledEpisode(
                    episode_id=episode.episode_id,
                    robot_id="" if fact_row.get("robot_id") is None else str(fact_row["robot_id"]),
                    mar=_optional_float(values.get("mar")),
                    pfi=_optional_float(values.get("pfi")),
                    # The recognizer calls this quantity robot_path_deviation_m; the fact column is robot_path_dev_m.
                    robot_path_dev_m=_optional_float(values.get("robot_path_deviation_m")),
                    kcsc_wh=_optional_float(values.get("kcsc_wh")),
                    relative_throughput=_optional_float(values.get("relative_throughput")),
                    has_pedestrian_twin=pedestrian_twin is not None,
                    has_robot_twin=robot_twin is not None,
                )
            )
    return ReconciliationPlan(
        episodes=tuple(reconciled),
        pedestrian_reasons=pedestrian_reasons,
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(typing.cast(float, value))


def _episode_null_reason(
    existing: object, reconciled: ReconciledEpisode, row: Mapping[str, typing.Any]
) -> str:
    """Existing row reason without the stale reconciled codes, plus the current typed codes."""
    parts = [part for part in str(existing or "").split(";") if part]
    kept = [part for part in parts if part.split(":", maxsplit=1)[0] not in RECONCILED_COLUMNS]
    for column in RECONCILED_COLUMNS:
        if column not in row:
            continue
        value = row[column]
        if value is not None:
            continue
        missing_reference = reconciled.missing_reference(column)
        code = (
            f"reference_missing:{missing_reference}"
            if missing_reference is not None
            else "no_samples"
        )
        kept.append(f"{column}:{code}")
    return ";".join(kept)


def _pedestrian_null_reason(existing: object, reason: str, *, value: object) -> str:
    """Existing pedestrian row reason without the stale deflection code, plus the twin-aware code."""
    parts = [part for part in str(existing or "").split(";") if part]
    kept = [
        part
        for part in parts
        if part.split(":", maxsplit=1)[0] != PEDESTRIAN_DEFLECTION_COLUMN
    ]
    if value is None:
        kept.append(f"{PEDESTRIAN_DEFLECTION_COLUMN}:{reason}")
    return ";".join(kept)


def patch_fact_episodes(
    frame: pl.DataFrame, plan: ReconciliationPlan
) -> pl.DataFrame:
    """Episode fact rows with the reconciled twin columns and their typed missingness codes."""
    if frame.height == 0 or not plan.episodes:
        return frame
    rows = frame.to_dicts()
    patched = False
    for row in rows:
        reconciled = plan.for_row(row.get("episode_id"), row.get("robot_id"))
        if reconciled is None:
            continue
        for column in RECONCILED_COLUMNS:
            if column in row:
                row[column] = reconciled.column_value(column)
        row["null_reason"] = _episode_null_reason(
            row.get("null_reason"), reconciled, row
        )
        patched = True
    return pl.DataFrame(rows, schema=frame.schema) if patched else frame


def patch_fact_pedestrians(
    frame: pl.DataFrame, plan: ReconciliationPlan
) -> pl.DataFrame:
    """Pedestrian fact rows with the twin-aware deflection missingness code."""
    if frame.height == 0 or not plan.pedestrian_reasons:
        return frame
    if PEDESTRIAN_DEFLECTION_COLUMN not in frame.columns:
        return frame
    rows = frame.to_dicts()
    patched = False
    for row in rows:
        reason = plan.pedestrian_reasons.get(int(typing.cast(int, row.get("episode_id"))))
        if reason is None:
            continue
        row["null_reason"] = _pedestrian_null_reason(
            row.get("null_reason"), reason, value=row.get(PEDESTRIAN_DEFLECTION_COLUMN)
        )
        patched = True
    return pl.DataFrame(rows, schema=frame.schema) if patched else frame


def apply_plan_to_disk(episode_dir: pathlib.Path, plan: ReconciliationPlan) -> bool:
    """Patch the published fact tables of one episode in place, True when a file was rewritten."""
    metrics_dir = episode_dir / "metrics"
    rewritten = False
    episodes_path = metrics_dir / _FACT_EPISODES_FILE
    if episodes_path.is_file():
        frame = pl.read_parquet(episodes_path)
        patched = patch_fact_episodes(frame, plan)
        if patched is not frame:
            patched.write_parquet(episodes_path)
            rewritten = True
    pedestrians_path = metrics_dir / _FACT_PEDESTRIANS_FILE
    if pedestrians_path.is_file():
        frame = pl.read_parquet(pedestrians_path)
        patched = patch_fact_pedestrians(frame, plan)
        if patched is not frame:
            patched.write_parquet(pedestrians_path)
            rewritten = True
    return rewritten
