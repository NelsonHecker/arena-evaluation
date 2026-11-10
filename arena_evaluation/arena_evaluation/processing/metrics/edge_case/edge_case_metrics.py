from __future__ import annotations

import re
import typing
from collections.abc import Iterable, Mapping

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.pedestrian_frames import pedestrians_at

from .criticality import SETTLE_S, Sample, formation_hold, score
from .provenance import Provenance, load_provenance, scenario_name

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SAMPLE_HZ = 10.0
PED_RADIUS = 0.3
PEDS_TOLERANCE_NS = 100_000_000
TIMELINE_KIND = "timeline"
STIMULUS_PREFIX = "stimulus."
FORM_NEED = "form"
BREAK_NEED = "break"

_ENV_PREFIX = re.compile(r"^/?env_\d+/")


def agent_name(recorded: str) -> str:
    """`/env_0/d_1/0` -> `d_1`, the name the scenario gives the pedestrian."""
    return _ENV_PREFIX.sub("", recorded, count=1).lstrip("/").split("/", 1)[0]


def _grid_step_ns() -> int:
    """Sampling step of the criticality grid in nanoseconds."""
    return int(1e9 / SAMPLE_HZ)


def _bare_ped_name_expression() -> pl.Expr:
    """Recorded pedestrian name with the env prefix and the trailing entity path removed."""
    return pl.col("ped_name").cast(pl.String).str.replace(_ENV_PREFIX.pattern, "").str.split("/").list.first()


def episode_provenance(episode: AlignedEpisodeBundle, world: str | None) -> Provenance | None:
    """The provenance of the scenario the episode ran, None when it has none."""
    world = episode.map or world
    scenario = scenario_name(episode.run)
    if not world or not scenario:
        return None
    return load_provenance(world, scenario, episode.run_dir)


def fired_stimuli(snapshot: pl.DataFrame | None, start_s: float | None) -> dict[str, dict[str, float]] | None:
    """Sim time in s each `stimulus.<agent>.<need>` of the `timeline` entity fired from `start_s` on, None without the entity."""
    if snapshot is None or len(snapshot) == 0 or "kind" not in snapshot.columns:
        return None
    rows = snapshot.filter(pl.col("kind") == TIMELINE_KIND)
    if len(rows) == 0:
        return None
    fired: dict[str, dict[str, float]] = {}
    for field, value in rows.filter((pl.col("field_kind") == "discrete") & pl.col("field_name").str.starts_with(STIMULUS_PREFIX)).select("field_name", "value_str").iter_rows():
        agent, _, need = field[len(STIMULUS_PREFIX) :].rpartition(".")
        try:
            at = float(value)
        except (TypeError, ValueError):
            continue
        if not agent or not need or (start_s is not None and at < start_s - PEDS_TOLERANCE_NS / 1e9):
            continue
        needs = fired.setdefault(agent, {})
        needs[need] = min(at, needs.get(need, at))
    return fired


def _first_fired(fired: Mapping[str, Mapping[str, float]], agents: Iterable[str], need: str | None, start_s: float) -> float | None:
    """Seconds after `start_s` the first of `agents` received `need` (any need when None), None when none did."""
    times = [at for agent in agents for name, at in fired.get(agent, {}).items() if need is None or name == need]
    return min(times) - start_s if times else None


def episode_samples(calc: BaseMetricCalculator, episode: AlignedEpisodeBundle, radii: Mapping[str, float]) -> list[Sample]:
    """The robot on a SAMPLE_HZ grid with the pedestrians of a peds message at most 0.1 s old, by scenario name and `radii`."""
    pos_x, pos_y, _yaw, t_ns = calc.resolve_native_pose(episode)
    if len(t_ns) < 2:
        return []
    grid = np.arange(int(t_ns[0]), int(t_ns[-1]) + 1, int(1e9 / SAMPLE_HZ), dtype=np.int64)
    rx = np.interp(grid, t_ns, pos_x)
    ry = np.interp(grid, t_ns, pos_y)
    vx, vy = calc.velocity_from_pose(rx, ry, grid)
    rows: list[dict[str, tuple[tuple[float, float], tuple[float, float], float]]] = [{} for _ in grid]
    peds = calc.native_ped_frame(episode)
    if peds is not None and peds.height > 0:
        held = pedestrians_at(pl.Series("t_ns", grid), peds, PEDS_TOLERANCE_NS).with_columns(
            frame_ix=((pl.col("t_ns") - int(grid[0])) // _grid_step_ns()).cast(pl.Int64),
            bare_name=_bare_ped_name_expression(),
        )
        for frame_ix, bare, ped_x, ped_y, ped_vx, ped_vy in zip(
            held.get_column("frame_ix").to_numpy(),
            held.get_column("bare_name").to_numpy(),
            held.get_column("x").to_numpy(),
            held.get_column("y").to_numpy(),
            held.get_column("vx").to_numpy(),
            held.get_column("vy").to_numpy(),
            strict=True,
        ):
            name = str(bare)
            rows[int(frame_ix)][name] = (
                (float(ped_x), float(ped_y)),
                (float(ped_vx), float(ped_vy)),
                radii.get(name, PED_RADIUS),
            )
    radius = calc.robot_params.robot_radius
    return [Sample(t=float(t) / 1e9, robot=(float(x), float(y)), robot_vel=(float(u), float(w)), robot_radius=radius, peds=rows[i]) for i, (t, x, y, u, w) in enumerate(zip(grid, rx, ry, vx, vy, strict=True))]


class EdgeCaseCriticalityCalculator(BaseMetricCalculator):
    """Criticality of an edge-case episode against the case's own agents: clearance, TTC, PET, freezes and the collision outcome."""

    NAME = "edge_case_criticality"
    CATEGORY = "edge_case"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE})
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_tracks"]

    UNITS = {
        "edge_case_id": "",
        "edge_case_steps": "",
        "edge_case_scope": "",
        "edge_case_agents": "",
        "edge_case_missing": "",
        "edge_case_played": "",
        "edge_case_unplayed": "",
        "edge_case_triggered_at": "s",
        "edge_case_min_ttc_s": "s",
        "edge_case_min_ttc_with": "",
        "edge_case_min_ttc_at": "s",
        "edge_case_min_clearance_m": "m",
        "edge_case_min_clearance_with": "",
        "edge_case_min_clearance_at": "s",
        "edge_case_pet_s": "s",
        "edge_case_pet_with": "",
        "edge_case_freeze_s": "s",
        "edge_case_intrusion_s": "s",
        "edge_case_collided": "",
        "edge_case_collision_fault": "",
        "edge_case_collision_with": "",
        "edge_case_collision_at": "s",
        "edge_case_collision_robot_closing": "m/s",
        "edge_case_collision_ped_closing": "m/s",
    }

    PRIMARY_OUTPUTS = ["edge_case_min_clearance_m", "edge_case_pet_s", "edge_case_freeze_s", "edge_case_collided"]
    OUTPUT_DIRECTIONS = {
        "edge_case_min_ttc_s": "higher",
        "edge_case_min_clearance_m": "higher",
        "edge_case_pet_s": "higher",
        "edge_case_freeze_s": "lower",
        "edge_case_intrusion_s": "lower",
        "edge_case_collided": "lower",
    }

    world: str | None = None

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        provenance = episode_provenance(episode, self.world)
        if provenance is None:
            return results
        samples = episode_samples(self, episode, provenance.radii)
        recorded = {name for s in samples for name in s.peds}
        seen = [name for name in provenance.agents if name in recorded]
        encountered = list(dict.fromkeys(e.agent for e in provenance.encounters))
        start_s = samples[0].t if samples else None
        fired = fired_stimuli(episode.semantic_snapshot, start_s)
        unplayed = [] if fired is None else [agent for agent in encountered if agent not in fired]
        scored = [name for name in seen if name not in unplayed]
        results["edge_case_id"] = provenance.id
        results["edge_case_steps"] = list(provenance.steps)
        results["edge_case_scope"] = "own" if scored else "unplayed" if seen else "all"
        results["edge_case_agents"] = scored
        results["edge_case_missing"] = [name for name in provenance.agents if name not in recorded]
        if fired is not None and encountered:
            results["edge_case_played"] = not unplayed
            results["edge_case_unplayed"] = unplayed
            if start_s is not None:
                results["edge_case_triggered_at"] = _first_fired(fired, encountered, None, start_s)
        if seen and not scored:
            return results
        encounters = [(e.agent, e.point) for e in provenance.encounters if e.agent not in unplayed]
        result = score(samples, only=scored or None, encounters=encounters, settle_s=SETTLE_S + provenance.hold_s)
        if not result.observed:
            return results
        results.update(
            {
                "edge_case_min_ttc_s": result.min_ttc_s,
                "edge_case_min_ttc_with": result.min_ttc_with or None,
                "edge_case_min_ttc_at": result.min_ttc_at,
                "edge_case_min_clearance_m": result.min_clearance_m,
                "edge_case_min_clearance_with": result.min_clearance_with or None,
                "edge_case_min_clearance_at": result.min_clearance_at,
                "edge_case_pet_s": result.pet_s,
                "edge_case_pet_with": result.pet_with or None,
                "edge_case_freeze_s": result.freeze_duration_s,
                "edge_case_intrusion_s": result.intrusion_time_s,
                "edge_case_collided": result.collided,
                "edge_case_collision_fault": result.collision_fault or None,
                "edge_case_collision_with": result.collision_with or None,
                "edge_case_collision_at": result.collision_at,
                "edge_case_collision_robot_closing": result.collision_robot_closing,
                "edge_case_collision_ped_closing": result.collision_ped_closing,
            }
        )
        return results

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE scope only: the case provenance and its agents are episode-level records."""
        if scope_slice is not None and scope_slice.scope != MetricScope.EPISODE:
            return dict.fromkeys(self.output_keys())
        return self.calculate(episode, prior_results)


class EdgeCaseFormationCalculator(BaseMetricCalculator):
    """Whether the formation a case's D step owns formed around its center, how long it held and when it broke."""

    NAME = "edge_case_formation"
    CATEGORY = "edge_case"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE})
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_tracks"]

    UNITS = {
        "edge_case_formation_variant": "",
        "edge_case_formation_members": "",
        "edge_case_formation_absent": "",
        "edge_case_formation_seen": "",
        "edge_case_formation_formed_at": "s",
        "edge_case_formation_held_s": "s",
        "edge_case_formation_broke_at": "s",
        "edge_case_formation_form_fired_at": "s",
        "edge_case_formation_break_fired_at": "s",
        "edge_case_formation_arrived": "",
    }

    PRIMARY_OUTPUTS = ["edge_case_formation_held_s"]
    OUTPUT_DIRECTIONS = {"edge_case_formation_held_s": "higher", "edge_case_formation_arrived": "higher"}

    world: str | None = None

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        provenance = episode_provenance(episode, self.world)
        members = provenance.owned.get("D", ()) if provenance is not None else ()
        if provenance is None or provenance.formation is None or not members:
            return results
        formation = provenance.formation
        samples = episode_samples(self, episode, provenance.radii)
        start_s = samples[0].t if samples else None
        fired = fired_stimuli(episode.semantic_snapshot, start_s) or {}
        form_at = None if start_s is None else _first_fired(fired, members, FORM_NEED, start_s)
        break_at = None if start_s is None else _first_fired(fired, members, BREAK_NEED, start_s)
        hold = formation_hold(samples, members, radius=formation.radius, center=formation.center, form_at=form_at)
        results["edge_case_formation_variant"] = formation.variant or None
        results["edge_case_formation_members"] = list(hold.members)
        results["edge_case_formation_absent"] = list(hold.absent)
        results["edge_case_formation_seen"] = hold.seen
        results["edge_case_formation_formed_at"] = hold.formed_at
        results["edge_case_formation_held_s"] = hold.held_s
        results["edge_case_formation_broke_at"] = hold.broke_at
        results["edge_case_formation_arrived"] = hold.arrived
        results["edge_case_formation_form_fired_at"] = form_at
        results["edge_case_formation_break_fired_at"] = break_at
        return results

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE scope only: the case provenance and its agents are episode-level records."""
        if scope_slice is not None and scope_slice.scope != MetricScope.EPISODE:
            return dict.fromkeys(self.output_keys())
        return self.calculate(episode, prior_results)
