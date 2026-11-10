from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_ENCOUNTER_DIST_M,
    DEFAULT_PED_RADIUS_M,
    NS_PER_S,
    PEDESTRIAN_TRACKS_TOPIC,
    ROBOT_TRACKS_TOPIC,
    EncounterSpec,
    PedTrack,
    Trajectory2D,
    detect_encounters,
    extract_episode_ped_tracks,
    resolve_ped_track,
    resolve_robot_trajectory,
)

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

NORMALIZATION_WINDOW_S: Final[float] = 12.0
PRE_CONTACT_PROBST_WINDOW_S: Final[float] = 3.0
PERSONAL_SPACE_RADIUS_M: Final[float] = 1.0
CONTACT_T0_CONFIG_KEYS: Final[tuple[str, ...]] = ("contact_t0_ns",)
SHARE_KEYS: Final[tuple[str, ...]] = ("r_robot", "r_target", "r_time", "e_robot", "e_target", "e_time")


@dataclass(frozen=True, slots=True)
class ProbstParts:
    """Additive conflict totals (Eq. 10) for rollup aggregation."""

    c_total: float
    cc_minus_robot: float
    cc_minus_target: float
    cc_minus_time: float
    cc_plus_robot: float
    cc_plus_target: float
    cc_plus_time: float


@dataclass(frozen=True, slots=True)
class ProbstResult:
    """Additive conflict totals and derived Responsibility/Engagement shares."""

    c_total: float | None = None
    cc_minus_robot: float | None = None
    cc_minus_target: float | None = None
    cc_minus_time: float | None = None
    cc_plus_robot: float | None = None
    cc_plus_target: float | None = None
    cc_plus_time: float | None = None
    r_robot: float | None = None
    r_target: float | None = None
    r_time: float | None = None
    e_robot: float | None = None
    e_target: float | None = None
    e_time: float | None = None
    cp_max: float | None = None
    pdce_min_m: float | None = None
    tce_s: float | None = None


@dataclass(frozen=True, slots=True)
class ProbstPair:
    """Aligned robot-target trajectory pair for Probst evaluation."""

    robot: Trajectory2D
    target: Trajectory2D
    window_size_s: float = NORMALIZATION_WINDOW_S


def distance_to_closest_encounter(
    state_robot: tuple[np.ndarray, np.ndarray] | np.ndarray,
    state_target: tuple[np.ndarray, np.ndarray] | np.ndarray,
) -> float:
    """Projected Distance of Closest Encounter (pDCE, Eq. 1) for a single state pair (pos, vel)."""
    pos_r, vel_r = (state_robot[0], state_robot[1]) if isinstance(state_robot, tuple) else (state_robot[:2], state_robot[2:4])
    pos_t, vel_t = (state_target[0], state_target[1]) if isinstance(state_target, tuple) else (state_target[:2], state_target[2:4])
    rel_pos = np.asarray(pos_t, dtype=np.float64) - np.asarray(pos_r, dtype=np.float64)
    rel_vel = np.asarray(vel_t, dtype=np.float64) - np.asarray(vel_r, dtype=np.float64)
    denom = float(np.dot(rel_vel, rel_vel))
    t_min = 0.0 if denom <= 1e-12 else max(0.0, -float(np.dot(rel_pos, rel_vel)) / denom)
    return float(np.linalg.norm(rel_pos + t_min * rel_vel))


def compute_pdce_series(
    robot_pos_vel: tuple[np.ndarray, np.ndarray],
    target_pos_vel: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Vectorized Projected Distance of Closest Encounter (pDCE, Eq. 1) over (T, 2) trajectories."""
    r_vec = np.asarray(target_pos_vel[0], dtype=np.float64) - np.asarray(robot_pos_vel[0], dtype=np.float64)
    v_vec = np.asarray(target_pos_vel[1], dtype=np.float64) - np.asarray(robot_pos_vel[1], dtype=np.float64)
    denom = np.sum(v_vec * v_vec, axis=1)
    dot_rv = np.sum(r_vec * v_vec, axis=1)
    t_min = np.where(denom > 1e-12, np.maximum(0.0, -dot_rv / np.where(denom > 1e-12, denom, 1.0)), 0.0)
    return np.linalg.norm(r_vec + t_min[:, np.newaxis] * v_vec, axis=1)


def compute_normalization(pair: ProbstPair, idx_tce: int) -> np.ndarray | None:
    """Linear ramp normalization N(t) over [TCE - N_W, TCE] (Eq. 3)."""
    n = len(pair.robot.t_ns)
    if n < 2 or idx_tce <= 0 or idx_tce >= n:
        return None

    times_s = pair.robot.times_s
    normalization = np.zeros(n, dtype=np.float64)
    dts = np.diff(times_s[: idx_tce + 1])
    is_uniform = len(dts) > 0 and bool(np.all(dts > 0.0)) and (float(np.max(dts) - np.min(dts)) <= 1e-6 * max(1.0, float(np.mean(dts))))
    if is_uniform:
        dt_mean = float(np.mean(dts))
        start_index = int(max(0, np.ceil(idx_tce - pair.window_size_s / dt_mean)))
        start_height = 1.0 - (idx_tce - start_index) * dt_mean / pair.window_size_s
        normalization[start_index:idx_tce] = np.linspace(start_height, 1.0, idx_tce - start_index)
        return normalization

    t_tce = float(times_s[idx_tce])
    t_prev = float(times_s[idx_tce - 1])
    in_window = (times_s[:idx_tce] >= t_tce - pair.window_size_s - 1e-9) & (times_s[:idx_tce] <= t_tce)
    if not np.any(in_window):
        normalization[idx_tce - 1] = 1.0
        return normalization
    start_idx = int(np.flatnonzero(in_window)[0])
    start_height = max(0.0, 1.0 - (t_tce - float(times_s[start_idx])) / pair.window_size_s)
    span = t_prev - float(times_s[start_idx])
    if span <= 1e-9:
        normalization[start_idx:idx_tce] = 1.0
    else:
        frac = (times_s[start_idx:idx_tce] - float(times_s[start_idx])) / span
        normalization[start_idx:idx_tce] = start_height + frac * (1.0 - start_height)
    return normalization


def derive_probst_shares(parts: ProbstParts | None) -> dict[str, float | None]:
    """Compute R_* and E_* shares from additive conflict parts (Eqs. 11-12)."""
    if parts is None or parts.c_total <= 0.0:
        unassigned: dict[str, float | None] = {key: None for key in SHARE_KEYS}
        return unassigned
    return {
        "r_robot": float(parts.cc_minus_robot / parts.c_total),
        "r_target": float(parts.cc_minus_target / parts.c_total),
        "r_time": float(parts.cc_minus_time / parts.c_total),
        "e_robot": float(parts.cc_plus_robot / parts.c_total),
        "e_target": float(parts.cc_plus_target / parts.c_total),
        "e_time": float(parts.cc_plus_time / parts.c_total),
    }


def _evaluate_pair_at_tce(pair: ProbstPair, idx_tce: int) -> ProbstResult:
    """Evaluate Probst et al. (ICRA 2026) Eqs. 1-12 for an aligned ProbstPair at a given TCE index."""
    r_sum = pair.robot.radius_m + pair.target.radius_m
    if len(pair.robot.t_ns) < 2 or len(pair.target.t_ns) != len(pair.robot.t_ns) or r_sum <= 0.0:
        return ProbstResult()

    norm = compute_normalization(pair, idx_tce)
    if norm is None:
        return ProbstResult()

    pos_r, vel_r = pair.robot.xy, pair.robot.vel
    pos_t, vel_t = pair.target.xy, pair.target.vel

    pdce = compute_pdce_series((pos_r, vel_r), (pos_t, vel_t))
    cp = np.maximum(0.0, 1.0 - pdce / r_sum)
    conflict = cp * norm

    pdce_r_no_change = compute_pdce_series((pos_r[1:], vel_r[:-1]), (pos_t[1:], vel_t[1:]))
    pdce_t_no_change = compute_pdce_series((pos_r[1:], vel_r[1:]), (pos_t[1:], vel_t[:-1]))

    conflict_r_no_change = np.maximum(0.0, 1.0 - pdce_r_no_change / r_sum) * norm[1:]
    conflict_t_no_change = np.maximum(0.0, 1.0 - pdce_t_no_change / r_sum) * norm[1:]

    cc_robot = conflict[1:] - conflict_r_no_change
    cc_target = conflict[1:] - conflict_t_no_change

    cc_plus_r = np.maximum(cc_robot, 0.0)
    cc_minus_r = -np.minimum(cc_robot, 0.0)
    cc_plus_t = np.maximum(cc_target, 0.0)
    cc_minus_t = -np.minimum(cc_target, 0.0)

    conflict_diff_time = np.diff(conflict) - cc_plus_r - cc_plus_t + cc_minus_r + cc_minus_t
    cc_plus_time = np.maximum(conflict_diff_time, 0.0)
    cc_minus_time = -np.minimum(conflict_diff_time, 0.0)

    parts = ProbstParts(
        c_total=float(np.sum(cc_plus_r) + np.sum(cc_plus_t) + np.sum(cc_plus_time) + conflict[0]),
        cc_minus_robot=float(np.sum(cc_minus_r)),
        cc_minus_target=float(np.sum(cc_minus_t)),
        cc_minus_time=float(np.sum(cc_minus_time) + conflict[idx_tce]),
        cc_plus_robot=float(np.sum(cc_plus_r)),
        cc_plus_target=float(np.sum(cc_plus_t)),
        cc_plus_time=float(np.sum(cc_plus_time) + conflict[0]),
    )
    shares = derive_probst_shares(parts)
    active = norm > 0.0
    times_s = pair.robot.times_s
    return ProbstResult(
        c_total=parts.c_total,
        cc_minus_robot=parts.cc_minus_robot,
        cc_minus_target=parts.cc_minus_target,
        cc_minus_time=parts.cc_minus_time,
        cc_plus_robot=parts.cc_plus_robot,
        cc_plus_target=parts.cc_plus_target,
        cc_plus_time=parts.cc_plus_time,
        r_robot=shares["r_robot"],
        r_target=shares["r_target"],
        r_time=shares["r_time"],
        e_robot=shares["e_robot"],
        e_target=shares["e_target"],
        e_time=shares["e_time"],
        cp_max=float(np.max(cp[active])) if np.any(active) else float(np.max(cp[: idx_tce + 1])),
        pdce_min_m=float(np.min(pdce[active])) if np.any(active) else float(np.min(pdce[: idx_tce + 1])),
        tce_s=float(times_s[idx_tce] - times_s[0]),
    )


def evaluate_probst(pair: ProbstPair) -> ProbstResult:
    """Evaluate Probst et al. (ICRA 2026) Eqs. 1-12 using the Euclidean closest encounter as TCE."""
    if len(pair.robot.t_ns) < 2:
        return ProbstResult()
    dists = np.linalg.norm(pair.target.xy - pair.robot.xy, axis=1)
    return _evaluate_pair_at_tce(pair, int(np.argmin(dists)))


def evaluate_pre_contact_probst(pair: ProbstPair) -> ProbstResult:
    """Evaluate Probst et al. (ICRA 2026) Eqs. 1-12 anchored at contact onset (final sample)."""
    if len(pair.robot.t_ns) < 2:
        return ProbstResult()
    return _evaluate_pair_at_tce(pair, len(pair.robot.t_ns) - 1)


def rollup_probst_results(results: Sequence[ProbstResult]) -> ProbstResult:
    """Aggregate multiple ProbstResult rows by summing additive parts and rebuilding shares."""
    valid = [result for result in results if result.c_total is not None]
    if not valid:
        return ProbstResult()

    parts = ProbstParts(
        c_total=float(sum(result.c_total for result in valid if result.c_total is not None)),
        cc_minus_robot=float(sum(result.cc_minus_robot for result in valid if result.cc_minus_robot is not None)),
        cc_minus_target=float(sum(result.cc_minus_target for result in valid if result.cc_minus_target is not None)),
        cc_minus_time=float(sum(result.cc_minus_time for result in valid if result.cc_minus_time is not None)),
        cc_plus_robot=float(sum(result.cc_plus_robot for result in valid if result.cc_plus_robot is not None)),
        cc_plus_target=float(sum(result.cc_plus_target for result in valid if result.cc_plus_target is not None)),
        cc_plus_time=float(sum(result.cc_plus_time for result in valid if result.cc_plus_time is not None)),
    )
    shares = derive_probst_shares(parts)
    cp_vals = [result.cp_max for result in valid if result.cp_max is not None]
    pdce_vals = [result.pdce_min_m for result in valid if result.pdce_min_m is not None]
    tce_vals = [result.tce_s for result in valid if result.tce_s is not None]

    return ProbstResult(
        c_total=parts.c_total,
        cc_minus_robot=parts.cc_minus_robot,
        cc_minus_target=parts.cc_minus_target,
        cc_minus_time=parts.cc_minus_time,
        cc_plus_robot=parts.cc_plus_robot,
        cc_plus_target=parts.cc_plus_target,
        cc_plus_time=parts.cc_plus_time,
        r_robot=shares["r_robot"],
        r_target=shares["r_target"],
        r_time=shares["r_time"],
        e_robot=shares["e_robot"],
        e_target=shares["e_target"],
        e_time=shares["e_time"],
        cp_max=max(cp_vals) if cp_vals else None,
        pdce_min_m=min(pdce_vals) if pdce_vals else None,
        tce_s=min(tce_vals) if tce_vals else None,
    )


def _contact_t0_ns(metrics_config: Mapping[str, typing.Any]) -> int | None:
    """Configured contact timestamp in nanoseconds."""
    for key in CONTACT_T0_CONFIG_KEYS:
        value = metrics_config.get(key)
        if value is not None:
            return int(value)
    return None


class ProbstResponsibilityCalculator(BaseMetricCalculator):
    """Probst et al. (ICRA 2026) Conflict, Responsibility and Engagement calculator (Eqs. 1-12)."""

    NAME = "probst_responsibility"
    CATEGORY = "social"
    SCOPE = MetricScope.ENCOUNTER
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.ENCOUNTER, MetricScope.CONTACT, MetricScope.EPISODE})
    REQUIRED_TOPICS = [ROBOT_TRACKS_TOPIC, PEDESTRIAN_TRACKS_TOPIC]

    @classmethod
    def required_topics_for_scope(cls, scope: MetricScope) -> tuple[str | tuple[str, ...], ...]:
        if scope == MetricScope.CONTACT:
            return (ROBOT_TRACKS_TOPIC,)
        return (ROBOT_TRACKS_TOPIC, PEDESTRIAN_TRACKS_TOPIC)

    UNITS = {
        "probst_c_total": "conflict",
        "probst_cc_minus_robot": "conflict",
        "probst_cc_minus_target": "conflict",
        "probst_cc_minus_time": "conflict",
        "probst_cc_plus_robot": "conflict",
        "probst_cc_plus_target": "conflict",
        "probst_cc_plus_time": "conflict",
        "probst_r_robot": "ratio",
        "probst_r_target": "ratio",
        "probst_r_time": "ratio",
        "probst_e_robot": "ratio",
        "probst_e_target": "ratio",
        "probst_e_time": "ratio",
        "probst_cp_max": "ratio",
        "probst_pdce_min_m": "m",
        "probst_tce_s": "s",
        "probst_ps_c_total": "conflict",
        "probst_ps_r_robot": "ratio",
        "probst_ps_r_target": "ratio",
        "probst_ps_r_time": "ratio",
        "probst_ps_e_robot": "ratio",
        "probst_ps_e_target": "ratio",
        "probst_ps_e_time": "ratio",
    }

    PRIMARY_OUTPUTS = [
        "probst_r_robot",
        "probst_e_robot",
        "probst_c_total",
    ]

    OUTPUT_DIRECTIONS = {
        "probst_r_robot": "higher",
        "probst_e_robot": "lower",
        "probst_c_total": "lower",
        "probst_cp_max": "lower",
        "probst_pdce_min_m": "higher",
        "probst_ps_r_robot": "higher",
        "probst_ps_e_robot": "lower",
        "probst_ps_c_total": "lower",
    }

    ROLLUP = {
        "probst_c_total": "sum",
        "probst_cc_minus_robot": "sum",
        "probst_cc_minus_target": "sum",
        "probst_cc_minus_time": "sum",
        "probst_cc_plus_robot": "sum",
        "probst_cc_plus_target": "sum",
        "probst_cc_plus_time": "sum",
        "probst_cp_max": "max",
        "probst_pdce_min_m": "min",
        "probst_ps_c_total": "sum",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        """Output keys produced by ProbstResponsibilityCalculator."""
        return [
            "probst_c_total",
            "probst_cc_minus_robot",
            "probst_cc_minus_target",
            "probst_cc_minus_time",
            "probst_cc_plus_robot",
            "probst_cc_plus_target",
            "probst_cc_plus_time",
            "probst_r_robot",
            "probst_r_target",
            "probst_r_time",
            "probst_e_robot",
            "probst_e_target",
            "probst_e_time",
            "probst_cp_max",
            "probst_pdce_min_m",
            "probst_tce_s",
            "probst_ps_c_total",
            "probst_ps_r_robot",
            "probst_ps_r_target",
            "probst_ps_r_time",
            "probst_ps_e_robot",
            "probst_ps_e_target",
            "probst_ps_e_time",
            # Fact table schema canonical aliases
            "probst_R_robot",
            "probst_R_actor2",
            "probst_R_time",
            "probst_E_robot",
            "probst_E_actor2",
        ]

    @classmethod
    def _empty_output(cls) -> dict[str, typing.Any]:
        """Every calculator output key mapped to None."""
        return {key: None for key in cls.output_keys()}

    @staticmethod
    def _format_output(coll: ProbstResult, ps: ProbstResult) -> dict[str, typing.Any]:
        """Format collision and personal-space ProbstResult instances into calculator output keys."""
        return {
            "probst_c_total": coll.c_total,
            "probst_cc_minus_robot": coll.cc_minus_robot,
            "probst_cc_minus_target": coll.cc_minus_target,
            "probst_cc_minus_time": coll.cc_minus_time,
            "probst_cc_plus_robot": coll.cc_plus_robot,
            "probst_cc_plus_target": coll.cc_plus_target,
            "probst_cc_plus_time": coll.cc_plus_time,
            "probst_r_robot": coll.r_robot,
            "probst_r_target": coll.r_target,
            "probst_r_time": coll.r_time,
            "probst_e_robot": coll.e_robot,
            "probst_e_target": coll.e_target,
            "probst_e_time": coll.e_time,
            "probst_cp_max": coll.cp_max,
            "probst_pdce_min_m": coll.pdce_min_m,
            "probst_tce_s": coll.tce_s,
            "probst_ps_c_total": ps.c_total,
            "probst_ps_r_robot": ps.r_robot,
            "probst_ps_r_target": ps.r_target,
            "probst_ps_r_time": ps.r_time,
            "probst_ps_e_robot": ps.e_robot,
            "probst_ps_e_target": ps.e_target,
            "probst_ps_e_time": ps.e_time,
            "probst_R_robot": coll.r_robot,
            "probst_R_actor2": coll.r_target,
            "probst_R_time": coll.r_time,
            "probst_E_robot": coll.e_robot,
            "probst_E_actor2": coll.e_target,
        }

    def _build_pair(
        self,
        robot: Trajectory2D,
        track: PedTrack,
        bounds_ns: tuple[int | None, int | None] = (None, None),
    ) -> ProbstPair | None:
        """Align robot and pedestrian trajectories within optional nanosecond bounds."""
        t_lo = max(int(robot.t_ns[0]), int(track.t_ns[0]))
        t_hi = min(int(robot.t_ns[-1]), int(track.t_ns[-1]))
        if bounds_ns[0] is not None:
            t_lo = max(t_lo, int(bounds_ns[0]))
        if bounds_ns[1] is not None:
            t_hi = min(t_hi, int(bounds_ns[1]))
        r_slice = robot.slice_ns(t_lo, t_hi)
        if len(r_slice.t_ns) < 2:
            return None
        ped_radius_m = track.radius_m if track.radius_m is not None else float(self.metrics_config.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M))
        t_slice = replace(track.trajectory.resample_onto(r_slice.t_ns), radius_m=ped_radius_m)
        return ProbstPair(robot=r_slice, target=t_slice, window_size_s=NORMALIZATION_WINDOW_S)

    @staticmethod
    def _personal_space_pair(pair: ProbstPair) -> ProbstPair:
        """Copy ProbstPair with personal-space radii (1.0 m per agent)."""
        return ProbstPair(
            robot=replace(pair.robot, radius_m=PERSONAL_SPACE_RADIUS_M),
            target=replace(pair.target, radius_m=PERSONAL_SPACE_RADIUS_M),
            window_size_s=pair.window_size_s,
        )

    def evaluate_encounter_track(
        self,
        robot: Trajectory2D,
        track: PedTrack,
        bounds_ns: tuple[int | None, int | None] = (None, None),
    ) -> tuple[ProbstResult, ProbstResult]:
        """Evaluate collision and personal-space Probst metrics for one encounter window."""
        pair = self._build_pair(robot, track, bounds_ns)
        if pair is None:
            return ProbstResult(), ProbstResult()
        return evaluate_probst(pair), evaluate_probst(self._personal_space_pair(pair))

    def calculate_pre_contact(
        self,
        episode: AlignedEpisodeBundle,
        t0_ns: int,
        target_ped_id: str,
    ) -> dict[str, typing.Any]:
        """Compute Probst responsibility over the pre-contact window [t0 - pre_window_s, t0]."""
        robot, _ = resolve_robot_trajectory(self, episode)
        tracks = extract_episode_ped_tracks(episode)
        target = tracks.get(target_ped_id)
        if len(robot.t_ns) < 2 or target is None:
            return self._empty_output()
        pre_s = float(self.metrics_config.get("pre_contact_window_s", PRE_CONTACT_PROBST_WINDOW_S))
        t_start_ns = int(t0_ns - round(pre_s * NS_PER_S))
        pair = self._build_pair(robot, target, (t_start_ns, int(t0_ns)))
        if pair is None:
            return self._empty_output()
        pre_pair = replace(pair, window_size_s=pre_s)
        coll = evaluate_pre_contact_probst(pre_pair)
        ps = evaluate_pre_contact_probst(self._personal_space_pair(pre_pair))
        return self._format_output(coll, ps)

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Compute Probst responsibility for an encounter/contact window or rolled up across episode encounters."""
        del prior_results
        robot, _ = resolve_robot_trajectory(self, episode)
        tracks = extract_episode_ped_tracks(episode)
        if len(robot.t_ns) < 2 or not tracks:
            return self._empty_output()

        target = resolve_ped_track(tracks, self.metrics_config.get("target_agent_id"))
        contact_t0_ns = _contact_t0_ns(self.metrics_config)
        if target is not None and contact_t0_ns is not None:
            return self.calculate_pre_contact(episode, contact_t0_ns, target.ped_id)

        if target is not None:
            bounds = (self.metrics_config.get("window_start_ns"), self.metrics_config.get("window_end_ns"))
            coll, ps = self.evaluate_encounter_track(robot, target, bounds)
            return self._format_output(coll, ps)

        if len(tracks) == 1:
            coll, ps = self.evaluate_encounter_track(robot, next(iter(tracks.values())))
            return self._format_output(coll, ps)

        spec = EncounterSpec(
            distance_threshold_m=float(self.metrics_config.get("encounter_distance_m", DEFAULT_ENCOUNTER_DIST_M)),
            default_ped_radius_m=float(self.metrics_config.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M)),
        )
        encounters = detect_encounters(robot, tracks, spec)
        evaluated = [self.evaluate_encounter_track(robot, tracks[encounter.ped_id], (encounter.t_start_ns, encounter.t_end_ns)) for encounter in encounters] if encounters else [self.evaluate_encounter_track(robot, track) for track in tracks.values()]
        coll_results = [coll for coll, _ in evaluated]
        ps_results = [ps for _, ps in evaluated]
        return self._format_output(rollup_probst_results(coll_results), rollup_probst_results(ps_results))

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """Compute Probst responsibility for the given scope slice without mutating calculator state."""
        if scope_slice is None:
            return self.calculate(episode, prior_results)

        if scope_slice.scope == MetricScope.CONTACT:
            collider_kind = str(scope_slice.identity.get("collider_kind", "pedestrian")).lower()
            obstacle_id = str(scope_slice.identity.get("obstacle_id", ""))
            if collider_kind != "pedestrian" or not obstacle_id:
                return self._empty_output()
            full_tracks = extract_episode_ped_tracks(scope_slice.full_episode)
            target = resolve_ped_track(full_tracks, obstacle_id)
            if target is None:
                return self._empty_output()
            return self.calculate_pre_contact(scope_slice.full_episode, int(scope_slice.t_start_ns), target.ped_id)

        if scope_slice.scope == MetricScope.ENCOUNTER:
            robot, _ = resolve_robot_trajectory(self, episode)
            tracks = extract_episode_ped_tracks(episode)
            if len(robot.t_ns) < 2 or not tracks:
                return self._empty_output()
            actor_2 = scope_slice.identity.get("actor_2")
            target = resolve_ped_track(tracks, actor_2) if actor_2 is not None else None
            if target is None and len(tracks) == 1:
                target = next(iter(tracks.values()))
            if target is None:
                return self._empty_output()
            coll, ps = self.evaluate_encounter_track(robot, target, (scope_slice.t_start_ns, scope_slice.t_end_ns))
            return self._format_output(coll, ps)

        return self.calculate(episode, prior_results)
