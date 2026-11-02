from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_PED_RADIUS_M,
    EncounterSpec,
    PedTrack,
    Trajectory2D,
    detect_encounters,
    extract_episode_ped_tracks,
    resolve_robot_trajectory,
)

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

NORMALIZATION_WINDOW_S: float = 12.0
PRE_CONTACT_PROBST_WINDOW_S: float = 3.0
PERSONAL_SPACE_RADIUS_M: float = 1.0


@dataclass(frozen=True)
class ProbstParts:
    """Additive conflict totals (Eq. 10) for rollup aggregation."""

    c_total: float
    cc_minus_robot: float
    cc_minus_target: float
    cc_minus_time: float
    cc_plus_robot: float
    cc_plus_target: float
    cc_plus_time: float


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    n = len(pair.robot.time_ns)
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
        return {
            "r_robot": None,
            "r_target": None,
            "r_time": None,
            "e_robot": None,
            "e_target": None,
            "e_time": None,
        }
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
    r_sum = float(pair.robot.radius + pair.target.radius)
    if len(pair.robot.time_ns) < 2 or len(pair.target.time_ns) != len(pair.robot.time_ns) or r_sum <= 0.0:
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
    if len(pair.robot.time_ns) < 2:
        return ProbstResult()
    dists = np.linalg.norm(pair.target.xy - pair.robot.xy, axis=1)
    return _evaluate_pair_at_tce(pair, int(np.argmin(dists)))


def evaluate_pre_contact_probst(pair: ProbstPair) -> ProbstResult:
    """Evaluate Probst et al. (ICRA 2026) Eqs. 1-12 anchored at contact onset (final sample)."""
    if len(pair.robot.time_ns) < 2:
        return ProbstResult()
    return _evaluate_pair_at_tce(pair, len(pair.robot.time_ns) - 1)


def rollup_probst_results(results: Sequence[ProbstResult]) -> ProbstResult:
    """Aggregate multiple ProbstResult rows by summing additive parts and rebuilding shares."""
    valid = [r for r in results if r.c_total is not None]
    if not valid:
        return ProbstResult()

    parts = ProbstParts(
        c_total=float(sum(r.c_total for r in valid if r.c_total is not None)),
        cc_minus_robot=float(sum(r.cc_minus_robot for r in valid if r.cc_minus_robot is not None)),
        cc_minus_target=float(sum(r.cc_minus_target for r in valid if r.cc_minus_target is not None)),
        cc_minus_time=float(sum(r.cc_minus_time for r in valid if r.cc_minus_time is not None)),
        cc_plus_robot=float(sum(r.cc_plus_robot for r in valid if r.cc_plus_robot is not None)),
        cc_plus_target=float(sum(r.cc_plus_target for r in valid if r.cc_plus_target is not None)),
        cc_plus_time=float(sum(r.cc_plus_time for r in valid if r.cc_plus_time is not None)),
    )
    shares = derive_probst_shares(parts)
    cp_vals = [r.cp_max for r in valid if r.cp_max is not None]
    pdce_vals = [r.pdce_min_m for r in valid if r.pdce_min_m is not None]
    tce_vals = [r.tce_s for r in valid if r.tce_s is not None]

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


class ProbstResponsibilityCalculator(BaseMetricCalculator):
    """Probst et al. (ICRA 2026) Conflict, Responsibility and Engagement calculator (Eqs. 1-12)."""

    NAME = "probst_responsibility"
    CATEGORY = "social"
    SCOPES = {"encounter", "contact"}
    REQUIRED_TOPICS = ["odom", "peds"]

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
        ]

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
        }

    def _build_pair(
        self,
        robot: Trajectory2D,
        track: PedTrack,
        bounds_ns: tuple[int | None, int | None] = (None, None),
    ) -> ProbstPair | None:
        """Align robot and pedestrian trajectories within optional nanosecond bounds."""
        t_lo = max(int(robot.time_ns[0]), int(track.time_ns[0]))
        t_hi = min(int(robot.time_ns[-1]), int(track.time_ns[-1]))
        if bounds_ns[0] is not None:
            t_lo = max(t_lo, int(bounds_ns[0]))
        if bounds_ns[1] is not None:
            t_hi = min(t_hi, int(bounds_ns[1]))
        r_slice = robot.slice_ns(t_lo, t_hi)
        if len(r_slice.time_ns) < 2:
            return None
        ped_r = float(track.radius) if track.radius is not None else float(self.metrics_config.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M))
        t_slice = replace(track.trajectory.resample_onto(r_slice.time_ns), radius=ped_r)
        return ProbstPair(robot=r_slice, target=t_slice, window_size_s=NORMALIZATION_WINDOW_S)

    @staticmethod
    def _personal_space_pair(pair: ProbstPair) -> ProbstPair:
        """Copy ProbstPair with personal-space radii (1.0 m per agent)."""
        return ProbstPair(
            robot=replace(pair.robot, radius=PERSONAL_SPACE_RADIUS_M),
            target=replace(pair.target, radius=PERSONAL_SPACE_RADIUS_M),
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
        target_agent_id: int,
    ) -> dict[str, typing.Any]:
        """Compute Probst responsibility over the pre-contact window [t0 - pre_window_s, t0]."""
        robot, _ = resolve_robot_trajectory(self, episode)
        tracks = extract_episode_ped_tracks(episode)
        if len(robot.time_ns) < 2 or target_agent_id not in tracks:
            return {k: None for k in self.output_keys()}
        pre_s = float(self.metrics_config.get("pre_contact_window_s", PRE_CONTACT_PROBST_WINDOW_S))
        t_start_ns = int(t0_ns - round(pre_s * 1e9))
        pair = self._build_pair(robot, tracks[target_agent_id], (t_start_ns, int(t0_ns)))
        if pair is None:
            return {k: None for k in self.output_keys()}
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
        if len(robot.time_ns) < 2 or not tracks:
            return {k: None for k in self.output_keys()}

        cfg_agent_id = self.metrics_config.get("target_agent_id")
        cfg_t0_ns = self.metrics_config.get("contact_t0_ns")
        if cfg_t0_ns is not None and cfg_agent_id is not None and int(cfg_agent_id) in tracks:
            return self.calculate_pre_contact(episode, int(cfg_t0_ns), int(cfg_agent_id))

        if cfg_agent_id is not None and int(cfg_agent_id) in tracks:
            bounds = (self.metrics_config.get("window_start_ns"), self.metrics_config.get("window_end_ns"))
            coll, ps = self.evaluate_encounter_track(robot, tracks[int(cfg_agent_id)], bounds)
            return self._format_output(coll, ps)

        if len(tracks) == 1:
            coll, ps = self.evaluate_encounter_track(robot, next(iter(tracks.values())))
            return self._format_output(coll, ps)

        spec = EncounterSpec(
            distance_threshold_m=float(self.metrics_config.get("encounter_distance_m", 4.0)),
            default_ped_radius_m=float(self.metrics_config.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M)),
        )
        encounters = detect_encounters(robot, tracks, spec)
        coll_list: list[ProbstResult] = []
        ps_list: list[ProbstResult] = []
        if encounters:
            for enc in encounters:
                c_res, ps_res = self.evaluate_encounter_track(robot, tracks[enc.agent_id], (enc.t_start_ns, enc.t_end_ns))
                coll_list.append(c_res)
                ps_list.append(ps_res)
        else:
            for track in tracks.values():
                c_res, ps_res = self.evaluate_encounter_track(robot, track)
                coll_list.append(c_res)
                ps_list.append(ps_res)

        return self._format_output(rollup_probst_results(coll_list), rollup_probst_results(ps_list))
