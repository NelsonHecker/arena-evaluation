from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_PED_RADIUS_M,
    EncounterSpec,
    PedTrack,
    Trajectory2D,
    detect_encounters,
    extract_episode_ped_tracks,
    extract_ped_tracks,
    merge_intervals_ns,
    resolve_robot_trajectory,
)
from arena_evaluation.processing.metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

P0_PA: float = 20e-6
P0_SQ_PA2: float = P0_PA * P0_PA
INTIMATE_ZONE_M: float = 0.45
PERSONAL_ZONE_M: float = 1.20
DEFAULT_P_IDLE_W: float = 15.0
DEFAULT_MASS_KG: float = 15.0
DEFAULT_MU: float = 0.02
DEFAULT_V_NOM_V: float = 24.0
DEFAULT_R_INT_OHM: float = 0.1
DEFAULT_L_BASE_DBA: float = 45.0
DEFAULT_K_V: float = 2.5
DEFAULT_K_OMEGA: float = 1.2
DEFAULT_K_A: float = 1.5
GRAVITY_MPS2: float = 9.81
MIN_ACOUSTIC_DIST_M: float = 0.2
SPHERICAL_SPREADING_OFFSET_DB: float = 8.0


@dataclass(frozen=True)
class RobotCostContext:
    """Robot trajectory, yaw, power profile and reference tracks for encounter cost evaluation."""

    robot: Trajectory2D
    yaw: np.ndarray
    power_w: np.ndarray
    robot_params: RobotParams
    reference_tracks_by_name: Mapping[str, PedTrack] = field(default_factory=dict)
    default_ped_radius_m: float = DEFAULT_PED_RADIUS_M


@dataclass(frozen=True)
class EncounterCostRecord:
    """Computed cost breakdown for a single robot-pedestrian encounter."""

    agent_id: int
    agent_name: str
    t_start_ns: int
    t_end_ns: int
    cost_time_s: float
    cost_path_m: float
    cost_energy_j: float
    cost_sound_pa2s: float
    cost_intimate_viol_s: float
    cost_personal_viol_s: float
    cost_ped_detour_m: float | None
    cost_ped_deflection_m: float | None
    cost_ped_delay_s: float | None


def _recorded_power_series(robot_time_ns: np.ndarray, episode: AlignedEpisodeBundle | None) -> np.ndarray | None:
    """Interpolate recorded total_power_w onto robot_time_ns when available."""
    if episode is None:
        return None
    topics = episode.topics if isinstance(episode.topics, dict) else {}
    pwr_df = topics.get("power")
    if isinstance(pwr_df, pl.LazyFrame):
        pwr_df = pwr_df.collect()
    if pwr_df is not None and len(pwr_df) > 0 and "time_ns" in pwr_df.columns and "total_power_w" in pwr_df.columns:
        p_sorted = pwr_df.drop_nulls(["time_ns", "total_power_w"]).sort("time_ns")
        if len(p_sorted) > 0:
            return np.interp(
                robot_time_ns.astype(np.float64),
                p_sorted["time_ns"].to_numpy().astype(np.float64),
                p_sorted["total_power_w"].to_numpy().astype(np.float64),
            )
    if episode.data is not None and "total_power_w" in episode.data.columns and len(episode.data) == len(robot_time_ns):
        arr = episode.data["total_power_w"].to_numpy().astype(np.float64)
        if not np.all(np.isnan(arr)):
            return np.nan_to_num(arr, nan=DEFAULT_P_IDLE_W)
    return None


def _kinematic_rates(robot: Trajectory2D, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (v_lin, v_ang, a_lin, a_ang) along the robot trajectory."""
    n = len(robot.time_ns)
    dt = np.diff(robot.time_ns).astype(np.float64) / 1e9
    dt_safe = np.where(dt <= 0.0, 1e-6, dt)

    v_lin = np.hypot(robot.vx, robot.vy)
    dyaw = (np.diff(yaw) + np.pi) % (2.0 * np.pi) - np.pi
    v_ang = np.zeros(n, dtype=np.float64)
    v_ang[1:] = dyaw / dt_safe
    v_ang[0] = v_ang[1]

    a_lin = np.zeros(n, dtype=np.float64)
    a_lin[1:] = np.diff(v_lin) / dt_safe
    a_ang = np.zeros(n, dtype=np.float64)
    a_ang[1:] = np.diff(v_ang) / dt_safe
    return v_lin, v_ang, a_lin, a_ang


def compute_robot_power_series(
    robot: Trajectory2D,
    yaw: np.ndarray,
    episode: AlignedEpisodeBundle | None = None,
) -> np.ndarray:
    """Instantaneous total electrical power P_total(t) in Watts on the robot time grid."""
    n = len(robot.time_ns)
    if n == 0:
        return np.array([], dtype=np.float64)
    recorded = _recorded_power_series(robot.time_ns, episode)
    if recorded is not None:
        return recorded
    if n < 2:
        return np.full(n, DEFAULT_P_IDLE_W, dtype=np.float64)

    v_lin, v_ang, a_lin, a_ang = _kinematic_rates(robot, yaw)
    m = DEFAULT_MASS_KG
    l_char = max(2.0 * float(robot.radius), 0.2)
    i_z = 0.5 * m * ((0.5 * l_char) ** 2)

    f_lin = m * (GRAVITY_MPS2 * DEFAULT_MU + np.maximum(0.0, a_lin))
    p_lin = np.abs(f_lin * v_lin)
    tau_rot = i_z * np.abs(a_ang) + DEFAULT_MU * (m * GRAVITY_MPS2 * l_char / 4.0)
    p_rot = np.abs(tau_rot * v_ang)

    p_mech = p_lin + p_rot
    p_therm = DEFAULT_R_INT_OHM * ((p_mech / DEFAULT_V_NOM_V) ** 2)
    return DEFAULT_P_IDLE_W + p_mech + p_therm


def compute_ped_sound_spl_dba(
    robot: Trajectory2D,
    yaw: np.ndarray,
    ped_xy: np.ndarray,
) -> np.ndarray:
    """Instantaneous A-weighted sound pressure level L_p(t) in dBA at the pedestrian."""
    n = len(robot.time_ns)
    if n == 0:
        return np.array([], dtype=np.float64)
    dist = np.maximum(np.hypot(ped_xy[:, 0] - robot.x, ped_xy[:, 1] - robot.y), MIN_ACOUSTIC_DIST_M)
    if n < 2:
        return np.maximum(0.0, DEFAULT_L_BASE_DBA - 20.0 * np.log10(dist) - SPHERICAL_SPREADING_OFFSET_DB)

    v_lin, v_ang, a_lin, _ = _kinematic_rates(robot, yaw)
    inner = np.maximum(1.0 + DEFAULT_K_V * (v_lin**2) + DEFAULT_K_OMEGA * (v_ang**2), 1e-6)
    l_source = DEFAULT_L_BASE_DBA + 10.0 * np.log10(inner) + DEFAULT_K_A * np.abs(a_lin)
    return np.maximum(0.0, l_source - 20.0 * np.log10(dist) - SPHERICAL_SPREADING_OFFSET_DB)


def _polyline_length(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean arc length of a 2D polyline."""
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def compare_pedestrian_to_reference_by_name(
    actual_track: PedTrack,
    reference_tracks_by_name: Mapping[str, PedTrack],
    bounds_ns: tuple[int, int] | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Compare a pedestrian trajectory to its same-named counterpart in the robot-free twin episode."""
    ref_track = reference_tracks_by_name.get(actual_track.agent_name)
    if ref_track is None or len(actual_track.time_ns) < 2 or len(ref_track.time_ns) < 2:
        return None, None, None

    act_sub = actual_track.trajectory.slice_ns(bounds_ns[0], bounds_ns[1]) if bounds_ns is not None else actual_track.trajectory
    if len(act_sub.time_ns) < 2:
        act_sub = actual_track.trajectory

    rel_start_ns = int(act_sub.time_ns[0] - actual_track.time_ns[0])
    rel_end_ns = int(act_sub.time_ns[-1] - actual_track.time_ns[0])
    ref_t0 = int(ref_track.time_ns[0])
    ref_sub = ref_track.trajectory.slice_ns(ref_t0 + rel_start_ns, ref_t0 + rel_end_ns)
    if len(ref_sub.time_ns) < 2:
        ref_sub = ref_track.trajectory

    act_len = _polyline_length(act_sub.x, act_sub.y)
    ref_len = _polyline_length(ref_sub.x, ref_sub.y)
    detour_m = float(max(0.0, act_len - ref_len))
    deflection_m = PedestrianDisturbanceCalculator._compute_trajectory_deflection(act_sub.xy, ref_track.trajectory.xy)

    ref_total_dt = float(ref_track.time_ns[-1] - ref_track.time_ns[0]) / 1e9
    ref_total_len = _polyline_length(ref_track.x, ref_track.y)
    ref_speed = (ref_total_len / ref_total_dt) if ref_total_dt > 1e-6 else 0.0
    act_dt = float(act_sub.time_ns[-1] - act_sub.time_ns[0]) / 1e9
    chord_len = float(np.hypot(act_sub.x[-1] - act_sub.x[0], act_sub.y[-1] - act_sub.y[0]))
    delay_s = float(max(0.0, act_dt - (chord_len / ref_speed))) if ref_speed > 1e-3 else 0.0
    return detour_m, deflection_m, delay_s


def extract_reference_tracks_by_name(
    episode: AlignedEpisodeBundle,
    metrics_config: Mapping[str, typing.Any] | None = None,
) -> dict[str, PedTrack]:
    """Resolve robot-free reference pedestrian tracks keyed by pedestrian name."""
    cfg = metrics_config or {}
    raw_ref = cfg.get("reference_peds") or cfg.get("reference_episode")
    if raw_ref is None:
        topics = episode.topics if isinstance(episode.topics, dict) else {}
        raw_ref = topics.get("reference_peds") or topics.get("unhindered_peds")
    if raw_ref is None:
        return {}

    if isinstance(raw_ref, Mapping):
        by_name: dict[str, PedTrack] = {}
        for k, v in raw_ref.items():
            if isinstance(v, PedTrack):
                by_name[v.agent_name] = v
                by_name[str(k)] = v
        if by_name:
            return by_name

    if isinstance(raw_ref, pl.LazyFrame):
        raw_ref = raw_ref.collect()
    if isinstance(raw_ref, pl.DataFrame):
        return {t.agent_name: t for t in extract_ped_tracks(raw_ref).values()}
    if isinstance(raw_ref, AlignedEpisodeBundle):
        return {t.agent_name: t for t in extract_episode_ped_tracks(raw_ref).values()}
    return {}


def evaluate_single_encounter_costs(
    track: PedTrack,
    bounds_ns: tuple[int, int],
    context: RobotCostContext,
) -> EncounterCostRecord:
    """Compute time, path, energy, sound dose and named-pedestrian reference detour for one encounter."""
    t_start_ns, t_end_ns = bounds_ns
    mask = (context.robot.time_ns >= t_start_ns) & (context.robot.time_ns <= t_end_ns)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(context.robot.time_ns - t_start_ns)))
        mask = np.zeros(len(context.robot.time_ns), dtype=bool)
        mask[idx] = True

    r_sub = context.robot.slice_ns(t_start_ns, t_end_ns)
    ryaw = context.yaw[mask]
    pwr = context.power_w[mask]
    t_q = r_sub.time_ns

    cost_time_s = float(max(0, int(t_q[-1] - t_q[0]))) / 1e9
    cost_path_m = _polyline_length(r_sub.x, r_sub.y)

    dt_s = np.maximum(0.0, np.diff(t_q).astype(np.float64) / 1e9) if len(t_q) >= 2 else np.array([], dtype=np.float64)
    cost_energy_j = float(np.sum(0.5 * (pwr[:-1] + pwr[1:]) * dt_s)) if len(dt_s) > 0 else 0.0

    p_sub = track.trajectory.resample_onto(t_q)
    spl_dba = compute_ped_sound_spl_dba(r_sub, ryaw, p_sub.xy)
    p_sq_pa2 = P0_SQ_PA2 * np.power(10.0, spl_dba / 10.0)
    cost_sound_pa2s = float(np.sum(0.5 * (p_sq_pa2[:-1] + p_sq_pa2[1:]) * dt_s)) if len(dt_s) > 0 else 0.0

    ped_r = track.radius if track.radius is not None else context.default_ped_radius_m
    clearance = np.hypot(p_sub.x - r_sub.x, p_sub.y - r_sub.y) - (float(context.robot_params.robot_radius) + ped_r)
    cost_intimate_s = float(np.sum(dt_s[clearance[:-1] < INTIMATE_ZONE_M])) if len(dt_s) > 0 else 0.0
    cost_personal_s = float(np.sum(dt_s[clearance[:-1] < PERSONAL_ZONE_M])) if len(dt_s) > 0 else 0.0

    detour_m: float | None = None
    deflection_m: float | None = None
    delay_s: float | None = None
    if context.reference_tracks_by_name:
        detour_m, deflection_m, delay_s = compare_pedestrian_to_reference_by_name(
            track,
            context.reference_tracks_by_name,
            bounds_ns,
        )

    return EncounterCostRecord(
        agent_id=track.agent_id,
        agent_name=track.agent_name,
        t_start_ns=int(t_q[0]),
        t_end_ns=int(t_q[-1]),
        cost_time_s=cost_time_s,
        cost_path_m=cost_path_m,
        cost_energy_j=cost_energy_j,
        cost_sound_pa2s=cost_sound_pa2s,
        cost_intimate_viol_s=cost_intimate_s,
        cost_personal_viol_s=cost_personal_s,
        cost_ped_detour_m=detour_m,
        cost_ped_deflection_m=deflection_m,
        cost_ped_delay_s=delay_s,
    )


def rollup_encounter_cost_records(
    records: Sequence[EncounterCostRecord],
    context: RobotCostContext,
) -> dict[str, typing.Any]:
    """Roll up encounter costs using interval union for robot-side costs and sum for per-pedestrian costs."""
    if not records:
        return {
            "cost_time_s": 0.0,
            "cost_path_m": 0.0,
            "cost_energy_j": 0.0,
            "cost_sound_pa2s": 0.0,
            "cost_intimate_viol_s": 0.0,
            "cost_personal_viol_s": 0.0,
            "cost_ped_detour_m": None,
            "cost_ped_deflection_m": None,
            "cost_ped_delay_s": None,
            "encounter_count": 0,
        }

    merged_spans = merge_intervals_ns([(r.t_start_ns, r.t_end_ns) for r in records])
    union_time_s = 0.0
    union_path_m = 0.0
    union_energy_j = 0.0
    for s_ns, e_ns in merged_spans:
        union_time_s += float(max(0, e_ns - s_ns)) / 1e9
        mask = (context.robot.time_ns >= s_ns) & (context.robot.time_ns <= e_ns)
        if int(np.sum(mask)) >= 2:
            t_seg = context.robot.time_ns[mask]
            pwr_seg = context.power_w[mask]
            union_path_m += _polyline_length(context.robot.x[mask], context.robot.y[mask])
            dt_s = np.maximum(0.0, np.diff(t_seg).astype(np.float64) / 1e9)
            union_energy_j += float(np.sum(0.5 * (pwr_seg[:-1] + pwr_seg[1:]) * dt_s))

    detours = [r.cost_ped_detour_m for r in records if r.cost_ped_detour_m is not None]
    deflections = [r.cost_ped_deflection_m for r in records if r.cost_ped_deflection_m is not None]
    delays = [r.cost_ped_delay_s for r in records if r.cost_ped_delay_s is not None]

    return {
        "cost_time_s": float(union_time_s),
        "cost_path_m": float(union_path_m),
        "cost_energy_j": float(union_energy_j),
        "cost_sound_pa2s": float(sum(r.cost_sound_pa2s for r in records)),
        "cost_intimate_viol_s": float(sum(r.cost_intimate_viol_s for r in records)),
        "cost_personal_viol_s": float(sum(r.cost_personal_viol_s for r in records)),
        "cost_ped_detour_m": float(sum(detours)) if detours else None,
        "cost_ped_deflection_m": float(sum(deflections)) if deflections else None,
        "cost_ped_delay_s": float(sum(delays)) if delays else None,
        "encounter_count": len(records),
    }


class EncounterCostsCalculator(BaseMetricCalculator):
    """Encounter-scoped time, path, energy, sound dose and reference-matched pedestrian detour calculator."""

    NAME = "encounter_costs"
    CATEGORY = "social"
    SCOPES = {"encounter"}
    REQUIRED_TOPICS = ["odom", "peds"]

    UNITS = {
        "cost_time_s": "s",
        "cost_path_m": "m",
        "cost_energy_j": "J",
        "cost_sound_pa2s": "Pa^2*s",
        "cost_intimate_viol_s": "s",
        "cost_personal_viol_s": "s",
        "cost_ped_detour_m": "m",
        "cost_ped_deflection_m": "m",
        "cost_ped_delay_s": "s",
        "encounter_count": "count",
    }

    PRIMARY_OUTPUTS = [
        "cost_time_s",
        "cost_energy_j",
        "cost_sound_pa2s",
        "cost_ped_detour_m",
    ]

    OUTPUT_DIRECTIONS = {
        "cost_time_s": "lower",
        "cost_path_m": "lower",
        "cost_energy_j": "lower",
        "cost_sound_pa2s": "lower",
        "cost_intimate_viol_s": "lower",
        "cost_personal_viol_s": "lower",
        "cost_ped_detour_m": "lower",
        "cost_ped_deflection_m": "lower",
        "cost_ped_delay_s": "lower",
    }

    ROLLUP = {
        "cost_time_s": "union",
        "cost_path_m": "union",
        "cost_energy_j": "union",
        "cost_sound_pa2s": "sum",
        "cost_intimate_viol_s": "sum",
        "cost_personal_viol_s": "sum",
        "cost_ped_detour_m": "sum",
        "cost_ped_deflection_m": "sum",
        "cost_ped_delay_s": "sum",
        "encounter_count": "sum",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        """Output keys produced by EncounterCostsCalculator."""
        return [
            "cost_time_s",
            "cost_path_m",
            "cost_energy_j",
            "cost_sound_pa2s",
            "cost_intimate_viol_s",
            "cost_personal_viol_s",
            "cost_ped_detour_m",
            "cost_ped_deflection_m",
            "cost_ped_delay_s",
            "encounter_count",
        ]

    def _build_context(
        self,
        episode: AlignedEpisodeBundle,
        reference_episode: AlignedEpisodeBundle | pl.DataFrame | Mapping[str, PedTrack] | None = None,
    ) -> tuple[RobotCostContext, dict[int, PedTrack]]:
        """Build RobotCostContext and extract pedestrian tracks for the episode."""
        robot, ryaw = resolve_robot_trajectory(self, episode)
        tracks = extract_episode_ped_tracks(episode)
        cfg = dict(self.metrics_config)
        if reference_episode is not None:
            cfg["reference_episode"] = reference_episode
        pwr = compute_robot_power_series(robot, ryaw, episode)
        ctx = RobotCostContext(
            robot=robot,
            yaw=ryaw,
            power_w=pwr,
            robot_params=self.robot_params,
            reference_tracks_by_name=extract_reference_tracks_by_name(episode, cfg),
            default_ped_radius_m=float(cfg.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M)),
        )
        return ctx, tracks

    def calculate_encounters(
        self,
        episode: AlignedEpisodeBundle,
        reference_episode: AlignedEpisodeBundle | pl.DataFrame | Mapping[str, PedTrack] | None = None,
    ) -> list[EncounterCostRecord]:
        """Compute EncounterCostRecord for every encounter in the episode."""
        context, tracks = self._build_context(episode, reference_episode)
        if len(context.robot.time_ns) < 2 or not tracks:
            return []

        target_aid = self.metrics_config.get("target_agent_id")
        target_name = self.metrics_config.get("target_agent_name")
        if target_name is not None and target_aid is None:
            for trk in tracks.values():
                if trk.agent_name == str(target_name):
                    target_aid = trk.agent_id
                    break

        if target_aid is not None and int(target_aid) in tracks:
            trk = tracks[int(target_aid)]
            t_s = int(self.metrics_config.get("window_start_ns", max(int(context.robot.time_ns[0]), int(trk.time_ns[0]))))
            t_e = int(self.metrics_config.get("window_end_ns", min(int(context.robot.time_ns[-1]), int(trk.time_ns[-1]))))
            return [evaluate_single_encounter_costs(trk, (t_s, t_e), context)]

        spec = EncounterSpec(
            distance_threshold_m=float(self.metrics_config.get("encounter_distance_m", 4.0)),
            default_ped_radius_m=context.default_ped_radius_m,
        )
        encounters = detect_encounters(context.robot, tracks, spec)
        return [evaluate_single_encounter_costs(tracks[enc.agent_id], (enc.t_start_ns, enc.t_end_ns), context) for enc in encounters]

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Compute encounter costs for a scoped encounter or rolled up across episode encounters."""
        del prior_results
        context, tracks = self._build_context(episode)
        if len(context.robot.time_ns) < 2 or not tracks:
            return {k: None for k in self.output_keys()}

        records = self.calculate_encounters(episode)
        if len(records) == 1:
            rec = records[0]
            return {
                "cost_time_s": rec.cost_time_s,
                "cost_path_m": rec.cost_path_m,
                "cost_energy_j": rec.cost_energy_j,
                "cost_sound_pa2s": rec.cost_sound_pa2s,
                "cost_intimate_viol_s": rec.cost_intimate_viol_s,
                "cost_personal_viol_s": rec.cost_personal_viol_s,
                "cost_ped_detour_m": rec.cost_ped_detour_m,
                "cost_ped_deflection_m": rec.cost_ped_deflection_m,
                "cost_ped_delay_s": rec.cost_ped_delay_s,
                "encounter_count": 1,
            }
        return rollup_encounter_cost_records(records, context)
