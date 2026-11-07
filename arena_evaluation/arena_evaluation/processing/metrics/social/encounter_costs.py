from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_ENCOUNTER_DIST_M,
    DEFAULT_PED_RADIUS_M,
    PEDESTRIAN_TRACKS_TOPIC,
    ROBOT_TRACKS_TOPIC,
    EncounterSpec,
    PedTrack,
    Trajectory2D,
    detect_encounters,
    episode_table,
    extract_episode_ped_tracks,
    extract_ped_tracks,
    merge_intervals_ns,
    resolve_ped_track,
    resolve_robot_trajectory,
)
from arena_evaluation.processing.metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

P0_PA: Final[float] = 20e-6
P0_SQ_PA2: Final[float] = P0_PA * P0_PA
INTIMATE_ZONE_M: Final[float] = 0.45
PERSONAL_ZONE_M: Final[float] = 1.20
DEFAULT_P_IDLE_W: Final[float] = 15.0
DEFAULT_MASS_KG: Final[float] = 15.0
DEFAULT_MU: Final[float] = 0.02
DEFAULT_V_NOM_V: Final[float] = 24.0
DEFAULT_R_INT_OHM: Final[float] = 0.1
DEFAULT_L_BASE_DBA: Final[float] = 45.0
DEFAULT_K_V: Final[float] = 2.5
DEFAULT_K_OMEGA: Final[float] = 1.2
DEFAULT_K_A: Final[float] = 1.5
GRAVITY_MPS2: Final[float] = 9.81
MIN_ACOUSTIC_DIST_M: Final[float] = 0.2
SPHERICAL_SPREADING_OFFSET_DB: Final[float] = 8.0

ROBOT_POWER_TOPIC: Final[str] = "robot_power"
ROBOT_ACOUSTICS_TOPIC: Final[str] = "robot_acoustics"
REFERENCE_CONFIG_KEYS: Final[tuple[str, ...]] = ("reference_peds", "reference_episode")

ReferenceSource = AlignedEpisodeBundle | pl.DataFrame | Mapping[str, PedTrack]


@dataclass(frozen=True, slots=True)
class RobotCostContext:
    """Robot trajectory, yaw, power profile and reference tracks for encounter cost evaluation."""

    robot: Trajectory2D
    yaw: np.ndarray
    power_w: np.ndarray
    robot_params: RobotParams
    reference_tracks_by_name: Mapping[str, PedTrack] = field(default_factory=dict)
    default_ped_radius_m: float = DEFAULT_PED_RADIUS_M
    acoustics_dba: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class EncounterCostRecord:
    """Computed cost breakdown for a single robot-pedestrian encounter."""

    ped_id: str
    ped_name: str
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
    """Interpolate recorded total_power_w onto robot_time_ns when the episode carries a robot_power table."""
    if episode is None or len(robot_time_ns) == 0:
        return None
    power = episode_table(episode, ROBOT_POWER_TOPIC)
    if power is None or len(power) == 0:
        return None
    series = power.select("t_ns", "total_power_w").drop_nulls(["t_ns", "total_power_w"]).sort("t_ns")
    if len(series) == 0:
        return None
    recorded_time_ns = series.get_column("t_ns").to_numpy().astype(np.float64)
    recorded_power_w = series.get_column("total_power_w").to_numpy().astype(np.float64)
    return np.interp(robot_time_ns.astype(np.float64), recorded_time_ns, recorded_power_w)


def _recorded_acoustics_series(robot_time_ns: np.ndarray, episode: AlignedEpisodeBundle | None) -> np.ndarray | None:
    """Interpolate recorded total_level_af_dba onto robot_time_ns when the episode carries a robot_acoustics table."""
    if episode is None or len(robot_time_ns) == 0:
        return None
    acoustics = episode_table(episode, ROBOT_ACOUSTICS_TOPIC)
    if acoustics is None or len(acoustics) == 0:
        return None
    series = acoustics.select("t_ns", "total_level_af_dba").drop_nulls(["t_ns", "total_level_af_dba"]).sort("t_ns")
    if len(series) == 0:
        return None
    recorded_time_ns = series.get_column("t_ns").to_numpy().astype(np.float64)
    recorded_level_dba = series.get_column("total_level_af_dba").to_numpy().astype(np.float64)
    return np.interp(robot_time_ns.astype(np.float64), recorded_time_ns, recorded_level_dba)


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
    l_char = max(2.0 * float(robot.radius_m), 0.2)
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
    reference_track = reference_tracks_by_name.get(actual_track.ped_name)
    if reference_track is None or len(actual_track.time_ns) < 2 or len(reference_track.time_ns) < 2:
        return None, None, None

    actual_window = actual_track.trajectory.slice_ns(bounds_ns[0], bounds_ns[1]) if bounds_ns is not None else actual_track.trajectory
    if len(actual_window.time_ns) < 2:
        actual_window = actual_track.trajectory

    relative_start_ns = int(actual_window.time_ns[0] - actual_track.time_ns[0])
    relative_end_ns = int(actual_window.time_ns[-1] - actual_track.time_ns[0])
    reference_start_ns = int(reference_track.time_ns[0])
    reference_window = reference_track.trajectory.slice_ns(reference_start_ns + relative_start_ns, reference_start_ns + relative_end_ns)
    if len(reference_window.time_ns) < 2:
        reference_window = reference_track.trajectory

    actual_length_m = _polyline_length(actual_window.x, actual_window.y)
    reference_length_m = _polyline_length(reference_window.x, reference_window.y)
    detour_m = float(max(0.0, actual_length_m - reference_length_m))
    deflection_m = PedestrianDisturbanceCalculator._compute_trajectory_deflection(actual_window.xy, reference_track.trajectory.xy)

    reference_duration_s = float(reference_track.time_ns[-1] - reference_track.time_ns[0]) / 1e9
    reference_total_length_m = _polyline_length(reference_track.x, reference_track.y)
    reference_speed_mps = (reference_total_length_m / reference_duration_s) if reference_duration_s > 1e-6 else 0.0
    actual_duration_s = float(actual_window.time_ns[-1] - actual_window.time_ns[0]) / 1e9
    chord_length_m = float(np.hypot(actual_window.x[-1] - actual_window.x[0], actual_window.y[-1] - actual_window.y[0]))
    delay_s = float(max(0.0, actual_duration_s - (chord_length_m / reference_speed_mps))) if reference_speed_mps > 1e-3 else 0.0
    return detour_m, deflection_m, delay_s


def extract_reference_tracks_by_name(
    episode: AlignedEpisodeBundle,
    metrics_config: Mapping[str, Any] | None = None,
) -> dict[str, PedTrack]:
    """Resolve robot-free reference pedestrian tracks keyed by pedestrian name."""
    raw_reference = _reference_source(metrics_config)
    if raw_reference is None:
        return {}
    if isinstance(raw_reference, Mapping):
        keyed: dict[str, PedTrack] = {}
        for key, track in raw_reference.items():
            if isinstance(track, PedTrack):
                keyed[str(key)] = track
                keyed[track.ped_name] = track
        return keyed
    if isinstance(raw_reference, pl.LazyFrame):
        raw_reference = raw_reference.collect()
    if isinstance(raw_reference, pl.DataFrame):
        return _keyed_by_name(extract_ped_tracks(raw_reference))
    return _keyed_by_name(extract_episode_ped_tracks(raw_reference))


def _reference_source(metrics_config: Mapping[str, Any] | None) -> ReferenceSource | None:
    """Reference pedestrian source of the metrics config: a robot-free twin episode, its flat pedestrian rows, or tracks.

    There is no topic fallback: a robot-free twin recording is a separate episode, matched upstream by
    stage and seed and injected through ``metrics_config``.
    """
    config = metrics_config or {}
    for key in REFERENCE_CONFIG_KEYS:
        configured = config.get(key)
        if configured is not None:
            return configured
    return None


def _keyed_by_name(tracks: Mapping[str, PedTrack]) -> dict[str, PedTrack]:
    """Reference tracks keyed by pedestrian name."""
    return {track.ped_name: track for track in tracks.values()}


def evaluate_single_encounter_costs(
    track: PedTrack,
    bounds_ns: tuple[int, int],
    context: RobotCostContext,
) -> EncounterCostRecord:
    """Compute time, path, energy, sound dose and named-pedestrian reference detour for one encounter."""
    t_start_ns, t_end_ns = bounds_ns
    mask = (context.robot.time_ns >= t_start_ns) & (context.robot.time_ns <= t_end_ns)
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(context.robot.time_ns - t_start_ns)))
        mask = np.zeros(len(context.robot.time_ns), dtype=bool)
        mask[nearest] = True

    robot_window = context.robot.slice_ns(t_start_ns, t_end_ns)
    yaw_window = context.yaw[mask]
    power_window = context.power_w[mask]
    window_time_ns = robot_window.time_ns

    cost_time_s = float(max(0, int(window_time_ns[-1] - window_time_ns[0]))) / 1e9
    cost_path_m = _polyline_length(robot_window.x, robot_window.y)

    dt_s = np.maximum(0.0, np.diff(window_time_ns).astype(np.float64) / 1e9) if len(window_time_ns) >= 2 else np.array([], dtype=np.float64)
    cost_energy_j = float(np.sum(0.5 * (power_window[:-1] + power_window[1:]) * dt_s)) if len(dt_s) > 0 else 0.0

    ped_window = track.trajectory.resample_onto(window_time_ns)
    if context.acoustics_dba is not None and len(context.acoustics_dba) == len(context.robot.time_ns):
        acoustics_window = context.acoustics_dba[mask]
        dist = np.maximum(np.hypot(ped_window.x - robot_window.x, ped_window.y - robot_window.y), MIN_ACOUSTIC_DIST_M)
        spl_dba = np.maximum(0.0, acoustics_window - 20.0 * np.log10(dist))
    else:
        spl_dba = compute_ped_sound_spl_dba(robot_window, yaw_window, ped_window.xy)
    p_sq_pa2 = P0_SQ_PA2 * np.power(10.0, spl_dba / 10.0)
    cost_sound_pa2s = float(np.sum(0.5 * (p_sq_pa2[:-1] + p_sq_pa2[1:]) * dt_s)) if len(dt_s) > 0 else 0.0

    ped_radius_m = track.radius_m if track.radius_m is not None else context.default_ped_radius_m
    clearance = np.hypot(ped_window.x - robot_window.x, ped_window.y - robot_window.y) - (float(context.robot_params.robot_radius) + ped_radius_m)
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
        ped_id=track.ped_id,
        ped_name=track.ped_name,
        t_start_ns=int(window_time_ns[0]),
        t_end_ns=int(window_time_ns[-1]),
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
) -> dict[str, Any]:
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
            "detour_time_s": None,
            "path_deviation_m": None,
            "encounter_energy_j": 0.0,
            "sound_exposure_dose_pa2s": 0.0,
            "intimate_space_violation_s": 0.0,
            "personal_space_violation_s": 0.0,
        }

    merged_spans = merge_intervals_ns([(record.t_start_ns, record.t_end_ns) for record in records])
    union_time_s = 0.0
    union_path_m = 0.0
    union_energy_j = 0.0
    for span_start_ns, span_end_ns in merged_spans:
        union_time_s += float(max(0, span_end_ns - span_start_ns)) / 1e9
        mask = (context.robot.time_ns >= span_start_ns) & (context.robot.time_ns <= span_end_ns)
        if int(np.sum(mask)) >= 2:
            span_time_ns = context.robot.time_ns[mask]
            span_power_w = context.power_w[mask]
            union_path_m += _polyline_length(context.robot.x[mask], context.robot.y[mask])
            span_dt_s = np.maximum(0.0, np.diff(span_time_ns).astype(np.float64) / 1e9)
            union_energy_j += float(np.sum(0.5 * (span_power_w[:-1] + span_power_w[1:]) * span_dt_s))

    detours = [record.cost_ped_detour_m for record in records if record.cost_ped_detour_m is not None]
    deflections = [record.cost_ped_deflection_m for record in records if record.cost_ped_deflection_m is not None]
    delays = [record.cost_ped_delay_s for record in records if record.cost_ped_delay_s is not None]

    return {
        "cost_time_s": float(union_time_s),
        "cost_path_m": float(union_path_m),
        "cost_energy_j": float(union_energy_j),
        "cost_sound_pa2s": float(sum(record.cost_sound_pa2s for record in records)),
        "cost_intimate_viol_s": float(sum(record.cost_intimate_viol_s for record in records)),
        "cost_personal_viol_s": float(sum(record.cost_personal_viol_s for record in records)),
        "cost_ped_detour_m": float(sum(detours)) if detours else None,
        "cost_ped_deflection_m": float(sum(deflections)) if deflections else None,
        "cost_ped_delay_s": float(sum(delays)) if delays else None,
        "encounter_count": len(records),
        "detour_time_s": float(sum(delays)) if delays else None,
        "path_deviation_m": float(sum(deflections)) if deflections else None,
        "encounter_energy_j": float(union_energy_j),
        "sound_exposure_dose_pa2s": float(sum(record.cost_sound_pa2s for record in records)),
        "intimate_space_violation_s": float(sum(record.cost_intimate_viol_s for record in records)),
        "personal_space_violation_s": float(sum(record.cost_personal_viol_s for record in records)),
    }


def _single_record_outputs(record: EncounterCostRecord) -> dict[str, Any]:
    """Outputs of an episode holding exactly one encounter."""
    return {
        "cost_time_s": record.cost_time_s,
        "cost_path_m": record.cost_path_m,
        "cost_energy_j": record.cost_energy_j,
        "cost_sound_pa2s": record.cost_sound_pa2s,
        "cost_intimate_viol_s": record.cost_intimate_viol_s,
        "cost_personal_viol_s": record.cost_personal_viol_s,
        "cost_ped_detour_m": record.cost_ped_detour_m,
        "cost_ped_deflection_m": record.cost_ped_deflection_m,
        "cost_ped_delay_s": record.cost_ped_delay_s,
        "encounter_count": 1,
        "detour_time_s": record.cost_ped_delay_s,
        "path_deviation_m": record.cost_ped_deflection_m,
        "encounter_energy_j": record.cost_energy_j,
        "sound_exposure_dose_pa2s": record.cost_sound_pa2s,
        "intimate_space_violation_s": record.cost_intimate_viol_s,
        "personal_space_violation_s": record.cost_personal_viol_s,
    }


class EncounterCostsCalculator(BaseMetricCalculator):
    """Encounter-scoped time, path, energy, sound dose and reference-matched pedestrian detour calculator."""

    NAME = "encounter_costs"
    CATEGORY = "social"
    SCOPE = MetricScope.ENCOUNTER
    SCOPES: ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.ENCOUNTER, MetricScope.EPISODE})
    REQUIRED_TOPICS = [ROBOT_TRACKS_TOPIC, PEDESTRIAN_TRACKS_TOPIC]

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
        "detour_time_s": "s",
        "path_deviation_m": "m",
        "encounter_energy_j": "J",
        "sound_exposure_dose_pa2s": "Pa^2*s",
        "intimate_space_violation_s": "s",
        "personal_space_violation_s": "s",
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
        "detour_time_s": "lower",
        "path_deviation_m": "lower",
        "encounter_energy_j": "lower",
        "sound_exposure_dose_pa2s": "lower",
        "intimate_space_violation_s": "lower",
        "personal_space_violation_s": "lower",
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
            "detour_time_s",
            "path_deviation_m",
            "encounter_energy_j",
            "sound_exposure_dose_pa2s",
            "intimate_space_violation_s",
            "personal_space_violation_s",
        ]

    def _build_context(
        self,
        episode: AlignedEpisodeBundle,
        reference_episode: ReferenceSource | None = None,
    ) -> tuple[RobotCostContext, dict[str, PedTrack]]:
        """Build the robot cost context and extract the pedestrian tracks of the episode."""
        robot, yaw = resolve_robot_trajectory(self, episode)
        tracks = extract_episode_ped_tracks(episode)
        metrics_config = dict(self.metrics_config)
        if reference_episode is not None:
            metrics_config["reference_episode"] = reference_episode
        context = RobotCostContext(
            robot=robot,
            yaw=yaw,
            power_w=compute_robot_power_series(robot, yaw, episode),
            robot_params=self.robot_params,
            reference_tracks_by_name=extract_reference_tracks_by_name(episode, metrics_config),
            default_ped_radius_m=float(metrics_config.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M)),
            acoustics_dba=_recorded_acoustics_series(robot.time_ns, episode),
        )
        return context, tracks

    def _records(self, context: RobotCostContext, tracks: Mapping[str, PedTrack]) -> list[EncounterCostRecord]:
        """Encounter cost records of the explicitly scoped pedestrian, else of every detected encounter."""
        target_track = resolve_ped_track(tracks, self.metrics_config.get("target_agent_id"), self.metrics_config.get("target_agent_name"))
        if target_track is not None:
            window_start_ns = int(self.metrics_config.get("window_start_ns", max(int(context.robot.time_ns[0]), int(target_track.time_ns[0]))))
            window_end_ns = int(self.metrics_config.get("window_end_ns", min(int(context.robot.time_ns[-1]), int(target_track.time_ns[-1]))))
            return [evaluate_single_encounter_costs(target_track, (window_start_ns, window_end_ns), context)]

        spec = EncounterSpec(
            distance_threshold_m=float(self.metrics_config.get("encounter_distance_m", DEFAULT_ENCOUNTER_DIST_M)),
            default_ped_radius_m=context.default_ped_radius_m,
        )
        encounters = detect_encounters(context.robot, tracks, spec)
        return [evaluate_single_encounter_costs(tracks[encounter.ped_id], (encounter.t_start_ns, encounter.t_end_ns), context) for encounter in encounters]

    def calculate_encounters(
        self,
        episode: AlignedEpisodeBundle,
        reference_episode: ReferenceSource | None = None,
    ) -> list[EncounterCostRecord]:
        """Compute an EncounterCostRecord for every encounter in the episode."""
        context, tracks = self._build_context(episode, reference_episode)
        if len(context.robot.time_ns) < 2 or not tracks:
            return []
        return self._records(context, tracks)

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compute encounter costs for a scoped encounter or rolled up across episode encounters."""
        del prior_results
        context, tracks = self._build_context(episode)
        if len(context.robot.time_ns) < 2 or not tracks:
            return {key: None for key in self.output_keys()}
        records = self._records(context, tracks)
        if len(records) == 1:
            return _single_record_outputs(records[0])
        return rollup_encounter_cost_records(records, context)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, Any]:
        """Compute encounter costs for a scope slice without mutating calculator state."""
        if scope_slice is None or scope_slice.scope != MetricScope.ENCOUNTER:
            return self.calculate(episode, prior_results)

        context, tracks = self._build_context(episode)
        if len(context.robot.time_ns) < 2 or not tracks:
            return {key: None for key in self.output_keys()}

        actor_2 = scope_slice.identity.get("actor_2")
        actor_2_kind = str(scope_slice.identity.get("actor_2_kind", "pedestrian")).lower()
        target_track = resolve_ped_track(tracks, actor_2) if actor_2 is not None else None
        if target_track is None and len(tracks) == 1:
            target_track = next(iter(tracks.values()))
        if target_track is None:
            return {key: None for key in self.output_keys()}

        record = evaluate_single_encounter_costs(
            target_track,
            (int(scope_slice.t_start_ns), int(scope_slice.t_end_ns)),
            context,
        )
        out = _single_record_outputs(record)

        if actor_2_kind == "robot" and out["path_deviation_m"] is None and out["detour_time_s"] is None:
            robot_window = context.robot.slice_ns(int(scope_slice.t_start_ns), int(scope_slice.t_end_ns))
            chord_m = float(np.hypot(robot_window.x[-1] - robot_window.x[0], robot_window.y[-1] - robot_window.y[0])) if len(robot_window.x) >= 2 else 0.0
            full_robot, _ = resolve_robot_trajectory(self, scope_slice.full_episode)
            v_nominal_mps = float(self.robot_params.max_linear_velocity) if self.robot_params.max_linear_velocity > 0.0 else (float(np.max(np.hypot(full_robot.vx, full_robot.vy))) if len(full_robot.time_ns) >= 2 else 1.0)
            if v_nominal_mps <= 1e-3:
                v_nominal_mps = 1.0
            out["path_deviation_m"] = float(max(0.0, record.cost_path_m - chord_m))
            out["detour_time_s"] = float(max(0.0, record.cost_time_s - (chord_m / v_nominal_mps)))
            out["cost_ped_deflection_m"] = out["path_deviation_m"]
            out["cost_ped_delay_s"] = out["detour_time_s"]

        return out
