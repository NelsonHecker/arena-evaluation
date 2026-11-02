from __future__ import annotations

import json
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import entity_name

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

DEFAULT_PED_RADIUS_M: float = 0.3
DEFAULT_ENCOUNTER_DIST_M: float = 4.0
DEFAULT_MERGE_GAP_S: float = 1.0


@dataclass(frozen=True)
class Trajectory2D:
    """Planar trajectory (x, y, vx, vy) on a sorted nanosecond time axis."""

    time_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    radius: float = DEFAULT_PED_RADIUS_M

    @property
    def xy(self) -> np.ndarray:
        """Positions as an (N, 2) float64 array."""
        return np.column_stack((self.x, self.y))

    @property
    def vel(self) -> np.ndarray:
        """Velocities as an (N, 2) float64 array."""
        return np.column_stack((self.vx, self.vy))

    @property
    def times_s(self) -> np.ndarray:
        """Timestamps in seconds."""
        return self.time_ns.astype(np.float64) / 1e9

    def slice_ns(self, t_start_ns: int, t_end_ns: int) -> Trajectory2D:
        """Sub-trajectory within [t_start_ns, t_end_ns]."""
        mask = (self.time_ns >= t_start_ns) & (self.time_ns <= t_end_ns)
        if not np.any(mask) and len(self.time_ns) > 0:
            nearest = int(np.argmin(np.abs(self.time_ns - t_start_ns)))
            mask = np.zeros(len(self.time_ns), dtype=bool)
            mask[nearest] = True
        return Trajectory2D(
            time_ns=self.time_ns[mask],
            x=self.x[mask],
            y=self.y[mask],
            vx=self.vx[mask],
            vy=self.vy[mask],
            radius=self.radius,
        )

    def resample_onto(self, query_time_ns: np.ndarray) -> Trajectory2D:
        """Linearly interpolate trajectory onto query_time_ns."""
        if len(self.time_ns) == 0 or len(query_time_ns) == 0:
            zeros = np.zeros(len(query_time_ns), dtype=np.float64)
            return Trajectory2D(query_time_ns, zeros, zeros.copy(), zeros.copy(), zeros.copy(), self.radius)
        t_src = self.time_ns.astype(np.float64)
        t_dst = query_time_ns.astype(np.float64)
        return Trajectory2D(
            time_ns=query_time_ns,
            x=np.interp(t_dst, t_src, self.x),
            y=np.interp(t_dst, t_src, self.y),
            vx=np.interp(t_dst, t_src, self.vx),
            vy=np.interp(t_dst, t_src, self.vy),
            radius=self.radius,
        )


@dataclass(frozen=True)
class PedTrack:
    """Identified pedestrian with its 2D trajectory."""

    agent_id: int
    agent_name: str
    trajectory: Trajectory2D
    radius: float | None = None
    desired_speed: float | None = None

    @property
    def time_ns(self) -> np.ndarray:
        """Sorted nanosecond timestamps."""
        return self.trajectory.time_ns

    @property
    def x(self) -> np.ndarray:
        """Planar X coordinates in meters."""
        return self.trajectory.x

    @property
    def y(self) -> np.ndarray:
        """Planar Y coordinates in meters."""
        return self.trajectory.y

    @property
    def vx(self) -> np.ndarray:
        """Planar X velocities in m/s."""
        return self.trajectory.vx

    @property
    def vy(self) -> np.ndarray:
        """Planar Y velocities in m/s."""
        return self.trajectory.vy


@dataclass(frozen=True)
class EncounterWindow:
    """Contiguous robot-pedestrian encounter interval."""

    agent_id: int
    agent_name: str
    t_start_ns: int
    t_end_ns: int
    t_min_ns: int
    min_dist_m: float
    min_clearance_m: float


@dataclass(frozen=True)
class EncounterSpec:
    """Parameters governing encounter segmentation."""

    distance_threshold_m: float = DEFAULT_ENCOUNTER_DIST_M
    merge_gap_s: float = DEFAULT_MERGE_GAP_S
    default_ped_radius_m: float = DEFAULT_PED_RADIUS_M


def resolve_robot_trajectory(calc: BaseMetricCalculator, episode: AlignedEpisodeBundle) -> tuple[Trajectory2D, np.ndarray]:
    """Build robot Trajectory2D and yaw array using BaseMetricCalculator pose/velocity helpers."""
    rx, ry, ryaw, r_time_ns = calc.resolve_native_pose(episode)
    if len(r_time_ns) < 2 and episode.data is not None and "time_ns" in episode.data.columns:
        rx, ry, ryaw, _, _, _ = calc.resolve_robot_pose(episode)
        r_time_ns = episode.data["time_ns"].to_numpy().astype(np.int64)
    rvx, rvy = calc.velocity_from_pose(rx, ry, r_time_ns)
    if len(rvx) >= 2:
        rvx[0], rvy[0] = rvx[1], rvy[1]
    traj = Trajectory2D(
        time_ns=r_time_ns,
        x=rx,
        y=ry,
        vx=rvx,
        vy=rvy,
        radius=float(calc.robot_params.robot_radius),
    )
    return traj, ryaw


def _build_ped_track(
    agent_id: int,
    agent_name: str,
    samples: tuple[list[int], list[float], list[float], list[float], list[float], bool, float | None, float | None],
) -> PedTrack:
    """Construct a sorted PedTrack from accumulated row samples."""
    t_vals, x_vals, y_vals, vx_vals, vy_vals, has_twist, rad, des = samples
    order = np.argsort(np.asarray(t_vals, dtype=np.int64), kind="mergesort")
    t_arr = np.asarray(t_vals, dtype=np.int64)[order]
    x_arr = np.asarray(x_vals, dtype=np.float64)[order]
    y_arr = np.asarray(y_vals, dtype=np.float64)[order]
    vx_arr = np.asarray(vx_vals, dtype=np.float64)[order]
    vy_arr = np.asarray(vy_vals, dtype=np.float64)[order]

    twist_is_zero = np.all(np.hypot(vx_arr, vy_arr) < 1e-9)
    position_moved = len(x_arr) >= 2 and bool(np.any(np.hypot(np.diff(x_arr), np.diff(y_arr)) > 1e-6))
    if not has_twist or (twist_is_zero and position_moved):
        vx_arr, vy_arr = BaseMetricCalculator.velocity_from_pose(x_arr, y_arr, t_arr)
        if len(vx_arr) >= 2:
            vx_arr[0], vy_arr[0] = vx_arr[1], vy_arr[1]

    eff_radius = rad if rad is not None else DEFAULT_PED_RADIUS_M
    traj = Trajectory2D(time_ns=t_arr, x=x_arr, y=y_arr, vx=vx_arr, vy=vy_arr, radius=eff_radius)
    return PedTrack(agent_id=agent_id, agent_name=agent_name, trajectory=traj, radius=rad, desired_speed=des)


def extract_ped_tracks(peds_df: pl.DataFrame | None) -> dict[int, PedTrack]:
    """Extract per-pedestrian tracks keyed by agent_id using BaseMetricCalculator._parse_peds."""
    if peds_df is None or len(peds_df) == 0 or "time_ns" not in peds_df.columns:
        return {}
    df = peds_df.sort("time_ns")
    cols = set(df.columns)
    pos_col = "peds_positions" if "peds_positions" in cols else ("positions" if "positions" in cols else None)
    if pos_col is None:
        return {}

    id_col = "peds_ids" if "peds_ids" in cols else ("agent_ids" if "agent_ids" in cols else None)
    name_col = "peds_names" if "peds_names" in cols else ("agent_names" if "agent_names" in cols else None)
    vel_col = "peds_twists" if "peds_twists" in cols else ("velocities" if "velocities" in cols else None)
    rad_col = "peds_radii" if "peds_radii" in cols else ("radii" if "radii" in cols else None)
    des_col = "peds_desired_speeds" if "peds_desired_speeds" in cols else None

    acc_t: dict[int, list[int]] = {}
    acc_x: dict[int, list[float]] = {}
    acc_y: dict[int, list[float]] = {}
    acc_vx: dict[int, list[float]] = {}
    acc_vy: dict[int, list[float]] = {}
    has_twist: dict[int, bool] = {}
    names: dict[int, str] = {}
    radii: dict[int, float] = {}
    desired: dict[int, float] = {}

    for row in df.iter_rows(named=True):
        t_raw = row["time_ns"]
        if t_raw is None:
            continue
        t_ns_val = int(t_raw)
        ids_seq = list(row[id_col]) if id_col and isinstance(row[id_col], (list, tuple, np.ndarray)) else []
        names_seq = list(row[name_col]) if name_col and isinstance(row[name_col], (list, tuple, np.ndarray)) else []
        rads_seq = list(row[rad_col]) if rad_col and isinstance(row[rad_col], (list, tuple, np.ndarray)) else []
        des_seq = list(row[des_col]) if des_col and isinstance(row[des_col], (list, tuple, np.ndarray)) else []

        hint = max(len(ids_seq), len(names_seq)) or None
        coords = BaseMetricCalculator._parse_peds(row[pos_col], hint)
        if len(coords) == 0:
            continue
        vels = BaseMetricCalculator._parse_peds(row[vel_col], len(coords)) if vel_col and row[vel_col] is not None else np.empty((0, 2))

        for idx in range(len(coords)):
            px_val, py_val = float(coords[idx, 0]), float(coords[idx, 1])
            if not (np.isfinite(px_val) and np.isfinite(py_val)):
                continue
            aid = int(ids_seq[idx]) if idx < len(ids_seq) and ids_seq[idx] is not None else idx
            acc_t.setdefault(aid, []).append(t_ns_val)
            acc_x.setdefault(aid, []).append(px_val)
            acc_y.setdefault(aid, []).append(py_val)
            if idx < len(vels) and np.isfinite(vels[idx, 0]) and np.isfinite(vels[idx, 1]):
                acc_vx.setdefault(aid, []).append(float(vels[idx, 0]))
                acc_vy.setdefault(aid, []).append(float(vels[idx, 1]))
                has_twist[aid] = True
            else:
                acc_vx.setdefault(aid, []).append(0.0)
                acc_vy.setdefault(aid, []).append(0.0)
            if idx < len(names_seq) and names_seq[idx] is not None and str(names_seq[idx]):
                names[aid] = entity_name(str(names_seq[idx]))
            elif aid not in names:
                names[aid] = str(aid)
            if idx < len(rads_seq) and rads_seq[idx] is not None and float(rads_seq[idx]) > 0.0:
                radii[aid] = float(rads_seq[idx])
            if idx < len(des_seq) and des_seq[idx] is not None and float(des_seq[idx]) > 0.0:
                desired[aid] = float(des_seq[idx])

    return {
        aid: _build_ped_track(
            aid,
            names.get(aid, str(aid)),
            (acc_t[aid], acc_x[aid], acc_y[aid], acc_vx[aid], acc_vy[aid], has_twist.get(aid, False), radii.get(aid), desired.get(aid)),
        )
        for aid in acc_t
    }


def extract_episode_ped_tracks(episode: AlignedEpisodeBundle) -> dict[int, PedTrack]:
    """Extract pedestrian tracks from episode.topics['peds'] or fall back to episode.data."""
    topics = episode.topics if isinstance(episode.topics, dict) else {}
    peds_df = topics.get("peds")
    if isinstance(peds_df, pl.LazyFrame):
        peds_df = peds_df.collect()
    tracks = extract_ped_tracks(peds_df)
    if tracks:
        return tracks
    if episode.data is not None and "peds_positions" in episode.data.columns and "time_ns" in episode.data.columns:
        return extract_ped_tracks(episode.data)
    return {}


def resolve_crowd_condition(
    episode: AlignedEpisodeBundle,
    metrics_config: Mapping[str, typing.Any] | None = None,
) -> typing.Literal["blind", "visible"]:
    """Determine whether the crowd condition for the episode is 'blind' or 'visible'."""
    cfg = metrics_config or {}
    if cfg.get("crowd_condition") in ("blind", "visible"):
        return "blind" if cfg["crowd_condition"] == "blind" else "visible"
    for key in ("crowd_visible", "robots_visible"):
        if cfg.get(key) is not None:
            return "visible" if bool(cfg[key]) else "blind"

    if episode.data is not None and len(episode.data) > 0:
        if "crowd_condition" in episode.data.columns:
            vals = episode.data["crowd_condition"].drop_nulls()
            if len(vals) > 0 and str(vals[0]) in ("blind", "visible"):
                return "blind" if str(vals[0]) == "blind" else "visible"
        for col in ("crowd_visible", "robots_visible"):
            if col in episode.data.columns:
                vals_b = episode.data[col].drop_nulls()
                if len(vals_b) > 0:
                    return "visible" if bool(vals_b[0]) else "blind"

    topics = episode.topics if isinstance(episode.topics, dict) else {}
    ep_rec = topics.get("episode_record")
    if isinstance(ep_rec, pl.LazyFrame):
        ep_rec = ep_rec.collect()
    if ep_rec is not None and len(ep_rec) > 0 and "human_params" in ep_rec.columns:
        for raw in reversed(ep_rec["human_params"].to_list()):
            if not raw:
                continue
            try:
                parsed = json.loads(str(raw))
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict) and "humansim.robots_visible" in parsed:
                return "visible" if bool(parsed["humansim.robots_visible"]) else "blind"
    return "visible"


def _agent_encounters(robot: Trajectory2D, track: PedTrack, spec: EncounterSpec) -> list[EncounterWindow]:
    """Detect contiguous encounter windows between the robot and a single pedestrian."""
    if len(robot.time_ns) == 0 or len(track.time_ns) == 0:
        return []
    t_lo = max(int(robot.time_ns[0]), int(track.time_ns[0]))
    t_hi = min(int(robot.time_ns[-1]), int(track.time_ns[-1]))
    if t_hi < t_lo:
        return []
    r_sub = robot.slice_ns(t_lo, t_hi)
    p_sub = track.trajectory.resample_onto(r_sub.time_ns)
    dists = np.hypot(p_sub.x - r_sub.x, p_sub.y - r_sub.y)
    indices = np.flatnonzero(dists <= spec.distance_threshold_m)
    if len(indices) == 0:
        return []

    merge_gap_ns = int(round(spec.merge_gap_s * 1e9))
    spans: list[tuple[int, int]] = []
    start_idx = int(indices[0])
    prev_idx = int(indices[0])
    for idx_val in indices[1:]:
        idx = int(idx_val)
        if int(r_sub.time_ns[idx] - r_sub.time_ns[prev_idx]) <= merge_gap_ns:
            prev_idx = idx
        else:
            spans.append((start_idx, prev_idx))
            start_idx = idx
            prev_idx = idx
    spans.append((start_idx, prev_idx))

    ped_r = track.radius if track.radius is not None else spec.default_ped_radius_m
    windows: list[EncounterWindow] = []
    for s_i, e_i in spans:
        seg_d = dists[s_i : e_i + 1]
        seg_t = r_sub.time_ns[s_i : e_i + 1]
        rel_min = int(np.argmin(seg_d))
        min_d = float(seg_d[rel_min])
        windows.append(
            EncounterWindow(
                agent_id=track.agent_id,
                agent_name=track.agent_name,
                t_start_ns=int(seg_t[0]),
                t_end_ns=int(seg_t[-1]),
                t_min_ns=int(seg_t[rel_min]),
                min_dist_m=min_d,
                min_clearance_m=min_d - (robot.radius + ped_r),
            )
        )
    return windows


DEFAULT_ENCOUNTER_SPEC = EncounterSpec()


def detect_encounters(
    robot: Trajectory2D,
    tracks: Mapping[int, PedTrack],
    spec: EncounterSpec = DEFAULT_ENCOUNTER_SPEC,
) -> list[EncounterWindow]:
    """Segment robot-pedestrian trajectories into sorted EncounterWindow instances."""
    windows: list[EncounterWindow] = []
    for _, track in sorted(tracks.items()):
        windows.extend(_agent_encounters(robot, track, spec))
    windows.sort(key=lambda w: (w.t_start_ns, w.agent_id))
    return windows


def merge_intervals_ns(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent [t_start_ns, t_end_ns] intervals into disjoint sorted intervals."""
    valid = [(int(a), int(b)) for a, b in intervals if b >= a]
    if not valid:
        return []
    valid.sort(key=lambda pair: (pair[0], pair[1]))
    merged: list[tuple[int, int]] = [valid[0]]
    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
