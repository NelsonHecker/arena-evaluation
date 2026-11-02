from __future__ import annotations

import re
import typing
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import entity_name
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_PED_RADIUS_M,
    PedTrack,
    Trajectory2D,
    extract_episode_ped_tracks,
    resolve_crowd_condition,
    resolve_robot_trajectory,
)

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PRE_CONTACT_WINDOW_S: float = 0.5
POST_CONTACT_WINDOW_S: float = 0.5
CONTACT_MERGE_GAP_S: float = 0.25
STOPPED_SPEED_MPS: float = 0.05
MOVING_SPEED_MPS: float = 0.15
DOMINANT_SHARE: float = 0.70

CloserLabel = typing.Literal["robot", "pedestrian", "mutual", "static"]
CounterfactualRegime = typing.Literal["blind_exact", "reactive_approx", "unavailable"]


@dataclass(frozen=True)
class ContactEvent:
    """Detected contact interval before kinematic attribution."""

    t0_ns: int
    t_end_ns: int
    obstacle_kind: str
    obstacle_id: str
    track: PedTrack | None = None
    static_point: tuple[float, float] | None = None


@dataclass(frozen=True)
class ContactContext:
    """Episode-level context needed to attribute a ContactEvent."""

    robot: Trajectory2D
    crowd_condition: typing.Literal["blind", "visible"] = "visible"
    default_ped_radius_m: float = DEFAULT_PED_RADIUS_M
    pre_window_s: float = PRE_CONTACT_WINDOW_S
    post_window_s: float = POST_CONTACT_WINDOW_S


@dataclass(frozen=True)
class ContactAttributionRecord:
    """Attribution record for a single contact event."""

    contact_id: int
    t0_ns: int
    t_end_ns: int
    t0_s: float
    duration_s: float
    obstacle_kind: str
    obstacle_id: str
    target_agent_id: int | None
    target_agent_name: str | None
    v_close_robot_mps: float
    v_close_target_mps: float
    s_robot: float
    closer_label: CloserLabel
    v_robot_impact_mps: float
    v_target_impact_mps: float
    relative_impact_speed_mps: float
    counterfactual_regime: CounterfactualRegime
    counterfactual_freeze_min_dist_m: float | None
    counterfactual_freeze_persists: bool | None
    is_robot_fault: bool | None
    is_robot_fault_lower_bound: bool | None


def classify_closer_label(
    v_close_robot_mps: float,
    v_close_target_mps: float,
    obstacle_kind: str = "pedestrian",
) -> tuple[CloserLabel, float]:
    """Classify contact closing speeds into closer_label and robot closing share s_robot."""
    if obstacle_kind != "pedestrian":
        return "static", (1.0 if v_close_robot_mps > 0.0 else 0.0)

    v_r_pos = max(0.0, float(v_close_robot_mps))
    v_t_pos = max(0.0, float(v_close_target_mps))
    denom = v_r_pos + v_t_pos
    s_robot = float(v_r_pos / denom) if denom > 1e-9 else 0.5

    if v_close_robot_mps <= STOPPED_SPEED_MPS and v_close_target_mps > MOVING_SPEED_MPS:
        return "pedestrian", s_robot
    if v_close_target_mps <= STOPPED_SPEED_MPS and v_close_robot_mps > MOVING_SPEED_MPS:
        return "robot", s_robot
    if v_close_robot_mps > STOPPED_SPEED_MPS or v_close_target_mps > STOPPED_SPEED_MPS:
        if s_robot >= DOMINANT_SHARE:
            return "robot", s_robot
        if s_robot <= (1.0 - DOMINANT_SHARE):
            return "pedestrian", s_robot
    return "mutual", s_robot


def compute_closing_speeds(event: ContactEvent, context: ContactContext) -> tuple[float, float]:
    """Mean robot and target closing speeds (m/s) over [t0 - pre_window_s, t0]."""
    win_start_ns = int(event.t0_ns - round(context.pre_window_s * 1e9))
    r_win = context.robot.slice_ns(win_start_ns, event.t0_ns)

    if event.track is not None and len(event.track.time_ns) > 0:
        p_win = event.track.trajectory.resample_onto(r_win.time_ns)
        dx = p_win.x - r_win.x
        dy = p_win.y - r_win.y
        dist = np.hypot(dx, dy)
        valid = dist > 1e-6
        if not np.any(valid):
            return float(np.mean(np.hypot(r_win.vx, r_win.vy))), float(np.mean(np.hypot(p_win.vx, p_win.vy)))
        ux, uy = dx[valid] / dist[valid], dy[valid] / dist[valid]
        v_close_r = r_win.vx[valid] * ux + r_win.vy[valid] * uy
        v_close_t = -(p_win.vx[valid] * ux + p_win.vy[valid] * uy)
        return float(np.mean(v_close_r)), float(np.mean(v_close_t))

    if event.static_point is not None:
        dx = float(event.static_point[0]) - r_win.x
        dy = float(event.static_point[1]) - r_win.y
        dist = np.hypot(dx, dy)
        valid = dist > 1e-6
        if np.any(valid):
            ux, uy = dx[valid] / dist[valid], dy[valid] / dist[valid]
            return float(np.mean(r_win.vx[valid] * ux + r_win.vy[valid] * uy)), 0.0

    return float(np.mean(np.hypot(r_win.vx, r_win.vy))), 0.0


def evaluate_frozen_robot_counterfactual(
    event: ContactEvent,
    context: ContactContext,
) -> tuple[CounterfactualRegime, float | None, bool | None]:
    """Frozen-robot counterfactual test over [t0 - pre_window_s, t0 + post_window_s]."""
    track = event.track
    if event.obstacle_kind != "pedestrian" or track is None or len(track.time_ns) == 0 or len(context.robot.time_ns) == 0:
        return "unavailable", None, None

    win_start_ns = int(event.t0_ns - round(context.pre_window_s * 1e9))
    win_end_ns = int(event.t0_ns + round(context.post_window_s * 1e9))
    ped_in_win = (track.time_ns >= win_start_ns) & (track.time_ns <= win_end_ns)
    if not np.any(ped_in_win):
        return "unavailable", None, None

    t_freeze_ns = max(int(context.robot.time_ns[0]), win_start_ns)
    frozen_pose = context.robot.resample_onto(np.asarray([t_freeze_ns], dtype=np.int64))
    freeze_x, freeze_y = float(frozen_pose.x[0]), float(frozen_pose.y[0])

    dists = np.hypot(track.x[ped_in_win] - freeze_x, track.y[ped_in_win] - freeze_y)
    min_dist_m = float(np.min(dists))
    ped_r = track.radius if track.radius is not None else context.default_ped_radius_m
    persists = bool(min_dist_m <= float(context.robot.radius + ped_r))
    regime: CounterfactualRegime = "blind_exact" if context.crowd_condition == "blind" else "reactive_approx"
    return regime, min_dist_m, persists


def attribute_single_contact(
    contact_id: int,
    event: ContactEvent,
    context: ContactContext,
) -> ContactAttributionRecord:
    """Compute full attribution record for a single ContactEvent."""
    v_close_r, v_close_t = compute_closing_speeds(event, context)
    closer_label, s_robot = classify_closer_label(v_close_r, v_close_t, event.obstacle_kind)

    t0_arr = np.asarray([event.t0_ns], dtype=np.int64)
    r_at_t0 = context.robot.resample_onto(t0_arr)
    rvx_0, rvy_0 = float(r_at_t0.vx[0]), float(r_at_t0.vy[0])
    v_robot_impact = float(np.hypot(rvx_0, rvy_0))

    if event.obstacle_kind == "pedestrian" and event.track is not None and len(event.track.time_ns) > 0:
        p_at_t0 = event.track.trajectory.resample_onto(t0_arr)
        pvx_0, pvy_0 = float(p_at_t0.vx[0]), float(p_at_t0.vy[0])
        v_target_impact = float(np.hypot(pvx_0, pvy_0))
        rel_impact = float(np.hypot(rvx_0 - pvx_0, rvy_0 - pvy_0))
    else:
        v_target_impact = 0.0
        rel_impact = v_robot_impact

    regime, min_dist_m, persists = evaluate_frozen_robot_counterfactual(event, context)
    fault_predicate = (closer_label in ("robot", "mutual", "static")) or (persists is False)
    is_robot_fault: bool | None = bool(fault_predicate) if regime == "blind_exact" else None
    is_robot_fault_lb: bool | None = bool(fault_predicate) if regime == "reactive_approx" else None

    ep_start_ns = int(context.robot.time_ns[0]) if len(context.robot.time_ns) > 0 else event.t0_ns
    return ContactAttributionRecord(
        contact_id=contact_id,
        t0_ns=int(event.t0_ns),
        t_end_ns=int(max(event.t0_ns, event.t_end_ns)),
        t0_s=float(event.t0_ns - ep_start_ns) / 1e9,
        duration_s=float(max(0, event.t_end_ns - event.t0_ns)) / 1e9,
        obstacle_kind=event.obstacle_kind,
        obstacle_id=event.obstacle_id,
        target_agent_id=event.track.agent_id if event.track is not None else None,
        target_agent_name=event.track.agent_name if event.track is not None else None,
        v_close_robot_mps=v_close_r,
        v_close_target_mps=v_close_t,
        s_robot=s_robot,
        closer_label=closer_label,
        v_robot_impact_mps=v_robot_impact,
        v_target_impact_mps=v_target_impact,
        relative_impact_speed_mps=rel_impact,
        counterfactual_regime=regime,
        counterfactual_freeze_min_dist_m=min_dist_m,
        counterfactual_freeze_persists=persists,
        is_robot_fault=is_robot_fault,
        is_robot_fault_lower_bound=is_robot_fault_lb,
    )


def _match_track(obstacle_id: str, robot_at_t0: Trajectory2D, tracks: Mapping[int, PedTrack]) -> PedTrack | None:
    """Resolve the PedTrack matching obstacle_id or nearest to the robot at t0_ns."""
    if not tracks:
        return None
    base = entity_name(obstacle_id.strip())
    for track in tracks.values():
        if track.agent_name in (obstacle_id.strip(), base) or str(track.agent_id) == base:
            return track
    digits = re.findall(r"\d+", base)
    if digits and int(digits[-1]) in tracks:
        return tracks[int(digits[-1])]
    best_track: PedTrack | None = None
    best_dist = float("inf")
    rx0, ry0 = float(robot_at_t0.x[0]), float(robot_at_t0.y[0])
    for track in tracks.values():
        if len(track.time_ns) == 0:
            continue
        p_t0 = track.trajectory.resample_onto(robot_at_t0.time_ns)
        d = float(np.hypot(p_t0.x[0] - rx0, p_t0.y[0] - ry0))
        if d < best_dist:
            best_dist = d
            best_track = track
    return best_track


def _group_contact_events(events: Sequence[ContactEvent], merge_gap_ns: int) -> list[ContactEvent]:
    """Merge contiguous ContactEvent rows for the same target within merge_gap_ns."""
    grouped: list[ContactEvent] = []
    for ev in events:
        merged = False
        for idx in range(len(grouped) - 1, -1, -1):
            prev = grouped[idx]
            same_target = (prev.obstacle_kind == ev.obstacle_kind) and ((prev.track is not None and ev.track is not None and prev.track.agent_id == ev.track.agent_id) or (prev.obstacle_id == ev.obstacle_id))
            if same_target and (ev.t0_ns - prev.t_end_ns) <= merge_gap_ns:
                grouped[idx] = ContactEvent(
                    t0_ns=prev.t0_ns,
                    t_end_ns=max(prev.t_end_ns, ev.t_end_ns),
                    obstacle_kind=prev.obstacle_kind,
                    obstacle_id=prev.obstacle_id,
                    track=prev.track,
                    static_point=prev.static_point or ev.static_point,
                )
                merged = True
                break
        if not merged:
            grouped.append(ev)
    return grouped


def _contacts_from_table(
    cc_df: pl.DataFrame,
    robot: Trajectory2D,
    tracks: Mapping[int, PedTrack],
) -> list[ContactEvent]:
    """Extract ContactEvent rows from a flat collision_contacts DataFrame."""
    sorted_cc = cc_df.sort("time_ns")
    cols = set(sorted_cc.columns)
    raw: list[ContactEvent] = []
    for row in sorted_cc.iter_rows(named=True):
        t_ns = int(row["time_ns"])
        kind_raw = str(row["kind"] or "static").lower() if "kind" in cols else "static"
        kind = "pedestrian" if ("ped" in kind_raw or "human" in kind_raw) else kind_raw
        obs_id = str(row["obstacle_id"] or "") if "obstacle_id" in cols else ""
        ox = row.get("obstacle_x") if "obstacle_x" in cols else None
        oy = row.get("obstacle_y") if "obstacle_y" in cols else None
        pt = (float(ox), float(oy)) if ox is not None and oy is not None and (float(ox) != 0.0 or float(oy) != 0.0) else None
        r_t0 = robot.resample_onto(np.asarray([t_ns], dtype=np.int64))
        track = _match_track(obs_id, r_t0, tracks) if kind == "pedestrian" else None
        if kind == "pedestrian" and track is not None and not obs_id:
            obs_id = track.agent_name
        raw.append(ContactEvent(t0_ns=t_ns, t_end_ns=t_ns, obstacle_kind=kind, obstacle_id=obs_id, track=track, static_point=pt))
    return raw


def _contacts_from_clearance(
    robot: Trajectory2D,
    tracks: Mapping[int, PedTrack],
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """Detect pedestrian contacts from surface-to-surface clearance <= 0."""
    events: list[ContactEvent] = []
    for _, track in sorted(tracks.items()):
        if len(track.time_ns) == 0:
            continue
        t_lo = max(int(robot.time_ns[0]), int(track.time_ns[0]))
        t_hi = min(int(robot.time_ns[-1]), int(track.time_ns[-1]))
        if t_hi < t_lo:
            continue
        r_sub = robot.slice_ns(t_lo, t_hi)
        p_sub = track.trajectory.resample_onto(r_sub.time_ns)
        ped_r = track.radius if track.radius is not None else DEFAULT_PED_RADIUS_M
        clearance = np.hypot(p_sub.x - r_sub.x, p_sub.y - r_sub.y) - (robot.radius + ped_r)
        c_indices = np.flatnonzero(clearance <= 0.0)
        if len(c_indices) == 0:
            continue
        c_start = int(c_indices[0])
        c_prev = int(c_indices[0])
        for c_idx_val in c_indices[1:]:
            c_idx = int(c_idx_val)
            if int(r_sub.time_ns[c_idx] - r_sub.time_ns[c_prev]) <= merge_gap_ns:
                c_prev = c_idx
            else:
                events.append(ContactEvent(int(r_sub.time_ns[c_start]), int(r_sub.time_ns[c_prev]), "pedestrian", track.agent_name, track))
                c_start = c_idx
                c_prev = c_idx
        events.append(ContactEvent(int(r_sub.time_ns[c_start]), int(r_sub.time_ns[c_prev]), "pedestrian", track.agent_name, track))
    return events


def _contacts_from_bumper_events(ce_df: pl.DataFrame | None) -> list[ContactEvent]:
    """Extract wall/static ContactEvent rows from collision_events counts."""
    if ce_df is None or len(ce_df) == 0 or "time_ns" not in ce_df.columns:
        return []
    cols = set(ce_df.columns)
    events: list[ContactEvent] = []
    for row in ce_df.sort("time_ns").iter_rows(named=True):
        t_ns = int(row["time_ns"])
        wall_c = int(row.get("wall_collisions") or 0) if "wall_collisions" in cols else 0
        stat_c = int(row.get("static_collisions") or 0) if "static_collisions" in cols else 0
        if wall_c > 0 or stat_c > 0:
            kind = "wall" if wall_c > 0 else "static"
            events.append(ContactEvent(t0_ns=t_ns, t_end_ns=t_ns, obstacle_kind=kind, obstacle_id=kind))
    return events


class ContactAttributionCalculator(BaseMetricCalculator):
    """Per-contact closing speed, closer_label, frozen-robot counterfactual and impact speed calculator."""

    NAME = "contact_attribution"
    CATEGORY = "social"
    SCOPES = {"contact"}
    REQUIRED_TOPICS = ["odom"]

    UNITS = {
        "contact_v_close_robot_mps": "m/s",
        "contact_v_close_target_mps": "m/s",
        "contact_s_robot": "ratio",
        "counterfactual_freeze_min_dist_m": "m",
        "contact_impact_speed_mps": "m/s",
        "contact_count": "count",
        "contact_closer_robot_count": "count",
        "contact_closer_pedestrian_count": "count",
        "contact_closer_mutual_count": "count",
        "contact_closer_static_count": "count",
        "contact_robot_fault_count": "count",
        "contact_passive_ped_impact_count": "count",
    }

    PRIMARY_OUTPUTS = [
        "closer_label",
        "is_robot_fault",
        "contact_impact_speed_mps",
        "contact_count",
    ]

    OUTPUT_DIRECTIONS = {
        "contact_v_close_robot_mps": "lower",
        "contact_s_robot": "lower",
        "counterfactual_freeze_min_dist_m": "higher",
        "is_robot_fault": "lower",
        "is_robot_fault_lower_bound": "lower",
        "contact_impact_speed_mps": "lower",
        "contact_count": "lower",
        "contact_closer_robot_count": "lower",
        "contact_closer_pedestrian_count": "lower",
        "contact_closer_mutual_count": "lower",
        "contact_closer_static_count": "lower",
        "contact_robot_fault_count": "lower",
        "contact_passive_ped_impact_count": "lower",
    }

    ROLLUP = {
        "contact_v_close_robot_mps": "max",
        "contact_v_close_target_mps": "max",
        "contact_s_robot": "max",
        "closer_label": "label_counts",
        "counterfactual_regime": "label_counts",
        "counterfactual_freeze_min_dist_m": "min",
        "contact_impact_speed_mps": "max",
        "contact_count": "sum",
        "contact_closer_robot_count": "sum",
        "contact_closer_pedestrian_count": "sum",
        "contact_closer_mutual_count": "sum",
        "contact_closer_static_count": "sum",
        "contact_robot_fault_count": "sum",
        "contact_passive_ped_impact_count": "sum",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        """Output keys produced by ContactAttributionCalculator."""
        return [
            "contact_v_close_robot_mps",
            "contact_v_close_target_mps",
            "contact_s_robot",
            "closer_label",
            "counterfactual_regime",
            "counterfactual_freeze_min_dist_m",
            "counterfactual_freeze_persists",
            "is_robot_fault",
            "is_robot_fault_lower_bound",
            "contact_impact_speed_mps",
            "contact_count",
            "contact_closer_robot_count",
            "contact_closer_pedestrian_count",
            "contact_closer_mutual_count",
            "contact_closer_static_count",
            "contact_robot_fault_count",
            "contact_passive_ped_impact_count",
            "contact_records",
        ]

    def _configured_contact(self, robot: Trajectory2D, tracks: Mapping[int, PedTrack]) -> list[ContactEvent] | None:
        """Build a single ContactEvent if contact_t0_ns is explicitly set in metrics_config."""
        cfg_t0_ns = self.metrics_config.get("contact_t0_ns")
        if cfg_t0_ns is None:
            return None
        t0 = int(cfg_t0_ns)
        t_end = int(self.metrics_config.get("contact_t_end_ns", t0))
        kind = str(self.metrics_config.get("obstacle_kind", "pedestrian"))
        obs_id = str(self.metrics_config.get("obstacle_id", ""))
        aid = self.metrics_config.get("target_agent_id")
        track = tracks.get(int(aid)) if aid is not None and int(aid) in tracks else None
        if track is None and kind == "pedestrian":
            track = _match_track(obs_id, robot.resample_onto(np.asarray([t0], dtype=np.int64)), tracks)
        static_pt = self.metrics_config.get("static_point")
        pt_tuple = (float(static_pt[0]), float(static_pt[1])) if isinstance(static_pt, (list, tuple)) and len(static_pt) >= 2 else None
        return [ContactEvent(t0, t_end, kind, obs_id or (track.agent_name if track else "static"), track, pt_tuple)]

    def _detect_contacts(
        self,
        episode: AlignedEpisodeBundle,
        robot: Trajectory2D,
        tracks: Mapping[int, PedTrack],
    ) -> list[ContactEvent]:
        """Detect all contact intervals in the episode from collision_contacts or clearance."""
        configured = self._configured_contact(robot, tracks)
        if configured is not None:
            return configured

        merge_gap_ns = int(round(float(self.metrics_config.get("contact_merge_gap_s", CONTACT_MERGE_GAP_S)) * 1e9))
        topics = self.native_topics(episode)
        cc_df = topics.get("collision_contacts")
        if isinstance(cc_df, pl.LazyFrame):
            cc_df = cc_df.collect()
        if cc_df is not None and len(cc_df) > 0 and "time_ns" in cc_df.columns:
            return _group_contact_events(_contacts_from_table(cc_df, robot, tracks), merge_gap_ns)

        events = _contacts_from_clearance(robot, tracks, merge_gap_ns)
        ce_df = topics.get("collision_events")
        if isinstance(ce_df, pl.LazyFrame):
            ce_df = ce_df.collect()
        events.extend(_contacts_from_bumper_events(ce_df))
        events.sort(key=lambda ev: (ev.t0_ns, ev.obstacle_kind, ev.obstacle_id))
        return events

    def attribute_all_contacts(self, episode: AlignedEpisodeBundle) -> list[ContactAttributionRecord]:
        """Detect and attribute every contact in the episode."""
        robot, _ = resolve_robot_trajectory(self, episode)
        if len(robot.time_ns) == 0:
            return []
        tracks = extract_episode_ped_tracks(episode)
        detected = self._detect_contacts(episode, robot, tracks)
        if not detected:
            return []

        context = ContactContext(
            robot=robot,
            crowd_condition=resolve_crowd_condition(episode, self.metrics_config),
            default_ped_radius_m=float(self.metrics_config.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M)),
            pre_window_s=float(self.metrics_config.get("pre_contact_window_s", PRE_CONTACT_WINDOW_S)),
            post_window_s=float(self.metrics_config.get("post_contact_window_s", POST_CONTACT_WINDOW_S)),
        )
        return [attribute_single_contact(idx, ev, context) for idx, ev in enumerate(detected)]

    @staticmethod
    def summarize_records(records: Sequence[ContactAttributionRecord]) -> dict[str, typing.Any]:
        """Build calculator output dictionary from a sequence of ContactAttributionRecord objects."""
        if not records:
            return {
                "contact_v_close_robot_mps": None,
                "contact_v_close_target_mps": None,
                "contact_s_robot": None,
                "closer_label": None,
                "counterfactual_regime": None,
                "counterfactual_freeze_min_dist_m": None,
                "counterfactual_freeze_persists": None,
                "is_robot_fault": None,
                "is_robot_fault_lower_bound": None,
                "contact_impact_speed_mps": None,
                "contact_count": 0,
                "contact_closer_robot_count": 0,
                "contact_closer_pedestrian_count": 0,
                "contact_closer_mutual_count": 0,
                "contact_closer_static_count": 0,
                "contact_robot_fault_count": None,
                "contact_passive_ped_impact_count": None,
                "contact_records": [],
            }

        primary = records[0]
        freeze_dists = [r.counterfactual_freeze_min_dist_m for r in records if r.counterfactual_freeze_min_dist_m is not None]
        blind_faults = [r.is_robot_fault for r in records if r.is_robot_fault is not None]
        reactive_faults = [r.is_robot_fault_lower_bound for r in records if r.is_robot_fault_lower_bound is not None]
        persists_vals = [r.counterfactual_freeze_persists for r in records if r.counterfactual_freeze_persists is not None]

        return {
            "contact_v_close_robot_mps": float(max(r.v_close_robot_mps for r in records)) if len(records) > 1 else primary.v_close_robot_mps,
            "contact_v_close_target_mps": float(max(r.v_close_target_mps for r in records)) if len(records) > 1 else primary.v_close_target_mps,
            "contact_s_robot": float(max(r.s_robot for r in records)) if len(records) > 1 else primary.s_robot,
            "closer_label": primary.closer_label,
            "counterfactual_regime": primary.counterfactual_regime,
            "counterfactual_freeze_min_dist_m": float(min(freeze_dists)) if freeze_dists else None,
            "counterfactual_freeze_persists": primary.counterfactual_freeze_persists if len(records) == 1 else (bool(all(persists_vals)) if persists_vals else None),
            "is_robot_fault": bool(any(blind_faults)) if blind_faults else None,
            "is_robot_fault_lower_bound": bool(any(reactive_faults)) if reactive_faults else None,
            "contact_impact_speed_mps": float(max(r.relative_impact_speed_mps for r in records)),
            "contact_count": len(records),
            "contact_closer_robot_count": sum(1 for r in records if r.closer_label == "robot"),
            "contact_closer_pedestrian_count": sum(1 for r in records if r.closer_label == "pedestrian"),
            "contact_closer_mutual_count": sum(1 for r in records if r.closer_label == "mutual"),
            "contact_closer_static_count": sum(1 for r in records if r.closer_label == "static"),
            "contact_robot_fault_count": sum(1 for f in blind_faults if f) if blind_faults else None,
            "contact_passive_ped_impact_count": sum(1 for f in blind_faults if not f) if blind_faults else None,
            "contact_records": [asdict(r) for r in records],
        }

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Compute contact attribution for a single contact window or all contacts in the episode."""
        del prior_results
        return self.summarize_records(self.attribute_all_contacts(episode))
