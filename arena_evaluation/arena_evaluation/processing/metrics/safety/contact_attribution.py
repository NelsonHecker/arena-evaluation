from __future__ import annotations

import re
import typing
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.ecological.mechanism_common import entity_name
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_PED_RADIUS_M,
    NS_PER_S,
    ROBOT_TRACKS_TOPIC,
    PedTrack,
    Trajectory2D,
    contiguous_spans,
    episode_table,
    extract_episode_ped_tracks,
    merge_intervals_ns,
    resolve_ped_track,
    resolve_robot_trajectory,
)
from arena_evaluation.storage.declarations import CrowdCondition, resolve_crowd_condition

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PRE_CONTACT_WINDOW_S: Final[float] = 0.5
POST_CONTACT_WINDOW_S: Final[float] = 0.5
#: Wall-clock gap, seconds, that still merges two recorded contact samples of the same target into one contact.
CONTACT_MERGE_GAP_S: Final[float] = 0.25
#: Share of the closing speed (dimensionless, 0..1) the robot must contribute to be the sole closer.
DOMINANT_CLOSING_SHARE: Final[float] = 0.70
PEDESTRIAN_KIND: Final[str] = "pedestrian"
ROBOT_KIND: Final[str] = "robot"
#: Collider kinds that move on their own and therefore get a closing-share attribution.
TRACKED_KINDS: Final[frozenset[str]] = frozenset({PEDESTRIAN_KIND, ROBOT_KIND})
PEDESTRIAN_KIND_PATTERN: Final[str] = "ped|human"
COLLISION_CONTACTS_TOPIC: Final[str] = "collision_contacts"
COLLISION_EVENTS_TOPIC: Final[str] = "collision_events"

CloserLabel = typing.Literal["robot", "pedestrian", "other_robot", "mutual", "static"]
FaultAttribution = typing.Literal["robot", "pedestrian", "other_robot", "mutual", "unavoidable"]
CounterfactualRegime = typing.Literal["blind_exact", "reactive_approx", "unavailable"]


@dataclass(frozen=True, slots=True)
class ContactEvent:
    """Detected contact interval before kinematic attribution."""

    t0_ns: int
    t_end_ns: int
    obstacle_kind: str
    obstacle_id: str
    track: PedTrack | None = None
    static_point: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class ContactContext:
    """Episode-level context needed to attribute a ContactEvent."""

    robot: Trajectory2D
    crowd_condition: CrowdCondition | None = None
    default_ped_radius_m: float = DEFAULT_PED_RADIUS_M
    pre_window_s: float = PRE_CONTACT_WINDOW_S
    post_window_s: float = POST_CONTACT_WINDOW_S


@dataclass(frozen=True, slots=True)
class ContactAttributionRecord:
    """Attribution record for a single contact event."""

    contact_id: int
    t0_ns: int
    t_end_ns: int
    t0_s: float
    duration_s: float
    obstacle_kind: str
    obstacle_id: str
    target_ped_id: str | None
    target_ped_name: str | None
    v_close_robot_mps: float
    v_close_target_mps: float
    s_robot: float | None
    closer_label: CloserLabel | None
    fault_attributed_to: FaultAttribution | None
    v_robot_impact_mps: float
    v_target_impact_mps: float
    relative_impact_speed_mps: float
    counterfactual_regime: CounterfactualRegime
    counterfactual_freeze_min_dist_m: float | None
    counterfactual_freeze_persists: bool | None
    is_robot_fault: bool | None
    is_robot_fault_lower_bound: bool | None


@dataclass(frozen=True, slots=True)
class FaultPolicy:
    """The one contact fault policy: who closed in, who is to blame and the robot-fault verdicts."""

    closer_label: CloserLabel | None
    fault_attributed_to: FaultAttribution | None
    robot_closing_share: float | None
    is_robot_fault: bool | None
    is_robot_fault_lower_bound: bool | None


def _closing_share(v_close_robot_mps: float, v_close_target_mps: float) -> float:
    """Robot share of the closing speed, 0.5 when neither body closes (two bodies at rest)."""
    closing_robot = max(0.0, float(v_close_robot_mps))
    closing_target = max(0.0, float(v_close_target_mps))
    total = closing_robot + closing_target
    return float(closing_robot / total) if total > 1e-9 else 0.5


def _policy(
    *,
    closer_label: CloserLabel | None,
    fault_attributed_to: FaultAttribution | None,
    share: float | None,
    verdict: bool | None,
    regime: CounterfactualRegime,
) -> FaultPolicy:
    """Gate one verdict by the counterfactual evidence standard: exact blame only in the blind regime."""
    match regime:
        case "blind_exact":
            return FaultPolicy(closer_label, fault_attributed_to, share, verdict, None)
        case "reactive_approx":
            return FaultPolicy(closer_label, fault_attributed_to, share, None, verdict)
        case _:
            return FaultPolicy(closer_label, fault_attributed_to, share, None, None)


def classify_contact_fault(
    *,
    collider_kind: str,
    v_close_robot_mps: float,
    v_close_target_mps: float,
    counterfactual_freeze_persists: bool | None,
    counterfactual_regime: CounterfactualRegime,
    target_speed_known: bool = True,
) -> FaultPolicy:
    """The single fault policy of a contact: closing shares and the freeze test into label, blame and verdicts.

    A wall or static collider is always the robot's fault. A moving collider is attributed by the
    robot's share of the closing speed: at or above :data:`DOMINANT_CLOSING_SHARE` the robot closed
    in, at or below its complement the other body did, otherwise the contact is mutual. A mutual
    contact is ``unavoidable`` when the frozen-robot counterfactual still collides, and only counts
    as the robot's fault when freezing the robot provably avoids it; two bodies at rest therefore
    stay unknown instead of silently becoming robot faults. Without a measured target speed the
    share is not computable and every verdict is NULL.
    """
    share = _closing_share(v_close_robot_mps, v_close_target_mps)
    if collider_kind not in TRACKED_KINDS:
        # A static-world impact needs no crowd reaction: the robot hit it in both evidence standards.
        return FaultPolicy(closer_label="static", fault_attributed_to="robot", robot_closing_share=share, is_robot_fault=True, is_robot_fault_lower_bound=True)
    if not target_speed_known:
        return _policy(closer_label=None, fault_attributed_to=None, share=None, verdict=None, regime=counterfactual_regime)

    target_label: CloserLabel = "pedestrian" if collider_kind == PEDESTRIAN_KIND else "other_robot"
    if share >= DOMINANT_CLOSING_SHARE:
        return _policy(closer_label="robot", fault_attributed_to="robot", share=share, verdict=True, regime=counterfactual_regime)
    if share <= 1.0 - DOMINANT_CLOSING_SHARE:
        return _policy(closer_label=target_label, fault_attributed_to=typing.cast(FaultAttribution, target_label), share=share, verdict=False, regime=counterfactual_regime)

    persists = counterfactual_freeze_persists
    unavoidable = persists is not False
    return _policy(
        closer_label="mutual",
        fault_attributed_to="unavoidable" if unavoidable else "mutual",
        share=share,
        verdict=None if unavoidable else True,
        regime=counterfactual_regime,
    )


def compute_closing_speeds(event: ContactEvent, context: ContactContext) -> tuple[float, float]:
    """Mean robot and target closing speeds (m/s) over [t0 - pre_window_s, t0]."""
    win_start_ns = int(event.t0_ns - round(context.pre_window_s * NS_PER_S))
    r_win = context.robot.slice_ns(win_start_ns, event.t0_ns)

    if event.track is not None and len(event.track.t_ns) > 0:
        p_win = event.track.trajectory.resample_onto(r_win.t_ns)
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
    """Frozen-robot counterfactual test over [t0 - pre_window_s, t0 + post_window_s].

    Only a pedestrian is not expected to react to the frozen robot, so only that pair yields an
    exact or approximated freeze verdict; another robot sees the stop and reacts, so its contact
    keeps the closing-share attribution alone. An unrecorded crowd condition is not blind: the
    test stays an approximation rather than claiming an exact counterfactual.
    """
    track = event.track
    if event.obstacle_kind != PEDESTRIAN_KIND or track is None or len(track.t_ns) == 0 or len(context.robot.t_ns) == 0:
        return "unavailable", None, None

    win_start_ns = int(event.t0_ns - round(context.pre_window_s * NS_PER_S))
    win_end_ns = int(event.t0_ns + round(context.post_window_s * NS_PER_S))
    ped_in_win = (track.t_ns >= win_start_ns) & (track.t_ns <= win_end_ns)
    if not np.any(ped_in_win):
        return "unavailable", None, None

    t_freeze_ns = max(int(context.robot.t_ns[0]), win_start_ns)
    frozen_pose = context.robot.resample_onto(np.asarray([t_freeze_ns], dtype=np.int64))
    freeze_x, freeze_y = float(frozen_pose.x[0]), float(frozen_pose.y[0])

    dists = np.hypot(track.x[ped_in_win] - freeze_x, track.y[ped_in_win] - freeze_y)
    min_dist_m = float(np.min(dists))
    ped_r = track.radius_m if track.radius_m is not None else context.default_ped_radius_m
    persists = bool(min_dist_m <= float(context.robot.radius_m + ped_r))
    regime: CounterfactualRegime = "blind_exact" if context.crowd_condition == "blind" else "reactive_approx"
    return regime, min_dist_m, persists


def attribute_single_contact(
    contact_id: int,
    event: ContactEvent,
    context: ContactContext,
) -> ContactAttributionRecord:
    """Compute full attribution record for a single ContactEvent, through the one fault policy."""
    v_close_r, v_close_t = compute_closing_speeds(event, context)
    target_known = event.obstacle_kind not in TRACKED_KINDS or (event.track is not None and len(event.track.t_ns) > 0)

    t0_arr = np.asarray([event.t0_ns], dtype=np.int64)
    r_at_t0 = context.robot.resample_onto(t0_arr)
    rvx_0, rvy_0 = float(r_at_t0.vx[0]), float(r_at_t0.vy[0])
    v_robot_impact = float(np.hypot(rvx_0, rvy_0))

    if target_known and event.obstacle_kind in TRACKED_KINDS and event.track is not None and len(event.track.t_ns) > 0:
        p_at_t0 = event.track.trajectory.resample_onto(t0_arr)
        pvx_0, pvy_0 = float(p_at_t0.vx[0]), float(p_at_t0.vy[0])
        v_target_impact = float(np.hypot(pvx_0, pvy_0))
        rel_impact = float(np.hypot(rvx_0 - pvx_0, rvy_0 - pvy_0))
    else:
        v_target_impact = 0.0
        rel_impact = v_robot_impact

    regime, min_dist_m, persists = evaluate_frozen_robot_counterfactual(event, context)
    policy = classify_contact_fault(
        collider_kind=event.obstacle_kind,
        v_close_robot_mps=v_close_r,
        v_close_target_mps=v_close_t,
        counterfactual_freeze_persists=persists,
        counterfactual_regime=regime,
        target_speed_known=target_known,
    )

    ep_start_ns = int(context.robot.t_ns[0]) if len(context.robot.t_ns) > 0 else event.t0_ns
    return ContactAttributionRecord(
        contact_id=contact_id,
        t0_ns=int(event.t0_ns),
        t_end_ns=int(max(event.t0_ns, event.t_end_ns)),
        t0_s=float(event.t0_ns - ep_start_ns) / NS_PER_S,
        duration_s=float(max(0, event.t_end_ns - event.t0_ns)) / NS_PER_S,
        obstacle_kind=event.obstacle_kind,
        obstacle_id=event.obstacle_id,
        target_ped_id=event.track.ped_id if event.track is not None else None,
        target_ped_name=event.track.ped_name if event.track is not None else None,
        v_close_robot_mps=v_close_r,
        v_close_target_mps=v_close_t,
        s_robot=policy.robot_closing_share,
        closer_label=policy.closer_label,
        fault_attributed_to=policy.fault_attributed_to,
        v_robot_impact_mps=v_robot_impact,
        v_target_impact_mps=v_target_impact,
        relative_impact_speed_mps=rel_impact,
        counterfactual_regime=regime,
        counterfactual_freeze_min_dist_m=min_dist_m,
        counterfactual_freeze_persists=persists,
        is_robot_fault=policy.is_robot_fault,
        is_robot_fault_lower_bound=policy.is_robot_fault_lower_bound,
    )


def _static_point(obstacle_x: float | None, obstacle_y: float | None) -> tuple[float, float] | None:
    """Static obstacle position of a collision row, None when the recording carries none or only the (0, 0) sentinel."""
    if obstacle_x is None or obstacle_y is None:
        return None
    point = (float(obstacle_x), float(obstacle_y))
    return None if point == (0.0, 0.0) else point


def _numeric_suffix(name: str) -> str | None:
    """Trailing number of an entity name (``Pedestrian_7`` -> ``7``), None when it has none."""
    digits = re.findall(r"\d+", name)
    return digits[-1] if digits else None


def _track_by_name_or_id(obstacle_id: str, tracks: Mapping[str, PedTrack]) -> PedTrack | None:
    """Track whose ped_id or ped_name equals the obstacle id or its env-stripped entity name."""
    normalized = obstacle_id.strip()
    if not normalized:
        return None
    base = entity_name(normalized)
    for track in tracks.values():
        if track.ped_id in (normalized, base) or track.ped_name in (normalized, base):
            return track
    return None


def _track_by_numeric_suffix(base: str, tracks: Mapping[str, PedTrack]) -> PedTrack | None:
    """Track whose ped_id or ped_name ends in the same number as the obstacle id."""
    suffix = _numeric_suffix(base)
    if suffix is None:
        return None
    for key, track in tracks.items():
        if _numeric_suffix(key) == suffix or _numeric_suffix(track.ped_name) == suffix:
            return track
    return None


def _nearest_track(tracks: Mapping[str, PedTrack], robot_at_t0: Trajectory2D) -> PedTrack | None:
    """Pedestrian whose resampled position is nearest to the robot pose at t0."""
    if len(robot_at_t0.t_ns) == 0:
        return None
    robot_x, robot_y = float(robot_at_t0.x[0]), float(robot_at_t0.y[0])
    nearest: PedTrack | None = None
    nearest_distance_m = float("inf")
    for track in tracks.values():
        if len(track.t_ns) == 0:
            continue
        pedestrian_at_t0 = track.trajectory.resample_onto(robot_at_t0.t_ns)
        distance_m = float(np.hypot(pedestrian_at_t0.x[0] - robot_x, pedestrian_at_t0.y[0] - robot_y))
        if distance_m < nearest_distance_m:
            nearest_distance_m = distance_m
            nearest = track
    return nearest


def _match_track(obstacle_id: str, tracks: Mapping[str, PedTrack], robot_at_t0: Trajectory2D) -> PedTrack | None:
    """Resolve the PedTrack of a contact: exact ped id/name, then numeric suffix, then nearest pose."""
    if not tracks:
        return None
    track = _track_by_name_or_id(obstacle_id, tracks)
    if track is not None:
        return track
    track = _track_by_numeric_suffix(entity_name(obstacle_id.strip()), tracks)
    if track is not None:
        return track
    return _nearest_track(tracks, robot_at_t0)


def _group_contact_rows(cc_df: pl.DataFrame, merge_gap_ns: int) -> pl.DataFrame:
    """One row per contiguous contact run of the same resolved target, within the merge gap."""
    contacts = cc_df.select(pl.col("t_ns").cast(pl.Int64), pl.col("obstacle_id"), pl.col("obstacle_kind"), pl.col("obstacle_x"), pl.col("obstacle_y")).with_columns((pl.col("obstacle_kind") + pl.lit("|") + pl.col("obstacle_id")).alias("target_key")).sort("target_key", "t_ns")

    same_target = pl.col("target_key").shift(1) == pl.col("target_key")
    beyond_gap = pl.col("t_ns").diff() > merge_gap_ns
    starts_group = ~same_target.fill_null(False) | beyond_gap.fill_null(True)

    return (
        contacts.with_columns(starts_group.cum_sum().alias("group_id"))
        .group_by("group_id")
        .agg(
            t0_ns=pl.col("t_ns").min(),
            t_end_ns=pl.col("t_ns").max(),
            obstacle_kind=pl.col("obstacle_kind").first(),
            obstacle_id=pl.col("obstacle_id").first(),
            obstacle_x=pl.col("obstacle_x").first(),
            obstacle_y=pl.col("obstacle_y").first(),
        )
        .sort("t0_ns", "obstacle_kind", "obstacle_id")
    )


def _resolve_contact_targets(cc_df: pl.DataFrame, robot: Trajectory2D, tracks: Mapping[str, PedTrack]) -> pl.DataFrame:
    """Raw contact rows whose pedestrian obstacle ids are canonical ``ped_id`` values, so later grouping never splits one track."""
    frame = _groupable_contacts_frame(cc_df)
    if frame.is_empty() or not tracks:
        return frame
    times_ns = frame.get_column("t_ns").to_numpy().astype(np.int64)
    kinds = frame.get_column("obstacle_kind").to_numpy()
    raw_ids = frame.get_column("obstacle_id").to_numpy()
    pose_by_time = {int(stamp): robot.resample_onto(np.asarray([stamp], dtype=np.int64)) for stamp in np.unique(times_ns)}
    resolved: list[str] = []
    for stamp, kind, raw_id in zip(times_ns, kinds, raw_ids, strict=True):
        if kind != PEDESTRIAN_KIND:
            resolved.append(str(raw_id))
            continue
        track = _track_by_name_or_id(str(raw_id), tracks) or _track_by_numeric_suffix(entity_name(str(raw_id).strip()), tracks)
        if track is None:
            track = _nearest_track(tracks, pose_by_time[int(stamp)])
        resolved.append(track.ped_id if track is not None else str(raw_id))
    return frame.with_columns(pl.Series("obstacle_id", resolved, dtype=pl.String)).select("t_ns", "obstacle_id", "obstacle_kind", "obstacle_x", "obstacle_y")


def _groupable_contacts_frame(cc_df: pl.DataFrame) -> pl.DataFrame:
    """Contact rows with the normalized obstacle kind and their grouping columns."""
    raw_kind = pl.col("kind").fill_null("").str.to_lowercase()
    obstacle_kind = pl.when(raw_kind == "").then(pl.lit("static")).when(raw_kind.str.contains(PEDESTRIAN_KIND_PATTERN)).then(pl.lit(PEDESTRIAN_KIND)).otherwise(raw_kind).alias("obstacle_kind")
    return cc_df.with_columns(obstacle_kind).select(
        pl.col("t_ns").cast(pl.Int64),
        pl.col("obstacle_id").fill_null("").alias("obstacle_id"),
        pl.col("obstacle_kind"),
        pl.col("obstacle_x").cast(pl.Float64),
        pl.col("obstacle_y").cast(pl.Float64),
    )


def _contacts_from_table(
    *,
    cc_df: pl.DataFrame,
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """ContactEvent rows grouped per resolved target from a flat collision_contacts table."""
    grouped = _group_contact_rows(_resolve_contact_targets(cc_df, robot, tracks), merge_gap_ns)
    t0_values = grouped.get_column("t0_ns").to_numpy()
    end_values = grouped.get_column("t_end_ns").to_numpy()
    kind_values = grouped.get_column("obstacle_kind").to_numpy()
    id_values = grouped.get_column("obstacle_id").to_numpy()
    x_values = grouped.get_column("obstacle_x").to_numpy()
    y_values = grouped.get_column("obstacle_y").to_numpy()
    events: list[ContactEvent] = []
    for t0_ns, t_end_ns, obstacle_kind, obstacle_id, obstacle_x, obstacle_y in zip(t0_values, end_values, kind_values, id_values, x_values, y_values, strict=True):
        robot_at_t0 = robot.resample_onto(np.asarray([int(t0_ns)], dtype=np.int64))
        track = _match_track(str(obstacle_id), tracks, robot_at_t0) if obstacle_kind in TRACKED_KINDS else None
        resolved_id = str(obstacle_id) or (track.ped_id if track is not None else PEDESTRIAN_KIND)
        events.append(
            ContactEvent(
                t0_ns=int(t0_ns),
                t_end_ns=int(t_end_ns),
                obstacle_kind=str(obstacle_kind),
                obstacle_id=resolved_id,
                track=track,
                static_point=_static_point(obstacle_x, obstacle_y),
            )
        )
    return events


def _contacts_from_clearance(
    *,
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """Detect pedestrian contacts from surface-to-surface clearance <= 0."""
    events: list[ContactEvent] = []
    for track in tracks.values():
        if len(track.t_ns) == 0 or len(robot.t_ns) == 0:
            continue
        t_lo = max(int(robot.t_ns[0]), int(track.t_ns[0]))
        t_hi = min(int(robot.t_ns[-1]), int(track.t_ns[-1]))
        if t_hi < t_lo:
            continue

        robot_window = robot.slice_ns(t_lo, t_hi)
        pedestrian_window = track.trajectory.resample_onto(robot_window.t_ns)
        ped_radius_m = track.radius_m if track.radius_m is not None else DEFAULT_PED_RADIUS_M
        # Surface-to-surface (edge) clearance in metres: the centre distance minus both body radii.
        edge_clearance_m = np.hypot(pedestrian_window.x - robot_window.x, pedestrian_window.y - robot_window.y) - (robot.radius_m + ped_radius_m)
        spans = contiguous_spans(edge_clearance_m <= 0.0, robot_window.t_ns, merge_gap_ns)
        events.extend(
            ContactEvent(
                t0_ns=int(robot_window.t_ns[start]),
                t_end_ns=int(robot_window.t_ns[end]),
                obstacle_kind=PEDESTRIAN_KIND,
                obstacle_id=track.ped_id,
                track=track,
            )
            for start, end in spans
        )
    return events


def _contacts_from_bumper_events(ce_df: pl.DataFrame | None) -> list[ContactEvent]:
    """ContactEvent rows for wall and static counts in the flat collision_events table."""
    if ce_df is None or len(ce_df) == 0:
        return []
    bumper_rows = ce_df.select("t_ns", "wall_count", "static_count").filter((pl.col("wall_count") > 0) | (pl.col("static_count") > 0)).sort("t_ns")
    events: list[ContactEvent] = []
    for t_ns, wall_count in zip(
        bumper_rows["t_ns"].to_numpy(),
        bumper_rows["wall_count"].to_numpy(),
        strict=True,
    ):
        kind = "wall" if int(wall_count) > 0 else "static"
        events.append(ContactEvent(t0_ns=int(t_ns), t_end_ns=int(t_ns), obstacle_kind=kind, obstacle_id=kind))
    return events


def _merge_contact_events(events: Sequence[ContactEvent]) -> list[ContactEvent]:
    """Union overlapping intervals of the same target across the clearance and bumper paths."""
    by_target: dict[tuple[str, str], list[ContactEvent]] = {}
    for event in events:
        by_target.setdefault((event.obstacle_kind, event.obstacle_id), []).append(event)

    merged: list[ContactEvent] = []
    for target_events in by_target.values():
        first = target_events[0]
        for start_ns, end_ns in merge_intervals_ns([(event.t0_ns, event.t_end_ns) for event in target_events]):
            merged.append(
                ContactEvent(
                    t0_ns=start_ns,
                    t_end_ns=end_ns,
                    obstacle_kind=first.obstacle_kind,
                    obstacle_id=first.obstacle_id,
                    track=first.track,
                    static_point=first.static_point,
                )
            )
    merged.sort(key=lambda event: (event.t0_ns, event.obstacle_kind, event.obstacle_id))
    return merged


def _configured_static_point(static_point: Sequence[float] | None, obstacle_kind: str) -> tuple[float, float] | None:
    """Static obstacle position of a configured non-pedestrian contact."""
    if obstacle_kind == PEDESTRIAN_KIND or static_point is None or len(static_point) < 2:
        return None
    return (float(static_point[0]), float(static_point[1]))


class ContactAttributionCalculator(BaseMetricCalculator):
    """Per-contact closing speed, closer_label, frozen-robot counterfactual and impact speed calculator."""

    NAME = "contact_attribution"
    CATEGORY = "safety"
    SCOPE = MetricScope.CONTACT
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.CONTACT, MetricScope.EPISODE})
    REQUIRED_TOPICS = [ROBOT_TRACKS_TOPIC]

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
            "impact_speed_mps",
            "robot_speed_at_impact_mps",
            "obstacle_speed_at_impact_mps",
            "closer_label",
            "robot_closing_share",
            "counterfactual_freeze_persists",
            "fault_attributed_to",
            "contact_v_close_robot_mps",
            "contact_v_close_target_mps",
            "contact_s_robot",
            "counterfactual_regime",
            "counterfactual_freeze_min_dist_m",
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

    def _configured_contact(self, robot: Trajectory2D, tracks: Mapping[str, PedTrack]) -> list[ContactEvent] | None:
        """Build a single ContactEvent if contact_t0_ns is explicitly set in metrics_config."""
        cfg_t0_ns = self.metrics_config.get("contact_t0_ns")
        if cfg_t0_ns is None:
            return None
        t0_ns = int(cfg_t0_ns)
        t_end_ns = int(self.metrics_config.get("contact_t_end_ns", t0_ns))
        obstacle_kind = str(self.metrics_config.get("obstacle_kind", PEDESTRIAN_KIND)).lower()
        obstacle_id = str(self.metrics_config.get("obstacle_id", ""))

        track = resolve_ped_track(tracks, self.metrics_config.get("target_agent_id")) if obstacle_kind in TRACKED_KINDS else None
        if track is None and obstacle_kind in TRACKED_KINDS:
            robot_at_t0 = robot.resample_onto(np.asarray([t0_ns], dtype=np.int64))
            track = _match_track(obstacle_id, tracks, robot_at_t0)
        resolved_id = obstacle_id or (track.ped_name if track is not None else obstacle_kind)
        return [
            ContactEvent(
                t0_ns=t0_ns,
                t_end_ns=t_end_ns,
                obstacle_kind=obstacle_kind,
                obstacle_id=resolved_id,
                track=track,
                static_point=_configured_static_point(self.metrics_config.get("static_point"), obstacle_kind),
            )
        ]

    def _detect_contacts(
        self,
        episode: AlignedEpisodeBundle,
        robot: Trajectory2D,
        tracks: Mapping[str, PedTrack],
    ) -> list[ContactEvent]:
        """Detect all contact intervals in the episode from collision_contacts or clearance."""
        configured = self._configured_contact(robot, tracks)
        if configured is not None:
            return configured

        merge_gap_ns = int(round(float(self.metrics_config.get("contact_merge_gap_s", CONTACT_MERGE_GAP_S)) * NS_PER_S))
        contact_rows = episode_table(episode, COLLISION_CONTACTS_TOPIC)
        if contact_rows is not None and len(contact_rows) > 0:
            return _contacts_from_table(cc_df=contact_rows, robot=robot, tracks=tracks, merge_gap_ns=merge_gap_ns)

        events = _contacts_from_clearance(robot=robot, tracks=tracks, merge_gap_ns=merge_gap_ns)
        events.extend(_contacts_from_bumper_events(episode_table(episode, COLLISION_EVENTS_TOPIC)))
        events.sort(key=lambda event: (event.t0_ns, event.obstacle_kind, event.obstacle_id))
        return _merge_contact_events(events)

    def attribute_all_contacts(self, episode: AlignedEpisodeBundle) -> list[ContactAttributionRecord]:
        """Detect and attribute every contact in the episode."""
        robot, _yaw = resolve_robot_trajectory(self, episode)
        if len(robot.t_ns) == 0:
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
        return [attribute_single_contact(contact_id, event, context) for contact_id, event in enumerate(detected)]

    @staticmethod
    def summarize_records(records: Sequence[ContactAttributionRecord]) -> dict[str, typing.Any]:
        """Build calculator output dictionary from a sequence of ContactAttributionRecord objects."""
        if not records:
            return {
                "contact_v_close_robot_mps": None,
                "contact_v_close_target_mps": None,
                "contact_s_robot": None,
                "closer_label": None,
                "fault_attributed_to": None,
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
        shares = [r.s_robot for r in records if r.s_robot is not None]

        return {
            "contact_v_close_robot_mps": float(max(r.v_close_robot_mps for r in records)) if len(records) > 1 else primary.v_close_robot_mps,
            "contact_v_close_target_mps": float(max(r.v_close_target_mps for r in records)) if len(records) > 1 else primary.v_close_target_mps,
            "contact_s_robot": float(max(shares)) if shares else None,
            "closer_label": primary.closer_label,
            "fault_attributed_to": primary.fault_attributed_to,
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
        res = self.summarize_records(self.attribute_all_contacts(episode))
        records = res.get("contact_records", [])
        rec = records[0] if records else {}
        return self._with_contact_fact_values(res, rec)

    @staticmethod
    def _with_contact_fact_values(res: dict[str, typing.Any], record: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
        """Add the fact-table aliases of one attribution record; the fault label is the policy's verdict, never re-derived."""
        res.update(
            {
                "impact_speed_mps": record.get("relative_impact_speed_mps"),
                "robot_speed_at_impact_mps": record.get("v_robot_impact_mps"),
                "obstacle_speed_at_impact_mps": record.get("v_target_impact_mps"),
                "robot_closing_share": record.get("s_robot"),
                "fault_attributed_to": record.get("fault_attributed_to"),
            }
        )
        return res

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """Compute contact attribution for a scope slice without mutating calculator state."""
        if scope_slice is None or scope_slice.scope != MetricScope.CONTACT:
            return self.calculate(episode, prior_results)

        t0_ns = int(scope_slice.t_start_ns)
        t_end_ns = int(scope_slice.t_end_ns)
        raw_collider_kind = str(scope_slice.identity.get("collider_kind", PEDESTRIAN_KIND)).lower()
        # Another robot stays a robot: routing it through the pedestrian path fabricated pedestrian
        # aliases and blamed the other robot as a pedestrian.
        obstacle_kind = raw_collider_kind
        obstacle_id = str(scope_slice.identity.get("obstacle_id", ""))
        static_point = _configured_static_point(scope_slice.identity.get("static_point"), obstacle_kind)
        configured_crowd = scope_slice.identity.get("crowd_condition")
        recorded_crowd = resolve_crowd_condition(episode, self.metrics_config)
        crowd_condition: CrowdCondition | None = configured_crowd if configured_crowd in ("blind", "visible") else recorded_crowd

        robot, _yaw = resolve_robot_trajectory(self, episode)
        if len(robot.t_ns) == 0:
            return self._with_contact_fact_values(self.summarize_records([]), {})

        tracks = extract_episode_ped_tracks(episode)
        tracked = obstacle_kind in TRACKED_KINDS
        track = resolve_ped_track(tracks, obstacle_id) if tracked else None
        if track is None and tracked:
            robot_at_t0 = robot.resample_onto(np.asarray([t0_ns], dtype=np.int64))
            track = _match_track(obstacle_id, tracks, robot_at_t0)

        resolved_id = obstacle_id or (track.ped_id if track is not None else obstacle_kind)
        event = ContactEvent(
            t0_ns=t0_ns,
            t_end_ns=t_end_ns,
            obstacle_kind=obstacle_kind,
            obstacle_id=resolved_id,
            track=track,
            static_point=static_point,
        )
        context = ContactContext(
            robot=robot,
            crowd_condition=crowd_condition,
            default_ped_radius_m=float(self.metrics_config.get("default_ped_radius_m", DEFAULT_PED_RADIUS_M)),
            pre_window_s=float(scope_slice.identity.get("pre_contact_window_s", self.metrics_config.get("pre_contact_window_s", PRE_CONTACT_WINDOW_S))),
            post_window_s=float(scope_slice.identity.get("post_contact_window_s", self.metrics_config.get("post_contact_window_s", POST_CONTACT_WINDOW_S))),
        )
        record = attribute_single_contact(0, event, context)
        return self._with_contact_fact_values(self.summarize_records([record]), asdict(record))
