from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

DEFAULT_PED_RADIUS_M: Final[float] = 0.3
DEFAULT_ENCOUNTER_DIST_M: Final[float] = 4.0
DEFAULT_MERGE_GAP_S: Final[float] = 1.0
NS_PER_S: Final[float] = 1e9
ROBOT_TRACKS_TOPIC: Final[str] = "robot_tracks"
PEDESTRIAN_TRACKS_TOPIC: Final[str] = "pedestrian_tracks"
ROBOT_VISIBILITY_PARAM: Final[str] = "humansim.robots_visible"

CrowdCondition = typing.Literal["blind", "visible"]


@dataclass(frozen=True, slots=True)
class Trajectory2D:
    """Planar trajectory (x, y, vx, vy) on a sorted nanosecond time axis."""

    time_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    radius_m: float = DEFAULT_PED_RADIUS_M

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
        return self.time_ns.astype(np.float64) / NS_PER_S

    def slice_ns(self, t_start_ns: int, t_end_ns: int) -> Trajectory2D:
        """Sub-trajectory within [t_start_ns, t_end_ns]; the nearest sample when the window holds none."""
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
            radius_m=self.radius_m,
        )

    def resample_onto(self, query_time_ns: np.ndarray) -> Trajectory2D:
        """Linearly interpolate trajectory onto query_time_ns."""
        if len(self.time_ns) == 0 or len(query_time_ns) == 0:
            zeros = np.zeros(len(query_time_ns), dtype=np.float64)
            return Trajectory2D(query_time_ns, zeros, zeros.copy(), zeros.copy(), zeros.copy(), self.radius_m)
        t_src = self.time_ns.astype(np.float64)
        t_dst = query_time_ns.astype(np.float64)
        return Trajectory2D(
            time_ns=query_time_ns,
            x=np.interp(t_dst, t_src, self.x),
            y=np.interp(t_dst, t_src, self.y),
            vx=np.interp(t_dst, t_src, self.vx),
            vy=np.interp(t_dst, t_src, self.vy),
            radius_m=self.radius_m,
        )


@dataclass(frozen=True, slots=True)
class PedTrack:
    """Identified pedestrian with its 2D trajectory."""

    ped_id: str
    ped_name: str
    trajectory: Trajectory2D
    radius_m: float | None = None
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


@dataclass(frozen=True, slots=True)
class EncounterWindow:
    """Contiguous robot-pedestrian encounter interval."""

    ped_id: str
    ped_name: str
    t_start_ns: int
    t_end_ns: int
    t_min_ns: int
    min_dist_m: float
    min_clearance_m: float


@dataclass(frozen=True, slots=True)
class EncounterSpec:
    """Parameters governing encounter segmentation."""

    distance_threshold_m: float = DEFAULT_ENCOUNTER_DIST_M
    merge_gap_s: float = DEFAULT_MERGE_GAP_S
    default_ped_radius_m: float = DEFAULT_PED_RADIUS_M


DEFAULT_ENCOUNTER_SPEC = EncounterSpec()


def episode_table(episode: AlignedEpisodeBundle, topic: str) -> pl.DataFrame | None:
    """Collected flat table of an episode topic, None when the recording carries none."""
    topics = episode.topics
    if topics is None:
        return None
    return topics.get(topic)


def resolve_robot_trajectory(calc: BaseMetricCalculator, episode: AlignedEpisodeBundle) -> tuple[Trajectory2D, np.ndarray]:
    """Robot ground-truth Trajectory2D and yaw array from the flat robot_tracks table."""
    radius_m = float(calc.robot_params.robot_radius)
    empty = np.array([], dtype=np.float64)
    empty_trajectory = Trajectory2D(time_ns=np.array([], dtype=np.int64), x=empty, y=empty.copy(), vx=empty.copy(), vy=empty.copy(), radius_m=radius_m)
    tracks = episode_table(episode, ROBOT_TRACKS_TOPIC)
    if tracks is None:
        return empty_trajectory, empty

    rows = tracks.select("t_ns", "x", "y", "yaw", "vx", "vy").drop_nulls(["t_ns", "x", "y"]).with_columns(pl.col("vx").fill_null(0.0), pl.col("vy").fill_null(0.0)).sort("t_ns")
    if len(rows) == 0:
        return empty_trajectory, empty

    return (
        Trajectory2D(
            time_ns=rows["t_ns"].to_numpy().astype(np.int64),
            x=rows["x"].to_numpy().astype(np.float64),
            y=rows["y"].to_numpy().astype(np.float64),
            vx=rows["vx"].to_numpy().astype(np.float64),
            vy=rows["vy"].to_numpy().astype(np.float64),
            radius_m=radius_m,
        ),
        rows["yaw"].to_numpy().astype(np.float64),
    )


def extract_episode_ped_tracks(episode: AlignedEpisodeBundle) -> dict[str, PedTrack]:
    """Pedestrian tracks of the episode keyed by canonical ped_id, read from the flat pedestrian_tracks table."""
    return extract_ped_tracks(episode_table(episode, PEDESTRIAN_TRACKS_TOPIC))


def extract_ped_tracks(peds: pl.DataFrame | None) -> dict[str, PedTrack]:
    """Pedestrian tracks keyed by canonical ped_id from a flat pedestrian_tracks frame."""
    if peds is None or len(peds) == 0:
        return {}

    rows = peds.select("t_ns", "ped_id", "ped_name", "x", "y", "vx", "vy", "radius_m", "desired_speed").filter(pl.col("x").is_finite() & pl.col("y").is_finite()).sort("t_ns")
    if len(rows) == 0:
        return {}

    return {str(frame["ped_id"][0]): _track_from_frame(frame) for frame in rows.partition_by("ped_id", maintain_order=True)}


def _track_from_frame(frame: pl.DataFrame) -> PedTrack:
    """PedTrack of one pedestrian's frames, in time order."""
    ped_id = str(frame["ped_id"][0])
    time_ns = frame["t_ns"].to_numpy().astype(np.int64)
    x = frame["x"].to_numpy().astype(np.float64)
    y = frame["y"].to_numpy().astype(np.float64)
    vx = frame["vx"].fill_null(0.0).to_numpy().astype(np.float64)
    vy = frame["vy"].fill_null(0.0).to_numpy().astype(np.float64)

    twist_is_zero = not bool(np.any(np.hypot(vx, vy) > 1e-9))
    position_moved = len(x) >= 2 and bool(np.any(np.hypot(np.diff(x), np.diff(y)) > 1e-6))
    if twist_is_zero and position_moved:
        vx, vy = BaseMetricCalculator.velocity_from_pose(x, y, time_ns)
        vx[0], vy[0] = vx[1], vy[1]

    radius_m = _first_float(frame["radius_m"])
    effective_radius_m = radius_m if radius_m is not None else DEFAULT_PED_RADIUS_M
    trajectory = Trajectory2D(time_ns=time_ns, x=x, y=y, vx=vx, vy=vy, radius_m=effective_radius_m)
    return PedTrack(
        ped_id=ped_id,
        ped_name=_first_text(frame["ped_name"], ped_id),
        trajectory=trajectory,
        radius_m=radius_m,
        desired_speed=_first_float(frame["desired_speed"]),
    )


def _first_text(series: pl.Series, default: str) -> str:
    """First non-null value of a text series, ``default`` when every value is null."""
    present = series.drop_nulls()
    return str(present[0]) if len(present) > 0 else default


def _first_float(series: pl.Series) -> float | None:
    """First non-null value of a float series, None when every value is null."""
    present = series.drop_nulls()
    return float(present[0]) if len(present) > 0 else None


def resolve_crowd_condition(
    episode: AlignedEpisodeBundle,
    metrics_config: Mapping[str, typing.Any] | None = None,
) -> CrowdCondition:
    """Crowd visibility of the episode: explicit metrics config first, else the recorded humansim parameter, else visible."""
    config = metrics_config or {}
    configured = config.get("crowd_condition")
    if configured in ("blind", "visible"):
        return "blind" if configured == "blind" else "visible"
    for key in ("crowd_visible", "robots_visible"):
        visible = config.get(key)
        if visible is not None:
            return "visible" if bool(visible) else "blind"

    recorded = _recorded_robot_visibility(episode)
    if recorded is not None:
        return "visible" if recorded else "blind"
    return "visible"


def _recorded_robot_visibility(episode: AlignedEpisodeBundle) -> bool | None:
    """Recorded ``humansim.robots_visible`` parameter, None when the recording carries none."""
    params = episode.human_params
    if params is None:
        return None
    value = params.get(ROBOT_VISIBILITY_PARAM)
    if value is not None:
        return bool(value)
    visibility_leaf = ROBOT_VISIBILITY_PARAM.rsplit(".", maxsplit=1)[-1]
    for name, parameter in params.items():
        if name.rsplit(".", maxsplit=1)[-1] == visibility_leaf:
            return bool(parameter)
    return None


def contiguous_spans(mask: np.ndarray, time_ns: np.ndarray, merge_gap_ns: int) -> list[tuple[int, int]]:
    """Inclusive index spans of the True runs of a mask, bridging gaps of at most merge_gap_ns."""
    hits = np.flatnonzero(mask)
    if len(hits) == 0:
        return []
    breaks = np.flatnonzero(np.diff(time_ns[hits]) > merge_gap_ns) + 1
    edges = np.concatenate(([0], breaks, [len(hits)]))
    return [(int(hits[start]), int(hits[end - 1])) for start, end in zip(edges[:-1], edges[1:], strict=True)]


def _agent_encounters(robot: Trajectory2D, track: PedTrack, spec: EncounterSpec) -> list[EncounterWindow]:
    """Encounter windows between the robot and a single pedestrian."""
    if len(robot.time_ns) == 0 or len(track.time_ns) == 0:
        return []
    t_lo = max(int(robot.time_ns[0]), int(track.time_ns[0]))
    t_hi = min(int(robot.time_ns[-1]), int(track.time_ns[-1]))
    if t_hi < t_lo:
        return []

    robot_window = robot.slice_ns(t_lo, t_hi)
    pedestrian_window = track.trajectory.resample_onto(robot_window.time_ns)
    distances = np.hypot(pedestrian_window.x - robot_window.x, pedestrian_window.y - robot_window.y)
    merge_gap_ns = int(round(spec.merge_gap_s * NS_PER_S))
    spans = contiguous_spans(distances <= spec.distance_threshold_m, robot_window.time_ns, merge_gap_ns)

    pedestrian_radius_m = track.radius_m if track.radius_m is not None else spec.default_ped_radius_m
    surface_margin_m = robot.radius_m + pedestrian_radius_m
    windows: list[EncounterWindow] = []
    for start, end in spans:
        span_distances = distances[start : end + 1]
        span_time_ns = robot_window.time_ns[start : end + 1]
        nearest = int(np.argmin(span_distances))
        min_dist_m = float(span_distances[nearest])
        windows.append(
            EncounterWindow(
                ped_id=track.ped_id,
                ped_name=track.ped_name,
                t_start_ns=int(span_time_ns[0]),
                t_end_ns=int(span_time_ns[-1]),
                t_min_ns=int(span_time_ns[nearest]),
                min_dist_m=min_dist_m,
                min_clearance_m=min_dist_m - surface_margin_m,
            )
        )
    return windows


def detect_encounters(
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    spec: EncounterSpec = DEFAULT_ENCOUNTER_SPEC,
) -> list[EncounterWindow]:
    """Segment robot-pedestrian trajectories into sorted EncounterWindow instances."""
    windows = [window for track in tracks.values() for window in _agent_encounters(robot, track, spec)]
    windows.sort(key=lambda window: (window.t_start_ns, window.ped_id))
    return windows


def merge_intervals_ns(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent [t_start_ns, t_end_ns] intervals into disjoint sorted intervals."""
    valid = sorted((int(start), int(end)) for start, end in intervals if end >= start)
    if not valid:
        return []
    merged: list[tuple[int, int]] = [valid[0]]
    for start, end in valid[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged
