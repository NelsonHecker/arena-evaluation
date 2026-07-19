from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np
import polars as pl

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ZoneGeometry:
    name: str
    polygon: object
    max_speed: float | None
    quiet: bool
    restricted: bool


@dataclasses.dataclass
class _DoorGeometry:
    name: str
    center_x: float
    center_y: float
    radius: float


def _extract_zone_geometry(flattened: typing.Any, require_annotation: bool = True) -> list[_ZoneGeometry]:
    import shapely

    zones: list[_ZoneGeometry] = []
    for zone in flattened.zones:
        if len(zone.corners) < 3:
            continue

        max_speed = None
        quiet = False
        restricted = False
        has_annotation = False
        for cfg in zone.semantics:
            if cfg.role == "state" and cfg.name == "max_speed":
                has_annotation = True
                if cfg.value is not None:
                    max_speed = float(cfg.value)
            elif cfg.role == "predicate" and cfg.name == "quiet":
                has_annotation = True
                quiet = bool(cfg.value)
            elif cfg.role == "predicate" and cfg.name == "restricted":
                has_annotation = True
                restricted = bool(cfg.value)

        if require_annotation and not has_annotation:
            continue

        polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.corners])
        zones.append(_ZoneGeometry(name=zone.name, polygon=polygon, max_speed=max_speed, quiet=quiet, restricted=restricted))
    return zones


def _extract_door_geometry(flattened: typing.Any) -> list[_DoorGeometry]:
    doors: list[_DoorGeometry] = []
    for door in flattened.all_doors:
        center_x = (door.start.x + door.end.x) / 2.0
        center_y = (door.start.y + door.end.y) / 2.0
        radius = max(door.activation_distance)
        doors.append(_DoorGeometry(name=door.name, center_x=center_x, center_y=center_y, radius=radius))
    return doors


def _offset_zones(zones: list[_ZoneGeometry], offset: tuple[float, float]) -> list[_ZoneGeometry]:
    """World-frame zones translated into the recorded map frame by the env packing offset."""
    if offset == (0.0, 0.0):
        return zones
    import shapely.affinity

    return [
        dataclasses.replace(zone, polygon=shapely.affinity.translate(zone.polygon, offset[0], offset[1]))
        for zone in zones
    ]


def _offset_doors(doors: list[_DoorGeometry], offset: tuple[float, float]) -> list[_DoorGeometry]:
    """World-frame door centers translated into the recorded map frame."""
    if offset == (0.0, 0.0):
        return doors
    return [
        dataclasses.replace(door, center_x=door.center_x + offset[0], center_y=door.center_y + offset[1])
        for door in doors
    ]


def _zone_membership(pos_x: np.ndarray, pos_y: np.ndarray, zones: list[_ZoneGeometry]) -> np.ndarray:
    import shapely

    idx = np.full(len(pos_x), -1, dtype=np.int64)
    for i in range(len(pos_x)):
        point = shapely.Point(pos_x[i], pos_y[i])
        for j, zone in enumerate(zones):
            if zone.polygon.covers(point):
                idx[i] = j
                break
    return idx


def _speed_zone_metrics(
    zone_idx: np.ndarray, speed: np.ndarray, dt: np.ndarray, zones: list[_ZoneGeometry]
) -> tuple[int, float]:
    mask = np.zeros(len(zone_idx), dtype=bool)
    for i, j in enumerate(zone_idx):
        if j == -1:
            continue
        max_speed = zones[j].max_speed
        if max_speed is not None and speed[i] > max_speed:
            mask[i] = True

    violations = 0
    prev = False
    for flag in mask:
        if flag and not prev:
            violations += 1
        prev = flag

    seconds = float(sum(dt[i] for i in range(len(dt)) if mask[i]))
    return violations, seconds


def _quiet_zone_dwell(zone_idx: np.ndarray, dt: np.ndarray, zones: list[_ZoneGeometry]) -> float:
    total = 0.0
    for i in range(len(dt)):
        j = zone_idx[i]
        if j != -1 and zones[j].quiet:
            total += dt[i]
    return total


def _restricted_zone_entries(zone_idx: np.ndarray, zones: list[_ZoneGeometry]) -> int:
    entries = 0
    for i in range(1, len(zone_idx)):
        j = zone_idx[i]
        if j == -1 or not zones[j].restricted:
            continue
        if zone_idx[i - 1] != j:
            entries += 1
    return entries


def _door_open_series(events: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-door sorted (time_ns, is_open) event series, name keyed by entity suffix."""
    door_events = events.filter((pl.col("kind") == "door") & (pl.col("field") == "state")).sort("time_ns")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if len(door_events) == 0:
        return result

    for _, group in door_events.group_by("entity"):
        g = group.sort("time_ns")
        name = str(g["entity"][0]).rsplit("/", 1)[-1]
        times = g["time_ns"].to_numpy()
        is_open = np.array([v == "open" for v in g["current"].to_list()])
        result[name] = (times, is_open)
    return result


def _is_open_at(door_series: dict[str, tuple[np.ndarray, np.ndarray]], name: str, t_ns: int) -> bool:
    entry = door_series.get(name)
    if entry is None:
        return False

    times, is_open = entry
    idx = int(np.searchsorted(times, t_ns, side="right")) - 1
    if idx < 0:
        return False
    return bool(is_open[idx])


class ComplianceMetricsCalculator(BaseMetricCalculator):
    """
    Compliance scoring against zone and door semantic annotations (SPEC_M1 M1.2).

    Zone membership is evaluated per odom sample against zone polygons loaded from
    the recorded world asset by name, not from the semantic snapshot (which only
    republishes the same literal YAML values), flattened into the single map frame
    the runtime renders via `WorldDescription.compact_world`. Doorway blocking
    replays `semantic_events` door state exactly as the seed metric's
    `_time_waiting_at_doors` replay does, seeded `"closed"` at episode start.
    Single-robot attribution assumption per SPEC_M1 M1.3. Returns schema-consistent
    `None` for every output key when the recorded world is not locally present.

    Elevator etiquette metrics (`blocked_closing_door`, `boarded_before_exit`,
    `stranded_ped_seconds`) are deferred to M2: v1 `occupants` is a scalar with no
    per-occupant identity or position (SPEC_M1 M1.2).
    """

    NAME = "compliance_metrics"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = ["odom"]

    UNITS = {
        "speed_zone_violations": "",
        "speed_zone_violation_seconds": "s",
        "quiet_zone_dwell_seconds": "s",
        "restricted_zone_entries": "",
        "doorway_blocking_time": "s",
    }

    STATIONARY_THRESHOLD = 0.05

    world: str | None = None

    def __init__(self, robot_params: typing.Any) -> None:
        super().__init__(robot_params)
        self._world_cache: dict[str, tuple[list[_ZoneGeometry], list[_DoorGeometry]] | None] = {}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "speed_zone_violations",
            "speed_zone_violation_seconds",
            "quiet_zone_dwell_seconds",
            "restricted_zone_entries",
            "doorway_blocking_time",
        ]

    def _load_world(self, world_name: str) -> tuple[list[_ZoneGeometry], list[_DoorGeometry]] | None:
        if world_name in self._world_cache:
            return self._world_cache[world_name]

        from arena_simulation_setup.tree.World import WorldIdentifier

        try:
            view = WorldIdentifier(world_name).resolve_sync()
            world = view.load()
        except FileNotFoundError as e:
            logger.warning("compliance_metrics: world '%s' not available locally: %s", world_name, e)
            self._world_cache[world_name] = None
            return None

        origins = view.level_origins()
        if origins is None:
            origins = {level_id: (0.0, 0.0) for level_id in world.levels}
        flattened = world.compact_world(origins)

        result = (_extract_zone_geometry(flattened), _extract_door_geometry(flattened))
        self._world_cache[world_name] = result
        return result

    def _doorway_blocking_time(
        self,
        episode: AlignedEpisodeBundle,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        speed: np.ndarray,
        time_ns: np.ndarray,
        doors: list[_DoorGeometry],
    ) -> float:
        if not doors:
            return 0.0

        events = episode.semantic_events
        if events is None or len(events) == 0 or "kind" not in events.columns:
            return 0.0

        door_series = _door_open_series(events)
        n = len(pos_x)
        if n < 2:
            return 0.0

        dt = np.diff(time_ns) / 1e9

        total = 0.0
        for i in range(n - 1):
            if speed[i] > self.STATIONARY_THRESHOLD:
                continue
            t = int(time_ns[i])
            for door in doors:
                dx = pos_x[i] - door.center_x
                dy = pos_y[i] - door.center_y
                if dx * dx + dy * dy <= door.radius**2 and _is_open_at(door_series, door.name, t):
                    total += dt[i]
                    break
        return total

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del prior_results
        empty = {k: None for k in self.output_keys()}

        if self.world is None:
            return empty
        if not episode.start_pos:
            return empty
        if episode.env_offset is None:
            return empty

        loaded = self._load_world(self.world)
        if loaded is None:
            return empty
        zones, doors = loaded
        zones = _offset_zones(zones, episode.env_offset)
        doors = _offset_doors(doors, episode.env_offset)

        pos_x, pos_y, _yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)

        if episode.data is None or len(episode.data) == 0 or "vel_linear" not in episode.data.columns:
            return empty

        speed = np.abs(episode.data["vel_linear"].to_numpy())
        time_ns = episode.data["time_ns"].to_numpy()
        n = len(pos_x)
        if n == 0 or len(speed) != n:
            return empty

        dt = np.diff(time_ns) / 1e9

        zone_idx = _zone_membership(pos_x, pos_y, zones)
        violations, violation_seconds = _speed_zone_metrics(zone_idx, speed, dt, zones)
        quiet_seconds = _quiet_zone_dwell(zone_idx, dt, zones)
        restricted_entries = _restricted_zone_entries(zone_idx, zones)
        doorway_blocking = self._doorway_blocking_time(episode, pos_x, pos_y, speed, time_ns, doors)

        return {
            "speed_zone_violations": violations,
            "speed_zone_violation_seconds": violation_seconds,
            "quiet_zone_dwell_seconds": quiet_seconds,
            "restricted_zone_entries": restricted_entries,
            "doorway_blocking_time": doorway_blocking,
        }
