from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import entity_name, robot_env_snapshot
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import shapely
    from arena_simulation_setup.tree.World import Level

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ZoneGeometry:
    name: str
    polygon: shapely.Polygon | shapely.MultiPolygon
    max_speed: float | None
    quiet: bool
    restricted: bool


@dataclasses.dataclass
class _DoorGeometry:
    name: str
    center_x: float
    center_y: float
    radius: float


def _extract_zone_geometry(flattened: Level, require_annotation: bool = True) -> list[_ZoneGeometry]:
    import shapely

    zones: list[_ZoneGeometry] = []
    for zone in flattened.zones:
        if len(zone.polygon) < 3:
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

        polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.polygon])
        zones.append(_ZoneGeometry(name=zone.name, polygon=polygon, max_speed=max_speed, quiet=quiet, restricted=restricted))
    return zones


def _extract_door_geometry(flattened: Level) -> list[_DoorGeometry]:
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

    return [dataclasses.replace(zone, polygon=shapely.affinity.translate(zone.polygon, offset[0], offset[1])) for zone in zones]


def _offset_doors(doors: list[_DoorGeometry], offset: tuple[float, float]) -> list[_DoorGeometry]:
    """World-frame door centers translated into the recorded map frame."""
    if offset == (0.0, 0.0):
        return doors
    return [dataclasses.replace(door, center_x=door.center_x + offset[0], center_y=door.center_y + offset[1]) for door in doors]


class _PolygonZone(typing.Protocol):
    @property
    def polygon(self) -> shapely.Polygon | shapely.MultiPolygon: ...


def _zone_membership(pos_x: np.ndarray, pos_y: np.ndarray, zones: Sequence[_PolygonZone]) -> np.ndarray:
    import shapely

    idx = np.full(len(pos_x), -1, dtype=np.int64)
    for i in range(len(pos_x)):
        point = shapely.Point(pos_x[i], pos_y[i])
        for j, zone in enumerate(zones):
            if zone.polygon.covers(point):
                idx[i] = j
                break
    return idx


def _speed_zone_metrics(zone_idx: np.ndarray, speed: np.ndarray, dt: np.ndarray, zones: list[_ZoneGeometry]) -> tuple[int, float]:
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


_RECONSTRUCTED_EVENTS_SCHEMA = {
    "time_ns": pl.Int64,
    "entity": pl.Utf8,
    "kind": pl.Utf8,
    "field": pl.Utf8,
    "previous": pl.Utf8,
    "current": pl.Utf8,
}


def _stringify_snapshot_value(row: dict) -> str:
    """Native snapshot value to the event wire convention: lowercase bool, str(float), raw string, members as sorted comma-joined names."""
    if row["field_kind"] == "predicate":
        return "true" if row["value_bool"] else "false"
    if row["field_kind"] == "continuous":
        return str(row["value_num"])
    return row["value_str"] if row["value_str"] is not None else ""


def _reconstruct_events(snapshot: pl.DataFrame | None) -> pl.DataFrame:
    """Event-equivalent (time_ns, entity, kind, field, previous, current) rows, derived by
    keeping only the change points of each entity+field's stepwise snapshot series. The first
    kept row per series is the seed value (from the episode's seed snapshot), `previous` empty.
    Members rows are kept as field `members` (see `_stringify_snapshot_value`), so membership
    changes are change points like any other field.
    """
    empty = pl.DataFrame(schema=_RECONSTRUCTED_EVENTS_SCHEMA)
    if snapshot is None or len(snapshot) == 0:
        return empty
    if not {"t_ns", "entity_name", "kind", "field_name"} <= set(snapshot.columns):
        return empty

    str_val = (
        pl.when(pl.col("field_kind") == "predicate")
        .then(pl.when(pl.col("value_bool").cast(pl.Boolean, strict=False).fill_null(False)).then(pl.lit("true")).otherwise(pl.lit("false")))
        .when(pl.col("field_kind") == "continuous")
        .then(pl.col("value_num").cast(pl.String))
        .otherwise(pl.col("value_str").fill_null(""))
    )

    group_keys = ["entity_name", "kind", "field_name"]
    return (
        snapshot.sort([*group_keys, "t_ns"])
        .with_columns(current=str_val)
        .with_columns(
            prev_raw=pl.col("current").shift(1).over(group_keys),
        )
        .filter(
            pl.col("prev_raw").is_null() | (pl.col("current") != pl.col("prev_raw"))
        )
        .select(
            time_ns=pl.col("t_ns"),
            entity=pl.col("entity_name"),
            kind=pl.col("kind"),
            field=pl.col("field_name"),
            previous=pl.col("prev_raw").fill_null(""),
            current=pl.col("current"),
        )
        .sort("time_ns")
    )


def _door_open_series(events: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-door sorted (time_ns, is_open) event series keyed by authored door name."""
    door_events = events.filter((pl.col("kind") == "door") & (pl.col("field") == "state")).sort("time_ns")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if len(door_events) == 0:
        return result

    for _, group in door_events.group_by("entity"):
        g = group.sort("time_ns")
        name = entity_name(str(g["entity"][0]))
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
    the runtime renders via `World.compact_world`. Doorway blocking
    replays the same door-state transitions reconstructed from `semantic_snapshot`
    that `_time_waiting_at_doors` uses, seeded from the entity's first recorded
    snapshot value.
    Single-robot attribution assumption per SPEC_M1 M1.3. Returns schema-consistent
    `None` for every output key when the recorded world is not locally present.

    `doorway_blocking_time` is robot-attributed (ego pose only). Elevator etiquette metrics
    (`blocked_closing_door`, `boarded_before_exit`, stranded peds) live in
    `mechanism_metrics`, which uses the per-occupant semantic `members`.
    """

    NAME = "compliance_metrics"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_tracks"]
    passing_convention = "right"

    UNITS = {
        "speed_zone_violations": "",
        "speed_zone_violation_seconds": "s",
        "quiet_zone_dwell_seconds": "s",
        "restricted_zone_entries": "",
        "doorway_blocking_time": "s",
        "passing_rule_compliance": "",
    }

    STATIONARY_THRESHOLD = 0.05

    world: str | None = None

    def __init__(self, robot_params: RobotParams) -> None:
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
            "passing_rule_compliance",
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

        events = _reconstruct_events(robot_env_snapshot(episode.semantic_snapshot, episode.robot_name))
        if len(events) == 0:
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

    def _compute_passing_compliance(self, episode: AlignedEpisodeBundle) -> float | None:
        peds_df = self.native_ped_frame(episode)
        if peds_df is None or "x" not in peds_df.columns:
            return None

        pos_x, pos_y, yaw, t_odom = self.resolve_native_pose(episode)
        if len(pos_x) == 0:
            if episode.data is None or len(episode.data) == 0:
                return None
            pos_x, pos_y, yaw, _, _, _ = self.resolve_robot_pose(episode)
            t_odom = (
                episode.data["time_ns"].to_numpy().astype(np.int64)
                if "time_ns" in episode.data.columns
                else np.arange(len(pos_x), dtype=np.int64)
            )

        if len(pos_x) < 2:
            return None

        peds_time_ns = peds_df["t_ns"].unique().sort().to_numpy()
        N = len(peds_time_ns)
        if N == 0:
            return None

        vx_full, vy_full = self.velocity_from_pose(pos_x, pos_y, t_odom)
        rvx = self.values_at_times(vx_full, t_odom, peds_time_ns)
        rvy = self.values_at_times(vy_full, t_odom, peds_time_ns)
        rpx, rpy, _ = self.pose_at_times(peds_time_ns, pos_x, pos_y, yaw, t_odom)

        robot_interp_lf = pl.LazyFrame(
            {
                "t_ns": peds_time_ns,
                "rx": rpx,
                "ry": rpy,
                "rvx": rvx,
                "rvy": rvy,
            }
        ).sort("t_ns")

        peds_lf = peds_df.lazy().sort(["ped_id", "t_ns"])
        dt_ped_s = pl.col("t_ns").diff().over("ped_id") * 1e-9
        dt_ped_safe = pl.when(dt_ped_s > 0).then(dt_ped_s).otherwise(1e-3)

        has_vx = "vx" in peds_df.columns
        has_vy = "vy" in peds_df.columns
        peds_with_vel = peds_lf.with_columns(
            vx=(
                pl.when(pl.col("vx").is_null() | pl.col("vx").is_nan())
                .then(pl.col("x").diff().over("ped_id") / dt_ped_safe)
                .otherwise(pl.col("vx"))
                if has_vx
                else pl.col("x").diff().over("ped_id") / dt_ped_safe
            ),
            vy=(
                pl.when(pl.col("vy").is_null() | pl.col("vy").is_nan())
                .then(pl.col("y").diff().over("ped_id") / dt_ped_safe)
                .otherwise(pl.col("vy"))
                if has_vy
                else pl.col("y").diff().over("ped_id") / dt_ped_safe
            ),
        ).with_columns(
            vx=pl.col("vx").fill_null(0.0),
            vy=pl.col("vy").fill_null(0.0),
        )

        joined = peds_with_vel.join(robot_interp_lf, on="t_ns", how="inner")

        dx = pl.col("x") - pl.col("rx")
        dy = pl.col("y") - pl.col("ry")
        dist = (dx.pow(2) + dy.pow(2)).sqrt()
        speed_r = (pl.col("rvx").pow(2) + pl.col("rvy").pow(2)).sqrt()
        speed_p = (pl.col("vx").pow(2) + pl.col("vy").pow(2)).sqrt()
        denom = speed_r * speed_p
        cos_angle = pl.when(denom > 1e-6).then((pl.col("rvx") * pl.col("vx") + pl.col("rvy") * pl.col("vy")) / denom).otherwise(0.0)

        is_head_on = (dist <= 4.0) & (speed_r > 0.05) & (speed_p > 0.05) & (cos_angle < -0.3)
        lat_offset = pl.col("rvx") * dy - pl.col("rvy") * dx

        pairs_analyzed = (
            joined.with_columns(
                dist=dist,
                is_head_on=is_head_on,
                lat_offset=lat_offset,
            )
            .group_by("t_ns")
            .agg(
                head_on_found=pl.col("is_head_on").any(),
                curr_min_dist=pl.col("dist").filter(pl.col("is_head_on")).min(),
                curr_lat=pl.col("lat_offset").filter(pl.col("is_head_on")).sort_by(pl.col("dist").filter(pl.col("is_head_on"))).first(),
            )
            .sort("t_ns")
            .with_columns(
                head_on_found=pl.col("head_on_found").fill_null(False),
            )
            .with_columns(
                bout_id=pl.col("head_on_found").rle_id(),
            )
            .collect()
        )

        bouts = pairs_analyzed.filter(pl.col("head_on_found")).partition_by("bout_id", maintain_order=True)
        if not bouts:
            return 1.0

        encounters_total = len(bouts)
        encounters_compliant = 0

        for bout in bouts:
            min_idx = bout["curr_min_dist"].arg_min()
            lat_at_min = bout["curr_lat"][min_idx]
            if self.passing_convention == "right" and lat_at_min >= 0.0:
                encounters_compliant += 1
            elif self.passing_convention == "left" and lat_at_min <= 0.0:
                encounters_compliant += 1

        return float(encounters_compliant) / encounters_total

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del prior_results
        result: dict[str, float | None] = {k: None for k in self.output_keys()}

        # 1. Passing Rule Compliance (Independent of world)
        result["passing_rule_compliance"] = self._compute_passing_compliance(episode)

        # 2. Zone/Door Metrics (Requires world)
        if self.world is not None and episode.start_pos:
            loaded = self._load_world(self.world)
            if loaded is not None:
                zones, doors = loaded
                pos_x, pos_y, _yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)

                if episode.data is not None and len(episode.data) > 0 and "vel_linear" in episode.data.columns:
                    speed = np.abs(episode.data["vel_linear"].to_numpy())
                    time_ns = episode.data["time_ns"].to_numpy()
                    n = len(pos_x)
                    if n > 0 and len(speed) == n:
                        dt = np.diff(time_ns) / 1e9
                        zone_idx = _zone_membership(pos_x, pos_y, zones)
                        violations, violation_seconds = _speed_zone_metrics(zone_idx, speed, dt, zones)
                        quiet_seconds = _quiet_zone_dwell(zone_idx, dt, zones)
                        restricted_entries = _restricted_zone_entries(zone_idx, zones)
                        doorway_blocking = self._doorway_blocking_time(episode, pos_x, pos_y, speed, time_ns, doors)

                        result.update(
                            {
                                "speed_zone_violations": violations,
                                "speed_zone_violation_seconds": violation_seconds,
                                "quiet_zone_dwell_seconds": quiet_seconds,
                                "restricted_zone_entries": restricted_entries,
                                "doorway_blocking_time": doorway_blocking,
                            }
                        )

        return result
