"""Vectorized scope-window detectors for the multi-resolution evaluator.

Every function here is a pure ``pl.LazyFrame -> pl.LazyFrame`` pipeline stage: encounter
bouts are extracted with run-length ids over boolean clearance masks, never with Python
row loops. The evaluator collects once, at the fact-table boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Final

import polars as pl

NS_PER_S: Final[float] = 1e9
DETECTOR_VERSION: Final[str] = "wp2-scope-detectors-v1"

ENCOUNTER_DISTANCE_M: Final[float] = 3.0
ENCOUNTER_MERGE_GAP_S: Final[float] = 1.0
CONTACT_MERGE_GAP_S: Final[float] = 0.25
ALIGN_TOLERANCE_NS: Final[int] = 100_000_000
DEFAULT_PED_RADIUS_M: Final[float] = 0.3
DEFAULT_ROBOT_RADIUS_M: Final[float] = 0.25

STATIONARY_SPEED_MPS: Final[float] = 0.15
HEAD_ON_ANGLE_RAD: Final[float] = 3.141592653589793 / 4.0
OVERTAKING_ALIGNMENT: Final[float] = 0.7
PEDESTRIAN_KIND_PATTERN: Final[str] = "ped|human"
ROBOT_KIND_PATTERN: Final[str] = "robot"
WALL_KIND_PATTERN: Final[str] = "wall"
TWO_PI_RAD: Final[float] = 6.283185307179586

ACTOR_PEDESTRIAN: Final[str] = "pedestrian"
ACTOR_ROBOT: Final[str] = "robot"
APPROACH_HEAD_ON: Final[str] = "head_on"
APPROACH_CROSSING: Final[str] = "crossing"
APPROACH_OVERTAKING: Final[str] = "overtaking"
APPROACH_STATIONARY: Final[str] = "stationary"

PAIR_SCHEMA = pl.Schema(
    {
        "robot_id": pl.String,
        "actor_id": pl.String,
        "actor_name": pl.String,
        "actor_kind": pl.String,
        "t_ns": pl.Int64,
        "robot_x": pl.Float64,
        "robot_y": pl.Float64,
        "robot_yaw": pl.Float64,
        "robot_vx": pl.Float64,
        "robot_vy": pl.Float64,
        "actor_x": pl.Float64,
        "actor_y": pl.Float64,
        "actor_vx": pl.Float64,
        "actor_vy": pl.Float64,
        "actor_radius_m": pl.Float64,
        "distance_m": pl.Float64,
        "edge_clearance_m": pl.Float64,
    }
)

ENCOUNTER_WINDOW_SCHEMA = pl.Schema(
    {
        "encounter_idx": pl.UInt32,
        "robot_id": pl.String,
        "actor_id": pl.String,
        "actor_name": pl.String,
        "actor_kind": pl.String,
        "t_start_ns": pl.Int64,
        "t_end_ns": pl.Int64,
        "min_distance_m": pl.Float64,
        "min_clearance_m": pl.Float64,
        "approach_type": pl.String,
        "detector_version": pl.String,
        "detector_params_hash": pl.String,
    }
)

CONTACT_WINDOW_SCHEMA = pl.Schema(
    {
        "contact_idx": pl.UInt32,
        "collider_kind": pl.String,
        "obstacle_id": pl.String,
        "t_onset_ns": pl.Int64,
        "t_end_ns": pl.Int64,
        "obstacle_x": pl.Float64,
        "obstacle_y": pl.Float64,
        "obstacle_z": pl.Float64,
        "encounter_idx": pl.Int64,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class DetectorParams:
    """Versioned parameters of the encounter and contact detectors."""

    encounter_distance_m: float = ENCOUNTER_DISTANCE_M
    encounter_merge_gap_s: float = ENCOUNTER_MERGE_GAP_S
    contact_merge_gap_s: float = CONTACT_MERGE_GAP_S
    align_tolerance_ns: int = ALIGN_TOLERANCE_NS

    def fingerprint(self) -> str:
        """Stable short hash of the parameters, stored next to every detected window."""
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


DEFAULT_DETECTOR_PARAMS = DetectorParams()


def pedestrian_pairs_lf(robot_tracks: pl.DataFrame | pl.LazyFrame, pedestrian_tracks: pl.DataFrame | pl.LazyFrame, robot_radius_m: float, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.LazyFrame:
    """One row per pedestrian sample with the robot pose at or before it: centre distance and edge clearance."""
    actors = (
        pedestrian_tracks.lazy()
        .select(
            pl.col("ped_id").alias("actor_id"),
            pl.col("ped_name").alias("actor_name"),
            pl.lit(ACTOR_PEDESTRIAN).alias("actor_kind"),
            pl.col("t_ns").cast(pl.Int64),
            pl.col("x").alias("actor_x"),
            pl.col("y").alias("actor_y"),
            pl.col("vx").alias("actor_vx"),
            pl.col("vy").alias("actor_vy"),
            pl.col("radius_m").fill_null(DEFAULT_PED_RADIUS_M).alias("actor_radius_m"),
        )
        .sort("t_ns")
    )
    return _pairs_lf(robot_tracks, actors, robot_radius_m, params)


def fleet_pairs_lf(robot_tracks: pl.DataFrame | pl.LazyFrame, fleet: dict[str, pl.DataFrame], robot_radius_m: float, actor_radius_m: float = DEFAULT_ROBOT_RADIUS_M, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.LazyFrame:
    """One row per other-robot task-pose sample with this robot's pose at or before it."""
    actor_frames = [_fleet_actor_frame(name, frame, actor_radius_m) for name, frame in fleet.items()]
    if not actor_frames:
        return pl.LazyFrame(schema=PAIR_SCHEMA)
    actors = pl.concat(actor_frames, how="vertical").sort("t_ns")
    return _pairs_lf(robot_tracks, actors, robot_radius_m, params)


def _fleet_actor_frame(actor_id: str, frame: pl.DataFrame, actor_radius_m: float) -> pl.LazyFrame:
    """Task poses of another robot as a pair-actor frame: position, derived velocity and heading."""
    actor = (
        frame.lazy()
        .select(
            pl.col("t_ns").cast(pl.Int64),
            pl.col("pos_x").alias("actor_x"),
            pl.col("pos_y").alias("actor_y"),
        )
        .sort("t_ns")
    )
    previous_time_s = pl.col("t_ns").diff().cast(pl.Float64) / NS_PER_S
    dt_s = pl.when(previous_time_s > 0.0).then(previous_time_s).otherwise(None)
    return actor.with_columns(
        actor_id=pl.lit(actor_id),
        actor_name=pl.lit(actor_id),
        actor_kind=pl.lit(ACTOR_ROBOT),
        actor_vx=(pl.col("actor_x").diff() / dt_s).fill_null(0.0),
        actor_vy=(pl.col("actor_y").diff() / dt_s).fill_null(0.0),
        actor_radius_m=pl.lit(actor_radius_m, dtype=pl.Float64),
    ).select("actor_id", "actor_name", "actor_kind", "t_ns", "actor_x", "actor_y", "actor_vx", "actor_vy", "actor_radius_m")


def _pairs_lf(robot_tracks: pl.DataFrame | pl.LazyFrame, actors: pl.LazyFrame, robot_radius_m: float, params: DetectorParams) -> pl.LazyFrame:
    """As-of join the actor samples onto the robot timeline and derive centre distance and edge clearance."""
    robot = (
        robot_tracks.lazy()
        .select(
            "robot_id",
            pl.col("t_ns").cast(pl.Int64).alias("robot_t_ns"),
            pl.col("x").alias("robot_x"),
            pl.col("y").alias("robot_y"),
            pl.col("yaw").alias("robot_yaw"),
            pl.col("vx").alias("robot_vx"),
            pl.col("vy").alias("robot_vy"),
        )
        .sort("robot_t_ns")
    )
    join_tolerance_ns = params.align_tolerance_ns
    return (
        actors.join_asof(robot, left_on="t_ns", right_on="robot_t_ns", strategy="backward", tolerance=join_tolerance_ns)
        .filter(pl.col("robot_x").is_not_null() & pl.col("actor_x").is_not_null())
        .with_columns(
            distance_m=(pl.col("actor_x") - pl.col("robot_x")).pow(2).add((pl.col("actor_y") - pl.col("robot_y")).pow(2)).sqrt(),
            edge_clearance_m=(pl.col("actor_x") - pl.col("robot_x")).pow(2).add((pl.col("actor_y") - pl.col("robot_y")).pow(2)).sqrt() - (pl.lit(robot_radius_m) + pl.col("actor_radius_m")),
        )
        .select(PAIR_SCHEMA.names())
    )


def detect_encounter_windows_lf(pairs: pl.LazyFrame, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.LazyFrame:
    """Contiguous bouts with centre distance <= encounter_distance_m, gaps shorter than the merge gap bridged.

    Run-length ids over the boolean clearance mask give the bouts; a second run-length id over the
    gap-bridged mask separates bouts that survive the merge window. The onset row of each bout keeps
    the kinematics that classify the approach.
    """
    merge_gap_ns = int(params.encounter_merge_gap_s * NS_PER_S)
    in_encounter = pl.col("distance_m") <= params.encounter_distance_m

    runs = (
        pairs.sort(["actor_id", "t_ns"])
        .with_columns(in_encounter=in_encounter)
        .with_columns(run_id=pl.col("in_encounter").rle_id().over("actor_id"))
        .group_by(["actor_id", "run_id"])
        .agg(
            run_in_encounter=pl.col("in_encounter").first(),
            run_start_ns=pl.col("t_ns").min(),
            run_end_ns=pl.col("t_ns").max(),
        )
        .sort(["actor_id", "run_id"])
    )
    interior_gap = (~pl.col("run_in_encounter")) & pl.col("run_in_encounter").shift(1).over("actor_id").fill_null(False) & pl.col("run_in_encounter").shift(-1).over("actor_id").fill_null(False) & ((pl.col("run_end_ns") - pl.col("run_start_ns")) <= merge_gap_ns)
    runs = runs.with_columns(bridgeable=interior_gap)
    flagged = (
        pairs.sort(["actor_id", "t_ns"])
        .with_columns(in_encounter=in_encounter)
        .with_columns(run_id=pl.col("in_encounter").rle_id().over("actor_id"))
        .join(runs.select("actor_id", "run_id", "bridgeable"), on=["actor_id", "run_id"], how="left")
        .with_columns(bridged=pl.col("in_encounter") | pl.col("bridgeable").fill_null(False))
        .with_columns(encounter_run=pl.col("bridged").rle_id().over("actor_id"))
        .filter(pl.col("bridged"))
    )

    onset = {column: pl.col(column).sort_by("t_ns").first() for column in ("robot_yaw", "robot_vx", "robot_vy", "robot_x", "robot_y", "actor_x", "actor_y", "actor_vx", "actor_vy")}
    windows = flagged.group_by(["actor_id", "encounter_run"]).agg(
        robot_id=pl.col("robot_id").first(),
        actor_name=pl.col("actor_name").first(),
        actor_kind=pl.col("actor_kind").first(),
        t_start_ns=pl.col("t_ns").min(),
        t_end_ns=pl.col("t_ns").max(),
        min_distance_m=pl.col("distance_m").min(),
        min_clearance_m=pl.col("edge_clearance_m").min(),
        **onset,
    )

    bearing_rad = pl.arctan2(pl.col("actor_y") - pl.col("robot_y"), pl.col("actor_x") - pl.col("robot_x"))
    heading_error_rad = ((pl.col("robot_yaw") - bearing_rad + pl.lit(3.141592653589793)) % pl.lit(TWO_PI_RAD) - pl.lit(3.141592653589793)).abs()
    ped_speed_mps = pl.col("actor_vx").pow(2).add(pl.col("actor_vy").pow(2)).sqrt()
    robot_speed_mps = pl.col("robot_vx").pow(2).add(pl.col("robot_vy").pow(2)).sqrt()
    robot_toward_actor = (pl.col("robot_vx") * bearing_rad.cos() + pl.col("robot_vy") * bearing_rad.sin()) > 0.0
    same_direction = (pl.col("actor_vx") * pl.col("robot_vx") + pl.col("actor_vy") * pl.col("robot_vy")) > (pl.lit(OVERTAKING_ALIGNMENT) * ped_speed_mps * robot_speed_mps)
    approach_type = (
        pl.when(ped_speed_mps < STATIONARY_SPEED_MPS).then(pl.lit(APPROACH_STATIONARY)).when((heading_error_rad <= HEAD_ON_ANGLE_RAD) & robot_toward_actor).then(pl.lit(APPROACH_HEAD_ON)).when(same_direction & (robot_speed_mps > ped_speed_mps)).then(pl.lit(APPROACH_OVERTAKING)).otherwise(pl.lit(APPROACH_CROSSING))
    )

    return (
        windows.with_columns(
            approach_type=approach_type,
            detector_version=pl.lit(DETECTOR_VERSION),
            detector_params_hash=pl.lit(params.fingerprint()),
        )
        .sort(["t_start_ns", "actor_kind", "actor_id"])
        .with_row_index("encounter_idx")
        .with_columns(pl.col("encounter_idx").cast(pl.UInt32))
        .select(ENCOUNTER_WINDOW_SCHEMA.names())
    )


def detect_contact_windows_lf(collision_contacts: pl.DataFrame | pl.LazyFrame, encounters: pl.DataFrame | pl.LazyFrame, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.LazyFrame:
    """Contact bouts from the recorded collision stream, each linked to its enclosing encounter."""
    merge_gap_ns = int(params.contact_merge_gap_s * NS_PER_S)
    normalized_kind = pl.col("kind").fill_null("static").str.to_lowercase()
    collider_kind = pl.when(normalized_kind.str.contains(PEDESTRIAN_KIND_PATTERN)).then(pl.lit(ACTOR_PEDESTRIAN)).when(normalized_kind.str.contains(ROBOT_KIND_PATTERN)).then(pl.lit(ACTOR_ROBOT)).when(normalized_kind.str.contains(WALL_KIND_PATTERN)).then(pl.lit("wall")).otherwise(pl.lit("static"))
    raw_obstacle_id = pl.col("obstacle_id").fill_null("")
    entity_obstacle_id = pl.col("raw_obstacle_id").str.replace(r"^env_[0-9]+/", "").str.split("/").list.first()
    contacts = (
        collision_contacts.lazy()
        .select(
            pl.col("t_ns").cast(pl.Int64),
            raw_obstacle_id.alias("raw_obstacle_id"),
            pl.col("kind").fill_null("static").str.to_lowercase().alias("kind"),
            pl.col("obstacle_x").cast(pl.Float64),
            pl.col("obstacle_y").cast(pl.Float64),
            pl.col("obstacle_z").cast(pl.Float64),
        )
        .with_columns(collider_kind=collider_kind)
        .with_columns(obstacle_id=pl.when(pl.col("collider_kind").is_in([ACTOR_PEDESTRIAN, ACTOR_ROBOT])).then(entity_obstacle_id).otherwise(pl.col("raw_obstacle_id")))
        .sort(["collider_kind", "obstacle_id", "t_ns"])
        .with_columns(
            same_target=(pl.col("collider_kind") == pl.col("collider_kind").shift(1)) & (pl.col("obstacle_id") == pl.col("obstacle_id").shift(1)),
            beyond_gap=pl.col("t_ns").diff() > merge_gap_ns,
        )
        .with_columns(group_id=(~(pl.col("same_target").fill_null(False)) | pl.col("beyond_gap").fill_null(True)).cum_sum())
    )
    grouped = contacts.group_by("group_id").agg(
        collider_kind=pl.col("collider_kind").first(),
        obstacle_id=pl.col("obstacle_id").first(),
        t_onset_ns=pl.col("t_ns").min(),
        t_end_ns=pl.col("t_ns").max(),
        obstacle_x=pl.col("obstacle_x").first(),
        obstacle_y=pl.col("obstacle_y").first(),
        obstacle_z=pl.col("obstacle_z").first(),
    )
    grouped = grouped.sort(["t_onset_ns", "collider_kind", "obstacle_id"]).with_row_index("contact_idx").with_columns(pl.col("contact_idx").cast(pl.UInt32))

    enclosing_windows = encounters.lazy().select(
        pl.col("actor_id").alias("obstacle_id"),
        pl.col("t_start_ns").alias("encounter_start_ns"),
        pl.col("t_end_ns").alias("encounter_end_ns"),
        pl.col("encounter_idx"),
    )
    enclosing = grouped.join(enclosing_windows, on="obstacle_id", how="inner").filter((pl.col("t_onset_ns") >= pl.col("encounter_start_ns")) & (pl.col("t_onset_ns") <= pl.col("encounter_end_ns"))).group_by("contact_idx").agg(encounter_idx=pl.col("encounter_idx").min())
    return grouped.join(enclosing, on="contact_idx", how="left").with_columns(pl.col("encounter_idx").cast(pl.Int64)).select(CONTACT_WINDOW_SCHEMA.names())


def empty_encounter_windows() -> pl.DataFrame:
    """Empty encounter window table with the detector schema."""
    return pl.DataFrame(schema=ENCOUNTER_WINDOW_SCHEMA)


def empty_contact_windows() -> pl.DataFrame:
    """Empty contact window table with the detector schema."""
    return pl.DataFrame(schema=CONTACT_WINDOW_SCHEMA)


def empty_pairs() -> pl.DataFrame:
    """Empty pair table with the detector schema."""
    return pl.DataFrame(schema=PAIR_SCHEMA)
