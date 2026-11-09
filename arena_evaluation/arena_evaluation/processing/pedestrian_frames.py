"""Pedestrian frames of an episode: the peak count inside a window, and each robot time's pedestrian states from the flat table.

``frame_geometry`` is the vectorized contract for calculators that need the robot state
sampled onto the pedestrian frame axis: one asof join against the robot track, then a
single long frame of frame x pedestrian rows. It replaces per-frame ``frames_of``
partitioning and Python-level per-frame selection in the metric hot path.
"""

from __future__ import annotations

import dataclasses
from typing import Final

import numpy as np
import polars as pl

ALIGN_TOLERANCE_NS: Final[int] = 100_000_000
NS_PER_S: Final[float] = 1e9

FRAME_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "frame_ix": pl.Int64,
        "t_ns": pl.Int64,
        "dt_s": pl.Float64,
        "robot_x_m": pl.Float64,
        "robot_y_m": pl.Float64,
        "robot_yaw_rad": pl.Float64,
        "robot_vx_mps": pl.Float64,
        "robot_vy_mps": pl.Float64,
        "robot_speed_mps": pl.Float64,
    }
)

PEDESTRIAN_FRAME_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "frame_ix": pl.Int64,
        "t_ns": pl.Int64,
        "ped_id": pl.String,
        "ped_x_m": pl.Float64,
        "ped_y_m": pl.Float64,
        "ped_yaw_rad": pl.Float64,
        "ped_vx_mps": pl.Float64,
        "ped_vy_mps": pl.Float64,
        "robot_yaw_rad": pl.Float64,
        "dx_m": pl.Float64,
        "dy_m": pl.Float64,
        "distance_m": pl.Float64,
        "rel_vx_mps": pl.Float64,
        "rel_vy_mps": pl.Float64,
        "rel_speed_mps": pl.Float64,
    }
)

ROBOT_TRACK_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "robot_t_ns": pl.Int64,
        "robot_x_m": pl.Float64,
        "robot_y_m": pl.Float64,
        "robot_yaw_rad": pl.Float64,
        "robot_vx_mps": pl.Float64,
        "robot_vy_mps": pl.Float64,
        "robot_speed_mps": pl.Float64,
    }
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FrameGeometry:
    """Robot state sampled onto the pedestrian frame axis, and the frame x pedestrian relative geometry.

    ``frames`` carries one row per distinct pedestrian frame timestamp in time order, with
    the frame-to-frame ``dt_s`` (the last frame carries a negligible 1e-6 s, as the
    integrators always did) and the robot pose/velocity of the latest robot sample within
    ``tolerance_ns``; that state is null where the recording has no such sample.
    ``pedestrians`` carries one row per pedestrian and frame with the robot-relative
    displacement, distance and relative velocity, null where either side is unknown.
    """

    frames: pl.DataFrame
    pedestrians: pl.DataFrame

    @property
    def observed(self) -> bool:
        """Whether at least one frame resolved a robot pose against its pedestrians."""
        return self.frames.filter(pl.col("robot_x_m").is_not_null()).height > 0


def empty_frame_geometry() -> FrameGeometry:
    """Geometry without frames, carrying both explicit schemas."""
    return FrameGeometry(
        frames=pl.DataFrame(schema=FRAME_SCHEMA),
        pedestrians=pl.DataFrame(schema=PEDESTRIAN_FRAME_SCHEMA),
    )


def robot_track_state(robot_tracks: pl.DataFrame) -> pl.DataFrame:
    """Robot ground-truth pose and finite-difference velocity on the track's own time axis.

    The velocity is the one-sided difference of the first sample carried forward as zero,
    which is the convention every finite-difference speed in the metrics package uses.
    """
    if robot_tracks.height == 0:
        return pl.DataFrame(schema=ROBOT_TRACK_SCHEMA)

    track = (
        robot_tracks.select(
            t_ns=pl.col("t_ns").cast(pl.Int64),
            x_m=pl.col("x").cast(pl.Float64),
            y_m=pl.col("y").cast(pl.Float64),
            yaw_rad=pl.col("yaw").cast(pl.Float64),
        )
        .drop_nulls(["t_ns", "x_m", "y_m"])
        .sort("t_ns")
        .with_columns(
            previous_t_ns=pl.col("t_ns").shift(1),
            previous_x_m=pl.col("x_m").shift(1),
            previous_y_m=pl.col("y_m").shift(1),
        )
    )
    step_s = (pl.col("t_ns") - pl.col("previous_t_ns")).cast(pl.Float64) / NS_PER_S
    safe_step_s = pl.when(step_s > 0.0).then(step_s).otherwise(1e-6)
    return (
        track.with_columns(
            vx_mps=((pl.col("x_m") - pl.col("previous_x_m")) / safe_step_s).fill_null(0.0),
            vy_mps=((pl.col("y_m") - pl.col("previous_y_m")) / safe_step_s).fill_null(0.0),
        )
        .with_columns(speed_mps=(pl.col("vx_mps").pow(2) + pl.col("vy_mps").pow(2)).sqrt())
        .select(
            robot_t_ns=pl.col("t_ns"),
            robot_x_m=pl.col("x_m"),
            robot_y_m=pl.col("y_m"),
            robot_yaw_rad=pl.col("yaw_rad"),
            robot_vx_mps=pl.col("vx_mps"),
            robot_vy_mps=pl.col("vy_mps"),
            robot_speed_mps=pl.col("speed_mps"),
        )
    )


def frame_geometry(
    peds: pl.DataFrame | None,
    robot_tracks: pl.DataFrame | None,
    *,
    tolerance_ns: int = ALIGN_TOLERANCE_NS,
) -> FrameGeometry:
    """Sample the robot track onto the pedestrian frame axis and attach every pedestrian's relative geometry."""
    if peds is None or robot_tracks is None or peds.height == 0 or robot_tracks.height == 0:
        return empty_frame_geometry()

    frames = peds.select(pl.col("t_ns").cast(pl.Int64)).unique().sort("t_ns").with_row_index("frame_ix").with_columns(dt_s=_frame_step_s()).with_columns(dt_s=_integration_step_s())
    sampled = frames.join_asof(
        robot_track_state(robot_tracks),
        left_on="t_ns",
        right_on="robot_t_ns",
        strategy="backward",
        tolerance=tolerance_ns,
    )
    pedestrians = (
        peds.select(
            t_ns=pl.col("t_ns").cast(pl.Int64),
            ped_id=pl.col("ped_id").cast(pl.String),
            ped_x_m=pl.col("x").cast(pl.Float64),
            ped_y_m=pl.col("y").cast(pl.Float64),
            ped_yaw_rad=pl.col("yaw").cast(pl.Float64),
            ped_vx_mps=pl.col("vx").cast(pl.Float64),
            ped_vy_mps=pl.col("vy").cast(pl.Float64),
        )
        .join(sampled, on="t_ns", how="inner")
        .with_columns(
            dx_m=pl.col("ped_x_m") - pl.col("robot_x_m"),
            dy_m=pl.col("ped_y_m") - pl.col("robot_y_m"),
            rel_vx_mps=pl.col("ped_vx_mps") - pl.col("robot_vx_mps"),
            rel_vy_mps=pl.col("ped_vy_mps") - pl.col("robot_vy_mps"),
        )
        .with_columns(
            distance_m=(pl.col("dx_m").pow(2) + pl.col("dy_m").pow(2)).sqrt(),
            rel_speed_mps=(pl.col("rel_vx_mps").pow(2) + pl.col("rel_vy_mps").pow(2)).sqrt(),
        )
        .select("frame_ix", "t_ns", "ped_id", "ped_x_m", "ped_y_m", "ped_yaw_rad", "ped_vx_mps", "ped_vy_mps", "robot_yaw_rad", "dx_m", "dy_m", "distance_m", "rel_vx_mps", "rel_vy_mps", "rel_speed_mps")
    )
    return FrameGeometry(frames=sampled.select("frame_ix", "t_ns", "dt_s", "robot_x_m", "robot_y_m", "robot_yaw_rad", "robot_vx_mps", "robot_vy_mps", "robot_speed_mps"), pedestrians=pedestrians)


def nearest_pedestrian_frames(geometry: FrameGeometry) -> pl.DataFrame:
    """The closest pedestrian of every frame, with the frame's robot velocity and integration step.

    Frames whose robot pose is unknown have no pedestrian distance and are therefore absent.
    """
    if geometry.pedestrians.height == 0:
        return pl.DataFrame(schema=PEDESTRIAN_FRAME_SCHEMA)
    ranked = geometry.pedestrians.filter(pl.col("distance_m").is_not_null()).with_columns(distance_rank=pl.col("distance_m").rank("ordinal").over("frame_ix"))
    nearest = ranked.filter(pl.col("distance_rank") == 1).drop("distance_rank")
    return nearest.join(
        geometry.frames.select("frame_ix", "dt_s", "robot_vx_mps", "robot_vy_mps", "robot_speed_mps"),
        on="frame_ix",
        how="inner",
    )


def _frame_step_s() -> pl.Expr:
    """Seconds to the next frame, null on the last frame."""
    return (pl.col("t_ns").shift(-1) - pl.col("t_ns")).cast(pl.Float64) / NS_PER_S


def _integration_step_s() -> pl.Expr:
    """Integration step of a frame: its forward difference, 1e-6 s on the last (or a repeated) frame."""
    step_s = _frame_step_s()
    return pl.when(step_s.is_null() | (step_s == 0.0)).then(1e-6).otherwise(step_s)


def pedestrians_by_time(time_ns: np.ndarray, peds: pl.DataFrame, tolerance_ns: int = ALIGN_TOLERANCE_NS) -> dict[int, pl.DataFrame]:
    """The pedestrians of the frame each robot time uses, keyed by robot time; times with no frame in tolerance are absent."""
    unique_times = np.unique(time_ns.astype(np.int64))
    joined = pedestrians_at(pl.Series("time_ns", unique_times), peds, tolerance_ns)
    return {int(key[0]): frame for key, frame in joined.partition_by("time_ns", as_dict=True).items()}


def frames_of(peds: pl.DataFrame) -> list[pl.DataFrame]:
    """The pedestrians of each frame, frames in time order: the flat table partitioned by t_ns."""
    if peds is None or len(peds) == 0 or "t_ns" not in peds.columns:
        return []
    return peds.sort("t_ns").partition_by("t_ns", maintain_order=True)


def peak_pedestrian_count(peds: pl.DataFrame | None, start_ns: int | None = None, end_ns: int | None = None) -> int:
    """Largest number of pedestrians recorded in one frame between start_ns and end_ns, bounds inclusive."""
    if peds is None or len(peds) == 0:
        return 0
    if start_ns is not None:
        peds = peds.filter(pl.col("t_ns") >= start_ns)
    if end_ns is not None:
        peds = peds.filter(pl.col("t_ns") <= end_ns)
    counts = peds.group_by("t_ns").len()["len"]
    return int(counts.max()) if len(counts) > 0 else 0  # pyright: ignore[reportArgumentType]


def distances_to_pedestrians(
    time_ns: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    peds: pl.DataFrame,
    tolerance_ns: int = ALIGN_TOLERANCE_NS,
) -> pl.DataFrame:
    """One row per robot sample and pedestrian in its frame: ``row`` (the sample index), ``ped_id`` and ``distance`` to the robot at (x, y)."""
    robot = pl.DataFrame({"row": np.arange(len(time_ns)), "time_ns": time_ns.astype(np.int64), "rx": x, "ry": y})
    frames = pedestrians_at(robot["time_ns"].unique(), peds, tolerance_ns).select("time_ns", "ped_id", "x", "y")
    return robot.join(frames, on="time_ns", how="inner").select(
        "row",
        "ped_id",
        ((pl.col("x") - pl.col("rx")) ** 2 + (pl.col("y") - pl.col("ry")) ** 2).sqrt().alias("distance"),
    )


def nearest_pedestrian_distance(
    time_ns: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    peds: pl.DataFrame,
    tolerance_ns: int = ALIGN_TOLERANCE_NS,
) -> np.ndarray:
    """Distance from the robot at (x, y) to its nearest pedestrian at each robot time, NaN where no frame is in tolerance."""
    nearest = distances_to_pedestrians(time_ns, x, y, peds, tolerance_ns).group_by("row").agg(pl.col("distance").min())
    distances = np.full(len(time_ns), np.nan)
    distances[nearest["row"].to_numpy()] = nearest["distance"].to_numpy()
    return distances


def pedestrians_at(time_ns: pl.Series, peds: pl.DataFrame, tolerance_ns: int = ALIGN_TOLERANCE_NS) -> pl.DataFrame:
    """Every pedestrian of the latest frame at or before each robot time, within the tolerance: one row per robot time and pedestrian.

    The frame a row came from is kept in ``frame_ns``; robot times with no frame in tolerance produce no rows.
    """
    frames = peds.select(pl.col("t_ns").alias("frame_ns")).unique().sort("frame_ns")
    robot_times = pl.DataFrame({"time_ns": time_ns.cast(pl.Int64)}).sort("time_ns")
    matched = robot_times.join_asof(frames, left_on="time_ns", right_on="frame_ns", strategy="backward", tolerance=tolerance_ns)
    return matched.drop_nulls("frame_ns").join(peds, left_on="frame_ns", right_on="t_ns", how="inner")
