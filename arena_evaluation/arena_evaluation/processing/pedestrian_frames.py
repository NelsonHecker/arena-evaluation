"""Pedestrian frames of an episode: the peak count inside a window, and each robot time's pedestrian states from the flat table."""

from __future__ import annotations

import numpy as np
import polars as pl

ALIGN_TOLERANCE_NS = 100_000_000


def pedestrians_by_time(time_ns: np.ndarray, peds: pl.DataFrame, tolerance_ns: int = ALIGN_TOLERANCE_NS) -> dict[int, pl.DataFrame]:
    """The pedestrians of the frame each robot time uses, keyed by robot time; times with no frame in tolerance are absent."""
    joined = pedestrians_at(pl.Series("time_ns", time_ns.astype(np.int64)), peds, tolerance_ns)
    return {int(key[0]): frame for key, frame in joined.partition_by("time_ns", as_dict=True).items()}


def frames_of(peds: pl.DataFrame) -> list[pl.DataFrame]:
    """The pedestrians of each frame, frames in time order: the flat table partitioned by t_ns."""
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
