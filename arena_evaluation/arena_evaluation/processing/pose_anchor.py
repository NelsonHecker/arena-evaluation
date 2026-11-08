"""The ground-truth pose stream an episode is evaluated on: the robot's /tf track, classified by how densely it is sampled."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import polars as pl

if typing.TYPE_CHECKING:
    from ..storage.schemas import TopicBundle

DENSE_GAP_NS = 200_000_000


@dataclasses.dataclass(frozen=True)
class PoseSource:
    """Which ground-truth stream an episode was evaluated on: tf, tf_sparse, or absent."""

    kind: str
    samples: int


def _collect(frame: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _windowed(frame: pl.DataFrame | None, window: tuple[int | None, int | None]) -> pl.DataFrame | None:
    """Rows inside the episode window, on log time."""
    if frame is None:
        return None
    start_ns, end_ns = window
    if start_ns is not None:
        frame = frame.filter(pl.col("t_ns") >= start_ns)
    if end_ns is not None:
        frame = frame.filter(pl.col("t_ns") <= end_ns)
    return frame


def classify(tracks: pl.DataFrame | pl.LazyFrame | None) -> str:
    """dense | sparse | absent, on the median gap between ground-truth stamps."""
    df = _collect(tracks)
    if df is None or len(df) == 0:
        return "absent"
    if len(df) < 2:
        return "sparse"
    gaps = np.diff(np.sort(df["t_ns"].to_numpy()))
    return "dense" if float(np.median(gaps)) <= DENSE_GAP_NS else "sparse"  # pyright: ignore[reportCallIssue, reportArgumentType]


def resolve_pose_source(bundle: TopicBundle, window: tuple[int | None, int | None]) -> tuple[pl.DataFrame | None, PoseSource]:
    """The robot's ground-truth track inside the window. Odometry is never a trajectory source."""
    tracks = _windowed(_collect(bundle.robot_tracks), window)
    if tracks is not None and len(tracks) > 0:
        kind = "tf" if classify(tracks) == "dense" else "tf_sparse"
        return tracks, PoseSource(kind, len(tracks))
    return None, PoseSource("absent", 0)
