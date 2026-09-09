"""Teleport detection on a recorded robot pose track."""
from __future__ import annotations

import numpy as np

MIN_JUMP_M = 0.5
MAX_SPEED_MPS = 5.0  # above any ground robot in the fleet; a reset moves the robot within one sample


def teleport_jumps(x, y, time_ns=None) -> np.ndarray:
    """Indices ``i`` where the step from sample ``i`` to ``i+1`` is a teleport."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return np.array([], dtype=np.int64)
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    big = dists > MIN_JUMP_M
    if time_ns is None:
        return np.where(big)[0]
    t = np.asarray(time_ns, dtype=np.float64)
    if len(t) != len(x):
        return np.where(big)[0]
    dt = np.diff(t) / 1e9
    speed = np.full_like(dists, np.inf)
    ok = dt > 0
    speed[ok] = dists[ok] / dt[ok]
    return np.where(big & (speed > MAX_SPEED_MPS))[0]
