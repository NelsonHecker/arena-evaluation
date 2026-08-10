# RECREATED 2026-08-10 from session context (original file lost in workspace deletion).
# Path: src/Arena/arena_evaluation/arena_evaluation/arena_evaluation/processing/acoustics/impedance_grid.py
import ctypes
import os
import subprocess
import time
from pathlib import Path
import numpy as np

# Find the C++ source file
_SRC_DIR = Path(__file__).parent
_CPP_FILE = _SRC_DIR / "solver.cpp"
_SO_FILE = _SRC_DIR / "solver.so"

def _compile_solver():
    if _SO_FILE.exists():
        # Check if cpp is newer
        if _CPP_FILE.stat().st_mtime <= _SO_FILE.stat().st_mtime:
            return

    print(f"Compiling C++ acoustic solver: {_CPP_FILE} -> {_SO_FILE}")
    cmd = [
        "g++",
        "-O3",
        "-ffast-math",
        "-fPIC",
        "-shared",
        "-std=c++11",
        str(_CPP_FILE),
        "-o",
        str(_SO_FILE)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to compile C++ solver:\n{result.stderr}")

# Compile on import
try:
    _compile_solver()
    _lib = ctypes.CDLL(str(_SO_FILE))

    _lib.solve_acoustic_field.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'), # grid
        ctypes.c_int,    # width
        ctypes.c_int,    # height
        ctypes.c_float,  # resolution
        ctypes.c_float,  # start_x
        ctypes.c_float,  # start_y
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # target_xs
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # target_ys
        ctypes.c_int,    # num_targets
        ctypes.c_float,  # wall_tl
        ctypes.c_float,  # mic_distance
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # out_attenuations
    ]
    _lib.solve_acoustic_field.restype = None

except Exception as e:
    print(f"Warning: Failed to load C++ acoustic solver. It will not be available. {e}")
    _lib = None

def compute_attenuations(
    occupancy_grid: np.ndarray,
    resolution: float,
    start_x_px: float,
    start_y_px: float,
    target_xs_px: np.ndarray,
    target_ys_px: np.ndarray,
    wall_tl: float = 47.0,
    mic_distance: float = 1.0
) -> np.ndarray:
    """
    Computes the shortest path acoustic attenuation from a start pixel to multiple target pixels.

    Args:
        occupancy_grid: 2D uint8 numpy array (0 = free, >0 = wall).
        resolution: Physical size of a pixel in meters.
        start_x_px, start_y_px: Start position in pixel coordinates.
        target_xs_px, target_ys_px: 1D arrays of target positions in pixel coordinates.
        wall_tl: Transmission loss added per wall crossing (dB).
        mic_distance: Minimum distance epsilon for self-noise (m).

    Returns:
        1D array of attenuations in dB (same length as targets).
    """
    if _lib is None:
        raise RuntimeError("C++ solver library not loaded.")

    if not isinstance(occupancy_grid, np.ndarray) or occupancy_grid.dtype != np.uint8:
        occupancy_grid = np.ascontiguousarray(occupancy_grid, dtype=np.uint8)

    if not isinstance(target_xs_px, np.ndarray) or target_xs_px.dtype != np.float32:
        target_xs_px = np.ascontiguousarray(target_xs_px, dtype=np.float32)

    if not isinstance(target_ys_px, np.ndarray) or target_ys_px.dtype != np.float32:
        target_ys_px = np.ascontiguousarray(target_ys_px, dtype=np.float32)

    num_targets = len(target_xs_px)
    if len(target_ys_px) != num_targets:
        raise ValueError("target_xs_px and target_ys_px must have the same length.")

    out_attenuations = np.empty(num_targets, dtype=np.float32)

    height, width = occupancy_grid.shape

    _lib.solve_acoustic_field(
        occupancy_grid,
        width,
        height,
        float(resolution),
        float(start_x_px),
        float(start_y_px),
        target_xs_px,
        target_ys_px,
        num_targets,
        float(wall_tl),
        float(mic_distance),
        out_attenuations
    )

    return out_attenuations


def downsample_occupancy(grid: np.ndarray, ds: int) -> np.ndarray:
    """Downsample an occupancy grid keeping walls (max-pool over ds x ds windows).

    Plain strided slicing (grid[::ds, ::ds]) drops 1-px walls entirely; max-pool
    keeps a wall if ANY sub-pixel is wall, preserving thin structures.
    Returns a C-contiguous uint8 array for the C++ solver.
    """
    if ds <= 1:
        return grid
    h, w = grid.shape
    h2, w2 = h // ds, w // ds
    g = grid[: h2 * ds, : w2 * ds].reshape(h2, ds, w2, ds)
    pooled = g.max(axis=(1, 3)).astype(np.uint8)
    return np.ascontiguousarray(pooled)
