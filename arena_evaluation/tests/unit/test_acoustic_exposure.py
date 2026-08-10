# RECREATED 2026-08-10 from session context.
# Path: src/Arena/arena_evaluation/arena_evaluation/tests/unit/test_acoustic_exposure.py
import numpy as np
import pytest
from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations

def test_acoustics_free_space():
    # 10x10 empty grid
    grid = np.zeros((10, 10), dtype=np.uint8)
    resolution = 1.0 # 1 meter per pixel for simplicity

    # Start at (0, 0)
    sx, sy = 0.0, 0.0

    # Target at (3, 4) -> distance = 5m
    tx = np.array([3.0], dtype=np.float32)
    ty = np.array([4.0], dtype=np.float32)

    att = compute_attenuations(
        grid, resolution, sx, sy, tx, ty,
        wall_tl=47.0, mic_distance=1.0
    )

    # Dijkstra on 8-connected grid: 3 diag + 1 straight = 3*sqrt(2) + 1 = 5.2426
    expected_dist = 3.0 * np.sqrt(2) + 1.0
    expected = 20.0 * np.log10(expected_dist + 1.0)
    assert np.isclose(att[0], expected, atol=0.1)

def test_acoustics_one_wall():
    # 10x10 grid with a vertical wall at x=2
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 2] = 255 # wall

    resolution = 1.0
    sx, sy = 0.0, 0.0

    tx = np.array([3.0], dtype=np.float32)
    ty = np.array([4.0], dtype=np.float32)

    att = compute_attenuations(
        grid, resolution, sx, sy, tx, ty,
        wall_tl=47.0, mic_distance=1.0
    )

    # The shortest path goes straight through the wall.
    # Dist = 5m, walls = 1
    # Dijkstra on 8-connected grid: 3 diag + 1 straight = 5.2426
    expected_dist = 3.0 * np.sqrt(2) + 1.0
    expected = 20.0 * np.log10(expected_dist + 1.0) + 47.0
    assert np.isclose(att[0], expected, atol=0.1)

def test_acoustics_pruning_dominance():
    # Grid where going around the wall is cheaper than going through it
    grid = np.zeros((10, 10), dtype=np.uint8)
    # Wall from y=0 to y=6 at x=2
    grid[0:7, 2] = 255

    resolution = 1.0
    sx, sy = 0.0, 3.0
    tx = np.array([4.0], dtype=np.float32)
    ty = np.array([3.0], dtype=np.float32)

    att = compute_attenuations(
        grid, resolution, sx, sy, tx, ty,
        wall_tl=47.0, mic_distance=1.0
    )

    # Path 1: through the wall: distance = 4m, walls = 1. Cost = 20log10(5) + 47 = 60.97
    # Path 2: around the wall: dist is roughly 4 + 4 + 4 = 12m. Cost = 20log10(13) = 22.2
    # So going around is much cheaper! The solver should return the path around.

    # Manually calculate roughly the around distance: (0,3) -> (2,7) -> (4,3)
    # dist = sqrt(2^2 + 4^2) * 2 = 2 * sqrt(20) = 8.94
    expected_around_cost = 20.0 * np.log10(8.94 + 1.0)

    # The actual solver might find a slightly different path on the 8-connected grid
    # But it must be < 40 dB
    assert att[0] < 40.0
