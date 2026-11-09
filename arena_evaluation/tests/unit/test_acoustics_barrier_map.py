"""Unit tests for the world-derived acoustic barrier grid (acoustics/barrier_map.py).

Geometry is built on the 5 cm lattice, so every expected cell is exact. Rooms are 1 m squares unless a test says
otherwise. The derived grid pads each side by five cells; the room (0, 0)-(1, 1) therefore starts at column and
row 5.
"""

from __future__ import annotations

import numpy as np
import pytest

from arena_evaluation.processing.acoustics.barrier_map import (
    DoorOpening,
    WallSegment,
    derive_barrier_map,
)

RES = 0.05
PAD = 5
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _free_count(barrier_map) -> int:
    return int((barrier_map.occupancy == 0).sum())


def _reachable_8_connected(occupancy: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {start}
    frontier = [start]
    while frontier:
        row, col = frontier.pop()
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                nxt = (row + d_row, col + d_col)
                if nxt in seen:
                    continue
                if not (0 <= nxt[0] < occupancy.shape[0] and 0 <= nxt[1] < occupancy.shape[1]):
                    continue
                if occupancy[nxt]:
                    continue
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def test_room_cells_are_free_and_origin_is_the_first_cell_centre():
    barrier_map = derive_barrier_map(rooms=[SQUARE], walls=[], doors=[])

    assert _free_count(barrier_map) == 20 * 20
    assert barrier_map.origin == pytest.approx(((0 - PAD + 0.5) * RES, (0 - PAD + 0.5) * RES))
    assert barrier_map.occupancy[PAD + 10, PAD + 10] == 0


def test_rounding_a_position_inside_a_room_lands_on_a_free_cell():
    barrier_map = derive_barrier_map(rooms=[SQUARE], walls=[], doors=[])
    ox, oy = barrier_map.origin
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(0.0, 1.0, 500), rng.uniform(0.0, 1.0, 500)

    cols = np.floor((xs - ox) / RES + 0.5).astype(int)
    rows = np.floor((ys - oy) / RES + 0.5).astype(int)

    assert not barrier_map.occupancy[rows, cols].any()


def test_wall_between_touching_rooms_takes_one_cell_on_the_high_side():
    rooms = [SQUARE, [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)]]
    wall = WallSegment(start=(1.0, 0.0), end=(1.0, 1.0))

    barrier_map = derive_barrier_map(rooms=rooms, walls=[wall], doors=[])

    assert _free_count(barrier_map) == 2 * 400 - 20
    assert barrier_map.occupancy[PAD : PAD + 20, PAD + 20].all()
    assert not barrier_map.occupancy[PAD : PAD + 20, PAD + 19].any()


def test_wall_on_the_outside_of_a_room_takes_no_room_cells():
    wall = WallSegment(start=(0.0, 0.0), end=(0.0, 1.0))

    barrier_map = derive_barrier_map(rooms=[SQUARE], walls=[wall], doors=[])

    assert _free_count(barrier_map) == 20 * 20


def test_door_opens_the_wall_it_sits_in():
    rooms = [SQUARE, [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)]]
    walls = [WallSegment(start=(1.0, 0.0), end=(1.0, 0.4)), WallSegment(start=(1.0, 0.6), end=(1.0, 1.0))]
    door = DoorOpening(name="between", start=(1.0, 0.4), end=(1.0, 0.6), width=0.1)

    barrier_map = derive_barrier_map(rooms=rooms, walls=walls, doors=[door])
    mask = barrier_map.door_masks["between"]

    assert mask.sum() == 4 * 2
    assert not barrier_map.occupancy[mask].any()
    assert barrier_map.occupancy[PAD : PAD + 8, PAD + 20].all()
    assert barrier_map.occupancy[PAD + 12 : PAD + 20, PAD + 20].all()


def test_partition_inside_a_room_splits_it():
    room = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    partition = WallSegment(start=(1.0, 0.0), end=(1.0, 1.0))

    barrier_map = derive_barrier_map(rooms=[room], walls=[partition], doors=[])

    assert _free_count(barrier_map) == 40 * 20 - 20


def test_corner_touching_rooms_do_not_connect_diagonally():
    rooms = [SQUARE, [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]]

    barrier_map = derive_barrier_map(rooms=rooms, walls=[], doors=[])
    reachable = _reachable_8_connected(barrier_map.occupancy, (PAD + 10, PAD + 10))

    assert (PAD + 30, PAD + 30) not in reachable


def test_sloped_wall_marks_the_cells_it_crosses():
    diagonal = WallSegment(start=(0.0, 0.0), end=(1.0, 1.0))

    barrier_map = derive_barrier_map(rooms=[SQUARE], walls=[diagonal], doors=[])

    assert barrier_map.occupancy[PAD + 10, PAD + 10] == 1
    assert _free_count(barrier_map) < 20 * 20


def test_duplicate_door_names_are_rejected():
    door = DoorOpening(name="same", start=(0.5, 0.0), end=(0.6, 0.0), width=0.1)

    with pytest.raises(ValueError, match="unique"):
        derive_barrier_map(rooms=[SQUARE], walls=[], doors=[door, door])
