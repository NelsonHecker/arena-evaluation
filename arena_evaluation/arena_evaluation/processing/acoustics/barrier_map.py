"""Acoustic barrier grid derived from the world model.

The grid is built from the zones (rooms), wall segments and doors of a world, not from the rendered map PNG. Walls
and doors therefore sit exactly where the world puts them. Furniture and other static objects are not part of it.

Conventions, one rule for rooms, walls, doors and agent positions:

- Cell (row, col) covers [col, col + 1) x [row, row + 1) cell widths from the lattice origin, which is anchored at the
  world origin, so cell edges fall on multiples of ``resolution``.
- ``origin`` is the world position of the centre of cell (0, 0). The solver and the door masks round a position to
  the nearest cell centre; on this lattice that is the cell that contains the position.
- A room cell is one whose centre lies inside a zone polygon.
- A wall on a lattice line takes the cell on its high side only where rooms lie on both sides of it (touching rooms
  and partitions). Elsewhere the room outline already makes the neighbouring cells barrier.
- A wall line between cell edges takes the cells it crosses.
- A door cell is one whose centre lies inside the door rectangle (span by width). Doors override walls.
- Two free cells that meet only at a corner are closed by a barrier cell, because the solver moves diagonally
  without checking the two cells beside the corner.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import numpy as np

_MICRO_PER_METRE = 1_000_000
_EDGE_TOLERANCE_M = 1e-9
_SLOPED_SAMPLES_PER_CELL = 8


@dataclasses.dataclass(frozen=True, slots=True)
class WallSegment:
    start: tuple[float, float]
    end: tuple[float, float]


@dataclasses.dataclass(frozen=True, slots=True)
class DoorOpening:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    width: float


@dataclasses.dataclass(frozen=True, slots=True)
class BarrierMap:
    """Barrier grid in solver orientation: row 0 is the lowest y, col 0 the lowest x."""

    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float]
    door_masks: dict[str, np.ndarray]


def derive_barrier_map(
    *,
    rooms: Sequence[Sequence[tuple[float, float]]],
    walls: Iterable[WallSegment],
    doors: Sequence[DoorOpening],
    resolution: float = 0.05,
    padding_cells: int = 5,
) -> BarrierMap:
    cell_um = _cell_micrometres(resolution)
    wall_list = list(walls)
    door_names = [door.name for door in doors]
    if len(set(door_names)) != len(door_names):
        raise ValueError(f"door names must be unique, got {sorted(door_names)}")

    points_um = [
        *(_to_micrometres(point) for room in rooms for point in room),
        *(_to_micrometres(corner) for door in doors for corner in _door_corners(door)),
        *(_to_micrometres(point) for wall in wall_list for point in (wall.start, wall.end)),
    ]
    if not points_um:
        raise ValueError("no geometry to derive a barrier map from")
    col0 = min(x for x, _ in points_um) // cell_um - padding_cells
    row0 = min(y for _, y in points_um) // cell_um - padding_cells
    col1 = _ceil_div(max(x for x, _ in points_um), cell_um) + padding_cells
    row1 = _ceil_div(max(y for _, y in points_um), cell_um) + padding_cells

    centres_x = (np.arange(col0, col1) + 0.5) * resolution
    centres_y = (np.arange(row0, row1) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(centres_x, centres_y)

    room_cells = np.zeros(grid_x.shape, dtype=bool)
    for room in rooms:
        room_cells |= _inside_polygon(grid_x, grid_y, np.asarray(room, dtype=float))

    barrier = ~room_cells
    for wall in wall_list:
        _mark_wall(barrier, room_cells, wall, cell_um=cell_um, col0=col0, row0=row0)

    door_masks: dict[str, np.ndarray] = {}
    door_union = np.zeros(grid_x.shape, dtype=bool)
    for door in doors:
        mask = _door_mask(grid_x, grid_y, door)
        door_masks[door.name] = mask
        door_union |= mask
    barrier &= ~door_union

    _close_corner_leaks(barrier, protected=door_union)

    return BarrierMap(
        occupancy=barrier.astype(np.uint8),
        resolution=resolution,
        origin=((col0 + 0.5) * resolution, (row0 + 0.5) * resolution),
        door_masks=door_masks,
    )


def _cell_micrometres(resolution: float) -> int:
    cell_um = round(resolution * _MICRO_PER_METRE)
    if cell_um <= 0 or abs(cell_um - resolution * _MICRO_PER_METRE) > 1e-6:
        raise ValueError(f"resolution {resolution} m must be a whole number of micrometres")
    return cell_um


def _to_micrometres(point: tuple[float, float]) -> tuple[int, int]:
    return round(point[0] * _MICRO_PER_METRE), round(point[1] * _MICRO_PER_METRE)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def _door_corners(door: DoorOpening) -> tuple[tuple[float, float], ...]:
    start, end, half_width = np.asarray(door.start, dtype=float), np.asarray(door.end, dtype=float), door.width / 2
    direction = (end - start) / float(np.hypot(*(end - start)))
    across = np.array([-direction[1], direction[0]]) * half_width
    return tuple(tuple(point) for point in (start + across, start - across, end - across, end + across))


def _inside_polygon(grid_x: np.ndarray, grid_y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(grid_x.shape, dtype=bool)
    xs, ys = polygon[:, 0], polygon[:, 1]
    next_xs, next_ys = np.roll(xs, -1), np.roll(ys, -1)
    for x_a, y_a, x_b, y_b in zip(xs, ys, next_xs, next_ys):
        if y_a == y_b:
            continue
        crosses = (y_a > grid_y) != (y_b > grid_y)
        x_at_height = (x_b - x_a) * (grid_y - y_a) / (y_b - y_a) + x_a
        inside ^= crosses & (grid_x < x_at_height)
    return inside


def _mark_wall(barrier: np.ndarray, room_cells: np.ndarray, wall: WallSegment, *, cell_um: int, col0: int, row0: int) -> None:
    (x_a, y_a), (x_b, y_b) = _to_micrometres(wall.start), _to_micrometres(wall.end)
    if (x_a, y_a) == (x_b, y_b):
        return
    if x_a == x_b:
        index = x_a // cell_um - col0
        rows = slice(min(y_a, y_b) // cell_um - row0, _ceil_div(max(y_a, y_b), cell_um) - row0)
        if x_a % cell_um == 0:
            barrier[rows, index] |= room_cells[rows, index - 1] & room_cells[rows, index]
        else:
            barrier[rows, index] = True
    elif y_a == y_b:
        index = y_a // cell_um - row0
        cols = slice(min(x_a, x_b) // cell_um - col0, _ceil_div(max(x_a, x_b), cell_um) - col0)
        if y_a % cell_um == 0:
            barrier[index, cols] |= room_cells[index - 1, cols] & room_cells[index, cols]
        else:
            barrier[index, cols] = True
    else:
        _mark_sloped_wall(barrier, (x_a, y_a), (x_b, y_b), cell_um=cell_um, col0=col0, row0=row0)


def _mark_sloped_wall(
    barrier: np.ndarray,
    start_um: tuple[int, int],
    end_um: tuple[int, int],
    *,
    cell_um: int,
    col0: int,
    row0: int,
) -> None:
    steps = max(abs(end_um[0] - start_um[0]), abs(end_um[1] - start_um[1])) // (cell_um // _SLOPED_SAMPLES_PER_CELL) + 1
    for step in range(steps + 1):
        x_um = start_um[0] + (end_um[0] - start_um[0]) * step // steps
        y_um = start_um[1] + (end_um[1] - start_um[1]) * step // steps
        barrier[y_um // cell_um - row0, x_um // cell_um - col0] = True


def _door_mask(grid_x: np.ndarray, grid_y: np.ndarray, door: DoorOpening) -> np.ndarray:
    start = np.asarray(door.start, dtype=float)
    span = np.asarray(door.end, dtype=float) - start
    length = float(np.hypot(*span))
    if length == 0.0:
        raise ValueError(f"door {door.name!r} has zero span")
    direction = span / length
    along = (grid_x - start[0]) * direction[0] + (grid_y - start[1]) * direction[1]
    across = (grid_y - start[1]) * direction[0] - (grid_x - start[0]) * direction[1]
    return (
        (along >= -_EDGE_TOLERANCE_M)
        & (along <= length + _EDGE_TOLERANCE_M)
        & (np.abs(across) <= door.width / 2 + _EDGE_TOLERANCE_M)
    )


def _close_corner_leaks(barrier: np.ndarray, *, protected: np.ndarray) -> None:
    """Close the free cell of every 2x2 checkerboard whose free pair touches only at the corner.

    Door cells are never closed; they are the openings the walls leave on purpose.
    """
    while True:
        south_west, south_east = barrier[:-1, :-1], barrier[:-1, 1:]
        north_west, north_east = barrier[1:, :-1], barrier[1:, 1:]
        close_north_east = ~south_west & ~north_east & north_west & south_east & ~protected[1:, 1:]
        close_south_east = ~north_west & ~south_east & south_west & north_east & ~protected[:-1, 1:]
        if not (close_north_east.any() or close_south_east.any()):
            return
        barrier[1:, 1:] |= close_north_east
        barrier[:-1, 1:] |= close_south_east
