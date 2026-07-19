import numpy as np
import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    ComplianceMetricsCalculator,
    _DoorGeometry,
    _ZoneGeometry,
    _door_open_series,
    _extract_door_geometry,
    _extract_zone_geometry,
    _is_open_at,
    _quiet_zone_dwell,
    _restricted_zone_entries,
    _speed_zone_metrics,
    _zone_membership,
)


def _calc():
    return ComplianceMetricsCalculator(RobotParams(0.2, 0.0, 10.0))


def _start_pos(data):
    if data is not None and "pos_x" in data.columns and len(data) > 0:
        return [data["pos_x"][0], data["pos_y"][0], data["yaw"][0]]
    return []


def _episode(data, semantic_events=None):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=_start_pos(data),
        goal_pos=[],
        semantic_events=semantic_events,
    )


def test_world_not_threaded_returns_none_defaults():
    calc = _calc()
    assert calc.world is None

    results = calc.calculate(
                _episode(pl.DataFrame({"time_ns": [0, 1_000_000_000], "pos_x": [0.0, 1.0], "pos_y": [0.0, 1.0], "yaw": [0.0, 0.0]})), {}
            )

    assert all(v is None for v in results.values())
    assert set(results) == set(calc.output_keys())


def test_world_not_locally_present_returns_none_and_warns():
    import logging

    from arena_evaluation.processing.metrics.ecological import compliance_metrics as cm

    records: list[logging.LogRecord] = []

    class _CollectingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _CollectingHandler()
    cm.logger.addHandler(handler)
    try:
        calc = _calc()
        calc.world = "definitely_nonexistent_world_xyz"
        results = calc.calculate(
                _episode(pl.DataFrame({"time_ns": [0, 1_000_000_000], "pos_x": [0.0, 1.0], "pos_y": [0.0, 1.0], "yaw": [0.0, 0.0]})), {}
            )
    finally:
        cm.logger.removeHandler(handler)

    assert all(v is None for v in results.values())
    assert any("definitely_nonexistent_world_xyz" in record.getMessage() for record in records)


def test_zone_membership_boundary_inclusive():
    import shapely

    square = shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])
    zones = [_ZoneGeometry(name="lobby", polygon=square, max_speed=None, quiet=False, restricted=False)]

    pos_x = np.array([4.0, 5.0])
    pos_y = np.array([2.0, 2.0])

    idx = _zone_membership(pos_x, pos_y, zones)

    assert idx[0] == 0  # on the edge, covers() counts as inside
    assert idx[1] == -1


def test_speed_zone_violations_counts_contiguous_runs():
    zones = [_ZoneGeometry(name="corridor", polygon=None, max_speed=1.0, quiet=False, restricted=False)]
    zone_idx = np.array([0, 0, 0, 0, 0, 0])
    speed = np.array([0.5, 1.5, 1.5, 0.5, 2.0, 0.5])
    dt = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    violations, seconds = _speed_zone_metrics(zone_idx, speed, dt, zones)

    assert violations == 2
    assert seconds == pytest.approx(3.0)


def test_speed_zone_violations_ignore_zones_without_max_speed():
    zones = [_ZoneGeometry(name="lobby", polygon=None, max_speed=None, quiet=True, restricted=False)]
    zone_idx = np.array([0, 0, 0])
    speed = np.array([5.0, 5.0, 5.0])
    dt = np.array([1.0, 1.0])

    violations, seconds = _speed_zone_metrics(zone_idx, speed, dt, zones)

    assert violations == 0
    assert seconds == pytest.approx(0.0)


def test_quiet_zone_dwell_seconds_sums_only_quiet_zones():
    zones = [
        _ZoneGeometry(name="ward", polygon=None, max_speed=None, quiet=True, restricted=False),
        _ZoneGeometry(name="corridor", polygon=None, max_speed=None, quiet=False, restricted=False),
    ]
    zone_idx = np.array([0, 0, -1, 1])
    dt = np.array([2.0, 3.0, 4.0])

    seconds = _quiet_zone_dwell(zone_idx, dt, zones)

    assert seconds == pytest.approx(5.0)


def test_restricted_zone_entries_counts_transitions_not_dwell():
    zones = [_ZoneGeometry(name="pharmacy", polygon=None, max_speed=None, quiet=False, restricted=True)]
    # outside, enter, stay, leave, re-enter
    zone_idx = np.array([-1, 0, 0, -1, 0])

    entries = _restricted_zone_entries(zone_idx, zones)

    assert entries == 2


def test_restricted_zone_entries_ignores_start_inside():
    zones = [_ZoneGeometry(name="pharmacy", polygon=None, max_speed=None, quiet=False, restricted=True)]
    zone_idx = np.array([0, 0, 0])

    entries = _restricted_zone_entries(zone_idx, zones)

    assert entries == 0


def test_door_open_series_and_query():
    events = pl.DataFrame({
        "time_ns": [1_000_000_000, 3_000_000_000],
        "entity": ["env_0/door_1", "env_0/door_1"],
        "kind": ["door", "door"],
        "field": ["state", "state"],
        "current": ["opening", "open"],
    })

    series = _door_open_series(events)

    assert _is_open_at(series, "door_1", 0) is False  # seeded closed before first event
    assert _is_open_at(series, "door_1", 1_000_000_000) is False  # opening != open
    assert _is_open_at(series, "door_1", 3_500_000_000) is True
    assert _is_open_at(series, "unknown_door", 5_000_000_000) is False


def test_doorway_blocking_time_requires_stationary_and_open():
    doors = [_DoorGeometry(name="door_1", center_x=0.0, center_y=0.0, radius=1.0)]
    events = pl.DataFrame({
        "time_ns": [0, 2_000_000_000],
        "entity": ["env_0/door_1", "env_0/door_1"],
        "kind": ["door", "door"],
        "field": ["state", "state"],
        "current": ["open", "open"],
    })
    data = pl.DataFrame({
        "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        "pos_x": [0.0, 0.0, 0.0, 0.0],
        "pos_y": [0.0, 0.0, 0.0, 0.0],
        "yaw": [0.0, 0.0, 0.0, 0.0],
        "vel_linear": [0.0, 0.0, 1.0, 0.0],  # moving through [2s, 3s)
    })
    episode = _episode(data, semantic_events=events)

    calc = _calc()
    pos_x, pos_y, _yaw, _ox, _oy, _oyaw = calc.resolve_robot_pose(episode)
    speed = np.abs(episode.data["vel_linear"].to_numpy())
    time_ns = episode.data["time_ns"].to_numpy()

    total = calc._doorway_blocking_time(episode, pos_x, pos_y, speed, time_ns, doors)

    # stationary intervals [0,1) and [1,2) both qualify, [2,3) does not (moving)
    assert total == pytest.approx(2.0)


def test_doorway_blocking_degrades_to_zero_without_events():
    doors = [_DoorGeometry(name="door_1", center_x=0.0, center_y=0.0, radius=1.0)]
    data = pl.DataFrame({
        "time_ns": [0, 1_000_000_000],
        "pos_x": [0.0, 0.0],
        "pos_y": [0.0, 0.0],
        "yaw": [0.0, 0.0],
        "vel_linear": [0.0, 0.0],
    })
    episode = _episode(data, semantic_events=None)
    calc = _calc()

    pos_x, pos_y, _yaw, _ox, _oy, _oyaw = calc.resolve_robot_pose(episode)
    speed = np.abs(episode.data["vel_linear"].to_numpy())
    time_ns = episode.data["time_ns"].to_numpy()

    total = calc._doorway_blocking_time(episode, pos_x, pos_y, speed, time_ns, doors)

    assert total == pytest.approx(0.0)


def test_extract_zone_and_door_geometry_reads_world_assets():
    from arena_simulation_setup.shared.semantics import SemanticCfg
    from arena_simulation_setup.shared.world import Door
    from arena_simulation_setup.tree.World.World import LevelDescription
    from arena_simulation_setup.utils.geometry import Position

    door = Door(name="door_1", start=Position(5.0, 1.0), end=Position(5.0, 2.0), activation_distance=(1.5, 1.5))
    zone = LevelDescription.Zone(
        name="lobby",
        corners=[Position(0.0, 0.0), Position(0.0, 4.0), Position(4.0, 4.0), Position(4.0, 0.0)],
        doors=[door],
        semantics=[
            SemanticCfg(role="state", name="max_speed", value=1.0),
            SemanticCfg(role="predicate", name="quiet", value=True),
        ],
    )
    level = LevelDescription(zones=[zone])

    zones = _extract_zone_geometry(level)
    doors = _extract_door_geometry(level)

    assert len(zones) == 1
    assert zones[0].name == "lobby"
    assert zones[0].max_speed == pytest.approx(1.0)
    assert zones[0].quiet is True
    assert zones[0].restricted is False

    assert len(doors) == 1
    assert doors[0].name == "door_1"
    assert doors[0].center_x == pytest.approx(5.0)
    assert doors[0].center_y == pytest.approx(1.5)
    assert doors[0].radius == pytest.approx(1.5)


def test_extract_zone_geometry_skips_unannotated_zones():
    from arena_simulation_setup.tree.World.World import LevelDescription
    from arena_simulation_setup.utils.geometry import Position

    zone = LevelDescription.Zone(
        name="plain_room",
        corners=[Position(0.0, 0.0), Position(0.0, 2.0), Position(2.0, 2.0), Position(2.0, 0.0)],
    )
    level = LevelDescription(zones=[zone])

    assert _extract_zone_geometry(level) == []


def test_calculate_end_to_end_with_cached_world():
    from arena_simulation_setup.shared.semantics import SemanticCfg
    from arena_simulation_setup.tree.World.World import LevelDescription
    from arena_simulation_setup.utils.geometry import Position

    zone = LevelDescription.Zone(
        name="lobby",
        corners=[Position(0.0, 0.0), Position(0.0, 4.0), Position(4.0, 4.0), Position(4.0, 0.0)],
        semantics=[
            SemanticCfg(role="state", name="max_speed", value=1.0),
            SemanticCfg(role="predicate", name="quiet", value=True),
        ],
    )
    level = LevelDescription(zones=[zone])

    calc = _calc()
    calc.world = "synthetic_world"
    calc._world_cache["synthetic_world"] = (_extract_zone_geometry(level), _extract_door_geometry(level))

    data = pl.DataFrame({
        "time_ns": [0, 1_000_000_000, 2_000_000_000],
        "pos_x": [1.0, 1.0, 1.0],
        "pos_y": [1.0, 1.0, 1.0],
        "yaw": [0.0, 0.0, 0.0],
        "vel_linear": [0.5, 2.0, 0.5],  # over 1.0 m/s cap during [1s, 2s)
    })
    episode = _episode(data)

    results = calc.calculate(episode, {})

    assert results["speed_zone_violations"] == 1
    assert results["speed_zone_violation_seconds"] == pytest.approx(1.0)
    assert results["quiet_zone_dwell_seconds"] == pytest.approx(2.0)
    assert results["restricted_zone_entries"] == 0
    assert results["doorway_blocking_time"] == pytest.approx(0.0)


def test_offset_zones_and_doors_translate_geometry():
    import shapely

    from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
        _offset_doors,
        _offset_zones,
    )

    zone = _ZoneGeometry(
        name="z",
        polygon=shapely.Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),
        max_speed=None,
        quiet=False,
        restricted=True,
    )
    door = _DoorGeometry(name="d", center_x=1.0, center_y=1.0, radius=2.0)

    (z,) = _offset_zones([zone], (5.0, 5.0))
    (d,) = _offset_doors([door], (5.0, 5.0))
    assert z.polygon.covers(shapely.Point(6.0, 6.0))
    assert not z.polygon.covers(shapely.Point(1.0, 1.0))
    assert (d.center_x, d.center_y) == (6.0, 6.0)
    assert z.restricted is True and d.radius == 2.0

    assert _offset_zones([zone], (0.0, 0.0))[0] is zone
