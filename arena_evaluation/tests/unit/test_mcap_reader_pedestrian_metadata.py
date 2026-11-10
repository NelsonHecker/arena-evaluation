"""Pedestrian agent attributes come from the engine table while poses stay in the arena frame."""

from __future__ import annotations

import polars as pl

from arena_evaluation.processing.mcap_reader import _with_engine_pedestrian_metadata


def _arena() -> pl.LazyFrame:
    """Arena pedestrian frame: map-frame poses with the engine attributes present but empty."""
    return pl.DataFrame(
        {
            "t_ns": [0, 0, 1_000_000_000],
            "env_id": ["env_0"] * 3,
            "ped_id": ["Pedestrian_1", "Pedestrian_2", "Pedestrian_1"],
            "ped_name": ["Pedestrian_1", "Pedestrian_2", "Pedestrian_1"],
            "x": [1.0, 2.0, 1.5],
            "y": [0.0, 0.0, 0.0],
            "yaw": [0.0] * 3,
            "vx": [0.1, 0.2, 0.1],
            "vy": [0.0] * 3,
            "speed_mps": [0.1, 0.2, 0.1],
            "radius_m": [None] * 3,
            "desired_speed": [None] * 3,
            "policy": [None] * 3,
            "agent_type": ["adult", "adult", "adult"],
        }
    ).lazy()


def _engine() -> pl.LazyFrame:
    """Engine pedestrian frame: anchor-offset poses with the recorded attributes."""
    return pl.DataFrame(
        {
            "t_ns": [0, 0],
            "env_id": ["env_0", "env_0"],
            "ped_id": ["Pedestrian_1", "Pedestrian_2"],
            "ped_name": ["Pedestrian_1", "Pedestrian_2"],
            "x": [101.0, 102.0],
            "y": [50.0, 50.0],
            "yaw": [0.0, 0.0],
            "vx": [0.0, 0.0],
            "vy": [0.0, 0.0],
            "speed_mps": [0.0, 0.0],
            "radius_m": [0.3, 0.25],
            "desired_speed": [1.2, 1.1],
            "policy": ["sfm", "orca"],
            "agent_type": [None, None],
        }
    ).lazy()


def test_engine_metadata_fills_the_empty_arena_attributes() -> None:
    merged = _with_engine_pedestrian_metadata(_arena(), _engine()).collect()
    assert merged.get_column("policy").to_list() == ["sfm", "orca", "sfm"]
    assert merged.get_column("desired_speed").to_list() == [1.2, 1.1, 1.2]
    assert merged.get_column("radius_m").to_list() == [0.3, 0.25, 0.3]


def test_positions_and_velocities_stay_in_the_arena_frame() -> None:
    merged = _with_engine_pedestrian_metadata(_arena(), _engine()).collect()
    assert merged.get_column("x").to_list() == [1.0, 2.0, 1.5]
    assert merged.get_column("y").to_list() == [0.0, 0.0, 0.0]
    assert merged.get_column("vx").to_list() == [0.1, 0.2, 0.1]


def test_arena_values_win_when_they_are_recorded() -> None:
    merged = _with_engine_pedestrian_metadata(_arena(), _engine()).collect()
    assert merged.get_column("agent_type").to_list() == ["adult", "adult", "adult"]
