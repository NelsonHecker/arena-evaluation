"""Unit tests for the CharacterizationCalculator (ecological metric).

Pure Python + Polars, no ROS, no Gazebo required.
"""

import polars as pl

from arena_evaluation.processing.metrics.ecological.characterization import (
    CharacterizationCalculator,
    _leq_power,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def _calc() -> CharacterizationCalculator:
    return CharacterizationCalculator(RobotParams.load("jackal"))


def test_output_keys_and_units():
    calc = _calc()
    keys = calc.output_keys()
    assert "timeseries_char_power_total_w" in keys
    assert "timeseries_char_phase_kind" in keys
    assert calc.UNITS["timeseries_char_power_total_w"] == "W"
    assert calc.UNITS["timeseries_char_dba"] == "dBA"
    assert len(keys) == len(set(keys))


def test_leq_power():
    import math

    s = pl.Series([50.0, 60.0])
    mean_power = float(_leq_power(s).mean())
    assert abs(10.0 * math.log10(mean_power) - 10.0 * math.log10((10**5 + 10**6) / 2)) < 1e-6


def _sample_episode() -> AlignedEpisodeBundle:
    """A small aligned frame: odom + power + joints + phase labels."""
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "pos_x": [0.0, 0.25, 0.75, 1.75],
            "pos_y": [0.0, 0.0, 0.0, 0.0],
            "vel_linear": [0.0, 0.25, 0.5, 1.0],
            "total_power_w": [50.0, 55.0, 60.0, 65.0],
            "total_level_af_dba": [42.0, 46.0, 51.0, 56.0],
            "velocity": [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]],
            "effort": [[0.1, 0.1], [0.5, -0.5], [1.0, 1.0], [1.0, -1.0]],
            "label": ["linear_vx_0.25", "linear_vx_0.25", "linear_vx_0.50", "linear_vx_0.50"],
        }
    )
    return AlignedEpisodeBundle(
        episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal"
    )


def test_calculate_produces_timeseries():
    calc = _calc()
    out = calc.calculate(_sample_episode(), {})
    assert out["timeseries_char_time_s"] == [0.0, 1.0, 2.0, 3.0]
    assert out["timeseries_char_power_total_w"] == [50.0, 55.0, 60.0, 65.0]
    assert out["timeseries_char_phase_kind"] == ["linear", "linear", "linear", "linear"]
    assert out["timeseries_char_vx_target"] == [0.25, 0.25, 0.5, 0.5]
    # P_mech = Sigma|tau*omega|: 2.0*0.5 + 2.0*0.5 = 2.0 at t=1s
    assert out["timeseries_char_power_mech_w"][1] == 2.0
    # Energy intensity = p*dt/ds: 55 W * 1s / 0.25 m = 220 J/m at t=1s
    assert out["timeseries_char_energy_intensity"][1] == 220.0
    assert len(out["timeseries_char_dba"]) == 4


def test_calculate_missing_power_and_acoustics():
    df = _sample_episode().data.drop(["total_power_w", "total_level_af_dba"])
    ep = AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    # Power degrades to 0; acoustics falls back to the joint model (non-null).
    assert out["timeseries_char_power_total_w"] == [0.0, 0.0, 0.0, 0.0]
    assert all(v is not None for v in out["timeseries_char_dba"])


def test_calculate_empty_returns_none_keys():
    ep = AlignedEpisodeBundle(
        episode_id=0, data=pl.DataFrame(), start_pos=[], goal_pos=[], robot_name="env_0_jackal"
    )
    out = _calc().calculate(ep, {})
    assert all(v is None for v in out.values())
