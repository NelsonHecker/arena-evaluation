"""Unit tests for the characterization calculator's pure vectorized helpers.

Pure Python + Polars — no ROS, no Gazebo required.
"""

import math

import polars as pl
import pytest

from arena_evaluation.characterization.ecological_characterization_calculator import (
    add_distance,
    add_mechanical_power,
    attach_phase_labels,
    leq_db,
    summarize_samples,
)
from arena_evaluation.characterization.maneuvers import Phase, PhaseKind, build_schedule


def test_leq_db():
    assert leq_db(pl.Series([50.0, 60.0])) == pytest.approx(
        10.0 * math.log10((10**5 + 10**6) / 2), abs=1e-6
    )
    # A single constant level maps to itself
    assert leq_db(pl.Series([55.0, 55.0])) == pytest.approx(55.0)
    assert math.isnan(leq_db(pl.Series([], dtype=pl.Float64)))


def test_add_distance():
    df = pl.DataFrame({"time_ns": [0, 1, 2], "pos_x": [0.0, 3.0, 3.0], "pos_y": [0.0, 4.0, 4.0]})
    out = add_distance(df)
    assert out["ds_m"].to_list() == pytest.approx([0.0, 5.0, 0.0])


def test_add_mechanical_power():
    df = pl.DataFrame(
        {
            "time_ns": [0, 1],
            "joint_velocity": [[2.0, 2.0], [0.0, 0.0]],
            "joint_effort": [[1.0, -3.0], [0.0, 0.0]],
        }
    )
    out = add_mechanical_power(df)
    # P = |1|*2 + |-3|*2 = 8 W
    assert out["p_mech_w"].to_list() == pytest.approx([8.0, 0.0])


def test_attach_phase_labels_carry_forward():
    odom = pl.DataFrame(
        {
            "time_ns": [10, 20, 30, 40],
            "cmd_linear_x": [1.0, 1.0, 0.0, 0.0],
            "cmd_angular_z": [0.0, 0.0, 0.0, 0.0],
        }
    )
    markers = pl.DataFrame({"time_ns": [5, 25], "label": ["linear_vx_1.00", "idle_mid"]})
    out = attach_phase_labels(odom, markers, build_schedule())
    assert out["phase_label"].to_list() == ["linear_vx_1.00", "linear_vx_1.00", "idle_mid", "idle_mid"]
    assert out["phase_kind"].to_list() == ["linear", "linear", "idle", "idle"]
    assert out["vx_target"].to_list() == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_attach_phase_labels_fallback_without_markers():
    odom = pl.DataFrame(
        {
            "time_ns": [10, 20, 30],
            "cmd_linear_x": [0.0, 1.5, 0.0],
            "cmd_angular_z": [0.0, 0.0, -1.0],
        }
    )
    out = attach_phase_labels(odom, None, build_schedule())
    kinds = out["phase_kind"].to_list()
    assert kinds == ["idle", "linear", "angular"]
    assert out["vx_target"].to_list() == pytest.approx([0.0, 1.5, 0.0])
    assert out["wz_target"].to_list() == pytest.approx([0.0, 0.0, -1.0])


def test_summarize_samples_energy_intensity():
    # 100 W for 5 s over 10 m → 500 J → 50 J/m
    df = pl.DataFrame(
        {
            "phase_kind": ["linear", "linear", "linear", "linear", "linear"],
            "vx_target": [1.0] * 5,
            "wz_target": [0.0] * 5,
            "p_total_w": [100.0] * 5,
            "p_mech_w": [80.0] * 5,
            "dba": [60.0] * 5,
            "_l_power": [10**6] * 5,
            "_e_total_j": [100.0] * 5,
            "_e_mech_j": [80.0] * 5,
            "ds_m": [2.0] * 5,
            "dt_s": [1.0] * 5,
            "vx_achieved": [1.0] * 5,
        }
    )
    out = summarize_samples(df)
    assert len(out) == 1
    row = out.row(0, named=True)
    assert row["energy_intensity_j_per_m"] == pytest.approx(50.0)
    assert row["mech_energy_intensity_j_per_m"] == pytest.approx(40.0)
    assert row["leq_af_dba"] == pytest.approx(60.0)
    assert row["lafmax_af_dba"] == pytest.approx(60.0)
    assert row["mean_power_total_w"] == pytest.approx(100.0)
