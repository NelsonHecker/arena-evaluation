import dataclasses

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.base import MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.ecological.energy import EnergyMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _episode(**columns):
    n = len(next(iter(columns.values())))
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": [i * SEC for i in range(n)], **columns}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0],
    )


@pytest.fixture
def calc():
    return EnergyMetricCalculator(RobotParams(robot_radius=0.25))


def test_missing_power_and_battery_columns_give_null_metrics(calc):
    results = calc.calculate(_episode(vel_linear=[0.1, 0.2, 0.3]), {})
    for key in (
        "energy_static_wh",
        "energy_mechanical_wh",
        "energy_thermal_wh",
        "energy_total_wh",
        "power_peak_w",
        "battery_soc_final",
        "battery_soc_drop_pct",
    ):
        assert results[key] is None, key
    assert set(results) == set(calc.output_keys())


def test_all_null_power_column_gives_null_metrics(calc):
    nulls = pl.Series([None, None, None], dtype=pl.Float64)
    results = calc.calculate(_episode(total_power_w=nulls, battery_soc_percent=nulls), {})
    assert results["energy_total_wh"] is None
    assert results["power_peak_w"] is None
    assert results["battery_soc_final"] is None


def test_leading_power_nulls_are_back_filled(calc):
    power = pl.Series([None, 36.0, 72.0], dtype=pl.Float64)
    results = calc.calculate(_episode(total_power_w=power), {})
    assert results["energy_total_wh"] == pytest.approx((36.0 + 72.0) / 3600.0)
    assert results["power_peak_w"] == pytest.approx(72.0)
    assert results["energy_static_wh"] is None


def test_cumulative_energy_topic_wins_without_power(calc):
    results = calc.calculate(_episode(total_energy_consumed_wh=[1.0, 1.5, 2.25]), {})
    assert results["energy_total_wh"] == pytest.approx(1.25)
    assert results["power_peak_w"] is None


def test_series_table_is_flat_one_row_per_sample(calc):
    power = pl.Series([None, 36.0, 72.0], dtype=pl.Float64)
    table = calc.series_table(_episode(total_power_w=power, vel_linear=[0.1, 0.2, 0.3]), run_id="run", robot_id="robot")
    expected = pl.DataFrame(
        {
            "run_id": ["run"] * 3,
            "episode_id": [1] * 3,
            "robot_id": ["robot"] * 3,
            "t_ns": [0, SEC, 2 * SEC],
            "time_s": [0.0, 1.0, 2.0],
            "power_total_w": [36.0, 36.0, 72.0],
            "power_static_w": [None, None, None],
            "power_mechanical_w": [None, None, None],
            "power_thermal_w": [None, None, None],
            "battery_soc_pct": [None, None, None],
            "velocity_linear_mps": [0.1, 0.2, 0.3],
        },
        schema={
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "time_s": pl.Float64,
            "power_total_w": pl.Float64,
            "power_static_w": pl.Float64,
            "power_mechanical_w": pl.Float64,
            "power_thermal_w": pl.Float64,
            "battery_soc_pct": pl.Float64,
            "velocity_linear_mps": pl.Float64,
        },
    )
    assert_frame_equal(table, expected, check_dtypes=True, rel_tol=1e-12)


def test_series_table_without_data_keeps_the_schema(calc):
    empty = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    assert len(calc.series_table(empty)) == 0
    assert calc.series_table(empty).schema == pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "time_s": pl.Float64,
            "power_total_w": pl.Float64,
            "power_static_w": pl.Float64,
            "power_mechanical_w": pl.Float64,
            "power_thermal_w": pl.Float64,
            "battery_soc_pct": pl.Float64,
            "velocity_linear_mps": pl.Float64,
        }
    )


# --- ENCOUNTER scope ---------------------------------------------------------


def _encounter_episode() -> AlignedEpisodeBundle:
    times = [0, SEC, 2 * SEC]
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": times}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[2.0, 0.0],
        topics={"robot_power": pl.DataFrame({"t_ns": times, "total_power_w": [36.0, 36.0, 36.0]})},
    )


def _encounter_slice(episode: AlignedEpisodeBundle) -> ScopeSlice:
    window = dataclasses.replace(episode, data=pl.DataFrame())
    return ScopeSlice(
        scope=MetricScope.ENCOUNTER,
        slice_id="enc-001",
        parent_id=None,
        t_start_ns=0,
        t_end_ns=2 * SEC,
        identity={},
        window_bundle=window,
        full_episode=episode,
    )


def test_encounter_scope_emits_every_key_from_the_window_power(calc):
    episode = _encounter_episode()
    encounter_slice = _encounter_slice(episode)
    result = calc.calculate_for_scope(encounter_slice.window_bundle, {}, encounter_slice)
    assert set(result) == set(calc.output_keys())
    assert result["energy_total_wh"] == pytest.approx(36.0 * 2.0 / 3600.0)
    assert result["power_peak_w"] == pytest.approx(36.0)
    assert result["energy_static_wh"] is None
