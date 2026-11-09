"""Shipped manifests read the 1NF series tables through the Star Schema catalog."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.schemas import PlotSpec
from star_benchmark import write_benchmark

_MANIFEST_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs" / "benchmark" / "manifests"

SERIES_SCHEMAS: dict[str, pl.Schema] = {
    "characterization": pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "time_s": pl.Float64,
            "power_total_w": pl.Float64,
            "power_mech_w": pl.Float64,
            "dba": pl.Float64,
            "vx_achieved_mps": pl.Float64,
            "vy_achieved_mps": pl.Float64,
            "wz_achieved_radps": pl.Float64,
            "speed_achieved_mps": pl.Float64,
            "accel_achieved_mps2": pl.Float64,
            "accel_target_mps2": pl.Float64,
            "efficiency": pl.Float64,
            "phase_kind": pl.String,
            "vx_target_mps": pl.Float64,
            "vy_target_mps": pl.Float64,
            "wz_target_radps": pl.Float64,
            "speed_target_mps": pl.Float64,
            "turn_radius_m": pl.Float64,
            "energy_intensity_j_m": pl.Float64,
            "energy_per_rad_j_rad": pl.Float64,
            "leq_power_w": pl.Float64,
        }
    ),
    "energy": pl.Schema(
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
    ),
    "acoustic_exposure": pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "ped_id": pl.String,
            "exposure_dba": pl.Float64,
            "attenuation_db": pl.Float64,
        }
    ),
    "clearance": pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "obstacle_clearance_m": pl.Float64,
            "pedestrian_clearance_m": pl.Float64,
            "combined_clearance_m": pl.Float64,
        }
    ),
}


def _catalog(tmp_path: pathlib.Path) -> BenchmarkCatalog:
    write_benchmark(
        tmp_path,
        [{"episode_id": 0, "planner": "dwb", "robot_id": "robot_0", "stage": "s1", "world": "map_empty", "is_reference": False}],
    )
    metrics = tmp_path / "metrics"
    for name, schema in SERIES_SCHEMAS.items():
        pl.DataFrame(schema=schema).write_parquet(metrics / f"series_{name}.parquet")
    return BenchmarkCatalog.from_directory(tmp_path)


def _manifests() -> list[tuple[str, dict]]:
    return [(path.name, yaml.safe_load(path.read_text(encoding="utf-8"))) for path in sorted(_MANIFEST_DIR.glob("*.yaml"))]


def _plot_columns(spec: PlotSpec) -> set[str]:
    columns: set[str] = set()
    if spec.data_key and spec.data_key != "*":
        columns.add(spec.data_key)
    for channel in (spec.channels.x, spec.channels.y, spec.channels.color, spec.channels.size, spec.channels.symbol):
        if isinstance(channel, str) and channel:
            columns.add(channel)
        elif isinstance(channel, list):
            columns.update(str(item) for item in channel if item)
    options = spec.options or {}
    for key in ("x", "y"):
        value = options.get(key)
        if isinstance(value, str) and value:
            columns.add(value)
    for key in ("metrics", "columns"):
        for item in options.get(key) or []:
            if isinstance(item, dict):
                metric = item.get("metric")
                if metric:
                    columns.add(str(metric))
            elif isinstance(item, str):
                columns.add(item)
    columns.update(str(key) for key in (spec.filter or {}))
    columns.update(str(item) for item in (spec.group_by or []) if item)
    return columns


def test_shipped_manifests_reference_no_legacy_timeseries_columns() -> None:
    for name, manifest in _manifests():
        text = yaml.safe_dump(manifest)
        assert "timeseries_char_" not in text, name
        assert "timeseries_time_s" not in text, name
        assert "combined_metrics" not in text, name


def test_series_plots_bind_only_to_the_series_contract(tmp_path: pathlib.Path) -> None:
    catalog = _catalog(tmp_path)
    used_series_sources: set[str] = set()
    for name, manifest in _manifests():
        for raw_plot in manifest.get("plots") or []:
            spec = PlotSpec.model_validate(raw_plot)
            source = spec.data.source if spec.data and spec.data.source else (spec.data_source or manifest.get("data_source"))
            if not source or not str(source).startswith("v_series_"):
                continue
            used_series_sources.add(str(source))
            columns = set(catalog.get_table(str(source)).collect_schema().names())
            missing = sorted(column for column in _plot_columns(spec) if column not in columns)
            assert not missing, f"{name}:{spec.id} references {missing}"

    assert used_series_sources == {"v_series_characterization", "v_series_energy", "v_series_clearance"}


def test_every_series_table_has_a_joined_view(tmp_path: pathlib.Path) -> None:
    catalog = _catalog(tmp_path)
    for name in SERIES_SCHEMAS:
        frame = catalog.get_table(f"v_series_{name}").collect()
        assert "planner" in frame.columns
        assert "world" in frame.columns
        assert "local_planner" in frame.columns


@pytest.mark.parametrize("table", ["v_series_characterization", "v_series_energy", "v_series_clearance"])
def test_an_absent_series_table_has_no_joined_view(tmp_path: pathlib.Path, table: str) -> None:
    write_benchmark(tmp_path, [{"episode_id": 0, "planner": "dwb"}])
    catalog = BenchmarkCatalog.from_directory(tmp_path)
    assert table not in catalog.table_names()