from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SECONDS_PER_HOUR: typing.Final[float] = 3600.0
NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
_TRACK_ALIGN_TOLERANCE_NS: typing.Final[int] = 100_000_000

SERIES_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
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

_POWER_COLUMNS: typing.Final[dict[str, str]] = {
    "power_total_w": "total_power_w",
    "power_static_w": "static_power_w",
    "power_mechanical_w": "total_mechanical_power_w",
    "power_thermal_w": "total_thermal_power_w",
}


def _forward_backward_fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaN gaps then back-fill the leading ones; an all-NaN array is returned unchanged."""
    missing = np.isnan(values)
    if missing.all():
        return values
    indices = np.where(~missing, np.arange(missing.shape[0]), 0)
    np.maximum.accumulate(indices, out=indices)
    filled = values[indices]
    remaining = np.isnan(filled)
    if np.any(remaining):
        first_valid = np.flatnonzero(~missing)[0]
        filled[remaining] = values[first_valid]
    return filled


class EnergyMetricCalculator(BaseMetricCalculator):
    NAME = "energy"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("robot_power", "robot_energy", "robot_odom")]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})

    UNITS = {
        "energy_static_wh": "Wh",
        "energy_mechanical_wh": "Wh",
        "energy_thermal_wh": "Wh",
        "energy_total_wh": "Wh",
        "power_peak_w": "W",
        "battery_soc_final": "%",
        "battery_soc_drop_pct": "%",
    }

    PRIMARY_OUTPUTS = ["energy_total_wh", "energy_mechanical_wh", "power_peak_w"]

    OUTPUT_DIRECTIONS = {"energy_total_wh": "lower", "energy_mechanical_wh": "lower", "power_peak_w": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "energy_static_wh",
            "energy_mechanical_wh",
            "energy_thermal_wh",
            "energy_total_wh",
            "power_peak_w",
            "battery_soc_final",
            "battery_soc_drop_pct",
        ]

    def _aligned_series(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Per-sample power/energy/velocity series on the aligned episode clock, every absent column all-null."""
        frame = episode.data
        if frame is None or len(frame) == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)

        time_ns = frame["time_ns"].cast(pl.Int64)
        columns: dict[str, pl.Series] = {}
        for target, source in _POWER_COLUMNS.items():
            values = frame[source].cast(pl.Float64).to_numpy() if source in frame.columns else np.full(len(frame), np.nan)
            columns[target] = pl.Series(target, _forward_backward_fill(values.astype(np.float64))).fill_nan(None)
        battery = frame["battery_soc_percent"].cast(pl.Float64).to_numpy() if "battery_soc_percent" in frame.columns else np.full(len(frame), np.nan)
        battery = _forward_backward_fill(battery.astype(np.float64))
        velocity = frame["vel_linear"].cast(pl.Float64).to_numpy() if "vel_linear" in frame.columns else np.full(len(frame), np.nan)
        columns["battery_soc_pct"] = pl.Series("battery_soc_pct", battery).fill_nan(None)
        columns["velocity_linear_mps"] = pl.Series("velocity_linear_mps", velocity).fill_nan(None)

        series = pl.DataFrame({"t_ns": time_ns, **columns})
        return series.with_columns(
            ((pl.col("t_ns").cast(pl.Float64) - pl.col("t_ns").cast(pl.Float64).min()) / NANOSECONDS_PER_SECOND).alias("time_s"),
            pl.lit("").alias("run_id"),
            pl.lit(int(episode.episode_id), dtype=pl.Int64).alias("episode_id"),
            pl.lit(episode.robot_name or "robot").alias("robot_id"),
        ).select(list(SERIES_SCHEMA))

    def _native_series(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Per-sample power/energy/velocity series from the native telemetry topics inside the scope window."""
        topics = self.native_topics(episode)
        power = topics.get("robot_power")
        if power is None or len(power) == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)

        power = power.select(
            pl.col("t_ns").cast(pl.Int64),
            pl.col("total_power_w").cast(pl.Float64),
            pl.col("static_power_w").cast(pl.Float64) if "static_power_w" in power.columns else pl.lit(None, dtype=pl.Float64).alias("static_power_w"),
            pl.col("total_mech_power_w").cast(pl.Float64) if "total_mech_power_w" in power.columns else pl.lit(None, dtype=pl.Float64).alias("total_mech_power_w"),
            pl.col("total_thermal_power_w").cast(pl.Float64) if "total_thermal_power_w" in power.columns else pl.lit(None, dtype=pl.Float64).alias("total_thermal_power_w"),
        ).sort("t_ns")

        for topic_name, columns in (
            ("robot_energy", ("total_energy_consumed_wh", "battery_soc_percent")),
            ("robot_tracks", ("speed_mps",)),
        ):
            table = topics.get(topic_name)
            if table is None or len(table) == 0:
                continue
            reference = table.select(pl.col("t_ns").cast(pl.Int64), *[pl.col(name).cast(pl.Float64) for name in columns if name in table.columns]).drop_nulls("t_ns").sort("t_ns")
            if len(reference.columns) > 1:
                power = power.join_asof(reference, on="t_ns", strategy="backward", tolerance=_TRACK_ALIGN_TOLERANCE_NS)

        rename = {
            "total_power_w": "power_total_w",
            "static_power_w": "power_static_w",
            "total_mech_power_w": "power_mechanical_w",
            "total_thermal_power_w": "power_thermal_w",
            "battery_soc_percent": "battery_soc_pct",
            "speed_mps": "velocity_linear_mps",
        }
        power = power.rename({source: target for source, target in rename.items() if source in power.columns})
        for column in SERIES_SCHEMA:
            if column not in power.columns and column not in ("run_id", "episode_id", "robot_id", "time_s"):
                power = power.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))

        return power.with_columns(
            ((pl.col("t_ns").cast(pl.Float64) - pl.col("t_ns").cast(pl.Float64).min()) / NANOSECONDS_PER_SECOND).alias("time_s"),
            pl.lit("").alias("run_id"),
            pl.lit(int(episode.episode_id), dtype=pl.Int64).alias("episode_id"),
            pl.lit(episode.robot_name or "robot").alias("robot_id"),
        ).select(list(SERIES_SCHEMA))

    @staticmethod
    def _summarize(series: pl.DataFrame) -> dict[str, float | None]:
        """Energy/power/battery aggregates over a per-sample series frame."""
        if len(series) == 0:
            return {key: None for key in EnergyMetricCalculator.output_keys()}

        time_s = series["time_s"].to_numpy().astype(np.float64)
        dt = np.diff(time_s, prepend=time_s[0] if len(time_s) else 0.0)
        dt[dt < 0.0] = 0.0

        def integrate(column: str) -> float | None:
            values = series[column].to_numpy().astype(np.float64)
            if np.isnan(values).all():
                return None
            return float(np.nansum(values * dt) / SECONDS_PER_HOUR)

        energy_static_wh = integrate("power_static_w")
        energy_mechanical_wh = integrate("power_mechanical_w")
        energy_thermal_wh = integrate("power_thermal_w")

        total_power = series["power_total_w"].to_numpy().astype(np.float64)
        energy_total_wh = None if np.isnan(total_power).all() else float(np.nansum(total_power * dt) / SECONDS_PER_HOUR)
        power_peak_w = None if np.isnan(total_power).all() else float(np.nanmax(total_power))

        battery = series["battery_soc_pct"].to_numpy().astype(np.float64)
        if np.isnan(battery).all():
            battery_soc_final = None
            battery_soc_drop_pct = None
        else:
            battery_soc_final = float(battery[-1])
            battery_soc_drop_pct = max(float(battery[0] - battery[-1]), 0.0)

        return {
            "energy_static_wh": energy_static_wh,
            "energy_mechanical_wh": energy_mechanical_wh,
            "energy_thermal_wh": energy_thermal_wh,
            "energy_total_wh": energy_total_wh,
            "power_peak_w": power_peak_w,
            "battery_soc_final": battery_soc_final,
            "battery_soc_drop_pct": battery_soc_drop_pct,
        }

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        series = self._aligned_series(episode)
        result = self._summarize(series)

        cumulative = episode.data["total_energy_consumed_wh"].cast(pl.Float64) if episode.data is not None and "total_energy_consumed_wh" in episode.data.columns else None
        if cumulative is not None:
            values = cumulative.drop_nulls().to_numpy().astype(np.float64)
            if len(values) >= 2:
                result["energy_total_wh"] = float(values[-1] - values[0])
        return result

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            return self._summarize(self._native_series(episode))
        raise ValueError(f"energy does not serve the {scope_slice.scope.value} scope")

    def series_table(self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = "") -> pl.DataFrame:
        """1NF flat series table of the power/energy/velocity timeseries, one row per sample."""
        series = self._aligned_series(episode)
        if len(series) == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        return series.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(robot_id or (episode.robot_name or "robot")).alias("robot_id"),
        ).select(list(SERIES_SCHEMA))