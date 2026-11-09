from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

STANDSTILL_SPEED_MPS: typing.Final[float] = 0.01
MIN_STANDSTILL_BLOCK_S: typing.Final[float] = 0.5
MIN_PATH_LENGTH_M: typing.Final[float] = 0.1
MIN_MEAN_POWER_W: typing.Final[float] = 0.01
SECONDS_PER_HOUR: typing.Final[float] = 3600.0
NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
TRACK_ALIGN_TOLERANCE_NS: typing.Final[int] = 100_000_000
_GRAVITY_MPS2: typing.Final[float] = 9.81
_DEFAULT_ROLLING_RESISTANCE: typing.Final[float] = 0.015


def _median_filter_window3(values: np.ndarray) -> np.ndarray:
    """3-sample moving median, to reject single-frame outliers."""
    length = len(values)
    if length < 3:
        return values.copy()
    filtered = np.empty_like(values)
    filtered[0] = np.median(values[:2])
    for index in range(1, length - 1):
        filtered[index] = np.median(values[index - 1 : index + 2])
    filtered[-1] = np.median(values[-2:])
    return filtered


def _filled(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaN gaps then back-fill the leading ones; an all-NaN array is returned unchanged."""
    missing = np.isnan(values)
    if missing.all():
        return values
    indices = np.where(~missing, np.arange(missing.shape[0]), 0)
    np.maximum.accumulate(indices, out=indices)
    filled = values[indices]
    remaining = np.isnan(filled)
    if np.any(remaining):
        filled[remaining] = values[np.flatnonzero(~missing)[0]]
    return filled


class EnergyExtendedCalculator(BaseMetricCalculator):
    """Extended ecological energy metrics.

    Computes Specific Cost of Transport, Energy per Meter, Peak-to-Mean Power
    Ratio and Standstill Energy Penalty on the native telemetry of the scope.
    All energy quantities are reported in watt-hours.
    """

    NAME = "energy_extended"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["energy", "path_metrics", "motion_metrics"]
    REQUIRED_TOPICS = [("robot_power", "robot_energy", "robot_odom")]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})

    UNITS = {
        "specific_cost_of_transport": "",
        "energy_per_meter": "Wh/m",
        "peak_to_mean_power_ratio": "",
        "standstill_energy_penalty_wh": "Wh",
        "kinetic_energy_demand_j": "J",
        "friction_dissipation_j": "J",
    }

    PRIMARY_OUTPUTS = ["specific_cost_of_transport", "energy_per_meter"]
    OUTPUT_DIRECTIONS = {
        "specific_cost_of_transport": "lower",
        "energy_per_meter": "lower",
        "peak_to_mean_power_ratio": "lower",
        "standstill_energy_penalty_wh": "lower",
        "kinetic_energy_demand_j": "lower",
        "friction_dissipation_j": "lower",
    }

    def __init__(self, robot_params: RobotParams):
        super().__init__(robot_params)
        self._rolling_resistance = self._load_rolling_resistance(robot_params.model)

    @staticmethod
    def _load_rolling_resistance(model: str) -> float:
        from pathlib import Path

        import yaml

        try:
            from ament_index_python.packages import get_package_share_directory

            power_path = Path(get_package_share_directory("arena_robots")) / "robots" / model / "telemetry" / "power.yaml"
            data = yaml.safe_load(power_path.read_text())
            return float(data.get("power_system", {}).get("rolling_resistance_crr", _DEFAULT_ROLLING_RESISTANCE))
        except (OSError, ImportError, ValueError, TypeError, AttributeError):
            return _DEFAULT_ROLLING_RESISTANCE

    @classmethod
    def required_topics_for_scope(cls, scope: MetricScope) -> tuple[str | tuple[str, ...], ...]:
        # A window may legitimately lack an optional sensor stream; a missing stream is then per-metric
        # `no_samples`, not a row-level `topic_missing` that would block the whole encounter.
        return () if scope == MetricScope.ENCOUNTER else super().required_topics_for_scope(scope)

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "specific_cost_of_transport",
            "energy_per_meter",
            "peak_to_mean_power_ratio",
            "standstill_energy_penalty_wh",
            "kinetic_energy_demand_j",
            "friction_dissipation_j",
        ]

    def _episode_series(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Velocity (m/s), total power (W) and elapsed time (s) of the episode on the aligned clock."""
        frame = episode.data
        if frame is None or len(frame) == 0 or "time_ns" not in frame.columns:
            return np.array([]), np.array([]), np.array([])
        time_s = (frame["time_ns"].to_numpy().astype(np.float64) - float(frame["time_ns"][0])) / NANOSECONDS_PER_SECOND
        power = _filled(frame["total_power_w"].cast(pl.Float64).to_numpy().astype(np.float64)) if "total_power_w" in frame.columns else np.full(len(frame), np.nan)
        velocity = np.asarray(prior_results.get("velocity") or [], dtype=np.float64)
        if len(velocity) != len(time_s):
            velocity = np.abs(frame["vel_linear"].cast(pl.Float64).to_numpy().astype(np.float64)) if "vel_linear" in frame.columns else np.full(len(frame), np.nan)
        return velocity, power, time_s

    def _native_series(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Velocity (m/s), total power (W) and elapsed time (s) from the window's native telemetry topics."""
        topics = self.native_topics(episode)
        power_table = topics.get("robot_power")
        if power_table is None or len(power_table) == 0 or "total_power_w" not in power_table.columns:
            return np.array([]), np.array([]), np.array([])
        power = power_table.select(pl.col("t_ns").cast(pl.Int64), pl.col("total_power_w").cast(pl.Float64)).drop_nulls("total_power_w").sort("t_ns")
        tracks = topics.get("robot_tracks")
        if tracks is not None and len(tracks) > 0 and "speed_mps" in tracks.columns:
            reference = tracks.select(pl.col("t_ns").cast(pl.Int64), pl.col("speed_mps").cast(pl.Float64).abs()).drop_nulls().sort("t_ns")
            power = power.join_asof(reference, on="t_ns", strategy="backward", tolerance=TRACK_ALIGN_TOLERANCE_NS)
        else:
            power = power.with_columns(pl.lit(None, dtype=pl.Float64).alias("speed_mps"))
        time_ns = power["t_ns"].to_numpy().astype(np.int64)
        power_w = power["total_power_w"].to_numpy().astype(np.float64)
        velocity = _filled(power["speed_mps"].to_numpy().astype(np.float64))
        time_s = (time_ns.astype(np.float64) - float(time_ns[0])) / NANOSECONDS_PER_SECOND
        return velocity, power_w, time_s

    def _summarize(
        self,
        velocity_mps: np.ndarray,
        power_w: np.ndarray,
        time_s: np.ndarray,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Extended energy metrics from the scoped velocity/power/time series and the parent-scope priors."""
        mass = self.robot_params.mass
        path_length_m = prior_results.get("path_length")
        energy_total_wh = prior_results.get("energy_total_wh")

        result: dict[str, typing.Any] = {}

        cost_of_transport = None
        if mass > 0 and energy_total_wh is not None and path_length_m is not None and path_length_m > MIN_PATH_LENGTH_M and energy_total_wh > 0:
            candidate = float(energy_total_wh) * SECONDS_PER_HOUR / (mass * _GRAVITY_MPS2 * float(path_length_m))
            cost_of_transport = candidate if np.isfinite(candidate) else None
        result["specific_cost_of_transport"] = cost_of_transport

        energy_per_meter = None
        if energy_total_wh is not None and path_length_m is not None and path_length_m > MIN_PATH_LENGTH_M and energy_total_wh > 0:
            candidate = float(energy_total_wh) / float(path_length_m)
            energy_per_meter = candidate if np.isfinite(candidate) else None
        result["energy_per_meter"] = energy_per_meter

        peak_to_mean = None
        valid_power = power_w[~np.isnan(power_w)]
        if len(valid_power) > 0:
            filtered = _median_filter_window3(valid_power)
            mean_power = float(np.mean(filtered))
            if mean_power > MIN_MEAN_POWER_W:
                candidate = float(np.max(filtered)) / mean_power
                peak_to_mean = candidate if np.isfinite(candidate) else None
        result["peak_to_mean_power_ratio"] = peak_to_mean

        if len(time_s) == 0:
            result["standstill_energy_penalty_wh"] = None
            result["kinetic_energy_demand_j"] = None
            result["friction_dissipation_j"] = None
            return result

        dt = np.diff(time_s, prepend=time_s[0])
        dt[dt < 0.0] = 0.0
        result["standstill_energy_penalty_wh"] = self._standstill_energy_wh(velocity_mps, power_w, dt)

        kinetic_energy_demand = None
        if mass > 0 and len(velocity_mps) > 1:
            inertia = 0.5 * mass * self.robot_params.robot_radius**2
            angular = np.asarray(prior_results.get("angular_velocity") or [], dtype=np.float64)
            if len(angular) != len(velocity_mps):
                angular = np.zeros_like(velocity_mps)
            kinetic = 0.5 * mass * velocity_mps**2 + 0.5 * inertia * angular**2
            kinetic_energy_demand = float(np.sum(np.abs(np.diff(kinetic))))
        result["kinetic_energy_demand_j"] = kinetic_energy_demand

        friction = None
        if mass > 0 and len(velocity_mps) > 1:
            friction = float(np.sum(self._rolling_resistance * mass * _GRAVITY_MPS2 * np.abs(velocity_mps) * dt))
        result["friction_dissipation_j"] = friction
        return result

    def _window_path_length_m(self, episode: AlignedEpisodeBundle) -> float | None:
        """Sum of the window's native /tf robot steps in metres, None without a usable track."""
        tracks = self.native_topics(episode).get("robot_tracks")
        if tracks is None or len(tracks) < 2:
            return None
        frame = (
            tracks.select(
                pl.col("t_ns").cast(pl.Int64),
                pl.col("x").cast(pl.Float64),
                pl.col("y").cast(pl.Float64),
            )
            .drop_nulls(["x", "y"])
            .sort("t_ns")
        )
        if len(frame) < 2:
            return None
        steps = (pl.col("x").diff().pow(2) + pl.col("y").diff().pow(2)).sqrt().alias("step_m")
        return float(frame.select(steps).get_column("step_m").drop_nulls().sum())

    @staticmethod
    def _integrated_energy_wh(power_w: np.ndarray, time_s: np.ndarray) -> float | None:
        """Trapezoid-free left-step integral of the window's total power, matching energy_metrics."""
        if len(time_s) == 0 or len(power_w) != len(time_s):
            return None
        if np.isnan(power_w).all():
            return None
        dt = np.diff(time_s, prepend=time_s[0])
        dt[dt < 0.0] = 0.0
        return float(np.nansum(power_w * dt) / SECONDS_PER_HOUR)

    @staticmethod
    def _standstill_energy_wh(velocity_mps: np.ndarray, power_w: np.ndarray, dt: np.ndarray) -> float | None:
        """Energy drawn while standing still in blocks of at least half a second."""
        if len(velocity_mps) == 0 or len(power_w) != len(velocity_mps) or np.isnan(velocity_mps).all():
            return None
        stationary = np.abs(velocity_mps) < STANDSTILL_SPEED_MPS
        if not np.any(stationary):
            return 0.0
        blocks = pl.DataFrame(
            {
                "stationary": stationary,
                "power_w": power_w,
                "dt_s": dt,
            }
        ).with_columns(
            (pl.col("stationary") != pl.col("stationary").shift(1)).fill_null(True).cum_sum().alias("block_id")
        )
        block_energy = (
            blocks.filter(pl.col("stationary"))
            .group_by("block_id")
            .agg(
                duration_s=pl.col("dt_s").sum(),
                energy_wh=(pl.col("power_w") * pl.col("dt_s")).sum() / SECONDS_PER_HOUR,
            )
            .filter(pl.col("duration_s") >= MIN_STANDSTILL_BLOCK_S)
        )
        return float(block_energy["energy_wh"].sum()) if len(block_energy) > 0 else 0.0

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        velocity_mps, power_w, time_s = self._episode_series(episode, prior_results)
        return self._summarize(velocity_mps, power_w, time_s, prior_results)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            velocity_mps, power_w, time_s = self._native_series(episode)
            window_priors = dict(prior_results)
            window_priors.setdefault("path_length", self._window_path_length_m(episode))
            window_priors.setdefault("energy_total_wh", self._integrated_energy_wh(power_w, time_s))
            return self._summarize(velocity_mps, power_w, time_s, window_priors)
        raise ValueError(f"energy_extended does not serve the {scope_slice.scope.value} scope")