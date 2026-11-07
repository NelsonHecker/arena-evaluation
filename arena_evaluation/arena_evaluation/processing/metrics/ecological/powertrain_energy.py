from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

METRES_PER_WH_J: typing.Final[float] = 3600.0


class RobotPowertrainCalculator(BaseMetricCalculator):
    """Robot-level powertrain energy, power, mechanical work, and friction scrub."""

    NAME = "robot_powertrain"
    CATEGORY = "ecological"
    SCOPE = MetricScope.ROBOT
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.ROBOT})
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = []

    UNITS = {
        "mean_power_w": "W",
        "peak_power_w": "W",
        "mechanical_work_j": "J",
        "thermal_energy_j": "J",
        "electrical_energy_wh": "Wh",
        "drivetrain_scrub_j": "J",
    }
    PRIMARY_OUTPUTS = ["electrical_energy_wh", "mechanical_work_j", "mean_power_w"]
    OUTPUT_DIRECTIONS = {
        "mean_power_w": "lower",
        "peak_power_w": "lower",
        "mechanical_work_j": "lower",
        "thermal_energy_j": "lower",
        "electrical_energy_wh": "lower",
        "drivetrain_scrub_j": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "mean_power_w",
            "peak_power_w",
            "mechanical_work_j",
            "thermal_energy_j",
            "electrical_energy_wh",
            "drivetrain_scrub_j",
        ]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        duration_s = prior_results.get("time_to_goal")
        electrical_energy_wh = prior_results.get("energy_total_wh")
        mechanical_energy_wh = prior_results.get("energy_mechanical_wh")
        thermal_energy_wh = prior_results.get("energy_thermal_wh")
        peak_power_w = prior_results.get("power_peak_w")
        friction_dissipation_j = prior_results.get("friction_dissipation_j")

        # Fallback to direct calculation from native robot_power table if not in prior_results
        power_table = self.native_topics(episode).get("robot_power")
        if power_table is not None and isinstance(power_table, pl.LazyFrame):
            power_table = power_table.collect()

        if electrical_energy_wh is None and power_table is not None and len(power_table) > 0:
            powers = power_table.get_column("total_power_w").drop_nulls()
            if len(powers) > 0:
                mean_w = float(powers.mean())
                if peak_power_w is None:
                    peak_power_w = float(powers.max())
                if duration_s is not None and float(duration_s) > 0.0:
                    electrical_energy_wh = mean_w * float(duration_s) / METRES_PER_WH_J

        mean_power_w = None
        if electrical_energy_wh is not None and duration_s is not None and float(duration_s) > 0.0:
            mean_power_w = float(electrical_energy_wh) * METRES_PER_WH_J / float(duration_s)
        elif power_table is not None and len(power_table) > 0:
            powers = power_table.get_column("total_power_w").drop_nulls()
            if len(powers) > 0:
                mean_power_w = float(powers.mean())

        mechanical_work_j = float(mechanical_energy_wh) * METRES_PER_WH_J if mechanical_energy_wh is not None else None
        thermal_energy_j = float(thermal_energy_wh) * METRES_PER_WH_J if thermal_energy_wh is not None else None

        return {
            "mean_power_w": mean_power_w,
            "peak_power_w": float(peak_power_w) if peak_power_w is not None else None,
            "mechanical_work_j": mechanical_work_j,
            "thermal_energy_j": thermal_energy_j,
            "electrical_energy_wh": float(electrical_energy_wh) if electrical_energy_wh is not None else None,
            "drivetrain_scrub_j": float(friction_dissipation_j) if friction_dissipation_j is not None else None,
        }


class SurgeEnergyCalculator(BaseMetricCalculator):
    """Multi-scope electrical energy demand and power surge ratio across episode, robot, and encounter scopes."""

    NAME = "surge_energy"
    CATEGORY = "ecological"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset(
        {
            MetricScope.EPISODE,
            MetricScope.ROBOT,
            MetricScope.ENCOUNTER,
        }
    )
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_power"]

    UNITS = {
        "surge_energy_j": "J",
        "surge_power_ratio": "",
    }
    PRIMARY_OUTPUTS = ["surge_energy_j", "surge_power_ratio"]
    OUTPUT_DIRECTIONS = {
        "surge_energy_j": "lower",
        "surge_power_ratio": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["surge_energy_j", "surge_power_ratio"]

    @classmethod
    def output_keys_for_scope(cls, scope: MetricScope) -> tuple[str, ...]:
        if scope == MetricScope.ENCOUNTER:
            return ("encounter_energy_j", "energy_surge_ratio")
        return ("surge_energy_j", "surge_power_ratio")

    def _compute_window_energy_and_surge(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> tuple[float | None, float | None]:
        prior_energy = prior_results.get("cost_energy_j", prior_results.get("encounter_energy_j"))
        prior_surge = prior_results.get("energy_surge_ratio", prior_results.get("surge_power_ratio"))

        power_table = self.native_topics(episode).get("robot_power")
        if power_table is not None and isinstance(power_table, pl.LazyFrame):
            power_table = power_table.collect()

        if power_table is None or len(power_table) < 2:
            return (
                float(prior_energy) if prior_energy is not None else None,
                float(prior_surge) if prior_surge is not None else None,
            )

        clean_power = power_table.select("t_ns", "total_power_w").drop_nulls().sort("t_ns")
        if len(clean_power) < 2:
            return (
                float(prior_energy) if prior_energy is not None else None,
                float(prior_surge) if prior_surge is not None else None,
            )

        t_ns = clean_power.get_column("t_ns").to_numpy().astype(np.int64)
        p_w = clean_power.get_column("total_power_w").to_numpy().astype(np.float64)
        dt_s = np.diff(t_ns).astype(np.float64) / 1e9

        energy_j = float(np.sum(0.5 * (p_w[:-1] + p_w[1:]) * dt_s)) if prior_energy is None else float(prior_energy)
        window_mean_power_w = float(clean_power.get_column("total_power_w").mean())

        episode_mean_power_w = prior_results.get("episode_mean_power_w")
        if episode_mean_power_w is None:
            episode_mean_power_w = prior_results.get("mean_power_w")
        if episode_mean_power_w is None:
            episode_mean_power_w = window_mean_power_w

        surge_ratio = None
        if prior_surge is not None:
            surge_ratio = float(prior_surge)
        elif window_mean_power_w is not None and episode_mean_power_w is not None and float(episode_mean_power_w) > 0.0:
            surge_ratio = window_mean_power_w / float(episode_mean_power_w)

        return energy_j, surge_ratio

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        energy_j, surge_ratio = self._compute_window_energy_and_surge(episode, prior_results)
        return {
            "surge_energy_j": energy_j,
            "surge_power_ratio": surge_ratio,
            "encounter_energy_j": energy_j,
            "energy_surge_ratio": surge_ratio,
        }

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        energy_j, surge_ratio = self._compute_window_energy_and_surge(episode, prior_results)
        if scope_slice is not None and scope_slice.scope == MetricScope.ENCOUNTER:
            return {
                "encounter_energy_j": energy_j,
                "energy_surge_ratio": surge_ratio,
            }
        return {
            "surge_energy_j": energy_j,
            "surge_power_ratio": surge_ratio,
        }


EncounterEnergyCalculator = SurgeEnergyCalculator
EpisodeSurgeEnergyCalculator = SurgeEnergyCalculator
