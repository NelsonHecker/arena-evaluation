from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

ACOUSTIC_REFERENCE_PRESSURE_PA: typing.Final[float] = 20e-6
ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2: typing.Final[float] = ACOUSTIC_REFERENCE_PRESSURE_PA * ACOUSTIC_REFERENCE_PRESSURE_PA


def compute_acoustic_dose_pa2s(
    acoustics_df: pl.DataFrame | pl.LazyFrame | None,
    t_start_ns: int | None = None,
    t_end_ns: int | None = None,
) -> float | None:
    """Integral of linearized emitted sound pressure squared over time in Pa^2*s."""
    if acoustics_df is None:
        return None
    if isinstance(acoustics_df, pl.LazyFrame):
        acoustics_df = acoustics_df.collect()
    if len(acoustics_df) < 2:
        return None
    frame = acoustics_df.select("t_ns", "total_level_af_dba").drop_nulls().sort("t_ns")
    if t_start_ns is not None:
        frame = frame.filter(pl.col("t_ns") >= t_start_ns)
    if t_end_ns is not None:
        frame = frame.filter(pl.col("t_ns") <= t_end_ns)
    if len(frame) < 2:
        return None
    time_ns = frame.get_column("t_ns").to_numpy().astype(np.int64)
    level_dba = frame.get_column("total_level_af_dba").to_numpy().astype(np.float64)
    dt_s = np.diff(time_ns).astype(np.float64) / 1e9
    linear_pressure_sq = ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2 * np.power(10.0, level_dba / 10.0)
    return float(np.sum(0.5 * (linear_pressure_sq[:-1] + linear_pressure_sq[1:]) * dt_s))


class AcousticsCalculator(BaseMetricCalculator):
    """Ecological acoustic metrics on the native /acoustics time base.

    Metrics:
    - L_Aeq,T (ISO 1996-1:2016): equivalent continuous A-weighted level.
    - L_A10 / L_A50 / L_A90: levels exceeded 10/50/90 % of the time.
    - ASF / LTR: max positive rise rate of the 3-sample median filtered
      A-weighted Fast level, gated on L_AF >= 65 dBA. A proxy for transients
      from jerky torque bursts, not the biological startle reflex (which
      needs >80 dBA and millisecond rise times).
    - ASI: variance of 1 s Leq windows.
    - acoustic_cost: linearized acoustic energy per meter.
    """

    NAME = "acoustics"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["path_metrics"]
    REQUIRED_TOPICS = ["robot_acoustics"]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})

    UNITS = {
        "l_aeq_t": "dBA",
        "l_a10": "dBA",
        "l_a50": "dBA",
        "l_a90": "dBA",
        "acoustic_startle_factor": "dBA/s",
        "acoustic_surge_index": "dBA²",
        "acoustic_cost": "s/m",
    }

    PRIMARY_OUTPUTS = ["l_aeq_t", "acoustic_startle_factor"]
    OUTPUT_DIRECTIONS = {
        "l_aeq_t": "lower",
        "acoustic_startle_factor": "lower",
        "acoustic_cost": "lower",
        "acoustic_surge_index": "lower",
        "l_a10": "lower",
        "l_a50": "lower",
        "l_a90": "lower",
    }

    _DEFAULT_DBA = 42.0  # dBA, idle noise floor
    _MIN_DBA = 1.0  # dBA, clamp before log
    _L_GATE_DBA = 65.0  # dBA, ASF level gate

    @classmethod
    def required_topics_for_scope(cls, scope: MetricScope) -> tuple[str | tuple[str, ...], ...]:
        # A window may legitimately lack an optional sensor stream; a missing stream is then per-metric
        # `no_samples`, not a row-level `topic_missing` that would block the whole encounter.
        return () if scope == MetricScope.ENCOUNTER else super().required_topics_for_scope(scope)

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "l_aeq_t",
            "l_a10",
            "l_a50",
            "l_a90",
            "acoustic_startle_factor",
            "acoustic_surge_index",
            "acoustic_cost",
        ]

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            return self.calculate(episode, {"path_length": self._window_path_length_m(episode)})
        raise ValueError(f"acoustics does not serve the {scope_slice.scope.value} scope")

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        topics = self.native_topics(episode)
        acous = topics.get("robot_acoustics")
        if acous is None or "total_level_af_dba" not in acous.columns or "t_ns" not in acous.columns:
            return {k: None for k in self.output_keys()}

        acous = acous.sort("t_ns")
        try:
            dba = acous["total_level_af_dba"].cast(pl.Float64).fill_null(strategy="forward").fill_null(self._DEFAULT_DBA).to_numpy()
        except (pl.ComputeError, TypeError):
            return {k: None for k in self.output_keys()}

        if len(dba) == 0 or np.all(np.isnan(dba)):
            return {k: None for k in self.output_keys()}

        dba = np.maximum(dba, self._MIN_DBA)
        time_ns = acous["t_ns"].to_numpy()
        if len(time_ns) < 2:
            return {k: None for k in self.output_keys()}

        dt = np.diff(time_ns) / 1e9
        dt = np.append(dt, dt[-1] if len(dt) > 0 else 0.01)
        dt = np.where(dt <= 0.0, 1e-6, dt)

        linear_energy = 10.0 ** (dba / 10.0)
        T = float(np.sum(dt))

        with np.errstate(divide="ignore", invalid="ignore"):
            l_aeq_t = 10.0 * np.log10(np.sum(linear_energy * dt) / T)
        if np.isnan(l_aeq_t) or np.isinf(l_aeq_t):
            l_aeq_t = None
        else:
            l_aeq_t = float(l_aeq_t)

        # L_AN convention: L_A10 is the level exceeded 10 % of the time
        l_a10 = float(np.percentile(dba, 90))
        l_a50 = float(np.percentile(dba, 50))
        l_a90 = float(np.percentile(dba, 10))

        dba_filt = self._median_filter_3(dba)
        dL = np.diff(dba_filt)
        dt_diff = np.diff(time_ns) / 1e9
        dt_diff = np.where(dt_diff <= 0.0, 1e-6, dt_diff)
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = dL / dt_diff

        # Only rises ending at or above the gate count as transients
        gated = np.where(dba_filt[1:] >= self._L_GATE_DBA, rates, 0.0)
        positive_rates = gated[gated > 0]
        acoustic_startle_factor = float(np.max(positive_rates)) if len(positive_rates) > 0 else 0.0

        window_s = 1.0
        if T >= window_s and len(dba) >= 2:
            cumsum_t = np.cumsum(dt)
            window_lecs = []
            t_start = 0.0
            while t_start + window_s <= T:
                t_end = t_start + window_s
                mask = (cumsum_t >= t_start) & (cumsum_t < t_end)
                if np.sum(mask) > 0:
                    window_le = np.mean(linear_energy[mask])
                    if window_le > 0:
                        window_lecs.append(10.0 * np.log10(window_le))
                t_start += window_s
            acoustic_surge_index = float(np.var(window_lecs)) if len(window_lecs) >= 2 else 0.0
        else:
            acoustic_surge_index = 0.0

        path_length = prior_results.get("path_length")
        if path_length is not None and path_length >= 0.1:
            tee_acoustic = float(np.sum(linear_energy * dt))
            acoustic_cost = float(tee_acoustic / path_length)
        else:
            acoustic_cost = None

        return {
            "l_aeq_t": l_aeq_t,
            "l_a10": l_a10,
            "l_a50": l_a50,
            "l_a90": l_a90,
            "acoustic_startle_factor": acoustic_startle_factor,
            "acoustic_surge_index": acoustic_surge_index,
            "acoustic_cost": acoustic_cost,
        }

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
    def _median_filter_3(signal: np.ndarray) -> np.ndarray:
        """3-sample median filter, edges mirrored."""
        if len(signal) < 3:
            return signal.copy()
        result = np.zeros_like(signal)
        for i in range(len(signal)):
            if i == 0:
                result[i] = np.median([signal[0], signal[0], signal[1]])
            elif i == len(signal) - 1:
                result[i] = np.median([signal[i - 1], signal[i], signal[i]])
            else:
                result[i] = np.median([signal[i - 1], signal[i], signal[i + 1]])
        return result


class AcousticDoseCalculator(BaseMetricCalculator):
    """Multi-scope acoustic emission and exposure dose calculator across episode, robot, encounter, and pedestrian scopes."""

    NAME = "acoustic_dose"
    CATEGORY = "ecological"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset(
        {
            MetricScope.EPISODE,
            MetricScope.ROBOT,
            MetricScope.ENCOUNTER,
            MetricScope.PEDESTRIAN,
        }
    )
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_acoustics"]

    UNITS = {
        "total_acoustic_dose_pa2s": "Pa²·s",
        "acoustic_emission_pa2s": "Pa²·s",
        "sound_exposure_dose_pa2s": "Pa²·s",
        "cumulative_sound_dose_pa2s": "Pa²·s",
    }
    PRIMARY_OUTPUTS = ["total_acoustic_dose_pa2s"]
    OUTPUT_DIRECTIONS = {
        "total_acoustic_dose_pa2s": "lower",
        "acoustic_emission_pa2s": "lower",
        "sound_exposure_dose_pa2s": "lower",
        "cumulative_sound_dose_pa2s": "lower",
    }

    @classmethod
    def required_topics_for_scope(cls, scope: MetricScope) -> tuple[str | tuple[str, ...], ...]:
        if scope in (MetricScope.ENCOUNTER, MetricScope.PEDESTRIAN):
            return ()
        return ("robot_acoustics",)

    @classmethod

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "total_acoustic_dose_pa2s",
            "acoustic_emission_pa2s",
            "sound_exposure_dose_pa2s",
            "cumulative_sound_dose_pa2s",
        ]

    @classmethod
    def output_keys_for_scope(cls, scope: MetricScope) -> tuple[str, ...]:
        if scope == MetricScope.ROBOT:
            return ("acoustic_emission_pa2s",)
        if scope == MetricScope.ENCOUNTER:
            return ("sound_exposure_dose_pa2s",)
        if scope == MetricScope.PEDESTRIAN:
            return ("cumulative_sound_dose_pa2s",)
        return ("total_acoustic_dose_pa2s",)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        topics = self.native_topics(episode)
        acoustics = topics.get("robot_acoustics")
        dose = compute_acoustic_dose_pa2s(acoustics)
        prior_total = prior_results.get("total_acoustic_dose_pa2s")
        emission = float(prior_total) if prior_total is not None else dose
        exposure = prior_results.get("sound_exposure_dose_pa2s", dose)
        cumulative = prior_results.get("cumulative_sound_dose_pa2s", prior_results.get("sound_exposure_dose_pa2s"))
        return {
            "total_acoustic_dose_pa2s": dose,
            "acoustic_emission_pa2s": emission,
            "sound_exposure_dose_pa2s": float(exposure) if exposure is not None else None,
            "cumulative_sound_dose_pa2s": float(cumulative) if cumulative is not None else None,
        }

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None:
            return self.calculate(episode, prior_results)

        if scope_slice.scope == MetricScope.PEDESTRIAN:
            dose = prior_results.get("cumulative_sound_dose_pa2s")
            if dose is None:
                dose = prior_results.get("sound_exposure_dose_pa2s")
            return {"cumulative_sound_dose_pa2s": float(dose) if dose is not None else None}

        if scope_slice.scope == MetricScope.ENCOUNTER:
            prior_exposure = prior_results.get("sound_exposure_dose_pa2s")
            if prior_exposure is not None:
                return {"sound_exposure_dose_pa2s": float(prior_exposure)}
            acoustics = self.native_topics(episode).get("robot_acoustics")
            return {"sound_exposure_dose_pa2s": compute_acoustic_dose_pa2s(acoustics)}

        if scope_slice.scope == MetricScope.ROBOT:
            prior_dose = prior_results.get("total_acoustic_dose_pa2s")
            if prior_dose is not None:
                return {"acoustic_emission_pa2s": float(prior_dose)}
            acoustics = self.native_topics(episode).get("robot_acoustics")
            return {"acoustic_emission_pa2s": compute_acoustic_dose_pa2s(acoustics)}

        acoustics = self.native_topics(episode).get("robot_acoustics")
        return {"total_acoustic_dose_pa2s": compute_acoustic_dose_pa2s(acoustics)}


EpisodeAcousticDoseCalculator = AcousticDoseCalculator
RobotAcousticEmissionCalculator = AcousticDoseCalculator
PedestrianAcousticDoseCalculator = AcousticDoseCalculator
