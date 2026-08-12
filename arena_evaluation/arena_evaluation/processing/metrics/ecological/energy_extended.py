from __future__ import annotations

import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


def _median_filter_window3(arr: np.ndarray) -> np.ndarray:
    """Apply a 3-sample moving median filter to reject single-frame outliers."""
    n = len(arr)
    if n < 3:
        return arr.copy()
    result = np.empty_like(arr)
    result[0] = np.median(arr[:2])
    for i in range(1, n - 1):
        result[i] = np.median(arr[i - 1 : i + 2])
    result[-1] = np.median(arr[-2:])
    return result


class EnergyExtendedCalculator(BaseMetricCalculator):
    """Extended ecological energy metrics.

    Computes Specific Cost of Transport, Energy per Meter, Peak-to-Mean Power
    Ratio, and Standstill Energy Penalty from prior energy, path, and motion
    results. All energy quantities are reported in watt-hours.
    """

    NAME = "energy_extended"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["energy", "path_metrics", "motion_metrics"]
    REQUIRED_TOPICS = ["power", "odom"]

    UNITS = {
        "specific_cost_of_transport": "",
        "energy_per_meter": "Wh/m",
        "peak_to_mean_power_ratio": "",
        "standstill_energy_penalty_wh": "Wh",
    }

    PRIMARY_OUTPUTS = ["specific_cost_of_transport", "energy_per_meter"]
    OUTPUT_DIRECTIONS = {
        "specific_cost_of_transport": "lower",
        "energy_per_meter": "lower",
        "peak_to_mean_power_ratio": "lower",
        "standstill_energy_penalty_wh": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "specific_cost_of_transport",
            "energy_per_meter",
            "peak_to_mean_power_ratio",
            "standstill_energy_penalty_wh",
        ]

    def calculate(
        self,
        episode: "AlignedEpisodeBundle",
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Calculate extended energy metrics from prior results.

        Parameters
        ----------
        episode : AlignedEpisodeBundle
            The aligned episode bundle (data, start_pos, etc.).
        prior_results : dict
            Results from upstream calculators: energy, path_metrics, motion_metrics.

        Returns
        -------
        dict
            All four output keys — scalar floats or None on failure.
        """
        if episode.data is None:
            return {k: None for k in self.output_keys()}

        # ── Physical constants ──
        g = 9.81                # gravitational acceleration [m/s²]
        v_thresh = 0.01         # standstill velocity threshold [m/s]
        mass = max(self.robot_params.mass, 0.1)  # clamp misconfigured-zero mass

        # ── Prior results ──
        path_length: float | None = prior_results.get("path_length")
        energy_total_wh: float | None = prior_results.get("energy_total_wh")
        velocity_list: list | None = prior_results.get("velocity")
        power_ts: list | None = prior_results.get("timeseries_power_total_w")
        time_ts: list | None = prior_results.get("timeseries_time_s")

        result: dict[str, typing.Any] = {}

        # =====================================================================
        # 1. Specific Cost of Transport  (dimensionless)
        #    Uses TOTAL energy consumed (including static/compute overhead) —
        #    the cost of the whole navigation task, not just locomotion.
        # =====================================================================
        cot = None
        if (
            energy_total_wh is not None
            and path_length is not None
            and path_length > 0.1
            and energy_total_wh > 0
        ):
            e_total_j = float(energy_total_wh) * 3600.0
            with np.errstate(divide="ignore", invalid="ignore"):
                cot = e_total_j / (mass * g * float(path_length))
                if not np.isfinite(cot):
                    cot = None
        result["specific_cost_of_transport"] = float(cot) if cot is not None else None

        # =====================================================================
        # 2. Energy per Meter  [Wh/m]
        # =====================================================================
        epm = None
        if (
            energy_total_wh is not None
            and path_length is not None
            and path_length > 0.1
            and energy_total_wh > 0
        ):
            with np.errstate(divide="ignore", invalid="ignore"):
                epm = float(energy_total_wh) / float(path_length)
                if not np.isfinite(epm):
                    epm = None
        result["energy_per_meter"] = float(epm) if epm is not None else None

        # =====================================================================
        # 3. Peak-to-Mean Power Ratio  (dimensionless)
        # =====================================================================
        pmpr = None
        if power_ts is not None and len(power_ts) > 0:
            p_arr = np.array(power_ts, dtype=float)
            # Median filter rejects single-frame physics glitches
            p_filt = _median_filter_window3(p_arr)
            p_mean = float(np.mean(p_filt))
            if p_mean > 0.01:
                p_max = float(np.max(p_filt))
                with np.errstate(divide="ignore", invalid="ignore"):
                    pmpr = p_max / p_mean
                    if not np.isfinite(pmpr):
                        pmpr = None
        result["peak_to_mean_power_ratio"] = float(pmpr) if pmpr is not None else None

        # =====================================================================
        # 4. Standstill Energy Penalty  [Wh]
        # =====================================================================
        vel = (
            np.array(velocity_list, dtype=float)
            if velocity_list is not None and len(velocity_list) > 0
            else np.array([])
        )
        p_ts = (
            np.array(power_ts, dtype=float)
            if power_ts is not None and len(power_ts) > 0
            else np.array([])
        )
        t_ts = (
            np.array(time_ts, dtype=float)
            if time_ts is not None and len(time_ts) > 0
            else np.array([])
        )

        # Truncate all timeseries to the shortest common length
        lengths = [len(vel), len(p_ts), len(t_ts)]
        min_len = min(lengths) if lengths else 0

        if min_len > 0:
            vel = vel[:min_len]
            p_ts = p_ts[:min_len]
            t_ts = t_ts[:min_len]

            # Time deltas consistent with energy calculator (np.diff with prepend)
            dt = np.diff(t_ts, prepend=0.0)
            # Clamp negative dt (should never happen) to zero
            dt[dt < 0] = 0.0

            is_stationary = np.abs(vel) < v_thresh  # boolean mask per frame
            sep_wh = 0.0
            if np.any(is_stationary):
                # Find contiguous stationary blocks via edge detection
                padded = np.concatenate([[False], is_stationary, [False]])
                edges = np.diff(padded.astype(np.int8))
                starts = np.where(edges == 1)[0]  # transition into stationary
                ends = np.where(edges == -1)[0]   # transition out of stationary

                for s, e in zip(starts, ends):
                    block_duration = float(np.sum(dt[s:e]))
                    if block_duration >= 0.5:  # hysteresis: ignore sub-0.5 s micro-jitter blocks
                        sep_wh += float(np.sum(p_ts[s:e] * dt[s:e])) / 3600.0

            result["standstill_energy_penalty_wh"] = float(sep_wh)
        else:
            result["standstill_energy_penalty_wh"] = None

        return result
