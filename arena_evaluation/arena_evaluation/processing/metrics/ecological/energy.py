from __future__ import annotations

import typing
import polars as pl
import numpy as np
from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class EnergyMetricCalculator(BaseMetricCalculator):
    NAME = "energy"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["power", "energy"]
    
    UNITS = {
        "energy_static_wh": "Wh",
        "energy_mechanical_wh": "Wh",
        "energy_thermal_wh": "Wh",
        "energy_total_wh": "Wh",
        "battery_soc_final": "%",
    }

    def output_keys(self) -> list[str]:
        return [
            # Scalars (aggregates for the episode)
            "energy_static_wh",
            "energy_mechanical_wh",
            "energy_thermal_wh",
            "energy_total_wh",
            "battery_soc_final",
            # Timeseries (arrays)
            "timeseries_power_total_w",
            "timeseries_power_static_w",
            "timeseries_power_mechanical_w",
            "timeseries_power_thermal_w",
            "timeseries_battery_soc",
            "timeseries_velocity_linear",
            "timeseries_time_s",
        ]

    def calculate(self, episode: "AlignedEpisodeBundle", dependencies: dict[str, typing.Any]) -> dict[str, typing.Any]:
        df = episode.data
        if df is None or df.is_empty():
            return {k: None for k in self.output_keys()}
            
        # We need a time axis in seconds relative to start
        if "time_ns" in df.columns:
            t_ns = df["time_ns"].to_numpy()
            t_s = (t_ns - t_ns[0]) * 1e-9
        else:
            t_s = np.zeros(len(df))

        # Power timeseries
        p_total = df["total_power_w"].to_numpy() if "total_power_w" in df.columns else np.zeros_like(t_s)
        p_static = df["static_power_w"].to_numpy() if "static_power_w" in df.columns else np.zeros_like(t_s)
        p_mech = df["total_mechanical_power_w"].to_numpy() if "total_mechanical_power_w" in df.columns else np.zeros_like(t_s)
        p_therm = df["total_thermal_power_w"].to_numpy() if "total_thermal_power_w" in df.columns else np.zeros_like(t_s)
        
        # We need to fill nulls which happen if the 'power' topic was joined but had missing data at odom timestamps (backward join leaves nulls at the start)
        def fill_nulls(arr):
            arr = np.array(arr, dtype=float)
            mask = np.isnan(arr)
            if np.all(mask):
                return np.zeros_like(arr)
            # Forward fill, then backward fill
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, out=idx)
            out = arr[idx]
            
            # backward fill remaining
            mask = np.isnan(out)
            if np.any(mask):
                valid_idx = np.where(~np.isnan(arr))[0]
                if len(valid_idx) > 0:
                    out[mask] = arr[valid_idx[0]]
                else:
                    out[mask] = 0.0
            return out

        p_total = fill_nulls(p_total)
        p_static = fill_nulls(p_static)
        p_mech = fill_nulls(p_mech)
        p_therm = fill_nulls(p_therm)
        
        # Velocity timeseries
        vel = df["vel_linear"].to_numpy() if "vel_linear" in df.columns else np.zeros_like(t_s)
        vel = fill_nulls(vel)
        
        # Battery timeseries
        batt = df["battery_soc_percent"].to_numpy() if "battery_soc_percent" in df.columns else np.zeros_like(t_s)
        batt = fill_nulls(batt)

        # Integration for total energy consumption over the episode
        # Energy = integral of Power dt
        dt = np.diff(t_s, prepend=0.0)
        e_static = np.sum(p_static * dt) / 3600.0
        e_mech = np.sum(p_mech * dt) / 3600.0
        e_therm = np.sum(p_therm * dt) / 3600.0
        
        # Alternative: use the final value from the /energy topic
        # The /energy topic publishes cumulative energy since node start. 
        # The energy used in THIS episode is the final value minus the initial value.
        if "total_energy_consumed_wh" in df.columns:
            energy_arr = fill_nulls(df["total_energy_consumed_wh"].to_numpy())
            if len(energy_arr) > 0:
                e_total = energy_arr[-1] - energy_arr[0]
            else:
                e_total = np.sum(p_total * dt) / 3600.0
        else:
            e_total = np.sum(p_total * dt) / 3600.0
            
        batt_final = batt[-1] if len(batt) > 0 else 0.0

        return {
            "energy_static_wh": float(e_static),
            "energy_mechanical_wh": float(e_mech),
            "energy_thermal_wh": float(e_therm),
            "energy_total_wh": float(e_total),
            "battery_soc_final": float(batt_final),
            "timeseries_power_total_w": p_total.tolist(),
            "timeseries_power_static_w": p_static.tolist(),
            "timeseries_power_mechanical_w": p_mech.tolist(),
            "timeseries_power_thermal_w": p_therm.tolist(),
            "timeseries_battery_soc": batt.tolist(),
            "timeseries_velocity_linear": vel.tolist(),
            "timeseries_time_s": t_s.tolist(),
        }
