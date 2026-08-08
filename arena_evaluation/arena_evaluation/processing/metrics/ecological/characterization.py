"""Open-loop characterization metrics (energy/acoustic profiles per working point).

Per-episode calculator: attaches the recorded ``characterization_phase`` markers
to every sample, computes per-sample power / mechanical power / acoustic level /
energy intensity, and exposes them as ``timeseries_char_*`` list columns in the
metrics row — the same wide per-episode shape as the energy calculator's
``timeseries_power_*`` columns. The report layer derives long-format frames and
per-working-point aggregates from these columns (see the ``line`` plot type and
the ``characterization`` report manifest).
"""

from __future__ import annotations

import typing

import polars as pl

from ..base import BaseMetricCalculator
from ....storage.schemas import AlignedEpisodeBundle

if typing.TYPE_CHECKING:
    from ....storage.schemas import RobotParams


_ACOUSTIC_DEFAULTS = {
    "L_base_0": 42.0,
    "beta_0": 45.0,
    "beta_1": 18.0,
    "beta_2": 5.0,
    "omega_ref": 5.0,
    "tau_ref": 10.0,
    "omega_active": 0.2,
}


def _leq_power(dba: pl.Series) -> pl.Series:
    """Linear acoustic power proxy 10^(L/10) — L_Aeq = 10·log10(mean(·))."""
    return 10.0 ** (dba / 10.0)


class CharacterizationCalculator(BaseMetricCalculator):
    NAME = "characterization"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["odom"]

    UNITS = {
        "timeseries_char_time_s": "s",
        "timeseries_char_power_total_w": "W",
        "timeseries_char_power_mech_w": "W",
        "timeseries_char_dba": "dBA",
        "timeseries_char_vx_achieved": "m/s",
        "timeseries_char_vx_target": "m/s",
        "timeseries_char_wz_target": "rad/s",
        "timeseries_char_energy_intensity": "J/m",
    }

    _TIMESERIES_KEYS = [
        "timeseries_char_time_s",
        "timeseries_char_power_total_w",
        "timeseries_char_power_mech_w",
        "timeseries_char_dba",
        "timeseries_char_vx_achieved",
        "timeseries_char_phase_kind",
        "timeseries_char_vx_target",
        "timeseries_char_wz_target",
        "timeseries_char_energy_intensity",
        "timeseries_char_leq_power",
    ]

    def output_keys(self) -> list[str]:
        return list(self._TIMESERIES_KEYS)

    # ── Phase labelling ──────────────────────────────────────────────────────

    def _phase_map(self) -> pl.DataFrame:
        """Schedule name → (kind, vx_target, wz_target) for this robot's envelope."""
        from task_generator.tasks.robots.characterization.schedule import (
            build_schedule,
            resolve_envelope,
        )

        envelope = resolve_envelope(self.robot_params.model)
        schedule = build_schedule(vx_max=envelope["vx_max"], wz_max=envelope["wz_max"])
        return pl.DataFrame(
            [
                {"phase_label": p.name, "phase_kind": p.kind.value, "vx_target": p.vx_target, "wz_target": p.wz_target}
                for p in schedule
            ]
        )

    def _attach_phases(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df
        if "label" not in out.columns:
            out = out.with_columns(pl.lit("unknown").alias("phase_label"))
        else:
            out = out.with_columns(pl.col("label").fill_null("unknown").alias("phase_label"))
        out = out.join(self._phase_map(), on="phase_label", how="left")

        # Fallback classification for unmarked samples (markers never recorded).
        cmd_vx = (
            pl.col("linear_x").cast(pl.Float64).fill_null(0.0)
            if "linear_x" in out.columns else pl.lit(0.0)
        )
        cmd_wz = (
            pl.col("angular_z").cast(pl.Float64).fill_null(0.0)
            if "angular_z" in out.columns else pl.lit(0.0)
        )
        fb_kind = (
            pl.when((cmd_vx == 0.0) & (cmd_wz == 0.0)).then(pl.lit("idle"))
            .when(cmd_wz != 0.0).then(pl.lit("angular"))
            .otherwise(pl.lit("linear"))
        )
        return out.with_columns(
            pl.col("phase_kind").fill_null(fb_kind).alias("phase_kind"),
            pl.col("vx_target").fill_null(pl.when(cmd_vx > 0.0).then(cmd_vx).otherwise(0.0)).alias("vx_target"),
            pl.col("wz_target").fill_null(pl.when(cmd_wz != 0.0).then(cmd_wz).otherwise(0.0)).alias("wz_target"),
        )

    # ── Acoustic fallback (mirrors acoustics_publisher.py) ────────────────────

    def _acoustic_model(self) -> dict:
        try:
            import yaml
            from ament_index_python.packages import get_package_share_directory
            from pathlib import Path

            share = Path(get_package_share_directory("arena_robots"))
            for cand in (
                share / "robots" / self.robot_params.model / "acoustic_profile.yaml",
                share / "config" / "acoustic_profile.yaml",
            ):
                if cand.is_file():
                    cfg = yaml.safe_load(cand.read_text())
                    return {
                        k: float(cfg.get(k, _ACOUSTIC_DEFAULTS[k]))
                        for k in _ACOUSTIC_DEFAULTS
                    }
        except Exception:
            pass
        return dict(_ACOUSTIC_DEFAULTS)

    # ── Per-sample enrichment ────────────────────────────────────────────────

    def _enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.with_columns(
            (pl.col("time_ns").diff().fill_null(0).cast(pl.Float64) / 1e9).clip(lower_bound=0.0).alias("_dt"),
            (
                (pl.col("pos_x").diff().fill_null(0.0).cast(pl.Float64) ** 2)
                + (pl.col("pos_y").diff().fill_null(0.0).cast(pl.Float64) ** 2)
            ).sqrt().clip(lower_bound=0.0).alias("_ds"),
        )
        # P_mech = Σ |τ·ω| over the joints (list columns, unprefixed).
        if "effort" in out.columns and "velocity" in out.columns:
            out = out.with_columns(
                (pl.col("effort") * pl.col("velocity")).list.eval(pl.element().abs()).list.sum()
                .fill_null(0.0).alias("_p_mech")
            )
        else:
            out = out.with_columns(pl.lit(0.0).alias("_p_mech"))

        out = out.with_columns(
            pl.col("total_power_w").cast(pl.Float64).fill_null(0.0).alias("_p_total")
            if "total_power_w" in out.columns else pl.lit(0.0).alias("_p_total"),
            pl.col("total_level_af_dba").cast(pl.Float64).alias("_dba")
            if "total_level_af_dba" in out.columns else pl.lit(None, dtype=pl.Float64).alias("_dba"),
        )
        if out["_dba"].null_count() == len(out) and "velocity" in out.columns:
            # Fallback: steady-state drivetrain model from joint states.
            m = self._acoustic_model()
            omega_eq = (
                (pl.col("velocity") ** 2).list.mean().sqrt()
                .cast(pl.Float64).fill_null(0.0).clip(lower_bound=m["omega_active"])
            )
            t_eq = (
                pl.col("effort").list.eval(pl.element().abs()).list.mean()
                .cast(pl.Float64).fill_null(0.0)
                if "effort" in out.columns else pl.lit(0.0)
            )
            p_drive = (
                (10.0 ** (m["beta_0"] / 10.0))
                * (omega_eq / m["omega_ref"]) ** (m["beta_1"] / 10.0)
                * (1.0 + t_eq / m["tau_ref"]) ** (m["beta_2"] / 10.0)
            )
            out = out.with_columns(
                (10.0 * ((10.0 ** (m["L_base_0"] / 10.0)) + p_drive).log10()).alias("_dba")
            )
        else:
            out = out.with_columns(pl.col("_dba").fill_null(0.0))

        return out

    def calculate(self, episode: AlignedEpisodeBundle, dependencies: dict[str, typing.Any]) -> dict[str, typing.Any]:
        df = episode.data
        if df is None or len(df) == 0:
            return {k: None for k in self.output_keys()}

        out = self._attach_phases(df)
        out = self._enrich(out)

        t_s = (out["time_ns"].cast(pl.Float64) - out["time_ns"].cast(pl.Float64).min()) / 1e9
        ds = out["_ds"].fill_null(0.0)
        dt = out["_dt"].fill_null(0.0)
        out = out.with_columns(
            pl.when(ds > 1e-6).then(out["_p_total"] * dt / ds).otherwise(None).alias("_e_per_m")
        )

        rows = {
            "timeseries_char_time_s": t_s.to_list(),
            "timeseries_char_power_total_w": out["_p_total"].to_list(),
            "timeseries_char_power_mech_w": out["_p_mech"].to_list(),
            "timeseries_char_dba": out["_dba"].to_list(),
            "timeseries_char_vx_achieved": out["vel_linear"].cast(pl.Float64).fill_null(0.0).to_list()
            if "vel_linear" in out.columns else [0.0] * len(out),
            "timeseries_char_phase_kind": out["phase_kind"].fill_null("unknown").to_list(),
            "timeseries_char_vx_target": out["vx_target"].fill_null(0.0).to_list(),
            "timeseries_char_wz_target": out["wz_target"].fill_null(0.0).to_list(),
            "timeseries_char_energy_intensity": out["_e_per_m"].to_list(),
            "timeseries_char_leq_power": _leq_power(out["_dba"].fill_null(0.0)).to_list(),
        }
        return rows
