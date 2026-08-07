"""Layer 3 offline characterization analysis.

Reads the extracted topic cache (Layer 3 Parquet), aligns multi-rate topics onto
the odometry axis, and maps energy expenditure (J/m, W) and acoustic estimates
(L_Aeq,T / L_AFmax) to each open-loop maneuver working point.

Vectorized Polars only — no Pandas, no Python loops over data rows.
"""

from __future__ import annotations

import math
import pathlib
import typing

import polars as pl

from ..processing.parquet_store import TopicParquetStore
from ..processing.topic_aligner import TopicAligner
from ..storage.folder_manager import FolderManager
from .maneuvers import Phase, PhaseKind, build_schedule, classify_cmd_point, resolve_envelope

if typing.TYPE_CHECKING:
    from ..storage.schemas import TopicBundle

# ── Acoustic fallback model (mirrors acoustics_publisher.py) ────────────────
# Generic defaults; the per-robot profile (robots/<model>/acoustic_profile.yaml)
# is authoritative and is only used when the /acoustics topic was not recorded.
_ACOUSTIC_MODEL = {
    "L_base_0": 42.0,
    "beta_0": 45.0,
    "beta_1": 18.0,
    "beta_2": 5.0,
    "omega_ref": 5.0,
    "tau_ref": 10.0,
    "omega_active": 0.2,
}


def _load_acoustic_profile(robot_name: str | None = None) -> dict:
    """Load the robot's acoustic profile yaml when available (fallback model)."""
    try:
        import yaml
        from ament_index_python.packages import get_package_share_directory
        from pathlib import Path as _P

        share = _P(get_package_share_directory("arena_robots"))
        candidates = []
        if robot_name:
            candidates.append(share / "robots" / robot_name / "telemetry" / "acoustics.yaml")
        candidates.append(share / "config" / "acoustic_profile.yaml")
        for cand in candidates:
            if cand.is_file():
                cfg = yaml.safe_load(cand.read_text())
                return {k: float(cfg.get(k, _ACOUSTIC_MODEL[k])) for k in _ACOUSTIC_MODEL}
    except Exception:
        pass
    return dict(_ACOUSTIC_MODEL)


# ── Pure vectorized helpers (unit-testable without ROS) ─────────────────────

def leq_db(levels: pl.Series) -> float:
    """Equivalent continuous level: L_Aeq = 10·log10(mean(10^(L/10)))."""
    levels = levels.cast(pl.Float64).drop_nulls()
    if len(levels) == 0:
        return float("nan")
    return float(10.0 * math.log10((10.0 ** (levels / 10.0)).mean()))


def add_distance(df: pl.DataFrame) -> pl.DataFrame:
    """Per-sample travelled distance (m) from odometry position diffs."""
    return df.with_columns(
        ((pl.col("pos_x").diff().fill_null(0.0).cast(pl.Float64) ** 2)
         + (pl.col("pos_y").diff().fill_null(0.0).cast(pl.Float64) ** 2))
        .sqrt()
        .clip(lower_bound=0.0)
        .alias("ds_m")
    )


def add_sample_dt(df: pl.DataFrame) -> pl.DataFrame:
    """Per-sample time step (s) from the odometry time axis."""
    return df.with_columns(
        (pl.col("time_ns").diff().fill_null(0).cast(pl.Float64) / 1e9)
        .clip(lower_bound=0.0)
        .alias("dt_s")
    )


def add_mechanical_power(df: pl.DataFrame) -> pl.DataFrame:
    """P_mech = Σ_i |τ_i|·ω_i over all joints, vectorized over list columns."""
    return df.with_columns(
        (
            pl.col("joint_effort").list.eval(pl.element().abs())
            * pl.col("joint_velocity")
        )
        .list.sum()
        .alias("p_mech_w")
    )


def add_fallback_acoustic_level(df: pl.DataFrame, model: dict | None = None) -> pl.DataFrame:
    """Steady-state dBA estimate from joint states when the /acoustics topic is absent."""
    m = model or _load_acoustic_profile()
    omega_ref = m["omega_ref"]
    p_base = 10.0 ** (m["L_base_0"] / 10.0)
    p_intercept = 10.0 ** (m["beta_0"] / 10.0)

    omega_eq = pl.col("joint_velocity").list.eval(
        pl.element().pow(2).mean().sqrt()
    ).cast(pl.Float64).fill_null(0.0).clip(lower_bound=m["omega_active"])
    t_eq = pl.col("joint_effort").list.eval(pl.element().abs().mean()).cast(pl.Float64).fill_null(0.0)

    p_drive = (
        p_intercept
        * (omega_eq / omega_ref) ** (m["beta_1"] / 10.0)
        * (1.0 + t_eq / m["tau_ref"]) ** (m["beta_2"] / 10.0)
    )
    return df.with_columns(
        (10.0 * (p_base + p_drive).log10()).alias("dba_estimated")
    )


def attach_phase_labels(df: pl.DataFrame, phase_df: pl.DataFrame | None, schedule: list[Phase]) -> pl.DataFrame:
    """Attach the maneuver phase to every sample.

    Phase markers are sparse and must carry forward indefinitely, so the asof
    join is done with strategy="backward" and NO tolerance. When markers are
    missing, falls back to classifying the recorded cmd_vel sample (vectorized).
    """
    out = df
    if phase_df is not None and len(phase_df) > 0:
        markers = phase_df.select(["time_ns", "label"]).sort("time_ns")
        out = out.sort("time_ns").join_asof(markers, on="time_ns", strategy="backward")
        out = out.with_columns(pl.col("label").fill_null("unknown").alias("phase_label"))
    else:
        out = out.with_columns(pl.lit("unknown").alias("phase_label"))

    phases = pl.DataFrame(
        [
            {"phase_label": p.name, "phase_kind": p.kind.value, "vx_target": p.vx_target, "wz_target": p.wz_target}
            for p in schedule
        ]
    )
    out = out.join(phases, on="phase_label", how="left")

    # Fallback classification for samples the markers never reached (e.g. the
    # recorder missed the marker topic): classify from the recorded cmd_vel.
    cmd_vx = pl.col("cmd_linear_x").cast(pl.Float64).fill_null(0.0)
    cmd_wz = pl.col("cmd_angular_z").cast(pl.Float64).fill_null(0.0)
    fb_kind = (
        pl.when((cmd_vx == 0.0) & (cmd_wz == 0.0)).then(pl.lit(PhaseKind.IDLE.value))
        .when(cmd_wz != 0.0).then(pl.lit(PhaseKind.ANGULAR.value))
        .otherwise(pl.lit(PhaseKind.LINEAR.value))
    )
    out = out.with_columns(
        pl.col("phase_kind").fill_null(fb_kind).alias("phase_kind"),
        pl.col("vx_target").fill_null(pl.when(cmd_vx > 0.0).then(cmd_vx).otherwise(0.0)).alias("vx_target"),
        pl.col("wz_target").fill_null(pl.when(cmd_wz != 0.0).then(cmd_wz).otherwise(0.0)).alias("wz_target"),
    )
    return out


def enrich_samples(df: pl.DataFrame, model: dict | None = None) -> pl.DataFrame:
    """Compute all per-sample characterization columns on an aligned frame."""
    out = add_distance(df)
    out = add_sample_dt(out)
    out = add_mechanical_power(out)
    out = out.with_columns(
        pl.col("power_total_power_w").cast(pl.Float64).alias("p_total_w"),
        pl.col("acous_total_level_af_dba").cast(pl.Float64).alias("dba"),
    )
    if out["dba"].null_count() == len(out):
        out = add_fallback_acoustic_level(out, model)
        out = out.with_columns(pl.col("dba_estimated").alias("dba"))
    out = out.with_columns((10.0 ** (pl.col("dba") / 10.0)).alias("_l_power"))
    out = out.with_columns(
        (pl.col("p_total_w") * pl.col("dt_s")).alias("_e_total_j"),
        (pl.col("p_mech_w") * pl.col("dt_s")).alias("_e_mech_j"),
        pl.col("vel_linear").cast(pl.Float64).alias("vx_achieved"),
    )
    return out


def summarize_samples(samples: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per (kind, vx_target, wz_target) working point."""
    key = ["phase_kind", "vx_target", "wz_target"]
    agg = (
        samples.group_by(key)
        .agg(
            pl.len().alias("n_samples"),
            pl.col("p_total_w").mean().alias("mean_power_total_w"),
            pl.col("p_total_w").std().alias("std_power_total_w"),
            pl.col("p_mech_w").mean().alias("mean_p_mech_w"),
            pl.col("_l_power").mean().alias("_mean_l_power"),
            pl.col("dba").max().alias("lafmax_af_dba"),
            pl.col("_e_total_j").sum().alias("e_total_j"),
            pl.col("_e_mech_j").sum().alias("e_mech_j"),
            pl.col("ds_m").sum().alias("dist_m"),
            pl.col("dt_s").sum().alias("duration_s"),
            pl.col("vx_achieved").mean().alias("mean_vx_achieved"),
        )
        .with_columns((10.0 * pl.col("_mean_l_power").log10()).alias("leq_af_dba"))
        .with_columns(
            pl.when(pl.col("dist_m") > 1e-6)
            .then(pl.col("e_total_j") / pl.col("dist_m"))
            .otherwise(None)
            .alias("energy_intensity_j_per_m"),
            pl.when(pl.col("dist_m") > 1e-6)
            .then(pl.col("e_mech_j") / pl.col("dist_m"))
            .otherwise(None)
            .alias("mech_energy_intensity_j_per_m"),
        )
        .sort(key)
        .drop("_mean_l_power")
    )
    return agg


def merge_episode_summaries(summaries: list[pl.DataFrame]) -> pl.DataFrame:
    """Combine per-episode summaries into one with mean/std across episodes."""
    if not summaries:
        return pl.DataFrame()
    combined = pl.concat(summaries, how="diagonal_relaxed")
    key = ["phase_kind", "vx_target", "wz_target"]
    numeric = [
        "mean_power_total_w", "mean_p_mech_w", "leq_af_dba", "lafmax_af_dba",
        "energy_intensity_j_per_m", "mech_energy_intensity_j_per_m",
        "mean_vx_achieved", "dist_m", "duration_s", "e_total_j",
    ]
    exprs = [pl.len().alias("n_episodes"), pl.col("n_samples").sum().alias("n_samples")]
    for col in numeric:
        exprs.append(pl.col(col).mean().alias(col))
        exprs.append(pl.col(col).std().alias(f"{col}_std"))
    return combined.group_by(key).agg(exprs).sort(key)


# ── Orchestration ────────────────────────────────────────────────────────────

def analyze_episode(topics_dir: pathlib.Path, schedule: list[Phase], model: dict | None = None) -> pl.DataFrame:
    """Compute per-sample characterization data for one extracted episode."""
    bundles = TopicParquetStore.read(topics_dir)
    if not bundles:
        raise FileNotFoundError(f"No extracted topics found in {topics_dir}")

    # The robot bundle is the one carrying odometry.
    bundle: TopicBundle | None = next(
        (b for b in bundles.values() if getattr(b, "odom", None) is not None and len(b.odom) > 0),
        None,
    )
    if bundle is None:
        raise ValueError(f"No odometry bundle found in {topics_dir}")

    aligned = TopicAligner().align(bundle)
    if aligned is None:
        raise ValueError(f"No aligned data produced for {topics_dir}")
    if isinstance(aligned, pl.LazyFrame):
        aligned = aligned.collect()

    aligned = attach_phase_labels(aligned, bundle.characterization_phase, schedule)
    samples = enrich_samples(aligned, model=model)
    samples = samples.filter(pl.col("phase_kind").is_not_null() & (pl.col("phase_kind") != "unknown"))
    return samples


def _episode_robot_name(episode_dir: pathlib.Path) -> str | None:
    """Robot model for an episode, read from its metadata yaml (robot_model[0])."""
    try:
        from ..storage.manifest import MetadataWriter

        yaml_path = episode_dir / f"{episode_dir.name}.yaml"
        if not yaml_path.exists():
            yaml_path = episode_dir / "metadata.yaml"
        if yaml_path.exists():
            meta = MetadataWriter.read(yaml_path)
            models = meta.robot_model or []
            return models[0] if models else None
    except Exception:
        pass
    return None


def run_characterization(benchmark_dir: pathlib.Path, output_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Run characterization over all episodes of a benchmark and write the summaries.

    Per-episode: the robot model from the episode metadata drives both the
    maneuver schedule envelope and the acoustic profile, so any robot recorded
    through the arena is characterized with its own operating limits.
    """
    fm = FolderManager(data_root=benchmark_dir.parent)
    episodes = fm.discover_episodes(benchmark_dir.name)
    if not episodes:
        raise FileNotFoundError(f"No episodes found for benchmark '{benchmark_dir.name}'")

    # Robot-specific schedule + acoustic profile, cached per robot model.
    schedule_cache: dict[str | None, list[Phase]] = {}
    model_cache: dict[str | None, dict] = {}

    all_samples: list[pl.DataFrame] = []
    summaries: list[pl.DataFrame] = []
    for ep in episodes:
        episode_dir = pathlib.Path(ep.episode_dir)
        robot = _episode_robot_name(episode_dir)
        if robot not in schedule_cache:
            envelope = resolve_envelope(robot)
            schedule_cache[robot] = build_schedule(vx_max=envelope["vx_max"], wz_max=envelope["wz_max"])
            model_cache[robot] = _load_acoustic_profile(robot)
        schedule = schedule_cache[robot]
        model = model_cache[robot]

        topics_dir = fm.extracted_topics_path_for_episode(episode_dir)
        try:
            samples = analyze_episode(topics_dir, schedule, model=model)
        except Exception as e:
            print(f"  [warn] episode_{ep.episode_id:03d} (robot={robot or 'unknown'}): {e}")
            continue
        samples = samples.with_columns(
            pl.lit(ep.episode_id).alias("episode_id"),
            pl.lit(robot or "unknown").alias("robot"),
        )
        all_samples.append(samples)
        summaries.append(summarize_samples(samples))
        print(f"  episode_{ep.episode_id:03d} (robot={robot or 'unknown'}): {len(samples)} samples, {len(summaries[-1])} working points")

    if not all_samples:
        raise FileNotFoundError(f"No usable characterization data in '{benchmark_dir.name}'")

    out = output_dir or benchmark_dir
    summary_path = out / "characterization_summary.parquet"
    samples_path = out / "characterization_samples.parquet"
    merged = merge_episode_summaries(summaries)
    merged.write_parquet(summary_path)
    merged.write_csv(summary_path.with_suffix(".csv"))
    pl.concat(all_samples, how="diagonal_relaxed").write_parquet(samples_path)
    print(f"Summary ({len(merged)} working points) → {summary_path}")
    print(f"Samples ({sum(len(s) for s in all_samples)} rows) → {samples_path}")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="characterize", description="Open-loop energy/acoustic characterization (Layer 3)")
    p.add_argument("--benchmark-dir", type=pathlib.Path, help="Benchmark directory containing episodes/")
    p.add_argument("--output-dir", type=pathlib.Path, default=None, help="Where to write the summaries (defaults to benchmark dir)")
    args = p.parse_args(argv)

    if not args.benchmark_dir or not args.benchmark_dir.is_dir():
        p.error("--benchmark-dir must point to a benchmark directory")
    run_characterization(args.benchmark_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
