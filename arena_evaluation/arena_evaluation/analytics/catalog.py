"""Benchmark OLAP Catalog (Layer 4).

Registers the nine Star Schema base tables (``dim_*`` and ``fact_*``) and
exposes canonical analytical views as pure Polars ``LazyFrame`` query plans.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping
from typing import Final

import polars as pl

from arena_evaluation.analytics.evidence import EVIDENCE_INDEX_SCHEMA
from arena_evaluation.storage.star_schema import (
    STAR_SCHEMA_SCHEMAS,
    STAR_TABLE_NAMES,
    StarSchemaTables,
    scan_star_schema,
)

__all__ = [
    "CANONICAL_VIEW_NAMES",
    "TELEMETRY_TABLE_NAMES",
    "TELEMETRY_TABLE_SCHEMAS",
    "BenchmarkCatalog",
    "StarCatalog",
    "synthesize_star_schema_from_combined",
]

CANONICAL_VIEW_NAMES: Final[tuple[str, ...]] = (
    "v_episode_summary",
    "v_planner_scorecard",
    "v_contact_blame_summary",
    "v_encounter_summary",
)

TELEMETRY_TABLE_NAMES: Final[tuple[str, ...]] = (
    "robot_tracks",
    "pedestrian_tracks",
    "collision_contacts",
    "robot_power",
    "robot_acoustics",
)

TELEMETRY_TABLE_SCHEMAS: Final[dict[str, dict[str, pl.DataType]]] = {
    "robot_tracks": {
        "episode_id": pl.Int64,
        "timestamp_ns": pl.Int64,
        "t_ns": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "yaw": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "omega": pl.Float64,
        "speed_mps": pl.Float64,
    },
    "pedestrian_tracks": {
        "episode_id": pl.Int64,
        "timestamp_ns": pl.Int64,
        "t_ns": pl.Int64,
        "ped_id": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "yaw": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "speed_mps": pl.Float64,
    },
    "robot_power": {
        "episode_id": pl.Int64,
        "timestamp_ns": pl.Int64,
        "t_ns": pl.Int64,
        "mechanical_power_w": pl.Float64,
        "electrical_power_w": pl.Float64,
        "static_power_w": pl.Float64,
        "total_power_w": pl.Float64,
    },
    "robot_acoustics": {
        "episode_id": pl.Int64,
        "timestamp_ns": pl.Int64,
        "t_ns": pl.Int64,
        "sound_pressure_pa": pl.Float64,
        "sound_exposure_dose_pa2s": pl.Float64,
    },
    "collision_contacts": {
        "episode_id": pl.Int64,
        "timestamp_ns": pl.Int64,
        "t_ns": pl.Int64,
        "contact_id": pl.String,
        "other_id": pl.String,
        "impact_speed_mps": pl.Float64,
        "robot_closing_share": pl.Float64,
        "closer_label": pl.String,
    },
}

_EPISODE_DIR_RE: Final[re.Pattern[str]] = re.compile(r"^episode_(\d+)$")


def synthesize_star_schema_from_combined(
    combined_lf: pl.LazyFrame,
    default_run_id: str = "default_run",
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Synthesize dim_episodes, dim_robots, and fact_episodes from combined_metrics.parquet."""
    names = set(combined_lf.collect_schema().names())

    if "run_id" in names:
        run_id_expr = pl.col("run_id").cast(pl.String)
    elif "benchmark_id" in names:
        run_id_expr = pl.col("benchmark_id").cast(pl.String)
    else:
        run_id_expr = pl.lit(default_run_id, dtype=pl.String)

    if "benchmark_id" in names:
        bench_id_expr = pl.col("benchmark_id").cast(pl.String)
    elif "run_id" in names:
        bench_id_expr = pl.col("run_id").cast(pl.String)
    else:
        bench_id_expr = pl.lit(default_run_id, dtype=pl.String)

    if "episode_id" in names:
        ep_id_expr = pl.col("episode_id").cast(pl.Int64)
    elif "episode" in names:
        ep_id_expr = pl.col("episode").cast(pl.Int64)
    else:
        ep_id_expr = pl.int_range(0, pl.len(), dtype=pl.Int64)

    if "robot_id" in names:
        robot_id_expr = pl.col("robot_id").cast(pl.String)
    elif "robot" in names:
        robot_id_expr = pl.col("robot").cast(pl.String)
    else:
        robot_id_expr = pl.lit("robot_0", dtype=pl.String)

    if "robot_model" in names:
        robot_model_expr = pl.col("robot_model").cast(pl.String)
    elif "robot" in names:
        robot_model_expr = pl.col("robot").cast(pl.String)
    else:
        robot_model_expr = pl.lit("unknown", dtype=pl.String)

    if "world" in names:
        world_expr = pl.col("world").cast(pl.String)
    elif "map" in names:
        world_expr = pl.col("map").cast(pl.String)
    else:
        world_expr = pl.lit("unknown", dtype=pl.String)

    if "stage" in names:
        stage_expr = pl.col("stage").cast(pl.String)
    elif "scenario" in names:
        stage_expr = pl.col("scenario").cast(pl.String)
    else:
        stage_expr = pl.lit("default", dtype=pl.String)

    if "scenario" in names:
        scenario_expr = pl.col("scenario").cast(pl.String)
    elif "stage" in names:
        scenario_expr = pl.col("stage").cast(pl.String)
    else:
        scenario_expr = pl.lit("default", dtype=pl.String)

    planner_expr = pl.col("planner").cast(pl.String) if "planner" in names else pl.lit("unknown", dtype=pl.String)
    inter_planner_expr = pl.col("inter_planner").cast(pl.String) if "inter_planner" in names else pl.lit("none", dtype=pl.String)
    dispatcher_expr = pl.col("dispatcher").cast(pl.String) if "dispatcher" in names else pl.lit("single", dtype=pl.String)
    crowd_cond_expr = pl.col("crowd_condition").cast(pl.String) if "crowd_condition" in names else pl.lit("default", dtype=pl.String)
    ped_policy_expr = pl.col("ped_policy_default").cast(pl.String) if "ped_policy_default" in names else pl.lit("none", dtype=pl.String)

    if "crowd_size" in names:
        crowd_size_expr = pl.col("crowd_size").cast(pl.Int64)
    elif "num_pedestrians" in names:
        crowd_size_expr = pl.col("num_pedestrians").cast(pl.Int64)
    else:
        crowd_size_expr = pl.lit(0, dtype=pl.Int64)

    seed_expr = pl.col("seed").cast(pl.Int64) if "seed" in names else pl.lit(0, dtype=pl.Int64)
    is_ref_expr = pl.col("is_reference").cast(pl.Boolean).fill_null(False) if "is_reference" in names else pl.lit(False)
    ref_type_expr = pl.col("reference_type").cast(pl.String) if "reference_type" in names else pl.lit(None, dtype=pl.String)

    if "duration_s" in names:
        duration_expr = pl.col("duration_s").cast(pl.Float64)
    elif "time_to_goal_s" in names:
        duration_expr = pl.col("time_to_goal_s").cast(pl.Float64)
    elif "time_to_goal" in names:
        duration_expr = pl.col("time_to_goal").cast(pl.Float64)
    elif "time" in names:
        duration_expr = pl.col("time").cast(pl.Float64)
    elif "outcome_info" in names:
        duration_expr = pl.col("outcome_info").str.extract(r"(\d+(?:\.\d+)?)s", 1).cast(pl.Float64, strict=False).fill_null(0.0)
    else:
        duration_expr = pl.lit(0.0, dtype=pl.Float64)

    if "outcome_state" in names:
        outcome_state_expr = pl.col("outcome_state").cast(pl.Int64)
    elif "success" in names:
        outcome_state_expr = pl.when(pl.col("success").fill_null(False)).then(pl.lit(2, dtype=pl.Int64)).otherwise(pl.lit(3, dtype=pl.Int64))
    else:
        outcome_state_expr = pl.lit(0, dtype=pl.Int64)

    if "outcome_info" in names:
        outcome_info_expr = pl.col("outcome_info").cast(pl.String)
    elif "result" in names:
        outcome_info_expr = pl.col("result").cast(pl.String)
    elif "status_reason" in names:
        outcome_info_expr = pl.col("status_reason").cast(pl.String)
    else:
        outcome_info_expr = pl.lit("", dtype=pl.String)

    dim_episodes = combined_lf.select(
        run_id=run_id_expr,
        episode_id=ep_id_expr,
        env_id=pl.col("env_id").cast(pl.String) if "env_id" in names else pl.lit("env_0", dtype=pl.String),
        robot_id=robot_id_expr,
        benchmark_id=bench_id_expr,
        suite_name=pl.col("suite_name").cast(pl.String) if "suite_name" in names else pl.lit("", dtype=pl.String),
        contest_name=pl.col("contest_name").cast(pl.String) if "contest_name" in names else pl.lit("", dtype=pl.String),
        stage=stage_expr,
        world=world_expr,
        scenario=scenario_expr,
        planner=planner_expr,
        inter_planner=inter_planner_expr,
        dispatcher=dispatcher_expr,
        robot_model=robot_model_expr,
        crowd_condition=crowd_cond_expr,
        ped_policy_default=ped_policy_expr,
        crowd_size=crowd_size_expr,
        seed=seed_expr,
        is_reference=is_ref_expr,
        reference_type=ref_type_expr,
        duration_s=duration_expr,
        outcome_state=outcome_state_expr,
        outcome_info=outcome_info_expr,
    )

    if "result" in names:
        result_expr = pl.col("result").cast(pl.String)
    elif "status" in names:
        result_expr = pl.col("status").cast(pl.String)
    else:
        result_expr = pl.lit("UNKNOWN", dtype=pl.String)

    success_expr = pl.col("success").cast(pl.Boolean).fill_null(False) if "success" in names else pl.lit(False, dtype=pl.Boolean)

    if "time_to_goal_s" in names:
        ttg_expr = pl.col("time_to_goal_s").cast(pl.Float64)
    elif "time_to_goal" in names:
        ttg_expr = pl.col("time_to_goal").cast(pl.Float64)
    elif "time" in names:
        ttg_expr = pl.col("time").cast(pl.Float64)
    elif "duration_s" in names:
        ttg_expr = pl.col("duration_s").cast(pl.Float64)
    elif "outcome_info" in names:
        ttg_expr = pl.col("outcome_info").str.extract(r"(\d+(?:\.\d+)?)s", 1).cast(pl.Float64, strict=False)
    else:
        ttg_expr = pl.lit(None, dtype=pl.Float64)

    if "path_length_m" in names:
        pl_expr = pl.col("path_length_m").cast(pl.Float64)
    elif "path_length" in names:
        pl_expr = pl.col("path_length").cast(pl.Float64)
    else:
        pl_expr = pl.lit(None, dtype=pl.Float64)

    if "path_efficiency" in names:
        pe_expr = pl.col("path_efficiency").cast(pl.Float64)
    elif "spl" in names:
        pe_expr = pl.col("spl").cast(pl.Float64)
    else:
        pe_expr = pl.lit(None, dtype=pl.Float64)

    if "total_energy_wh" in names:
        energy_expr = pl.col("total_energy_wh").cast(pl.Float64)
    elif "energy_total_wh" in names:
        energy_expr = pl.col("energy_total_wh").cast(pl.Float64)
    else:
        energy_expr = pl.lit(None, dtype=pl.Float64)

    if "energy_per_meter_j_m" in names:
        epm_expr = pl.col("energy_per_meter_j_m").cast(pl.Float64)
    elif "energy_per_meter" in names:
        epm_expr = pl.col("energy_per_meter").cast(pl.Float64)
    else:
        epm_expr = pl.lit(None, dtype=pl.Float64)

    if "mean_velocity_mps" in names:
        vel_expr = pl.col("mean_velocity_mps").cast(pl.Float64)
    elif "velocity_mean" in names:
        vel_expr = pl.col("velocity_mean").cast(pl.Float64)
    else:
        vel_expr = pl.lit(None, dtype=pl.Float64)

    if "max_acceleration_mps2" in names:
        acc_expr = pl.col("max_acceleration_mps2").cast(pl.Float64)
    elif "acceleration_max" in names:
        acc_expr = pl.col("acceleration_max").cast(pl.Float64)
    else:
        acc_expr = pl.lit(None, dtype=pl.Float64)

    if "jerk_max_mps3" in names:
        jerk_expr = pl.col("jerk_max_mps3").cast(pl.Float64)
    elif "jerk_max" in names:
        jerk_expr = pl.col("jerk_max").cast(pl.Float64)
    else:
        jerk_expr = pl.lit(None, dtype=pl.Float64)

    if "min_obstacle_clearance_m" in names:
        moc_expr = pl.col("min_obstacle_clearance_m").cast(pl.Float64)
    elif "min_obstacle_clearance" in names:
        moc_expr = pl.col("min_obstacle_clearance").cast(pl.Float64)
    else:
        moc_expr = pl.lit(None, dtype=pl.Float64)

    if "min_ped_clearance_m" in names:
        mpc_expr = pl.col("min_ped_clearance_m").cast(pl.Float64)
    elif "min_pedestrian_clearance" in names:
        mpc_expr = pl.col("min_pedestrian_clearance").cast(pl.Float64)
    else:
        mpc_expr = pl.lit(None, dtype=pl.Float64)

    if "total_collisions_ped" in names:
        tcp_expr = pl.col("total_collisions_ped").cast(pl.Int64)
    elif "collision_amount_pedestrian" in names:
        tcp_expr = pl.col("collision_amount_pedestrian").cast(pl.Int64)
    else:
        tcp_expr = pl.lit(0, dtype=pl.Int64)

    if "total_collisions_static" in names:
        tcs_expr = pl.col("total_collisions_static").cast(pl.Int64)
    elif "collision_amount_static" in names:
        tcs_expr = pl.col("collision_amount_static").cast(pl.Int64)
    elif "collision_amount_wall" in names:
        tcs_expr = pl.col("collision_amount_wall").cast(pl.Int64)
    elif "collision_amount" in names:
        tcs_expr = pl.col("collision_amount").cast(pl.Int64)
    else:
        tcs_expr = pl.lit(0, dtype=pl.Int64)

    if "success_no_robot_fault" in names:
        snrf_expr = pl.col("success_no_robot_fault").cast(pl.Boolean).fill_null(False)
    else:
        snrf_expr = success_expr

    mar_expr = pl.col("mar").cast(pl.Float64) if "mar" in names else pl.lit(None, dtype=pl.Float64)
    pfi_expr = pl.col("pfi").cast(pl.Float64) if "pfi" in names else pl.lit(None, dtype=pl.Float64)

    if "robot_path_dev_m" in names:
        rpd_expr = pl.col("robot_path_dev_m").cast(pl.Float64)
    elif "robot_path_deviation_m" in names:
        rpd_expr = pl.col("robot_path_deviation_m").cast(pl.Float64)
    else:
        rpd_expr = pl.lit(None, dtype=pl.Float64)

    if "total_acoustic_dose_pa2s" in names:
        dose_expr = pl.col("total_acoustic_dose_pa2s").cast(pl.Float64)
    elif "acoustic_cost" in names:
        dose_expr = pl.col("acoustic_cost").cast(pl.Float64)
    else:
        dose_expr = pl.lit(None, dtype=pl.Float64)

    if "null_reason" in names:
        null_reason_expr = pl.col("null_reason").cast(pl.String)
    elif "status_reason" in names:
        null_reason_expr = pl.col("status_reason").cast(pl.String)
    else:
        null_reason_expr = pl.lit(None, dtype=pl.String)

    fact_episodes = combined_lf.select(
        run_id=run_id_expr,
        episode_id=ep_id_expr,
        robot_id=robot_id_expr,
        result=result_expr,
        success=success_expr,
        time_to_goal_s=ttg_expr,
        path_length_m=pl_expr,
        path_efficiency=pe_expr,
        total_energy_wh=energy_expr,
        energy_per_meter_j_m=epm_expr,
        mean_velocity_mps=vel_expr,
        max_acceleration_mps2=acc_expr,
        jerk_max_mps3=jerk_expr,
        min_obstacle_clearance_m=moc_expr,
        min_ped_clearance_m=mpc_expr,
        total_collisions_ped=tcp_expr,
        total_collisions_static=tcs_expr,
        success_no_robot_fault=snrf_expr,
        mar=mar_expr,
        pfi=pfi_expr,
        robot_path_dev_m=rpd_expr,
        total_acoustic_dose_pa2s=dose_expr,
        null_reason=null_reason_expr,
    ).with_columns(
        time_to_goal=pl.col("time_to_goal_s"),
        path_length=pl.col("path_length_m"),
    )

    dim_robots = combined_lf.select(
        run_id=run_id_expr,
        episode_id=ep_id_expr,
        env_id=pl.col("env_id").cast(pl.String) if "env_id" in names else pl.lit("env_0", dtype=pl.String),
        robot_id=robot_id_expr,
        robot_model=robot_model_expr,
        planner=planner_expr,
        inter_planner=inter_planner_expr,
        dispatcher=dispatcher_expr,
        max_linear_vel=pl.col("max_linear_vel").cast(pl.Float64) if "max_linear_vel" in names else pl.lit(None, dtype=pl.Float64),
        max_angular_vel=pl.col("max_angular_vel").cast(pl.Float64) if "max_angular_vel" in names else pl.lit(None, dtype=pl.Float64),
        linear_acc=pl.col("linear_acc").cast(pl.Float64) if "linear_acc" in names else pl.lit(None, dtype=pl.Float64),
        linear_decel=pl.col("linear_decel").cast(pl.Float64) if "linear_decel" in names else pl.lit(None, dtype=pl.Float64),
        robot_radius=pl.col("robot_radius").cast(pl.Float64) if "robot_radius" in names else pl.lit(None, dtype=pl.Float64),
        robot_mass_kg=pl.col("robot_mass_kg").cast(pl.Float64) if "robot_mass_kg" in names else pl.lit(None, dtype=pl.Float64),
        static_power_w=pl.col("static_power_w").cast(pl.Float64) if "static_power_w" in names else pl.lit(None, dtype=pl.Float64),
        lidar_variant=pl.col("lidar_variant").cast(pl.String) if "lidar_variant" in names else pl.lit(None, dtype=pl.String),
        start_x=pl.col("start_x").cast(pl.Float64) if "start_x" in names else pl.lit(None, dtype=pl.Float64),
        start_y=pl.col("start_y").cast(pl.Float64) if "start_y" in names else pl.lit(None, dtype=pl.Float64),
        goal_x=pl.col("goal_x").cast(pl.Float64) if "goal_x" in names else pl.lit(None, dtype=pl.Float64),
        goal_y=pl.col("goal_y").cast(pl.Float64) if "goal_y" in names else pl.lit(None, dtype=pl.Float64),
    )

    return dim_episodes, dim_robots, fact_episodes


class BenchmarkCatalog:
    """Lazy Polars catalog of Star Schema tables, analytical views, and telemetry tables."""

    def __init__(
        self,
        source: pathlib.Path | StarSchemaTables | Mapping[str, pl.LazyFrame | pl.DataFrame] | None = None,
    ) -> None:
        self._tables: dict[str, pl.LazyFrame] = {}

        if isinstance(source, pathlib.Path):
            self._tables.update(scan_star_schema(source))
            self._register_telemetry_views(source)
            self._register_evidence_index(source)

            fact_table = self._tables.get("fact_episodes")
            has_fact_ep = False
            if fact_table is not None:
                try:
                    has_fact_ep = fact_table.select(pl.len()).collect().item() > 0
                except Exception:
                    has_fact_ep = False

            combined_file = source / "combined_metrics.parquet"
            if not combined_file.is_file():
                combined_file = source / "metrics" / "combined_metrics.parquet"

            if not has_fact_ep and combined_file.is_file():
                comb_lf = pl.scan_parquet(combined_file)
                dim_ep, dim_rob, fact_ep = synthesize_star_schema_from_combined(comb_lf, default_run_id=source.name)
                dim_table = self._tables.get("dim_episodes")
                dim_has_rows = False
                if dim_table is not None:
                    try:
                        dim_has_rows = dim_table.select(pl.len()).collect().item() > 0
                    except Exception:
                        pass
                if not dim_has_rows:
                    self._tables["dim_episodes"] = dim_ep

                rob_table = self._tables.get("dim_robots")
                rob_has_rows = False
                if rob_table is not None:
                    try:
                        rob_has_rows = rob_table.select(pl.len()).collect().item() > 0
                    except Exception:
                        pass
                if not rob_has_rows:
                    self._tables["dim_robots"] = dim_rob

                self._tables["fact_episodes"] = fact_ep

            if combined_file.is_file() and "combined_metrics" not in self._tables:
                self._tables["combined_metrics"] = pl.scan_parquet(combined_file)
        elif isinstance(source, StarSchemaTables):
            self._tables.update(source.lazy_by_name())
        elif isinstance(source, Mapping):
            for name, schema in STAR_SCHEMA_SCHEMAS.items():
                raw = source.get(name)
                if isinstance(raw, pl.LazyFrame):
                    self._tables[name] = raw
                elif isinstance(raw, pl.DataFrame):
                    self._tables[name] = raw.lazy()
                else:
                    self._tables[name] = pl.LazyFrame(schema=schema)
            for name, table in source.items():
                if name not in self._tables:
                    self._tables[name] = table.lazy() if isinstance(table, pl.DataFrame) else table
        else:
            for name, schema in STAR_SCHEMA_SCHEMAS.items():
                self._tables[name] = pl.LazyFrame(schema=schema)

        if "evidence_index" not in self._tables:
            self._tables["evidence_index"] = pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA)

        for tname, tschema in TELEMETRY_TABLE_SCHEMAS.items():
            if tname not in self._tables:
                lf = pl.LazyFrame(schema=tschema)
                self._tables[tname] = lf
                self._tables[f"telemetry_{tname}"] = lf

        for tname in TELEMETRY_TABLE_NAMES:
            if tname in self._tables:
                sch_names = set(self._tables[tname].collect_schema().names())
                if "timestamp_ns" in sch_names and "t_ns" not in sch_names:
                    self._tables[tname] = self._tables[tname].with_columns(t_ns=pl.col("timestamp_ns"))
                    self._tables[f"telemetry_{tname}"] = self._tables[tname]
                elif "t_ns" in sch_names and "timestamp_ns" not in sch_names:
                    self._tables[tname] = self._tables[tname].with_columns(timestamp_ns=pl.col("t_ns"))
                    self._tables[f"telemetry_{tname}"] = self._tables[tname]

        fact_names = set(self._tables["fact_episodes"].collect_schema().names())
        alias_exprs: list[pl.Expr] = []
        if "time_to_goal" not in fact_names and "time_to_goal_s" in fact_names:
            alias_exprs.append(pl.col("time_to_goal_s").alias("time_to_goal"))
        elif "time_to_goal_s" not in fact_names and "time_to_goal" in fact_names:
            alias_exprs.append(pl.col("time_to_goal").alias("time_to_goal_s"))
        if "path_length" not in fact_names and "path_length_m" in fact_names:
            alias_exprs.append(pl.col("path_length_m").alias("path_length"))
        elif "path_length_m" not in fact_names and "path_length" in fact_names:
            alias_exprs.append(pl.col("path_length").alias("path_length_m"))
        if alias_exprs:
            self._tables["fact_episodes"] = self._tables["fact_episodes"].with_columns(alias_exprs)

        self._register_canonical_views()

    @classmethod
    def from_directory(cls, benchmark_dir: pathlib.Path) -> BenchmarkCatalog:
        """Create a catalog scanning Parquet tables from ``benchmark_dir``."""
        return cls(benchmark_dir)

    @classmethod
    def from_tables(cls, tables: StarSchemaTables) -> BenchmarkCatalog:
        """Create a catalog from in-memory ``StarSchemaTables``."""
        return cls(tables)

    def register_table(self, name: str, frame: pl.LazyFrame | pl.DataFrame) -> None:
        """Register or replace a lazy table or derived view in the catalog."""
        if not name or not name.strip():
            raise ValueError("Table name must be a non-empty string.")
        self._tables[name] = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        if name in STAR_TABLE_NAMES:
            self._register_canonical_views()

    def get_table(self, name: str) -> pl.LazyFrame:
        """Return the ``LazyFrame`` registered under ``name``."""
        if name not in self._tables:
            raise KeyError(f"Unknown table or view '{name}'. Available: {sorted(self._tables)}")
        return self._tables[name]

    def table_names(self) -> tuple[str, ...]:
        """Return all registered base, view, and derived table names."""
        return tuple(self._tables.keys())

    def tables(self) -> dict[str, pl.LazyFrame]:
        """Return a shallow copy of all registered ``LazyFrame`` tables."""
        return dict(self._tables)

    def _register_telemetry_views(self, benchmark_dir: pathlib.Path) -> None:
        """Lazily scan episode telemetry Parquet tables across ``benchmark_dir / 'episodes'``."""
        episodes_root = benchmark_dir / "episodes"
        if not episodes_root.is_dir():
            return

        ep_dirs = sorted(
            {path for pattern in ("episode_*", "*/episode_*") for path in episodes_root.glob(pattern) if path.is_dir() and _EPISODE_DIR_RE.match(path.name) is not None},
            key=lambda p: int(p.name.split("_")[-1]),
        )
        if not ep_dirs:
            return

        frames_by_table: dict[str, list[pl.LazyFrame]] = {}
        for ep_dir in ep_dirs:
            match = _EPISODE_DIR_RE.match(ep_dir.name)
            if match is None:
                continue
            ep_id = int(match.group(1))
            telemetry_dir = ep_dir / "telemetry"
            source_dir = telemetry_dir if telemetry_dir.is_dir() else (ep_dir / "topics")
            if not source_dir.is_dir():
                continue
            for parquet_path in sorted(source_dir.glob("*.parquet")):
                stem = parquet_path.stem
                if stem in STAR_TABLE_NAMES:
                    continue
                lf = pl.scan_parquet(parquet_path).with_columns(
                    episode_id=pl.lit(ep_id, dtype=pl.Int64),
                )
                frames_by_table.setdefault(stem, []).append(lf)

        for table_name, frames in frames_by_table.items():
            combined = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")
            self._tables[table_name] = combined
            self._tables[f"telemetry_{table_name}"] = combined

    def _register_evidence_index(self, benchmark_dir: pathlib.Path) -> None:
        """Register evidence_index from benchmark_dir / 'evidence' / 'evidence_index.parquet'."""
        evidence_index_path = benchmark_dir / "evidence" / "evidence_index.parquet"
        if evidence_index_path.is_file():
            self._tables["evidence_index"] = pl.scan_parquet(evidence_index_path)
        else:
            self._tables["evidence_index"] = pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA)

    def _register_canonical_views(self) -> None:
        dim_ep = self._tables["dim_episodes"]
        dim_rob = self._tables["dim_robots"]
        fact_ep = self._tables["fact_episodes"]
        fact_enc = self._tables["fact_encounters"]
        fact_con = self._tables["fact_contacts"]

        dim_ep_cols = dim_ep.select(
            "run_id",
            "episode_id",
            "robot_id",
            "benchmark_id",
            "suite_name",
            "contest_name",
            "stage",
            "world",
            "scenario",
            "planner",
            "inter_planner",
            "dispatcher",
            "robot_model",
            "crowd_condition",
            "ped_policy_default",
            "crowd_size",
            "seed",
            "is_reference",
            "reference_type",
            "duration_s",
            "outcome_state",
            "outcome_info",
        )
        dim_rob_cols = dim_rob.select(
            "run_id",
            "episode_id",
            "robot_id",
            "max_linear_vel",
            "max_angular_vel",
            "linear_acc",
            "linear_decel",
            "robot_radius",
            "robot_mass_kg",
            "static_power_w",
            "lidar_variant",
            "start_x",
            "start_y",
            "goal_x",
            "goal_y",
        )

        v_episode_summary = fact_ep.join(dim_ep_cols, on=["run_id", "episode_id", "robot_id"], how="left").join(dim_rob_cols, on=["run_id", "episode_id", "robot_id"], how="left")
        self._tables["v_episode_summary"] = v_episode_summary

        enc_counts = fact_enc.group_by("run_id", "episode_id", "robot_id").agg(
            encounter_count=pl.len().cast(pl.Int64),
        )
        con_counts = fact_con.group_by("run_id", "episode_id", "robot_id").agg(
            contact_count=pl.len().cast(pl.Int64),
            robot_blame_contacts=(pl.col("closer_label") == "robot").sum().cast(pl.Int64),
            pedestrian_blame_contacts=(pl.col("closer_label") == "pedestrian").sum().cast(pl.Int64),
            mutual_contacts=(pl.col("closer_label") == "mutual").sum().cast(pl.Int64),
            static_contacts=(pl.col("closer_label") == "static").sum().cast(pl.Int64),
            unavoidable_contacts=pl.col("counterfactual_freeze_persists").fill_null(False).sum().cast(pl.Int64),
        )

        v_planner_scorecard = (
            v_episode_summary.join(enc_counts, on=["run_id", "episode_id", "robot_id"], how="left")
            .join(con_counts, on=["run_id", "episode_id", "robot_id"], how="left")
            .group_by("world", "scenario", "crowd_condition", "planner", "dispatcher", "robot_model")
            .agg(
                episodes_count=pl.len().cast(pl.Int64),
                success_rate=pl.col("success").cast(pl.Float64).mean(),
                success_no_robot_fault_rate=pl.col("success_no_robot_fault").cast(pl.Float64).mean(),
                mean_time_to_goal_s=pl.col("time_to_goal_s").mean(),
                mean_path_efficiency=pl.col("path_efficiency").mean(),
                mean_energy_wh=pl.col("total_energy_wh").mean(),
                total_encounters=pl.col("encounter_count").fill_null(0).sum().cast(pl.Int64),
                total_contacts=pl.col("contact_count").fill_null(0).sum().cast(pl.Int64),
                robot_blame_contacts=pl.col("robot_blame_contacts").fill_null(0).sum().cast(pl.Int64),
                pedestrian_blame_contacts=pl.col("pedestrian_blame_contacts").fill_null(0).sum().cast(pl.Int64),
                mutual_contacts=pl.col("mutual_contacts").fill_null(0).sum().cast(pl.Int64),
                static_contacts=pl.col("static_contacts").fill_null(0).sum().cast(pl.Int64),
                unavoidable_contacts=pl.col("unavoidable_contacts").fill_null(0).sum().cast(pl.Int64),
            )
            .sort("world", "scenario", "crowd_condition", "planner", "dispatcher", "robot_model")
            .with_columns(
                mean_time_to_goal=pl.col("mean_time_to_goal_s"),
            )
        )
        self._tables["v_planner_scorecard"] = v_planner_scorecard

        dim_treatment = dim_ep.select(
            "run_id",
            "episode_id",
            "robot_id",
            "world",
            "scenario",
            "crowd_condition",
            "planner",
            "dispatcher",
            "robot_model",
        )

        v_contact_blame_summary = (
            fact_con.join(dim_treatment, on=["run_id", "episode_id", "robot_id"], how="left")
            .group_by("crowd_condition", "planner", "closer_label", "fault_attributed_to")
            .agg(
                total_contacts=pl.len().cast(pl.Int64),
                mean_robot_closing_share=pl.col("robot_closing_share").mean(),
                mean_impact_speed_mps=pl.col("impact_speed_mps").mean(),
                unavoidable_contacts=pl.col("counterfactual_freeze_persists").fill_null(False).sum().cast(pl.Int64),
            )
            .with_columns(
                closer_label_share=pl.col("total_contacts").cast(pl.Float64) / pl.col("total_contacts").sum().over("crowd_condition", "planner").cast(pl.Float64),
            )
            .sort("crowd_condition", "planner", "closer_label", "fault_attributed_to")
        )
        self._tables["v_contact_blame_summary"] = v_contact_blame_summary

        v_encounter_summary = (
            fact_enc.join(dim_treatment, on=["run_id", "episode_id", "robot_id"], how="left")
            .group_by("crowd_condition", "planner", "approach_type")
            .agg(
                encounter_count=pl.len().cast(pl.Int64),
                mean_duration_s=pl.col("duration_s").mean(),
                mean_min_clearance_m=pl.col("min_clearance_m").mean(),
                mean_detour_time_s=pl.col("detour_time_s").mean(),
                mean_path_deviation_m=pl.col("path_deviation_m").mean(),
                mean_encounter_energy_j=pl.col("encounter_energy_j").mean(),
                mean_energy_surge_ratio=pl.col("energy_surge_ratio").mean(),
                mean_sound_exposure_dose_pa2s=pl.col("sound_exposure_dose_pa2s").mean(),
                mean_probst_R_robot=pl.col("probst_R_robot").mean(),
                mean_probst_R_actor2=pl.col("probst_R_actor2").mean(),
                mean_probst_R_time=pl.col("probst_R_time").mean(),
                mean_probst_E_robot=pl.col("probst_E_robot").mean(),
                mean_probst_E_actor2=pl.col("probst_E_actor2").mean(),
                contact_encounters=pl.col("had_contact").fill_null(False).sum().cast(pl.Int64),
            )
            .sort("crowd_condition", "planner", "approach_type")
        )
        self._tables["v_encounter_summary"] = v_encounter_summary


StarCatalog = BenchmarkCatalog
