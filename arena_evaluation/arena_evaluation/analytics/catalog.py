"""Benchmark OLAP Catalog (Layer 4).

Registers the nine Star Schema base tables (``dim_*`` and ``fact_*``) and
exposes canonical analytical views as pure Polars ``LazyFrame`` query plans.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Final

import polars as pl

from arena_evaluation.analytics.evidence import EVIDENCE_INDEX_SCHEMA
from arena_evaluation.storage.star_schema import (
    STAR_SCHEMA_SCHEMAS,
    STAR_TABLE_NAMES,
    StarSchemaTables,
    episode_directories,
    scan_star_schema,
)

__all__ = [
    "CANONICAL_VIEW_NAMES",
    "SERIES_TABLE_PREFIX",
    "TELEMETRY_TABLE_NAMES",
    "TELEMETRY_TABLE_SCHEMAS",
    "BenchmarkCatalog",
    "StarCatalog",
]

SERIES_TABLE_PREFIX: Final[str] = "series_"

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

TELEMETRY_TABLE_SCHEMAS: Final[dict[str, dict[str, pl.DataType | type[pl.DataType]]]] = {
    "robot_tracks": {
        "episode_id": pl.Int64,
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "robot_id": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
        "yaw": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "vz": pl.Float64,
        "omega_z": pl.Float64,
        "speed_mps": pl.Float64,
        "pose_source": pl.String,
    },
    "pedestrian_tracks": {
        "episode_id": pl.Int64,
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "ped_id": pl.String,
        "ped_name": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
        "yaw": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "vz": pl.Float64,
        "speed_mps": pl.Float64,
        "radius_m": pl.Float64,
        "desired_speed": pl.Float64,
        "policy": pl.String,
        "agent_type": pl.String,
        "animation_state": pl.Int64,
        "interaction_id": pl.Int64,
        "interaction_type": pl.Int64,
        "gait_phase": pl.Float64,
        "gait_cadence": pl.Float64,
    },
    "robot_power": {
        "episode_id": pl.Int64,
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "robot_id": pl.String,
        "total_power_w": pl.Float64,
        "static_power_w": pl.Float64,
        "total_mech_power_w": pl.Float64,
        "total_thermal_power_w": pl.Float64,
    },
    "robot_acoustics": {
        "episode_id": pl.Int64,
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "robot_id": pl.String,
        "total_level_af_dba": pl.Float64,
        "total_level_zf_db": pl.Float64,
        "baseline_level_dba": pl.Float64,
        "drivetrain_level_dba": pl.Float64,
        "operating_state": pl.String,
        "calibration_status": pl.String,
        "uncertainty_dba": pl.Float64,
        "validity_flags": pl.Int64,
    },
    "collision_contacts": {
        "episode_id": pl.Int64,
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "robot_id": pl.String,
        "obstacle_id": pl.String,
        "kind": pl.String,
        "obstacle_x": pl.Float64,
        "obstacle_y": pl.Float64,
        "obstacle_z": pl.Float64,
        "distance": pl.Float64,
    },
}

_EPISODE_DIR_RE: Final[re.Pattern[str]] = re.compile(r"^episode_(\d+)$")

_JOIN_KEYS: Final[tuple[str, ...]] = ("run_id", "episode_id", "robot_id")

_DIM_EPISODE_VIEW_COLS: Final[tuple[str, ...]] = (
    *_JOIN_KEYS,
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

_DIM_ROBOT_VIEW_COLS: Final[tuple[str, ...]] = (
    *_JOIN_KEYS,
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

_DIM_TREATMENT_COLS: Final[tuple[str, ...]] = (
    *_JOIN_KEYS,
    "world",
    "scenario",
    "crowd_condition",
    "planner",
    "dispatcher",
    "robot_model",
)

_BLAME_LABELS: Final[dict[str, str]] = {
    "robot_blame_contacts": "robot",
    "pedestrian_blame_contacts": "pedestrian",
    "mutual_contacts": "mutual",
    "static_contacts": "static",
}

_ENCOUNTER_MEAN_METRICS: Final[tuple[str, ...]] = (
    "duration_s",
    "min_clearance_m",
    "detour_time_s",
    "path_deviation_m",
    "encounter_energy_j",
    "energy_surge_ratio",
    "sound_exposure_dose_pa2s",
    "probst_R_robot",
    "probst_R_actor2",
    "probst_R_time",
    "probst_E_robot",
    "probst_E_actor2",
)


def _to_lazy(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def _sync_bidirectional_aliases(lf: pl.LazyFrame, pairs: Sequence[tuple[str, str]]) -> pl.LazyFrame:
    """Ensure both column names in each `(col_a, col_b)` pair exist if either is present."""
    schema_names = set(lf.collect_schema().names())
    exprs: list[pl.Expr] = []
    for col_a, col_b in pairs:
        if col_a in schema_names and col_b not in schema_names:
            exprs.append(pl.col(col_a).alias(col_b))
        elif col_b in schema_names and col_a not in schema_names:
            exprs.append(pl.col(col_b).alias(col_a))
    return lf.with_columns(exprs) if exprs else lf


class BenchmarkCatalog:
    """Lazy Polars catalog of Star Schema tables, analytical views, and telemetry tables."""

    def __init__(
        self,
        source: pathlib.Path | StarSchemaTables | Mapping[str, pl.LazyFrame | pl.DataFrame] | None = None,
    ) -> None:
        self._tables: dict[str, pl.LazyFrame] = {}
        self._load_source(source)
        self._ensure_default_tables_and_aliases()
        self._register_canonical_views()
        self._register_series_views()

    def _load_source(
        self,
        source: pathlib.Path | StarSchemaTables | Mapping[str, pl.LazyFrame | pl.DataFrame] | None,
    ) -> None:
        if isinstance(source, pathlib.Path):
            self._load_from_directory(source)
        elif isinstance(source, StarSchemaTables):
            self._tables.update(source.lazy_by_name())
        elif isinstance(source, Mapping):
            self._load_from_mapping(source)
        else:
            self._load_empty_star_schema()

    def _load_from_directory(self, source_dir: pathlib.Path) -> None:
        """Scan the canonical Star Schema Parquet tables, plus series, telemetry and evidence tables."""
        self._tables.update(scan_star_schema(source_dir))
        self._register_series_tables(source_dir)
        self._register_telemetry_views(source_dir)
        self._register_evidence_index(source_dir)

    def _register_series_tables(self, benchmark_dir: pathlib.Path) -> None:
        """Register one lazy view per ``series_*`` stem over the per-episode stored files (D6/D7)."""
        paths_by_stem: dict[str, list[pathlib.Path]] = {}
        for episode_dir in episode_directories(benchmark_dir):
            metrics_dir = episode_dir / "metrics"
            if not metrics_dir.is_dir():
                continue
            for path in sorted(metrics_dir.glob(f"{SERIES_TABLE_PREFIX}*.parquet")):
                paths_by_stem.setdefault(path.stem, []).append(path)
        for stem, paths in paths_by_stem.items():
            self._tables[stem] = pl.scan_parquet(paths)

    def _load_from_mapping(self, source: Mapping[str, pl.LazyFrame | pl.DataFrame]) -> None:
        for name, schema in STAR_SCHEMA_SCHEMAS.items():
            raw = source.get(name)
            self._tables[name] = _to_lazy(raw) if raw is not None else pl.LazyFrame(schema=schema)

        for name, table in source.items():
            if name not in self._tables:
                self._tables[name] = _to_lazy(table)

    def _load_empty_star_schema(self) -> None:
        for name, schema in STAR_SCHEMA_SCHEMAS.items():
            self._tables[name] = pl.LazyFrame(schema=schema)

    def _ensure_default_tables_and_aliases(self) -> None:
        self._tables.setdefault("evidence_index", pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA))

        for tname, tschema in TELEMETRY_TABLE_SCHEMAS.items():
            if tname not in self._tables:
                self._tables[tname] = pl.LazyFrame(schema=tschema)
            self._tables[f"telemetry_{tname}"] = self._tables[tname]

        self._tables["fact_episodes"] = _sync_bidirectional_aliases(
            self._tables["fact_episodes"],
            [("time_to_goal", "time_to_goal_s"), ("path_length", "path_length_m")],
        )

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
        self._tables[name] = _to_lazy(frame)
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

    def _register_series_views(self) -> None:
        """Expose every ``series_*`` table joined with the episode identity columns of the summary view."""
        series_names = [name for name in self._tables if name.startswith(SERIES_TABLE_PREFIX)]
        if not series_names:
            return
        identity = self._tables["v_episode_summary"].select(
            [
                *_JOIN_KEYS,
                "local_planner",
                "planner",
                "inter_planner",
                "dispatcher",
                "robot_model",
                "stage",
                "world",
                "scenario",
                "crowd_condition",
                "seed",
                "is_reference",
            ]
        )
        for name in series_names:
            self._tables[f"v_{name}"] = self._tables[name].join(identity, on=list(_JOIN_KEYS), how="left")

    def _register_telemetry_views(self, benchmark_dir: pathlib.Path) -> None:
        """Lazily scan episode telemetry Parquet tables across ``benchmark_dir / 'episodes'``."""
        episodes_root = benchmark_dir / "episodes"
        if not episodes_root.is_dir():
            return

        frames_by_table: dict[str, list[pl.LazyFrame]] = {}
        for ep_dir in episode_directories(benchmark_dir):
            match = _EPISODE_DIR_RE.match(ep_dir.name)
            if match is None:
                continue
            ep_id = int(match.group(1))
            source_dir = ep_dir / "topics"
            if not source_dir.is_dir():
                continue

            for parquet_path in sorted(source_dir.glob("*.parquet")):
                stem = parquet_path.stem
                if stem not in STAR_TABLE_NAMES:
                    lf = pl.scan_parquet(parquet_path).with_columns(episode_id=pl.lit(ep_id, dtype=pl.Int64))
                    frames_by_table.setdefault(stem, []).append(lf)

        for table_name, frames in frames_by_table.items():
            combined = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")
            self._tables[table_name] = combined
            self._tables[f"telemetry_{table_name}"] = combined

    def _register_evidence_index(self, benchmark_dir: pathlib.Path) -> None:
        """Register evidence_index from benchmark_dir / 'evidence' / 'evidence_index.parquet'."""
        evidence_index_path = benchmark_dir / "evidence" / "evidence_index.parquet"
        self._tables["evidence_index"] = pl.scan_parquet(evidence_index_path) if evidence_index_path.is_file() else pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA)

    def _register_canonical_views(self) -> None:
        dim_ep = self._tables["dim_episodes"]
        dim_rob = self._tables["dim_robots"]
        fact_ep = self._tables["fact_episodes"]
        fact_enc = self._tables["fact_encounters"]
        fact_con = self._tables["fact_contacts"]

        # local_planner is the planner token a contest entry compares on: contest
        # names carry the inter-planner suffix, the local planner is the first
        # segment for a plain name and the middle segment for a three-part name.
        planner_parts = pl.col("planner").str.split("-")
        local_planner = pl.when(pl.col("planner").is_null() | (pl.col("planner") == "")).then(pl.lit("unknown")).when(planner_parts.list.len() >= 3).then(planner_parts.list.get(1, null_on_oob=True)).otherwise(planner_parts.list.get(0, null_on_oob=True)).alias("local_planner")
        v_episode_summary = fact_ep.join(dim_ep.select(_DIM_EPISODE_VIEW_COLS), on=list(_JOIN_KEYS), how="left").join(dim_rob.select(_DIM_ROBOT_VIEW_COLS), on=list(_JOIN_KEYS), how="left").with_columns(local_planner)
        self._tables["v_episode_summary"] = v_episode_summary

        enc_counts = fact_enc.group_by(_JOIN_KEYS).agg(encounter_count=pl.len().cast(pl.Int64))
        con_counts = fact_con.group_by(_JOIN_KEYS).agg(
            contact_count=pl.len().cast(pl.Int64),
            **{col_name: (pl.col("closer_label") == label).sum().cast(pl.Int64) for col_name, label in _BLAME_LABELS.items()},
            unavoidable_contacts=pl.col("counterfactual_freeze_persists").fill_null(False).sum().cast(pl.Int64),
        )

        scorecard_group_cols = ("world", "scenario", "crowd_condition", "planner", "dispatcher", "robot_model")
        contact_sum_cols = (*_BLAME_LABELS.keys(), "unavoidable_contacts")

        self._tables["v_planner_scorecard"] = (
            v_episode_summary.join(enc_counts, on=list(_JOIN_KEYS), how="left")
            .join(con_counts, on=list(_JOIN_KEYS), how="left")
            .group_by(scorecard_group_cols)
            .agg(
                episodes_count=pl.len().cast(pl.Int64),
                success_rate=pl.col("success").cast(pl.Float64).mean(),
                success_no_robot_fault_rate=pl.col("success_no_robot_fault").cast(pl.Float64).mean(),
                mean_time_to_goal_s=pl.col("time_to_goal_s").mean(),
                mean_path_efficiency=pl.col("path_efficiency").mean(),
                mean_energy_wh=pl.col("total_energy_wh").mean(),
                total_encounters=pl.col("encounter_count").fill_null(0).sum().cast(pl.Int64),
                total_contacts=pl.col("contact_count").fill_null(0).sum().cast(pl.Int64),
                **{col: pl.col(col).fill_null(0).sum().cast(pl.Int64) for col in contact_sum_cols},
            )
            .sort(scorecard_group_cols)
            .with_columns(mean_time_to_goal=pl.col("mean_time_to_goal_s"))
        )

        dim_treatment = dim_ep.select(_DIM_TREATMENT_COLS)
        blame_group_cols = ("crowd_condition", "planner", "closer_label", "fault_attributed_to")

        self._tables["v_contact_blame_summary"] = (
            fact_con.join(dim_treatment, on=list(_JOIN_KEYS), how="left")
            .group_by(blame_group_cols)
            .agg(
                total_contacts=pl.len().cast(pl.Int64),
                mean_robot_closing_share=pl.col("robot_closing_share").mean(),
                mean_impact_speed_mps=pl.col("impact_speed_mps").mean(),
                unavoidable_contacts=pl.col("counterfactual_freeze_persists").fill_null(False).sum().cast(pl.Int64),
            )
            .with_columns(
                closer_label_share=pl.col("total_contacts").cast(pl.Float64) / pl.col("total_contacts").sum().over("crowd_condition", "planner").cast(pl.Float64),
            )
            .sort(blame_group_cols)
        )

        encounter_group_cols = ("crowd_condition", "planner", "approach_type")
        self._tables["v_encounter_summary"] = (
            fact_enc.join(dim_treatment, on=list(_JOIN_KEYS), how="left")
            .group_by(encounter_group_cols)
            .agg(
                encounter_count=pl.len().cast(pl.Int64),
                **{f"mean_{col}": pl.col(col).mean() for col in _ENCOUNTER_MEAN_METRICS},
                contact_encounters=pl.col("had_contact").fill_null(False).sum().cast(pl.Int64),
            )
            .sort(encounter_group_cols)
        )


StarCatalog = BenchmarkCatalog
