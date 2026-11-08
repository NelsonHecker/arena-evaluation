"""Benchmark OLAP Catalog (Layer 4).

Registers the nine Star Schema base tables (``dim_*`` and ``fact_*``) and
exposes canonical analytical views as pure Polars ``LazyFrame`` query plans.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

TELEMETRY_TABLE_SCHEMAS: Final[dict[str, dict[str, pl.DataType | type[pl.DataType]]]] = {
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
_DURATION_REGEX: Final[str] = r"(\d+(?:\.\d+)?)s"

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


@dataclass(frozen=True, slots=True)
class _SchemaResolver:
    """Declarative helper to resolve Polars expressions against an existing column set."""

    available_columns: frozenset[str]

    def has(self, col_name: str) -> bool:
        return col_name in self.available_columns

    def pick(
        self,
        candidates: Sequence[str],
        dtype: pl.DataType | type[pl.DataType],
        default: str | float | bool | None = None,
        *,
        fallback_expr: pl.Expr | None = None,
        fill_null: str | float | bool | None = None,
    ) -> pl.Expr:
        """Select the first matching column from `candidates`, or fall back to a default/expression."""
        expr: pl.Expr | None = None
        for col in candidates:
            if col in self.available_columns:
                expr = pl.col(col).cast(dtype)
                break

        if expr is None:
            expr = fallback_expr if fallback_expr is not None else pl.lit(default, dtype=dtype)

        if fill_null is not None:
            expr = expr.fill_null(fill_null)

        return expr

    def first_rule(self, rules: Sequence[tuple[str, Callable[[], pl.Expr]]], default_expr: pl.Expr) -> pl.Expr:
        """Evaluate and return the first expression whose trigger column exists."""
        for col, builder in rules:
            if col in self.available_columns:
                return builder()
        return default_expr


def _to_lazy(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def _has_rows(lf: pl.LazyFrame | None) -> bool:
    if lf is None:
        return False
    try:
        return lf.select(pl.len()).collect().item() > 0
    except Exception:
        return False


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


def synthesize_star_schema_from_combined(
    combined_lf: pl.LazyFrame,
    default_run_id: str = "default_run",
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Synthesize dim_episodes, dim_robots, and fact_episodes from combined_metrics.parquet."""
    res = _SchemaResolver(frozenset(combined_lf.collect_schema().names()))

    # Shared keys & common attributes
    run_id_expr = res.pick(("run_id", "benchmark_id"), pl.String, default_run_id)
    bench_id_expr = res.pick(("benchmark_id", "run_id"), pl.String, default_run_id)
    ep_id_expr = res.pick(
        ("episode_id", "episode"),
        pl.Int64,
        fallback_expr=pl.int_range(0, pl.len(), dtype=pl.Int64),
    )
    env_id_expr = res.pick(("env_id",), pl.String, "env_0")
    robot_id_expr = res.pick(("robot_id", "robot"), pl.String, "robot_0")
    robot_model_expr = res.pick(("robot_model", "robot"), pl.String, "unknown")
    planner_expr = res.pick(("planner",), pl.String, "unknown")
    inter_planner_expr = res.pick(("inter_planner",), pl.String, "none")
    dispatcher_expr = res.pick(("dispatcher",), pl.String, "single")

    # Complex / conditional expressions
    duration_fallback = pl.col("outcome_info").str.extract(_DURATION_REGEX, 1).cast(pl.Float64, strict=False).fill_null(0.0) if res.has("outcome_info") else pl.lit(0.0, dtype=pl.Float64)
    duration_expr = res.pick(
        ("duration_s", "time_to_goal_s", "time_to_goal", "time"),
        pl.Float64,
        fallback_expr=duration_fallback,
    )

    ttg_fallback = pl.col("outcome_info").str.extract(_DURATION_REGEX, 1).cast(pl.Float64, strict=False) if res.has("outcome_info") else pl.lit(None, dtype=pl.Float64)
    ttg_expr = res.pick(
        ("time_to_goal_s", "time_to_goal", "time", "duration_s"),
        pl.Float64,
        fallback_expr=ttg_fallback,
    )

    outcome_state_expr = res.first_rule(
        rules=[
            ("outcome_state", lambda: pl.col("outcome_state").cast(pl.Int64)),
            (
                "success",
                lambda: pl.when(pl.col("success").fill_null(False)).then(pl.lit(2, dtype=pl.Int64)).otherwise(pl.lit(3, dtype=pl.Int64)),
            ),
        ],
        default_expr=pl.lit(0, dtype=pl.Int64),
    )

    success_expr = res.pick(("success",), pl.Boolean, False, fill_null=False)
    snrf_expr = res.pick(("success_no_robot_fault",), pl.Boolean, fallback_expr=success_expr, fill_null=False)

    dim_episodes = combined_lf.select(
        run_id=run_id_expr,
        episode_id=ep_id_expr,
        env_id=env_id_expr,
        robot_id=robot_id_expr,
        benchmark_id=bench_id_expr,
        suite_name=res.pick(("suite_name",), pl.String, ""),
        contest_name=res.pick(("contest_name",), pl.String, ""),
        stage=res.pick(("stage", "scenario"), pl.String, "default"),
        world=res.pick(("world", "map"), pl.String, "unknown"),
        scenario=res.pick(("scenario", "stage"), pl.String, "default"),
        planner=planner_expr,
        inter_planner=inter_planner_expr,
        dispatcher=dispatcher_expr,
        robot_model=robot_model_expr,
        crowd_condition=res.pick(("crowd_condition",), pl.String, "default"),
        ped_policy_default=res.pick(("ped_policy_default",), pl.String, "none"),
        crowd_size=res.pick(("crowd_size", "num_pedestrians"), pl.Int64, 0),
        seed=res.pick(("seed",), pl.Int64, 0),
        is_reference=res.pick(("is_reference",), pl.Boolean, False, fill_null=False),
        reference_type=res.pick(("reference_type",), pl.String, None),
        duration_s=duration_expr,
        outcome_state=outcome_state_expr,
        outcome_info=res.pick(("outcome_info", "result", "status_reason"), pl.String, ""),
    )

    fact_episodes = combined_lf.select(
        run_id=run_id_expr,
        episode_id=ep_id_expr,
        robot_id=robot_id_expr,
        result=res.pick(("result", "status"), pl.String, "UNKNOWN"),
        success=success_expr,
        time_to_goal_s=ttg_expr,
        path_length_m=res.pick(("path_length_m", "path_length"), pl.Float64, None),
        path_efficiency=res.pick(("path_efficiency", "spl"), pl.Float64, None),
        total_energy_wh=res.pick(("total_energy_wh", "energy_total_wh"), pl.Float64, None),
        energy_per_meter_j_m=res.pick(("energy_per_meter_j_m", "energy_per_meter"), pl.Float64, None),
        mean_velocity_mps=res.pick(("mean_velocity_mps", "velocity_mean"), pl.Float64, None),
        max_acceleration_mps2=res.pick(("max_acceleration_mps2", "acceleration_max"), pl.Float64, None),
        jerk_max_mps3=res.pick(("jerk_max_mps3", "jerk_max"), pl.Float64, None),
        min_obstacle_clearance_m=res.pick(("min_obstacle_clearance_m", "min_obstacle_clearance"), pl.Float64, None),
        min_ped_clearance_m=res.pick(("min_ped_clearance_m", "min_pedestrian_clearance"), pl.Float64, None),
        total_collisions_ped=res.pick(("total_collisions_ped", "collision_amount_pedestrian"), pl.Int64, 0),
        total_collisions_static=res.pick(
            ("total_collisions_static", "collision_amount_static", "collision_amount_wall", "collision_amount"),
            pl.Int64,
            0,
        ),
        success_no_robot_fault=snrf_expr,
        mar=res.pick(("mar",), pl.Float64, None),
        pfi=res.pick(("pfi",), pl.Float64, None),
        robot_path_dev_m=res.pick(("robot_path_dev_m", "robot_path_deviation_m"), pl.Float64, None),
        total_acoustic_dose_pa2s=res.pick(("total_acoustic_dose_pa2s", "acoustic_cost"), pl.Float64, None),
        null_reason=res.pick(("null_reason", "status_reason"), pl.String, None),
    ).with_columns(
        time_to_goal=pl.col("time_to_goal_s"),
        path_length=pl.col("path_length_m"),
    )

    optional_robot_floats = (
        "max_linear_vel",
        "max_angular_vel",
        "linear_acc",
        "linear_decel",
        "robot_radius",
        "robot_mass_kg",
        "static_power_w",
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
    )

    dim_robots = combined_lf.select(
        run_id=run_id_expr,
        episode_id=ep_id_expr,
        env_id=env_id_expr,
        robot_id=robot_id_expr,
        robot_model=robot_model_expr,
        planner=planner_expr,
        inter_planner=inter_planner_expr,
        dispatcher=dispatcher_expr,
        **{col: res.pick((col,), pl.Float64, None) for col in optional_robot_floats},
        lidar_variant=res.pick(("lidar_variant",), pl.String, None),
    )

    return dim_episodes, dim_robots, fact_episodes


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
        self._tables.update(scan_star_schema(source_dir))
        self._register_telemetry_views(source_dir)
        self._register_evidence_index(source_dir)

        combined_file = next(
            (p for p in (source_dir / "combined_metrics.parquet", source_dir / "metrics" / "combined_metrics.parquet") if p.is_file()),
            None,
        )
        if combined_file is None:
            return

        if not _has_rows(self._tables.get("fact_episodes")):
            comb_lf = pl.scan_parquet(combined_file)
            dim_ep, dim_rob, fact_ep = synthesize_star_schema_from_combined(comb_lf, default_run_id=source_dir.name)
            for name, synthesized_lf in (("dim_episodes", dim_ep), ("dim_robots", dim_rob)):
                if not _has_rows(self._tables.get(name)):
                    self._tables[name] = synthesized_lf
            self._tables["fact_episodes"] = fact_ep

        self._tables.setdefault("combined_metrics", pl.scan_parquet(combined_file))

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

            synced_lf = _sync_bidirectional_aliases(self._tables[tname], [("timestamp_ns", "t_ns")])
            self._tables[tname] = synced_lf
            self._tables[f"telemetry_{tname}"] = synced_lf

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

    def _register_telemetry_views(self, benchmark_dir: pathlib.Path) -> None:
        """Lazily scan episode telemetry Parquet tables across ``benchmark_dir / 'episodes'``."""
        episodes_root = benchmark_dir / "episodes"
        if not episodes_root.is_dir():
            return

        ep_dirs = sorted(
            {path for pattern in ("episode_*", "*/episode_*") for path in episodes_root.glob(pattern) if path.is_dir() and _EPISODE_DIR_RE.match(path.name) is not None},
            key=lambda p: int(p.name.split("_")[-1]),
        )

        frames_by_table: dict[str, list[pl.LazyFrame]] = {}
        for ep_dir in ep_dirs:
            match = _EPISODE_DIR_RE.match(ep_dir.name)
            if match is None:
                continue
            ep_id = int(match.group(1))
            source_dir = next((d for d in (ep_dir / "telemetry", ep_dir / "topics") if d.is_dir()), None)
            if source_dir is None:
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

        v_episode_summary = fact_ep.join(dim_ep.select(_DIM_EPISODE_VIEW_COLS), on=list(_JOIN_KEYS), how="left").join(dim_rob.select(_DIM_ROBOT_VIEW_COLS), on=list(_JOIN_KEYS), how="left")
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
