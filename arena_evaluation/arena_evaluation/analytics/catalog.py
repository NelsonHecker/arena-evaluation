"""Benchmark OLAP Catalog (Layer 4).

Registers the nine Star Schema base tables (``dim_*`` and ``fact_*``) and
exposes canonical analytical views as pure Polars ``LazyFrame`` query plans.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Final

import polars as pl

from arena_evaluation.storage.star_schema import (
    STAR_SCHEMA_SCHEMAS,
    STAR_TABLE_NAMES,
    StarSchemaTables,
    scan_star_schema,
)

__all__ = [
    "CANONICAL_VIEW_NAMES",
    "BenchmarkCatalog",
]

CANONICAL_VIEW_NAMES: Final[tuple[str, ...]] = (
    "v_episode_summary",
    "v_planner_scorecard",
    "v_contact_blame_summary",
    "v_encounter_summary",
)


class BenchmarkCatalog:
    """Lazy Polars catalog of Star Schema tables, analytical views, and derived tables."""

    def __init__(
        self,
        source: pathlib.Path | StarSchemaTables | Mapping[str, pl.LazyFrame | pl.DataFrame] | None = None,
    ) -> None:
        self._tables: dict[str, pl.LazyFrame] = {}

        if isinstance(source, pathlib.Path):
            self._tables.update(scan_star_schema(source))
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
            "env_id",
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
