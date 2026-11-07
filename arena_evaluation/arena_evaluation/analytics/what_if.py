"""In-Memory Sensitivity Analysis & Parameter Sweeps (Layer 4).

Evaluates how contact blame attribution and encounter cost aggregations shift
across parameter thresholds without re-running physical simulations.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import Final

import polars as pl

__all__ = [
    "CLOSER_SWEEP_METRIC_SCHEMA",
    "DEFAULT_CLEARANCE_THRESHOLDS_M",
    "DEFAULT_DOMINANT_SHARE_THRESHOLDS",
    "ENCOUNTER_SWEEP_METRIC_SCHEMA",
    "sweep_closer_share_thresholds",
    "sweep_encounter_clearance_thresholds",
]

DEFAULT_DOMINANT_SHARE_THRESHOLDS: Final[tuple[float, ...]] = (0.60, 0.65, 0.70, 0.75, 0.80)
DEFAULT_CLEARANCE_THRESHOLDS_M: Final[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0, 3.0)

CLOSER_SWEEP_METRIC_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "total_contacts": pl.Int64,
        "closer_robot_count": pl.Int64,
        "closer_pedestrian_count": pl.Int64,
        "closer_mutual_count": pl.Int64,
        "closer_static_count": pl.Int64,
        "fault_robot_count": pl.Int64,
        "fault_pedestrian_count": pl.Int64,
        "fault_unavoidable_count": pl.Int64,
    }
)

ENCOUNTER_SWEEP_METRIC_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "encounter_count": pl.Int64,
        "mean_detour_time_s": pl.Float64,
        "mean_path_deviation_m": pl.Float64,
        "mean_energy_surge_ratio": pl.Float64,
        "mean_encounter_energy_j": pl.Float64,
        "mean_probst_R_robot": pl.Float64,
        "mean_sound_exposure_dose_pa2s": pl.Float64,
        "contact_encounters": pl.Int64,
    }
)


def _join_with_dim_episodes(
    fact_lf: pl.LazyFrame,
    dim_episodes: pl.LazyFrame | pl.DataFrame | None,
    group_cols: Sequence[str],
) -> pl.LazyFrame:
    if dim_episodes is None:
        return fact_lf
    dim_lf = dim_episodes.lazy() if isinstance(dim_episodes, pl.DataFrame) else dim_episodes
    join_keys = ["run_id", "episode_id", "robot_id"]
    extra_cols = [c for c in group_cols if c not in join_keys]
    dim_projected = dim_lf.select([*join_keys, *extra_cols])
    return fact_lf.join(dim_projected, on=join_keys, how="left")


@typing.overload
def sweep_closer_share_thresholds(
    fact_contacts: pl.LazyFrame,
    dim_episodes: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    dominant_share_thresholds: Sequence[float] = DEFAULT_DOMINANT_SHARE_THRESHOLDS,
    group_cols: Sequence[str] = ("crowd_condition", "planner"),
) -> pl.LazyFrame: ...


@typing.overload
def sweep_closer_share_thresholds(
    fact_contacts: pl.DataFrame,
    dim_episodes: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    dominant_share_thresholds: Sequence[float] = DEFAULT_DOMINANT_SHARE_THRESHOLDS,
    group_cols: Sequence[str] = ("crowd_condition", "planner"),
) -> pl.DataFrame: ...


def sweep_closer_share_thresholds(
    fact_contacts: pl.LazyFrame | pl.DataFrame,
    dim_episodes: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    dominant_share_thresholds: Sequence[float] = DEFAULT_DOMINANT_SHARE_THRESHOLDS,
    group_cols: Sequence[str] = ("crowd_condition", "planner"),
) -> pl.LazyFrame | pl.DataFrame:
    """Evaluate how closer_label and fault_attributed_to counts shift across dominant_share thresholds."""
    thresholds = [float(t) for t in dominant_share_thresholds]
    if not thresholds:
        raise ValueError("dominant_share_thresholds must not be empty.")
    for t in thresholds:
        if not (0.5 <= t <= 1.0):
            raise ValueError(f"Each dominant_share_threshold must be in [0.5, 1.0], got {t}")

    is_eager = isinstance(fact_contacts, pl.DataFrame)
    contacts_lf = fact_contacts.lazy() if is_eager else fact_contacts
    group_list = list(group_cols)

    joined_lf = _join_with_dim_episodes(contacts_lf, dim_episodes, group_list)
    thresholds_lf = pl.LazyFrame(
        {"dominant_share_threshold": thresholds},
        schema={"dominant_share_threshold": pl.Float64},
    )

    is_static = pl.col("collider_kind").is_in(["wall", "static"]) | (pl.col("closer_label") == "static")
    s_robot = pl.col("robot_closing_share")
    tau = pl.col("dominant_share_threshold")

    swept_closer = pl.when(is_static).then(pl.lit("static")).when(s_robot.is_null()).then(pl.col("closer_label")).when(s_robot >= tau).then(pl.lit("robot")).when(s_robot <= (1.0 - tau)).then(pl.lit("pedestrian")).otherwise(pl.lit("mutual")).alias("swept_closer_label")

    freeze_false = pl.col("counterfactual_freeze_persists").fill_null(True).not_()
    freeze_true = pl.col("counterfactual_freeze_persists").fill_null(False)

    swept_fault = pl.when(is_static).then(pl.lit("robot")).when(pl.col("swept_closer_label").is_in(["robot", "mutual"]) | freeze_false).then(pl.lit("robot")).when((pl.col("swept_closer_label") == "pedestrian") & freeze_true).then(pl.lit("pedestrian")).otherwise(pl.lit("unavoidable")).alias("swept_fault_attributed_to")

    agg_keys = ["dominant_share_threshold", *group_list]
    swept_lf = (
        joined_lf.join(thresholds_lf, how="cross")
        .with_columns(swept_closer)
        .with_columns(swept_fault)
        .group_by(agg_keys)
        .agg(
            total_contacts=pl.len().cast(pl.Int64),
            closer_robot_count=(pl.col("swept_closer_label") == "robot").sum().cast(pl.Int64),
            closer_pedestrian_count=(pl.col("swept_closer_label") == "pedestrian").sum().cast(pl.Int64),
            closer_mutual_count=(pl.col("swept_closer_label") == "mutual").sum().cast(pl.Int64),
            closer_static_count=(pl.col("swept_closer_label") == "static").sum().cast(pl.Int64),
            fault_robot_count=(pl.col("swept_fault_attributed_to") == "robot").sum().cast(pl.Int64),
            fault_pedestrian_count=(pl.col("swept_fault_attributed_to") == "pedestrian").sum().cast(pl.Int64),
            fault_unavoidable_count=(pl.col("swept_fault_attributed_to") == "unavoidable").sum().cast(pl.Int64),
        )
        .sort(agg_keys)
    )

    return swept_lf.collect() if is_eager else swept_lf


@typing.overload
def sweep_encounter_clearance_thresholds(
    fact_encounters: pl.LazyFrame,
    dim_episodes: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    max_min_clearance_m: Sequence[float] = DEFAULT_CLEARANCE_THRESHOLDS_M,
    group_cols: Sequence[str] = ("crowd_condition", "planner"),
) -> pl.LazyFrame: ...


@typing.overload
def sweep_encounter_clearance_thresholds(
    fact_encounters: pl.DataFrame,
    dim_episodes: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    max_min_clearance_m: Sequence[float] = DEFAULT_CLEARANCE_THRESHOLDS_M,
    group_cols: Sequence[str] = ("crowd_condition", "planner"),
) -> pl.DataFrame: ...


def sweep_encounter_clearance_thresholds(
    fact_encounters: pl.LazyFrame | pl.DataFrame,
    dim_episodes: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    max_min_clearance_m: Sequence[float] = DEFAULT_CLEARANCE_THRESHOLDS_M,
    group_cols: Sequence[str] = ("crowd_condition", "planner"),
) -> pl.LazyFrame | pl.DataFrame:
    """Evaluate encounter counts, detour time, energy surge, and Probst R_robot across clearance thresholds."""
    clearances = [float(c) for c in max_min_clearance_m]
    if not clearances:
        raise ValueError("max_min_clearance_m must not be empty.")

    is_eager = isinstance(fact_encounters, pl.DataFrame)
    encounters_lf = fact_encounters.lazy() if is_eager else fact_encounters
    group_list = list(group_cols)

    joined_lf = _join_with_dim_episodes(encounters_lf, dim_episodes, group_list)
    thresholds_lf = pl.LazyFrame(
        {"max_min_clearance_m": clearances},
        schema={"max_min_clearance_m": pl.Float64},
    )

    agg_keys = ["max_min_clearance_m", *group_list]
    if group_list:
        groups_Source = (dim_episodes.lazy() if isinstance(dim_episodes, pl.DataFrame) else dim_episodes) if dim_episodes is not None else joined_lf
        distinct_groups = groups_Source.select(group_list).unique()
        grid_lf = thresholds_lf.join(distinct_groups, how="cross")
    else:
        grid_lf = thresholds_lf

    filtered_agg = (
        joined_lf.join(thresholds_lf, how="cross")
        .filter(pl.col("min_clearance_m") <= pl.col("max_min_clearance_m"))
        .group_by(agg_keys)
        .agg(
            encounter_count=pl.len().cast(pl.Int64),
            mean_detour_time_s=pl.col("detour_time_s").mean(),
            mean_path_deviation_m=pl.col("path_deviation_m").mean(),
            mean_energy_surge_ratio=pl.col("energy_surge_ratio").mean(),
            mean_encounter_energy_j=pl.col("encounter_energy_j").mean(),
            mean_probst_R_robot=pl.col("probst_R_robot").mean(),
            mean_sound_exposure_dose_pa2s=pl.col("sound_exposure_dose_pa2s").mean(),
            contact_encounters=pl.col("had_contact").fill_null(False).sum().cast(pl.Int64),
        )
    )

    swept_lf = (
        grid_lf.join(filtered_agg, on=agg_keys, how="left")
        .with_columns(
            encounter_count=pl.col("encounter_count").fill_null(0).cast(pl.Int64),
            contact_encounters=pl.col("contact_encounters").fill_null(0).cast(pl.Int64),
        )
        .sort(agg_keys)
    )

    return swept_lf.collect() if is_eager else swept_lf
