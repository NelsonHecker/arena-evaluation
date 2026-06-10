from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.schemas import PlotSpec

# Ordered priority list of identity columns.
# The detector checks these and reports which ones carry more than one unique value.
IDENTITY_COLS: list[str] = ["planner", "robot", "stage", "map", "benchmark_id"]

# Name of the synthetic compound-label column added to the DataFrame when
# multiple dimensions vary simultaneously.
COMPOUND_LABEL_COL = "__label__"


def detect_varying_dims(df: pl.DataFrame) -> list[str]:
    """
    Return identity columns that have more than one unique value in df.
    """
    varying: list[str] = []
    for col in IDENTITY_COLS:
        if col not in df.columns:
            continue
        series = df[col].drop_nulls()
        if series.n_unique() > 1:
            varying.append(col)
    return varying


def build_label_column(df: pl.DataFrame, dims: list[str]) -> pl.DataFrame:
    """
    Add (or replace) a __label__ column whose value is a human-readable
    compound of the given dims.
    """
    if not dims:
        return df

    if len(dims) == 1:
        # Simple — just alias the single varying column
        return df.with_columns(pl.col(dims[0]).alias(COMPOUND_LABEL_COL))

    # Compound — join the values of all varying dims
    parts = [pl.col(d).cast(pl.Utf8) for d in dims]
    label_expr = pl.concat_str(parts, separator=" / ")
    return df.with_columns(label_expr.alias(COMPOUND_LABEL_COL))


def resolve_differentiate(spec: "PlotSpec", df: pl.DataFrame) -> tuple[str, pl.DataFrame]:
    """
    Determine the effective differentiation column for spec given df.
    """
    auto = getattr(spec, "auto_differentiate", True)
    requested = spec.differentiate  # may be None

    # Rule 1 – explicit opt-out of auto detection
    if not auto:
        fallback = requested if (requested and requested in df.columns) else "planner"
        return fallback, df

    varying = detect_varying_dims(df)

    # Rule 2 – single varying dim, requested col matches → no change needed
    if len(varying) <= 1:
        col = requested if (requested and requested in df.columns) else (varying[0] if varying else "planner")
        return col, df

    # Multiple dims vary — we need a compound label regardless of what was requested.
    # Rule 3 & 4
    df = build_label_column(df, varying)
    return COMPOUND_LABEL_COL, df
