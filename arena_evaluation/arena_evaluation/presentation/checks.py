"""Quality and variance diagnostic checks for plot datasets (non-blocking).

Collects sample count, zero variance, all-null columns, and cardinality as
informative metadata attached to rendered plots. Never raises rendering exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from arena_evaluation.storage.schemas import PlotSpec

__all__ = [
    "DiagnosticInspector",
    "PlotDiagnostics",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PlotDiagnostics:
    """Informative diagnostic metadata collected before rendering (non-blocking)."""

    row_count: int
    zero_variance_columns: list[str]
    null_columns: list[str]
    cardinality: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        """Return True if any diagnostic warnings or anomalies were flagged."""
        return bool(self.warnings or self.zero_variance_columns or self.null_columns)

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics into a JSON-compatible dictionary."""
        return {
            "row_count": self.row_count,
            "zero_variance_columns": list(self.zero_variance_columns),
            "null_columns": list(self.null_columns),
            "cardinality": dict(self.cardinality),
            "warnings": list(self.warnings),
        }


class DiagnosticInspector:
    """Inspects a plot DataFrame and generates non-blocking diagnostic metadata."""

    @staticmethod
    def inspect(df: pl.DataFrame, spec: PlotSpec | None = None) -> PlotDiagnostics:
        """Inspect data characteristics without throwing exceptions."""
        row_count = df.height
        null_cols: list[str] = []
        zero_var_cols: list[str] = []
        cardinality: dict[str, int] = {}
        warnings: list[str] = []

        if row_count < 10:
            warnings.append(f"Low sample size: N={row_count} (< 10 samples)")

        target_cols: set[str] = set()
        if spec is not None:
            if spec.data_key and spec.data_key != "*":
                target_cols.add(spec.data_key)
            ch = spec.channels
            for val in (ch.x, ch.y, ch.color, ch.facet_col, ch.facet_row, ch.size, ch.symbol):
                if isinstance(val, str) and val:
                    target_cols.add(val)
                elif isinstance(val, list):
                    target_cols.update(item for item in val if isinstance(item, str) and item)

        available_cols = set(df.columns)
        for col in df.columns:
            series = df[col]
            n_null = series.null_count()
            if n_null == row_count:
                null_cols.append(col)
                warnings.append(f"Column '{col}' has only null values")
                cardinality[col] = 0
                continue

            n_uniq = series.n_unique()
            cardinality[col] = n_uniq
            if n_uniq <= 1:
                zero_var_cols.append(col)
                warnings.append(f"Column '{col}' has zero variance (constant value)")
            elif n_uniq > 50 and series.dtype in (pl.Utf8, pl.Categorical):
                warnings.append(f"Column '{col}' has high cardinality ({n_uniq} distinct categories)")

        if spec is not None and spec.checks:
            req_cols = spec.checks.get("require_columns", [])
            if isinstance(req_cols, list):
                for rc in req_cols:
                    if isinstance(rc, str) and rc not in available_cols:
                        warnings.append(f"Required column '{rc}' not found in data frame")

        return PlotDiagnostics(
            row_count=row_count,
            zero_variance_columns=zero_var_cols,
            null_columns=null_cols,
            cardinality=cardinality,
            warnings=warnings,
        )
