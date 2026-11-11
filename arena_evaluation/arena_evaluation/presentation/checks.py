"""Quality and variance diagnostic checks for plot datasets.

Provides automated plot quality and integrity gate checks before rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from arena_evaluation.storage.schemas import PlotSpec

__all__ = [
    "BuildChecks",
    "CheckStage",
    "DiagnosticInspector",
    "PlotCheck",
    "PlotDiagnostics",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PlotDiagnostics:
    """Informative diagnostic metadata collected before rendering."""

    status: str = "PASSED"
    issues: list[str] = field(default_factory=list)
    sample_count: int = 0
    null_ratio: float = 0.0
    zero_variance: bool = False

    row_count: int = 0
    zero_variance_columns: list[str] = field(default_factory=list)
    null_columns: list[str] = field(default_factory=list)
    cardinality: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        """Return True if any diagnostic warnings or anomalies were flagged."""
        return self.status in ("WARNING", "FAILED") or bool(self.warnings or self.issues or self.zero_variance_columns or self.null_columns)

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics into a JSON-compatible dictionary."""
        return {
            "status": self.status,
            "issues": list(self.issues),
            "sample_count": self.sample_count,
            "null_ratio": self.null_ratio,
            "zero_variance": self.zero_variance,
            "row_count": self.row_count,
            "zero_variance_columns": list(self.zero_variance_columns),
            "null_columns": list(self.null_columns),
            "cardinality": dict(self.cardinality),
            "warnings": list(self.warnings),
        }


class CheckStage(StrEnum):
    """The build step that refused a plot."""

    LOAD = "load"
    QUALITY = "quality gate"
    STATIC = "static plot"
    INTERACTIVE = "interactive plot"


@dataclass(frozen=True, slots=True, kw_only=True)
class PlotCheck:
    """One plot that failed a build step, with the reason the build recorded."""

    plot_id: str
    title: str
    stage: CheckStage
    detail: str


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildChecks:
    """The failed checks of one report build against the plots its manifest asked for."""

    plots_total: int
    failures: tuple[PlotCheck, ...]

    @property
    def failed_count(self) -> int:
        """Plots that failed at least one build step."""
        return len({failure.plot_id for failure in self.failures})

    @property
    def passed_count(self) -> int:
        """Plots that rendered on every step of the build."""
        return self.plots_total - self.failed_count

    def for_plot(self, plot_id: str) -> tuple[PlotCheck, ...]:
        """Every failed step of one plot, in build order."""
        return tuple(failure for failure in self.failures if failure.plot_id == plot_id)


class DiagnosticInspector:
    """Inspects a plot DataFrame and generates quality gate diagnostic metadata."""

    @staticmethod
    def inspect(df: pl.DataFrame, spec: PlotSpec | None = None) -> PlotDiagnostics:
        """Inspect data characteristics and determine gate pass, warning, or failure status."""
        row_count = df.height
        available_cols = set(df.columns)

        if row_count == 0:
            issues = ["Resolved DataFrame contains 0 rows"]
            warnings = ["Resolved DataFrame contains 0 rows"]
            if spec is not None and spec.checks:
                req_cols = spec.checks.get("require_columns", [])
                if isinstance(req_cols, list):
                    for rc in req_cols:
                        if isinstance(rc, str) and rc not in available_cols:
                            msg = f"Required column '{rc}' not found in data frame"
                            issues.append(msg)
                            warnings.append(msg)
            return PlotDiagnostics(
                status="FAILED",
                issues=issues,
                sample_count=0,
                null_ratio=0.0,
                zero_variance=False,
                row_count=0,
                zero_variance_columns=[],
                null_columns=[],
                cardinality={},
                warnings=warnings,
            )

        null_cols: list[str] = []
        zero_var_cols: list[str] = []
        cardinality: dict[str, int] = {}
        warnings: list[str] = []
        issues: list[str] = []
        fatal: bool = False
        overall_zero_variance: bool = False
        overall_null_ratio: float = 0.0

        if row_count < 10:
            warn = f"Low sample size: N={row_count} (< 10 samples)"
            warnings.append(warn)
            issues.append(warn)

        for col in df.columns:
            series = df[col]
            effective = series.fill_nan(None) if series.dtype.is_float() else series
            n_null = effective.null_count()
            if n_null == row_count:
                null_cols.append(col)
                warn = f"Column '{col}' has only null values"
                warnings.append(warn)
                issues.append(warn)
                cardinality[col] = 0
                continue

            n_uniq = effective.n_unique()
            cardinality[col] = n_uniq
            if n_uniq <= 1:
                zero_var_cols.append(col)
                warn = f"Column '{col}' has zero variance (constant value)"
                warnings.append(warn)
                issues.append(warn)
            elif n_uniq > 50 and (series.dtype == pl.String or series.dtype == pl.Categorical):
                warn = f"Column '{col}' has high cardinality ({n_uniq} distinct categories)"
                warnings.append(warn)
                issues.append(warn)

        if spec is not None and spec.checks:
            req_cols = spec.checks.get("require_columns", [])
            if isinstance(req_cols, list):
                for rc in req_cols:
                    if isinstance(rc, str) and rc not in available_cols:
                        err = f"Required column '{rc}' not found in data frame"
                        warnings.append(err)
                        issues.append(err)
                        fatal = True

        metric_cols: list[str] = []
        if spec is not None:
            if spec.channels.y:
                if isinstance(spec.channels.y, str):
                    metric_cols.append(spec.channels.y)
                elif isinstance(spec.channels.y, list):
                    metric_cols.extend(col for col in spec.channels.y if isinstance(col, str) and col)
            elif spec.data_key and spec.data_key != "*":
                metric_cols.append(spec.data_key)

        if metric_cols:
            for col in metric_cols:
                if col not in available_cols:
                    err = f"Metric column '{col}' not found in data frame"
                    warnings.append(err)
                    issues.append(err)
                    fatal = True
                    continue

                series = df[col]
                effective = series.fill_nan(None) if series.dtype.is_float() else series
                n_null = effective.null_count()
                col_null_ratio = n_null / row_count
                if col_null_ratio > overall_null_ratio:
                    overall_null_ratio = col_null_ratio

                if col_null_ratio == 1.0:
                    err = f"Metric '{col}' is 100% NULL across all {row_count} rows"
                    warnings.append(err)
                    issues.append(err)
                    fatal = True
                else:
                    if series.dtype.is_numeric():
                        non_null = effective.drop_nulls()
                        if len(non_null) > 0 and non_null.min() == non_null.max():
                            overall_zero_variance = True
                            warn = f"Metric '{col}' has zero variance (constant value)"
                            warnings.append(warn)
                            issues.append(warn)
        else:
            if zero_var_cols:
                for col in zero_var_cols:
                    if df[col].dtype.is_numeric():
                        overall_zero_variance = True
                        break
            if null_cols and len(null_cols) == len(df.columns) and len(df.columns) > 0:
                overall_null_ratio = 1.0
            elif len(df.columns) > 0:
                overall_null_ratio = max((df[c].fill_nan(None) if df[c].dtype.is_float() else df[c]).null_count() / row_count for c in df.columns)

        unique_issues = list(dict.fromkeys(issues))
        unique_warnings = list(dict.fromkeys(warnings))

        if fatal:
            status = "FAILED"
        elif unique_warnings or overall_zero_variance:
            status = "WARNING"
        else:
            status = "PASSED"

        return PlotDiagnostics(
            status=status,
            issues=unique_issues,
            sample_count=row_count,
            null_ratio=overall_null_ratio,
            zero_variance=overall_zero_variance,
            row_count=row_count,
            zero_variance_columns=zero_var_cols,
            null_columns=null_cols,
            cardinality=cardinality,
            warnings=unique_warnings,
        )
