"""Parity certificate: frozen legacy metrics versus the canonical catalog views.

The legacy baseline stores one ``metrics.parquet`` frame per episode, frozen before the
star-schema port. The new run stores facts once per episode under
``episodes/episode_NNN/metrics/fact_*.parquet``; :class:`BenchmarkCatalog` exposes them
through the canonical views, so this harness never reads a benchmark-level
``combined_metrics.parquet`` or a stale ``metrics/fact_*.parquet``.

Run it with the two explicit paths::

    python tests/parity/certify.py \
      --baseline-dir /path/to/_parity_baseline/20261109 \
      --run-dir /path/to/benchmarks/<run-id>

Statuses (every row is reported; nothing is silently accepted):

* ``exact`` -- every compared episode carries the same value.
* ``within_tolerance`` -- values differ only inside ``--rel-tol``.
* ``expected_diff`` -- every difference is covered by ``expected_diffs.yaml``.
* ``missing`` -- no usable comparison: an unexplained difference, no joined episode,
  all values null on both sides, or the legacy column is absent.
* ``no_producer`` -- the mapped catalog view or column does not exist in this run.

``missing`` and ``no_producer`` fail the gate (exit code 1); a certificate cannot go
green on null==null.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import enum
import pathlib
import sys
import typing

import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.declarations import FACT_TABLE_NAMES

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "Certificate",
    "CertificateRow",
    "DiffKind",
    "ExpectedDiff",
    "ExpectedManifest",
    "MetricMapping",
    "ParityInputError",
    "ParitySchemaError",
    "ParityStatus",
    "certify",
    "load_baseline",
    "load_expected_manifest",
    "load_mappings",
    "main",
    "status_counts",
]

_HERE = pathlib.Path(__file__).resolve().parent

REL_TOL = 1.0e-5
_EPISODE_VIEW = "v_episode_summary"
_LEGACY_RUN_COLUMN = "benchmark_id"
_LEGACY_EPISODE_COLUMN = "episode"
_LEGACY_ROBOT_COLUMN = "robot"
_STAR_RUN_COLUMN = "run_id"
_STAR_EPISODE_COLUMN = "episode_id"
_STAR_ROBOT_COLUMN = "robot_id"
_ROBOT_KEY = "robot_key"
_JOIN_KEYS: typing.Final[tuple[str, ...]] = (
    _STAR_RUN_COLUMN,
    _STAR_EPISODE_COLUMN,
    _ROBOT_KEY,
)
_ROBOT_SUFFIX = r"_\d+$"
_STAR_TABLES: typing.Final[tuple[str, ...]] = tuple(f"fact_{name}" for name in FACT_TABLE_NAMES)
_EXAMPLE_LIMIT = 5


class ParityInputError(RuntimeError):
    """The baseline or run directory cannot be certified as given."""


class ParitySchemaError(RuntimeError):
    """A reviewed manifest or a catalog table violates the harness contract."""


class ParityStatus(enum.StrEnum):
    """Verdict of one mapped legacy-to-catalog comparison."""

    EXACT = "exact"
    WITHIN_TOLERANCE = "within_tolerance"
    EXPECTED_DIFF = "expected_diff"
    MISSING = "missing"
    NO_PRODUCER = "no_producer"


_FAILING_STATUSES: typing.Final[frozenset[ParityStatus]] = frozenset({ParityStatus.MISSING, ParityStatus.NO_PRODUCER})


class DiffKind(enum.StrEnum):
    """Reviewed explanation of an allowed legacy-to-catalog difference."""

    UNIT_SCALE = "unit_scale"
    DERIVED = "derived"
    GROUND_TRUTH = "ground_truth"
    NULL_FILL = "null_fill"
    DTYPE = "dtype"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MetricMapping:
    """Reviewed correspondence between one legacy column and one catalog column."""

    metric_key: str
    star_table: str
    star_column: str
    legacy_columns: tuple[str, ...]
    legacy_combine: str = "first"
    legacy_reduction: str = "identity"
    star_reduction: str = "identity"
    note: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedDiff:
    """Reviewed explanation of why a legacy value may legitimately differ."""

    metric_key: str
    kind: DiffKind
    reason: str
    star_table: str = ""
    star_column: str = ""
    factor: float | None = None
    max_rel: float | None = None
    allows_null: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedManifest:
    """The reviewed entries plus their cross-cutting notes."""

    entries: tuple[ExpectedDiff, ...]
    notes: tuple[Mapping[str, str], ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CertificateRow:
    """One certificate line: one legacy key against one catalog column."""

    metric_key: str
    legacy_column: str
    star_table: str
    star_column: str
    status: ParityStatus
    episodes_compared: int
    null_compared: int
    nonnull_compared: int
    max_abs_diff: float | None
    max_rel_diff: float | None
    detail: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Certificate:
    """Every certificate row of one baseline/run pair."""

    rows: tuple[CertificateRow, ...]
    run_id: str
    baseline_episodes: int
    run_episodes: int
    expected_diff_entries: int
    expected_diff_pending: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonStats:
    """Per-mapping comparison outcome before it is folded into a status."""

    episodes_compared: int
    null_compared: int
    nonnull_compared: int
    max_abs_diff: float | None
    max_rel_diff: float | None
    unexpected_count: int
    expected_diff_count: int
    tolerance_count: int
    example: str


# ---------------------------------------------------------------------------
# Reviewed manifests
# ---------------------------------------------------------------------------


def _as_mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ParitySchemaError(f"{where}: expected a mapping, got {type(value).__name__}")
    return typing.cast("Mapping[str, object]", value)


def _as_sequence(value: object, *, where: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ParitySchemaError(f"{where}: expected a list, got {type(value).__name__}")
    return typing.cast("Sequence[object]", value)


def _string_item(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParitySchemaError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _text(data: Mapping[str, object], key: str, *, where: str, default: str | None = None) -> str:
    return _string_item(data.get(key, default), where=f"{where}.{key}")


def _optional_text(data: Mapping[str, object], key: str, *, where: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise ParitySchemaError(f"{where}.{key}: expected a string, got {value!r}")
    return value


def _optional_float(data: Mapping[str, object], key: str, *, where: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParitySchemaError(f"{where}.{key}: expected a number, got {value!r}")
    return float(value)


def _optional_bool(data: Mapping[str, object], key: str, *, where: str) -> bool:
    value = data.get(key, False)
    if not isinstance(value, bool):
        raise ParitySchemaError(f"{where}.{key}: expected a boolean, got {value!r}")
    return value


def _load_yaml(path: pathlib.Path) -> object:
    import yaml

    if not path.is_file():
        raise ParityInputError(f"missing reviewed manifest: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ParitySchemaError(f"{path}: invalid YAML: {error}") from error


def load_mappings(path: pathlib.Path) -> tuple[MetricMapping, ...]:
    """Read the reviewed legacy-key -> catalog-column map."""
    document = _as_mapping(_load_yaml(path), where=path.name)
    entries = _as_sequence(document.get("mappings"), where=f"{path.name}.mappings")
    mappings: list[MetricMapping] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(entries):
        where = f"{path.name}.mappings[{index}]"
        entry = _as_mapping(raw, where=where)
        legacy_columns = tuple(_string_item(column, where=f"{where}.legacy_columns") for column in _as_sequence(entry.get("legacy_columns"), where=f"{where}.legacy_columns"))
        if not legacy_columns:
            raise ParitySchemaError(f"{where}: legacy_columns is empty")
        mapping = MetricMapping(
            metric_key=_text(entry, "metric_key", where=where),
            star_table=_text(entry, "star_table", where=where),
            star_column=_text(entry, "star_column", where=where),
            legacy_columns=legacy_columns,
            legacy_combine=_text(entry, "legacy_combine", where=where, default="first"),
            legacy_reduction=_text(entry, "legacy_reduction", where=where, default="identity"),
            star_reduction=_text(entry, "star_reduction", where=where, default="identity"),
            note=_optional_text(entry, "note", where=where),
        )
        target = (mapping.metric_key, mapping.star_table, mapping.star_column)
        if target in seen:
            raise ParitySchemaError(f"duplicate mapping {target}")
        if mapping.star_table not in _STAR_TABLES:
            raise ParitySchemaError(f"mapping {target}: unknown star table {mapping.star_table}")
        if mapping.legacy_combine == "sum" and len(mapping.legacy_columns) < 2:
            raise ParitySchemaError(f"mapping {target}: legacy_combine=sum needs at least two columns")
        if mapping.legacy_combine == "first" and len(mapping.legacy_columns) != 1:
            raise ParitySchemaError(f"mapping {target}: legacy_combine=first needs exactly one column")
        seen.add(target)
        mappings.append(mapping)
    if not mappings:
        raise ParitySchemaError(f"{path}: empty mappings")
    return tuple(mappings)


def load_expected_manifest(path: pathlib.Path) -> ExpectedManifest:
    """Read the reviewed expected-difference entries and their notes."""
    document = _as_mapping(_load_yaml(path), where=path.name)
    entries = _as_sequence(document.get("entries", []), where=f"{path.name}.entries")
    expected: list[ExpectedDiff] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(entries):
        where = f"{path.name}.entries[{index}]"
        entry = _as_mapping(raw, where=where)
        kind = DiffKind(_text(entry, "kind", where=where))
        factor = _optional_float(entry, "factor", where=where)
        max_rel = _optional_float(entry, "max_rel", where=where)
        if kind is DiffKind.UNIT_SCALE and factor is None:
            raise ParitySchemaError(f"{where}: unit_scale needs a factor")
        if kind is DiffKind.DERIVED and max_rel is None:
            raise ParitySchemaError(f"{where}: derived needs a reviewed max_rel bound")
        expected_diff = ExpectedDiff(
            metric_key=_text(entry, "metric_key", where=where),
            kind=kind,
            reason=_text(entry, "reason", where=where),
            star_table=_optional_text(entry, "star_table", where=where),
            star_column=_optional_text(entry, "star_column", where=where),
            factor=factor,
            max_rel=max_rel,
            allows_null=_optional_bool(entry, "allows_null", where=where),
        )
        target = (
            expected_diff.metric_key,
            expected_diff.star_table,
            expected_diff.star_column,
        )
        if target in seen:
            raise ParitySchemaError(f"duplicate expected diff {target}")
        if bool(expected_diff.star_table) != bool(expected_diff.star_column):
            raise ParitySchemaError(f"expected diff {expected_diff.metric_key}: star_table and star_column must both be set or both empty")
        seen.add(target)
        expected.append(expected_diff)
    notes = tuple(_as_mapping(note, where=f"{path.name}.notes[{index}]") for index, note in enumerate(_as_sequence(document.get("notes", []), where=f"{path.name}.notes")))
    return ExpectedManifest(entries=tuple(expected), notes=notes)


def _expected_index(
    entries: tuple[ExpectedDiff, ...],
) -> Mapping[tuple[str, str, str], ExpectedDiff]:
    """Exact target first; an entry without a target applies to every mapping of its key."""
    index: dict[tuple[str, str, str], ExpectedDiff] = {}
    for entry in entries:
        index[(entry.metric_key, entry.star_table, entry.star_column)] = entry
    return index


# ---------------------------------------------------------------------------
# Baseline and run loading
# ---------------------------------------------------------------------------


def _robot_key(column: pl.Expr) -> pl.Expr:
    """Strip the instance suffix of a runtime robot id so ``env_0_jackal`` meets ``env_0_jackal_0``."""
    return column.str.replace(_ROBOT_SUFFIX, "").alias(_ROBOT_KEY)


def load_baseline(directory: pathlib.Path, *, benchmark: str | None = None) -> pl.DataFrame:
    """Concatenate the frozen per-episode legacy frames, normalized to the catalog join keys."""
    baseline_dir = pathlib.Path(directory)
    paths = sorted(baseline_dir.glob("*.parquet"))
    if not paths:
        raise ParityInputError(f"{baseline_dir} holds no per-episode baseline parquet files")
    baseline = pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed")
    required = {_LEGACY_RUN_COLUMN, _LEGACY_EPISODE_COLUMN, _LEGACY_ROBOT_COLUMN}
    missing = sorted(required - set(baseline.collect_schema().names()))
    if missing:
        raise ParityInputError(f"{baseline_dir} is not a legacy per-episode baseline: missing {missing}")
    if benchmark is not None:
        baseline = baseline.filter(pl.col(_LEGACY_RUN_COLUMN) == benchmark)
    if baseline.is_empty():
        benchmark_note = f" for benchmark {benchmark!r}" if benchmark else ""
        raise ParityInputError(f"{baseline_dir} holds no baseline episode{benchmark_note}")
    return baseline.with_columns(
        pl.col(_LEGACY_RUN_COLUMN).alias(_STAR_RUN_COLUMN),
        pl.col(_LEGACY_EPISODE_COLUMN).cast(pl.Int64).alias(_STAR_EPISODE_COLUMN),
        _robot_key(pl.col(_LEGACY_ROBOT_COLUMN)),
    )


def load_run(run_dir: pathlib.Path) -> tuple[BenchmarkCatalog, pl.DataFrame]:
    """The catalog over the per-episode facts and its episode-level canonical view."""
    catalog = BenchmarkCatalog.from_directory(pathlib.Path(run_dir))
    summary = catalog.get_table(_EPISODE_VIEW).collect()
    if summary.is_empty():
        raise ParityInputError(f"{run_dir} holds no per-episode fact rows, so {_EPISODE_VIEW} is empty")
    run_ids = sorted(str(value) for value in summary.get_column(_STAR_RUN_COLUMN).unique().to_list())
    if len(run_ids) != 1:
        raise ParityInputError(f"{run_dir} mixes run ids {run_ids}; certify one benchmark directory at a time")
    return catalog, summary


def _catalog_table(catalog: BenchmarkCatalog, star_table: str) -> pl.LazyFrame:
    """The canonical view for episode facts, the scanned fact view for every other scope."""
    return catalog.get_table(_EPISODE_VIEW if star_table == "fact_episodes" else star_table)


def _with_robot_key(frame: pl.DataFrame) -> pl.DataFrame:
    if _STAR_ROBOT_COLUMN not in frame.collect_schema().names():
        return frame
    return frame.with_columns(_robot_key(pl.col(_STAR_ROBOT_COLUMN)))


# ---------------------------------------------------------------------------
# Joining the legacy and catalog grains
# ---------------------------------------------------------------------------


def _legacy_value_expression(mapping: MetricMapping) -> pl.Expr:
    expressions = [pl.col(column) for column in mapping.legacy_columns]
    combined = expressions[0]
    for expression in expressions[1:]:
        combined = combined + expression
    reducers = {
        "identity": combined,
        "list_mean": combined.list.mean(),
        "list_min": combined.list.min(),
        "list_max": combined.list.max(),
        "list_sum": combined.list.sum(),
        "list_len": combined.list.len().cast(pl.Float64),
    }
    if mapping.legacy_reduction not in reducers:
        raise ParitySchemaError(f"{mapping.metric_key}: unknown legacy_reduction {mapping.legacy_reduction!r}")
    return reducers[mapping.legacy_reduction].alias("legacy_value")


def _legacy_value_frame(baseline: pl.DataFrame, mapping: MetricMapping) -> pl.DataFrame | None:
    names = set(baseline.collect_schema().names())
    if any(column not in names for column in mapping.legacy_columns):
        return None
    return baseline.select([*_JOIN_KEYS, _legacy_value_expression(mapping)])


def _star_value_frame(catalog: BenchmarkCatalog, mapping: MetricMapping) -> pl.DataFrame | None:
    frame = _with_robot_key(_catalog_table(catalog, mapping.star_table).collect())
    names = set(frame.collect_schema().names())
    if mapping.star_column not in names:
        return None
    missing_keys = [key for key in _JOIN_KEYS if key not in names]
    if missing_keys:
        raise ParitySchemaError(f"{mapping.star_table}: missing join keys {missing_keys}")
    if mapping.star_reduction == "identity":
        duplicates = frame.group_by(list(_JOIN_KEYS)).len().filter(pl.col("len") > 1)
        if not duplicates.is_empty():
            raise ParitySchemaError(f"{mapping.star_table}.{mapping.star_column}: {duplicates.height} duplicated key rows; declare a star_reduction instead of identity")
        return frame.select([*_JOIN_KEYS, pl.col(mapping.star_column).alias("star_value")])
    reducers = {
        "mean": pl.col(mapping.star_column).mean(),
        "min": pl.col(mapping.star_column).min(),
        "max": pl.col(mapping.star_column).max(),
        "sum": pl.col(mapping.star_column).sum(),
        "count": pl.len().cast(pl.Float64),
    }
    if mapping.star_reduction not in reducers:
        raise ParitySchemaError(f"{mapping.metric_key}: unknown star_reduction {mapping.star_reduction!r}")
    return frame.group_by(list(_JOIN_KEYS)).agg(reducers[mapping.star_reduction].alias("star_value"))


# ---------------------------------------------------------------------------
# One comparison
# ---------------------------------------------------------------------------


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _example_text(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return ""
    shown = frame.head(_EXAMPLE_LIMIT)
    rendered = shown.select(
        pl.format(
            "ep={} robot={} legacy={} catalog={}",
            pl.col(_STAR_EPISODE_COLUMN),
            pl.col(_ROBOT_KEY),
            pl.col("legacy_value"),
            pl.col("star_value"),
        )
    ).to_series()
    return "; ".join(str(value) for value in rendered.to_list())


def _numeric_stats(joined: pl.DataFrame, *, expected: ExpectedDiff | None, rel_tol: float) -> ComparisonStats:
    factor = expected.factor if expected is not None and expected.factor is not None else 1.0
    legacy = pl.col("legacy_value").cast(pl.Float64)
    star = pl.col("star_value").cast(pl.Float64)
    target = legacy * pl.lit(factor)
    both_present = legacy.is_not_null() & star.is_not_null()
    both_null = legacy.is_null() & star.is_null()
    nan_both = legacy.is_nan() & star.is_nan()
    exact = nan_both | (both_present & (star == target))
    scale = pl.max_horizontal(target.abs(), star.abs())
    abs_diff = (star - target).abs()
    rel_diff = pl.when(scale > pl.lit(0.0)).then(abs_diff / scale).otherwise(pl.lit(0.0))
    tolerance = both_present & (rel_diff <= pl.lit(rel_tol))
    expected_difference = pl.lit(False)
    allowed_null = pl.lit(False)
    if expected is not None:
        if expected.kind is DiffKind.UNIT_SCALE:
            bound = expected.max_rel if expected.max_rel is not None else 1.0e-9
            expected_difference = both_present & ~exact & ~tolerance & (rel_diff <= pl.lit(bound))
        elif expected.kind is DiffKind.DERIVED:
            expected_difference = both_present & ~exact & ~tolerance & (rel_diff <= pl.lit(expected.max_rel or 0.0))
        elif expected.kind is DiffKind.GROUND_TRUTH:
            expected_difference = both_present & ~exact & ~tolerance
        elif expected.kind is DiffKind.NULL_FILL:
            allowed_null = star.is_null() & legacy.is_not_null()
            expected_difference = allowed_null
        elif expected.kind is DiffKind.DTYPE:
            expected_difference = pl.lit(False)
        if expected.allows_null:
            allowed_null = star.is_null() & legacy.is_not_null()
    unexpected_value = both_present & ~exact & ~tolerance & ~expected_difference
    unexpected_null = star.is_null() & legacy.is_not_null() & ~allowed_null
    unexpected_legacy_null = legacy.is_null() & star.is_not_null()
    enriched = joined.with_columns(
        exact.alias("_exact"),
        tolerance.alias("_tolerance"),
        expected_difference.alias("_expected"),
        (unexpected_value | unexpected_null | unexpected_legacy_null).alias("_unexpected"),
        abs_diff.alias("_abs_diff"),
        rel_diff.alias("_rel_diff"),
    )
    non_null = enriched.filter(both_present)
    both_null_frame = enriched.filter(both_null)
    unexpected = enriched.filter(pl.col("_unexpected"))
    return ComparisonStats(
        episodes_compared=enriched.height,
        null_compared=both_null_frame.height,
        nonnull_compared=non_null.height,
        max_abs_diff=None if non_null.is_empty() else float(non_null["_abs_diff"].max()),
        max_rel_diff=None if non_null.is_empty() else float(non_null["_rel_diff"].max()),
        unexpected_count=unexpected.height,
        expected_diff_count=enriched.filter(pl.col("_expected")).height,
        tolerance_count=enriched.filter(pl.col("_tolerance") & ~pl.col("_exact")).height,
        example=_example_text(unexpected),
    )


def _non_numeric_stats(joined: pl.DataFrame, *, expected: ExpectedDiff | None) -> ComparisonStats:
    legacy_dtype = joined.collect_schema()["legacy_value"]
    star_dtype = joined.collect_schema()["star_value"]
    dtype_allowed = expected is not None and expected.kind is DiffKind.DTYPE
    if legacy_dtype != star_dtype and not dtype_allowed:
        return ComparisonStats(
            episodes_compared=joined.height,
            null_compared=0,
            nonnull_compared=0,
            max_abs_diff=None,
            max_rel_diff=None,
            unexpected_count=joined.height,
            expected_diff_count=0,
            tolerance_count=0,
            example=f"dtype change {legacy_dtype} -> {star_dtype} has no reviewed dtype entry",
        )
    both_null = joined["legacy_value"].is_null() & joined["star_value"].is_null()
    equal = both_null | (joined["legacy_value"].cast(pl.String) == joined["star_value"].cast(pl.String))
    difference_allowed = expected is not None and expected.kind is DiffKind.GROUND_TRUTH
    expected_difference = (~equal) if difference_allowed else pl.Series([False] * joined.height)
    allowed_null = pl.Series([False] * joined.height)
    if expected is not None and (expected.allows_null or expected.kind is DiffKind.NULL_FILL):
        allowed_null = joined["star_value"].is_null() & joined["legacy_value"].is_not_null()
    unexpected_null = joined["star_value"].is_null() & joined["legacy_value"].is_not_null() & ~allowed_null
    unexpected_legacy_null = joined["legacy_value"].is_null() & joined["star_value"].is_not_null()
    unexpected = ~(equal | expected_difference) | unexpected_null | unexpected_legacy_null
    non_null = joined.filter(joined["legacy_value"].is_not_null() & joined["star_value"].is_not_null())
    return ComparisonStats(
        episodes_compared=joined.height,
        null_compared=joined.filter(both_null).height,
        nonnull_compared=non_null.height,
        max_abs_diff=None,
        max_rel_diff=None,
        unexpected_count=joined.filter(unexpected).height,
        expected_diff_count=joined.filter(expected_difference).height,
        tolerance_count=0,
        example=_example_text(joined.filter(unexpected)),
    )


def _comparison_stats(joined: pl.DataFrame, *, expected: ExpectedDiff | None, rel_tol: float) -> ComparisonStats:
    legacy_dtype = joined.collect_schema()["legacy_value"]
    star_dtype = joined.collect_schema()["star_value"]
    if legacy_dtype == pl.Null or star_dtype == pl.Null:
        legacy_is_null = joined["legacy_value"].is_null()
        star_is_null = joined["star_value"].is_null()
        both_null = legacy_is_null & star_is_null
        unexpected = legacy_is_null != star_is_null
        return ComparisonStats(
            episodes_compared=joined.height,
            null_compared=joined.filter(both_null).height,
            nonnull_compared=0,
            max_abs_diff=None,
            max_rel_diff=None,
            unexpected_count=joined.filter(unexpected).height,
            expected_diff_count=0,
            tolerance_count=0,
            example=_example_text(joined.filter(unexpected)),
        )
    if _is_numeric(legacy_dtype) and _is_numeric(star_dtype):
        return _numeric_stats(joined, expected=expected, rel_tol=rel_tol)
    return _non_numeric_stats(joined, expected=expected)


def _status_for(stats: ComparisonStats) -> tuple[ParityStatus, str]:
    if stats.unexpected_count > 0:
        detail = f"UNEXPLAINED difference on {stats.unexpected_count} episode(s)"
        if stats.example:
            detail = f"{detail}: {stats.example}"
        return ParityStatus.MISSING, detail
    if stats.episodes_compared > 0 and stats.null_compared == stats.episodes_compared:
        return (
            ParityStatus.MISSING,
            f"both sides are null for all {stats.episodes_compared} compared episode(s)",
        )
    if stats.expected_diff_count > 0:
        return (
            ParityStatus.EXPECTED_DIFF,
            f"{stats.expected_diff_count} reviewed difference(s)",
        )
    if stats.nonnull_compared == 0:
        return ParityStatus.MISSING, "no non-null comparison"
    if stats.tolerance_count > 0:
        return (
            ParityStatus.WITHIN_TOLERANCE,
            f"{stats.tolerance_count} difference(s) inside the relative tolerance",
        )
    return ParityStatus.EXACT, ""


def _compare_mapping(
    *,
    catalog: BenchmarkCatalog,
    baseline: pl.DataFrame,
    mapping: MetricMapping,
    expected: ExpectedDiff | None,
    rel_tol: float,
) -> CertificateRow:
    legacy_columns = ", ".join(mapping.legacy_columns)
    legacy = _legacy_value_frame(baseline, mapping)
    if legacy is None:
        return CertificateRow(
            metric_key=mapping.metric_key,
            legacy_column=legacy_columns,
            star_table=mapping.star_table,
            star_column=mapping.star_column,
            status=ParityStatus.MISSING,
            episodes_compared=0,
            null_compared=0,
            nonnull_compared=0,
            max_abs_diff=None,
            max_rel_diff=None,
            detail=f"legacy column absent from the frozen baseline: {legacy_columns}",
        )
    star = _star_value_frame(catalog, mapping)
    if star is None:
        return CertificateRow(
            metric_key=mapping.metric_key,
            legacy_column=legacy_columns,
            star_table=mapping.star_table,
            star_column=mapping.star_column,
            status=ParityStatus.NO_PRODUCER,
            episodes_compared=0,
            null_compared=0,
            nonnull_compared=0,
            max_abs_diff=None,
            max_rel_diff=None,
            detail=f"mapped catalog column absent from {mapping.star_table}",
        )
    joined = legacy.join(star, on=list(_JOIN_KEYS), how="inner")
    if joined.is_empty():
        return CertificateRow(
            metric_key=mapping.metric_key,
            legacy_column=legacy_columns,
            star_table=mapping.star_table,
            star_column=mapping.star_column,
            status=ParityStatus.MISSING,
            episodes_compared=0,
            null_compared=0,
            nonnull_compared=0,
            max_abs_diff=None,
            max_rel_diff=None,
            detail="no baseline episode matches the run on (run_id, episode_id, robot)",
        )
    stats = _comparison_stats(joined, expected=expected, rel_tol=rel_tol)
    status, detail = _status_for(stats)
    if status is ParityStatus.EXPECTED_DIFF and expected is not None:
        detail = f"{detail}: {expected.reason}"
    elif not detail:
        detail = mapping.note
    return CertificateRow(
        metric_key=mapping.metric_key,
        legacy_column=legacy_columns,
        star_table=mapping.star_table,
        star_column=mapping.star_column,
        status=status,
        episodes_compared=stats.episodes_compared,
        null_compared=stats.null_compared,
        nonnull_compared=stats.nonnull_compared,
        max_abs_diff=stats.max_abs_diff,
        max_rel_diff=stats.max_rel_diff,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------


def _expected_for(mapping: MetricMapping, index: Mapping[tuple[str, str, str], ExpectedDiff]) -> ExpectedDiff | None:
    exact = index.get((mapping.metric_key, mapping.star_table, mapping.star_column))
    if exact is not None:
        return exact
    return index.get((mapping.metric_key, "", ""))


def certify(
    *,
    catalog: BenchmarkCatalog,
    baseline: pl.DataFrame,
    run_summary: pl.DataFrame,
    mappings: tuple[MetricMapping, ...],
    expected_diffs: tuple[ExpectedDiff, ...] = (),
    rel_tol: float = REL_TOL,
) -> Certificate:
    """Compare every reviewed mapping, returning the certificate bundle."""
    run_ids = sorted(str(value) for value in run_summary.get_column(_STAR_RUN_COLUMN).unique().to_list())
    run_id = run_ids[0]
    index = _expected_index(expected_diffs)
    rows = [
        _compare_mapping(
            catalog=catalog,
            baseline=baseline.filter(pl.col(_STAR_RUN_COLUMN) == run_id),
            mapping=mapping,
            expected=_expected_for(mapping, index),
            rel_tol=rel_tol,
        )
        for mapping in mappings
    ]
    exercised = {mapping.metric_key for mapping in mappings}
    pending = sum(1 for entry in expected_diffs if entry.metric_key not in exercised)
    return Certificate(
        rows=tuple(sorted(rows, key=lambda row: (row.metric_key, row.star_table, row.star_column))),
        run_id=run_id,
        baseline_episodes=baseline.height,
        run_episodes=run_summary.height,
        expected_diff_entries=len(expected_diffs),
        expected_diff_pending=pending,
    )


def status_counts(certificate: Certificate) -> Mapping[ParityStatus, int]:
    """Count certificate rows per status, with a zero for every known status."""
    counts: dict[ParityStatus, int] = {status: 0 for status in ParityStatus}
    for row in certificate.rows:
        counts[row.status] += 1
    return counts


def write_certificate_csv(path: pathlib.Path, certificate: Certificate) -> None:
    """Write the flat certificate table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric_key",
                "legacy_column",
                "star_table",
                "star_column",
                "status",
                "episodes_compared",
                "null_compared",
                "nonnull_compared",
                "max_abs_diff",
                "max_rel_diff",
                "detail",
            ]
        )
        for row in certificate.rows:
            writer.writerow(
                [
                    row.metric_key,
                    row.legacy_column,
                    row.star_table,
                    row.star_column,
                    row.status.value,
                    row.episodes_compared,
                    row.null_compared,
                    row.nonnull_compared,
                    "" if row.max_abs_diff is None else f"{row.max_abs_diff:.12g}",
                    "" if row.max_rel_diff is None else f"{row.max_rel_diff:.12g}",
                    row.detail,
                ]
            )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_check_frames() -> tuple[BenchmarkCatalog, pl.DataFrame, pl.DataFrame]:
    fact = pl.DataFrame(
        {
            "run_id": ["run"] * 3,
            "episode_id": [0, 1, 2],
            "robot_id": ["env_0_jackal_0"] * 3,
            "success": [True, False, True],
            "time_to_goal_s": [10.0, 20.0, 30.0],
            "path_efficiency": [None, 0.9, 0.8],
            "null_reason": ["path_efficiency:not_recorded", None, None],
        }
    )
    from arena_evaluation.storage.star_schema import (
        DIM_EPISODES_SCHEMA,
        DIM_ROBOTS_SCHEMA,
    )

    def dimension(schema: pl.Schema) -> pl.DataFrame:
        rows = []
        for episode_id in (0, 1, 2):
            row = {name: None for name in schema.names()}
            row.update(
                run_id="run",
                episode_id=episode_id,
                env_id="env_0",
                robot_id="env_0_jackal_0",
                benchmark_id="run",
                planner="dwb",
            )
            rows.append(row)
        return pl.DataFrame(rows, schema=schema)

    catalog = BenchmarkCatalog(
        {
            "fact_episodes": fact,
            "dim_episodes": dimension(DIM_EPISODES_SCHEMA),
            "dim_robots": dimension(DIM_ROBOTS_SCHEMA),
        }
    )
    baseline = pl.DataFrame(
        {
            "benchmark_id": ["run"] * 3,
            "episode": [0, 1, 2],
            "robot": ["env_0_jackal"] * 3,
            "run_id": ["run"] * 3,
            "episode_id": [0, 1, 2],
            "robot_key": ["env_0_jackal"] * 3,
            "time_to_goal": [10.0, 20.0, 30.0],
            "path_efficiency": [0.0, 0.9, 0.8],
        }
    )
    return catalog, fact, baseline


def self_check() -> int:
    """Exercise the comparison engine on synthetic frames without touching a benchmark."""
    catalog, fact, baseline = _self_check_frames()
    mappings = (
        MetricMapping(
            metric_key="time_to_goal",
            star_table="fact_episodes",
            star_column="time_to_goal_s",
            legacy_columns=("time_to_goal",),
        ),
        MetricMapping(
            metric_key="path_efficiency",
            star_table="fact_episodes",
            star_column="path_efficiency",
            legacy_columns=("path_efficiency",),
        ),
    )
    expected = (
        ExpectedDiff(
            metric_key="path_efficiency",
            kind=DiffKind.NULL_FILL,
            reason="synthetic null fill",
        ),
    )
    certificate = certify(
        catalog=catalog,
        baseline=baseline,
        run_summary=fact,
        mappings=mappings,
        expected_diffs=expected,
    )
    rows = {row.metric_key: row for row in certificate.rows}
    failures: list[str] = []
    if rows["time_to_goal"].status is not ParityStatus.EXACT:
        failures.append(f"time_to_goal: expected exact, got {rows['time_to_goal'].status}")
    if rows["path_efficiency"].status is not ParityStatus.EXPECTED_DIFF:
        failures.append(f"path_efficiency: expected expected_diff, got {rows['path_efficiency'].status}")
    for failure in failures:
        print(f"SELF-CHECK FAILED: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("self-check: comparison engine OK")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parity certificate between the frozen legacy per-episode metrics and the canonical catalog views.")
    parser.add_argument(
        "--baseline-dir",
        type=pathlib.Path,
        help="Directory of frozen per-episode legacy metrics.parquet files",
    )
    parser.add_argument(
        "--run-dir",
        type=pathlib.Path,
        help="Processed benchmark directory holding per-episode facts",
    )
    parser.add_argument(
        "--benchmark",
        help="Baseline benchmark id, inferred from the run's dim_episodes run_id when omitted",
    )
    parser.add_argument(
        "--mappings",
        type=pathlib.Path,
        default=_HERE / "metric_map.yaml",
        help="Reviewed legacy -> catalog map",
    )
    parser.add_argument(
        "--expected-diffs",
        type=pathlib.Path,
        default=_HERE / "expected_diffs.yaml",
        help="Reviewed expected-difference manifest",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=REL_TOL,
        help="Relative tolerance for within_tolerance",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, help="Write certificate.csv here")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Exercise the comparison engine on synthetic frames and exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    if args.self_check:
        return self_check()
    if args.baseline_dir is None or args.run_dir is None:
        print(
            "error: --baseline-dir and --run-dir are required unless --self-check is set",
            file=sys.stderr,
        )
        return 2
    try:
        catalog, summary = load_run(args.run_dir)
        run_id = str(summary.get_column(_STAR_RUN_COLUMN).unique().to_list()[0])
        baseline = load_baseline(args.baseline_dir, benchmark=args.benchmark or run_id)
        mappings = load_mappings(args.mappings)
        manifest = load_expected_manifest(args.expected_diffs)
        certificate = certify(
            catalog=catalog,
            baseline=baseline,
            run_summary=summary,
            mappings=mappings,
            expected_diffs=manifest.entries,
            rel_tol=args.rel_tol,
        )
    except (ParityInputError, ParitySchemaError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.output_dir is not None:
        write_certificate_csv(args.output_dir / "certificate.csv", certificate)
    counts = status_counts(certificate)
    for status in ParityStatus:
        print(f"{status.value}: {counts[status]}")
    print(f"run_id: {certificate.run_id}  baseline_episodes: {certificate.baseline_episodes}  run_episodes: {certificate.run_episodes}")
    print(f"expected_diff_entries: {certificate.expected_diff_entries}  pending: {certificate.expected_diff_pending}")
    return 1 if any(counts[status] for status in _FAILING_STATUSES) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
