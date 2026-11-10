"""G1 parity certificate: compare the legacy combined metrics against the star schema.

The dual-path pipeline writes one legacy frame (``combined_metrics.parquet``) and one
star-schema lakehouse (``dim_episodes.parquet`` + ``metrics/fact_*.parquet``) per
benchmark. This generator joins them on ``(run_id, episode_id[, robot_id])`` and emits
a per-metric certificate plus the inventory gap list.

Run it against a benchmark that has both artifacts::

    python tests/parity/certify.py --benchmark-dir /path/to/benchmark

A run that predates the star schema (legacy only) cannot be certified; reprocess it
first::

    arena evaluation process --benchmark-dir /path/to/benchmark --workers 1

Statuses:

* ``EXACT`` -- every compared episode carries the same value in both tables.
* ``WITHIN_TOL`` -- numeric values agree within ``REL_TOL`` (relative, per episode).
* ``EXPECTED_DIFF`` -- a reviewed ``expected_diff.yaml`` entry explains the difference.
* ``MISSING_STAR`` -- the inventory key has no scoped producer in this pipeline revision.
* ``MISSING_LEGACY`` -- a mapped legacy column is absent from this run's frame.
* ``NULL_REASON`` -- the legacy value is ``0.0`` and the star value is null with a reason.
* ``MISMATCH`` -- an unexplained difference; the certificate fails (exit code 1).
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

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_HERE = pathlib.Path(__file__).resolve().parent
_RESULTS_DIR = _HERE / "results"

REL_TOL = 1.0e-5
_DEFAULT_UNIT_SCALE_TOL = 1.0e-9
_EXAMPLE_LIMIT = 5
_LEGACY_RUN_COLUMN = "benchmark_id"
_LEGACY_EPISODE_COLUMN = "episode"
_LEGACY_ROBOT_COLUMN = "robot"
_STAR_RUN_COLUMN = "run_id"
_STAR_EPISODE_COLUMN = "episode_id"
_STAR_ROBOT_COLUMN = "robot_id"

_REQUIRED_FACT_TABLES: tuple[str, ...] = (
    "fact_episodes",
    "fact_encounters",
    "fact_contacts",
    "fact_pedestrians",
    "fact_robots",
)
_CHILD_FACT_TABLES: tuple[str, ...] = (
    "fact_edge_cases",
    "fact_formations",
    "fact_phase_outcomes",
    "fact_judge_disagreements",
    "fact_mechanism_states",
    "fact_collisions",
)
STAR_TABLES: tuple[str, ...] = (
    "dim_episodes",
    *_REQUIRED_FACT_TABLES,
    *_CHILD_FACT_TABLES,
)
_ROBOT_KEYED_TABLES: frozenset[str] = frozenset(
    {
        "fact_episodes",
        "fact_encounters",
        "fact_contacts",
        "fact_robots",
        *_CHILD_FACT_TABLES,
    }
)
# Child tables are optional: a run processed before they landed carries only the five
# scoped fact tables and must still certify.
_FACT_TABLES: tuple[str, ...] = (*_REQUIRED_FACT_TABLES, *_CHILD_FACT_TABLES)
_KEY_COLUMNS: frozenset[str] = frozenset(
    {
        "run_id",
        "episode_id",
        "env_id",
        "robot_id",
        "ped_id",
        "encounter_id",
        "contact_id",
        "null_reason",
        "case_idx",
        "formation_idx",
        "phase_idx",
        "disagreement_idx",
        "ordinal",
        "collision_idx",
        "entity",
        "level",
        "kind",
        "field",
        "role",
        "t_ns",
        "value",
        "x_m",
        "y_m",
    }
)


class ParityStatus(enum.StrEnum):
    """Verdict of one metric comparison."""

    EXACT = "EXACT"
    WITHIN_TOL = "WITHIN_TOL"
    EXPECTED_DIFF = "EXPECTED_DIFF"
    UNMAPPED = "UNMAPPED"
    NO_PRODUCER = "NO_PRODUCER"
    MISSING_LEGACY = "MISSING_LEGACY"
    NULL_REASON = "NULL_REASON"
    RELOCATED_SERIES = "RELOCATED_SERIES"
    CERT_NULL_ONLY = "CERT_NULL_ONLY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MISMATCH = "MISMATCH"


_FAILING_STATUSES: frozenset[ParityStatus] = frozenset(
    {
        ParityStatus.MISMATCH,
        ParityStatus.CERT_NULL_ONLY,
        ParityStatus.INSUFFICIENT_EVIDENCE,
    }
)

_ACCEPTED_STATUSES: frozenset[ParityStatus] = frozenset(
    {
        ParityStatus.EXACT,
        ParityStatus.WITHIN_TOL,
        ParityStatus.EXPECTED_DIFF,
        ParityStatus.UNMAPPED,
        ParityStatus.NO_PRODUCER,
        ParityStatus.MISSING_LEGACY,
        ParityStatus.NULL_REASON,
        ParityStatus.RELOCATED_SERIES,
    }
)


class StarReduction(enum.StrEnum):
    """How a mapping collapses a multi-row star table onto the episode grain."""

    IDENTITY = "identity"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"


class LegacyReduction(enum.StrEnum):
    """How a mapping reduces a legacy list/timeseries column to a scalar."""

    IDENTITY = "identity"
    LIST_MEAN = "list_mean"
    LIST_MIN = "list_min"
    LIST_MAX = "list_max"
    LIST_SUM = "list_sum"
    LIST_LEN = "list_len"


class LegacyCombine(enum.StrEnum):
    """How a mapping combines several legacy columns into one value."""

    FIRST = "first"
    SUM = "sum"


class DiffKind(enum.StrEnum):
    """Reviewed explanation of an allowed legacy-to-star difference."""

    UNIT_SCALE = "unit_scale"
    DERIVED = "derived"
    GROUND_TRUTH = "ground_truth"
    NULL_FILL = "null_fill"
    DTYPE = "dtype"


_TEAM_OWNER_BY_CATEGORY: Mapping[str, str] = {
    "social": "port-social",
    "safety": "port-social",
    "edge_case": "port-social",
    "holistic": "port-social",
    "ecological": "port-eco",
    "performance": "port-eco",
    "naturalness": "port-eco",
}
_G1_OWNER = "parity-g1"


class ParityInputError(RuntimeError):
    """The benchmark directory cannot be certified as given."""


class ParitySchemaError(RuntimeError):
    """A reviewed manifest or the benchmark schema violates the harness contract."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InventoryRow:
    """One legacy metric key the port must reproduce somewhere in the star schema."""

    metric_key: str
    calculator: str
    category: str
    inventory_owner: str
    target_scope: str
    target_table: str
    test_file: str
    port_status: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MetricMapping:
    """Reviewed correspondence between one inventory key and one star-schema column."""

    metric_key: str
    star_table: str
    star_column: str
    legacy_columns: tuple[str, ...]
    legacy_combine: LegacyCombine = LegacyCombine.FIRST
    legacy_reduction: LegacyReduction = LegacyReduction.IDENTITY
    star_reduction: StarReduction = StarReduction.IDENTITY
    note: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedDiff:
    """Reviewed explanation of why a legacy value may legitimately differ in the star.

    An entry with an empty ``star_table``/``star_column`` applies to every mapping of
    ``metric_key``; a pending entry documents a reviewed change whose star producer has
    not landed yet.
    """

    metric_key: str
    kind: DiffKind
    reason: str
    star_table: str = ""
    star_column: str = ""
    factor: float | None = None
    max_rel: float | None = None
    allows_null: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SeriesRelocation:
    """A legacy list/timeseries column whose per-sample data moved to a 1NF series table."""

    metric_key: str
    reason: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StructuralReplacement:
    """A new star key that replaces a removed legacy column; it is not a parity key."""

    metric_key: str
    reason: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ListViolation:
    """A legacy list column that is still not in 1NF and is knowingly left as a gap."""

    metric_key: str
    reason: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedManifest:
    """The reviewed expected-difference manifest with its routing sections."""

    entries: tuple[ExpectedDiff, ...]
    relocated_series: tuple[SeriesRelocation, ...]
    structural_replacements: tuple[StructuralReplacement, ...]
    list_violations: tuple[ListViolation, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedIndex:
    """Lookup of the reviewed entries, exact target first, metric-key-only second."""

    exact: Mapping[tuple[str, str, str], ExpectedDiff]
    by_key: Mapping[str, ExpectedDiff]

    def for_mapping(self, mapping: MetricMapping) -> ExpectedDiff | None:
        """Reviewed entry for a concrete mapping, or a metric-key-only entry."""
        found = self.exact.get((mapping.metric_key, mapping.star_table, mapping.star_column))
        return found if found is not None else self.by_key.get(mapping.metric_key)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkArtifacts:
    """Both outputs of one dual-path benchmark run."""

    benchmark_dir: pathlib.Path
    legacy: pl.DataFrame
    star: Mapping[str, pl.DataFrame]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CertificateRow:
    """One certificate line: one inventory key against one star column."""

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
class GapRow:
    """One inventory key that still has no scoped star-schema producer."""

    metric_key: str
    calculator: str
    category: str
    inventory_owner: str
    owner: str
    target_scope: str
    target_table: str
    legacy_dtype: str
    note: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UnmappedStarColumn:
    """A numeric star-schema metric column the reviewed map does not claim."""

    star_table: str
    star_column: str
    dtype: str
    note: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UnmappedKey:
    """An inventory key with no reviewed mapping whose star column does exist."""

    metric_key: str
    calculator: str
    category: str
    inventory_owner: str
    target_table: str
    star_table: str
    star_column: str
    legacy_dtype: str
    note: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Certificate:
    """The full G1 evidence bundle for one benchmark run."""

    rows: tuple[CertificateRow, ...]
    gaps: tuple[GapRow, ...]
    unmapped: tuple[UnmappedKey, ...]
    unmapped_star_columns: tuple[UnmappedStarColumn, ...]
    relocated_series: tuple[SeriesRelocation, ...]
    structural_replacements: tuple[StructuralReplacement, ...]
    list_violations: tuple[ListViolation, ...]
    run_ids: tuple[str, ...]
    inventory_keys: int
    comparison_rows: int
    mapped_keys: int
    expected_diff_count: int
    expected_diff_pending: int
    min_nonnull_episodes: int
    legacy_episodes: int
    star_episodes: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonStats:
    """Per-mapping comparison outcome before it is folded into a status."""

    episodes_compared: int
    null_compared: int
    nonnull_compared: int
    max_abs_diff: float | None
    max_rel_diff: float | None
    hard_mismatch_count: int
    expected_diff_count: int
    null_reason_count: int
    tolerance_count: int
    example: str


# ---------------------------------------------------------------------------
# Loading the reviewed artefacts
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


def load_inventory(path: pathlib.Path) -> tuple[InventoryRow, ...]:
    """Read the reviewed metric inventory (BOM-tolerant CSV of the 267 legacy keys)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[InventoryRow] = []
        for line, raw in enumerate(reader, start=2):
            fields = {key: (raw.get(key) or "") for key in ("metric_key", "calculator", "category", "owner", "target_scope", "target_table", "test_file", "status")}
            if not fields["metric_key"]:
                raise ParitySchemaError(f"{path.name}:{line}: empty metric_key")
            rows.append(
                InventoryRow(
                    metric_key=fields["metric_key"],
                    calculator=fields["calculator"],
                    category=fields["category"],
                    inventory_owner=fields["owner"],
                    target_scope=fields["target_scope"],
                    target_table=fields["target_table"],
                    test_file=fields["test_file"],
                    port_status=fields["status"],
                )
            )
    if not rows:
        raise ParitySchemaError(f"{path}: empty inventory")
    return tuple(rows)


def load_mappings(path: pathlib.Path) -> tuple[MetricMapping, ...]:
    """Read the reviewed metric_key -> star column correspondence map."""
    document = _as_mapping(_load_yaml(path), where=path.name)
    entries = _as_sequence(document.get("mappings"), where=f"{path.name}.mappings")
    mappings: list[MetricMapping] = []
    for index, raw in enumerate(entries):
        where = f"{path.name}.mappings[{index}]"
        entry = _as_mapping(raw, where=where)
        legacy_columns = tuple(_string_item(column, where=f"{where}.legacy_columns") for column in _as_sequence(entry.get("legacy_columns"), where=f"{where}.legacy_columns"))
        if not legacy_columns:
            raise ParitySchemaError(f"{where}: legacy_columns is empty")
        mappings.append(
            MetricMapping(
                metric_key=_text(entry, "metric_key", where=where),
                star_table=_text(entry, "star_table", where=where),
                star_column=_text(entry, "star_column", where=where),
                legacy_columns=legacy_columns,
                legacy_combine=LegacyCombine(_text(entry, "legacy_combine", where=where, default=LegacyCombine.FIRST.value)),
                legacy_reduction=LegacyReduction(_text(entry, "legacy_reduction", where=where, default=LegacyReduction.IDENTITY.value)),
                star_reduction=StarReduction(_text(entry, "star_reduction", where=where, default=StarReduction.IDENTITY.value)),
                note=_optional_text(entry, "note", where=where),
            )
        )
    _validate_mappings(tuple(mappings))
    return tuple(mappings)


def _group_metric_keys(group: Mapping[str, object], *, where: str) -> tuple[str, ...]:
    """Keys of one group: either a `metric_keys` list or a single `metric_key`."""
    raw_list = group.get("metric_keys")
    if raw_list is not None:
        return tuple(_string_item(key, where=f"{where}.metric_keys") for key in _as_sequence(raw_list, where=f"{where}.metric_keys"))
    raw_single = group.get("metric_key")
    if isinstance(raw_single, str) and raw_single:
        return (raw_single,)
    raise ParitySchemaError(f"{where}: expected metric_keys or metric_key")


def _load_grouped_keys(document: Mapping[str, object], *, section: str, where: str) -> tuple[tuple[str, str], ...]:
    """(metric_key, reason) pairs of a manifest section that groups keys under a reason."""
    groups = _as_sequence(document.get(section), where=f"{where}.{section}")
    pairs: list[tuple[str, str]] = []
    for index, raw in enumerate(groups):
        group_where = f"{where}.{section}[{index}]"
        group = _as_mapping(raw, where=group_where)
        reason = _text(group, "reason", where=group_where)
        for key in _group_metric_keys(group, where=group_where):
            pairs.append((key, reason))
    return tuple(pairs)


def _unique_pairs(pairs: tuple[tuple[str, str], ...], *, section: str) -> Mapping[str, str]:
    grouped: dict[str, str] = {}
    for metric_key, reason in pairs:
        if metric_key in grouped:
            raise ParitySchemaError(f"duplicate {section} key {metric_key}")
        grouped[metric_key] = reason
    return grouped


def load_expected_manifest(path: pathlib.Path) -> ExpectedManifest:
    """Read the reviewed allowlist plus its relocation, replacement and violation sections."""
    document = _as_mapping(_load_yaml(path), where=path.name)
    entries = _as_sequence(document.get("entries"), where=f"{path.name}.entries")
    expected: list[ExpectedDiff] = []
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
        expected.append(
            ExpectedDiff(
                metric_key=_text(entry, "metric_key", where=where),
                kind=kind,
                reason=_text(entry, "reason", where=where),
                star_table=_optional_text(entry, "star_table", where=where),
                star_column=_optional_text(entry, "star_column", where=where),
                factor=factor,
                max_rel=max_rel,
                allows_null=_optional_bool(entry, "allows_null", where=where),
            )
        )
    _validate_expected_diffs(tuple(expected))
    relocated = _unique_pairs(_load_grouped_keys(document, section="relocated_series", where=path.name), section="relocated_series")
    structural = _unique_pairs(_load_grouped_keys(document, section="structural_replacements", where=path.name), section="structural_replacements")
    violations = _unique_pairs(_load_grouped_keys(document, section="list_violations", where=path.name), section="list_violations")
    return ExpectedManifest(
        entries=tuple(expected),
        relocated_series=tuple(SeriesRelocation(metric_key=key, reason=reason) for key, reason in relocated.items()),
        structural_replacements=tuple(StructuralReplacement(metric_key=key, reason=reason) for key, reason in structural.items()),
        list_violations=tuple(ListViolation(metric_key=key, reason=reason) for key, reason in violations.items()),
    )


def _load_yaml(path: pathlib.Path) -> object:
    import yaml

    if not path.is_file():
        raise ParityInputError(f"missing reviewed manifest: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ParitySchemaError(f"{path}: invalid YAML: {error}") from error


def _validate_mappings(mappings: tuple[MetricMapping, ...]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for mapping in mappings:
        key = (mapping.metric_key, mapping.star_table, mapping.star_column)
        if key in seen:
            raise ParitySchemaError(f"duplicate mapping {key}")
        seen.add(key)
        if mapping.star_table not in STAR_TABLES:
            raise ParitySchemaError(f"mapping {key}: unknown star table {mapping.star_table}")
        if mapping.legacy_combine is LegacyCombine.SUM and len(mapping.legacy_columns) < 2:
            raise ParitySchemaError(f"mapping {key}: legacy_combine=sum needs at least two columns")
        if mapping.legacy_combine is LegacyCombine.FIRST and len(mapping.legacy_columns) != 1:
            raise ParitySchemaError(f"mapping {key}: legacy_combine=first needs exactly one column")
        if mapping.legacy_reduction is not LegacyReduction.IDENTITY and len(mapping.legacy_columns) != 1:
            raise ParitySchemaError(f"mapping {key}: a list reduction needs exactly one legacy column")


def _validate_expected_diffs(expected: tuple[ExpectedDiff, ...]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for entry in expected:
        key = (entry.metric_key, entry.star_table, entry.star_column)
        if key in seen:
            raise ParitySchemaError(f"duplicate expected diff {key}")
        seen.add(key)


# ---------------------------------------------------------------------------
# Reading a dual-path benchmark
# ---------------------------------------------------------------------------


def _fact_table_path(root: pathlib.Path, table: str) -> pathlib.Path | None:
    """A fact table lives under metrics/ in canonical runs, or at the benchmark root in OEC runs."""
    for candidate in (root / "metrics" / f"{table}.parquet", root / f"{table}.parquet"):
        if candidate.is_file():
            return candidate
    return None


def load_benchmark_artifacts(benchmark_dir: pathlib.Path) -> BenchmarkArtifacts:
    """Read the legacy frame and all star tables of one processed benchmark run."""
    root = benchmark_dir.resolve()
    missing = [name for name in ("combined_metrics.parquet", "dim_episodes.parquet") if not (root / name).is_file()]
    required_paths = {name: _fact_table_path(root, name) for name in _REQUIRED_FACT_TABLES}
    missing.extend(name for name, path in required_paths.items() if path is None)
    if missing:
        raise ParityInputError(f"{root} is not a dual-path benchmark: missing {', '.join(missing)}. Historic runs predate the star schema; reprocess the run with `arena evaluation process --benchmark-dir <dir> --workers 1` so both artifacts are written in one pass.")
    star: dict[str, pl.DataFrame] = {"dim_episodes": pl.read_parquet(root / "dim_episodes.parquet")}
    for name in _FACT_TABLES:
        path = _fact_table_path(root, name)
        if path is None and name in _REQUIRED_FACT_TABLES:
            raise ParityInputError(f"{root} is not a dual-path benchmark: missing {name}")
        if path is not None:
            star[name] = pl.read_parquet(path)
    return BenchmarkArtifacts(benchmark_dir=root, legacy=pl.read_parquet(root / "combined_metrics.parquet"), star=star)


# ---------------------------------------------------------------------------
# Joining the two grains
# ---------------------------------------------------------------------------


def _star_keys(star_table: str) -> tuple[str, ...]:
    if star_table in _ROBOT_KEYED_TABLES:
        return (_STAR_RUN_COLUMN, _STAR_EPISODE_COLUMN, _STAR_ROBOT_COLUMN)
    return (_STAR_RUN_COLUMN, _STAR_EPISODE_COLUMN)


def _legacy_key_expressions(star_table: str) -> tuple[pl.Expr, ...]:
    keys: list[pl.Expr] = [
        pl.col(_LEGACY_RUN_COLUMN).alias(_STAR_RUN_COLUMN),
        pl.col(_LEGACY_EPISODE_COLUMN).cast(pl.Int64).alias(_STAR_EPISODE_COLUMN),
    ]
    if star_table in _ROBOT_KEYED_TABLES:
        keys.append(pl.col(_LEGACY_ROBOT_COLUMN).alias(_STAR_ROBOT_COLUMN))
    return tuple(keys)


def _schema_names(frame: pl.DataFrame) -> frozenset[str]:
    return frozenset(frame.collect_schema().names())


def _false_series(height: int) -> pl.Series:
    """An all-False boolean column for a frame of ``height`` rows."""
    return pl.repeat(False, height, eager=True)


def _legacy_value_expression(mapping: MetricMapping) -> pl.Expr:
    expressions = [pl.col(column) for column in mapping.legacy_columns]
    combined = expressions[0]
    for expression in expressions[1:]:
        combined = combined + expression
    if mapping.legacy_reduction is LegacyReduction.IDENTITY:
        return combined.alias("legacy_value")
    reducers = {
        LegacyReduction.LIST_MEAN: combined.list.mean(),
        LegacyReduction.LIST_MIN: combined.list.min(),
        LegacyReduction.LIST_MAX: combined.list.max(),
        LegacyReduction.LIST_SUM: combined.list.sum(),
        LegacyReduction.LIST_LEN: combined.list.len().cast(pl.Float64),
    }
    return reducers[mapping.legacy_reduction].alias("legacy_value")


def _star_value_frame(artifacts: BenchmarkArtifacts, mapping: MetricMapping) -> pl.DataFrame | None:
    frame = artifacts.star.get(mapping.star_table)
    if frame is None:
        return None
    names = _schema_names(frame)
    if mapping.star_column not in names:
        return None
    keys = _star_keys(mapping.star_table)
    missing_keys = [key for key in keys if key not in names]
    if missing_keys:
        raise ParitySchemaError(f"{mapping.star_table}: missing key columns {missing_keys}")
    if mapping.star_reduction is StarReduction.IDENTITY:
        duplicates = frame.group_by(list(keys)).len().filter(pl.col("len") > 1)
        if not duplicates.is_empty():
            raise ParitySchemaError(f"{mapping.star_table}.{mapping.star_column}: {duplicates.height} duplicated key rows; declare a star_reduction instead of identity")
        return frame.select([*keys, pl.col(mapping.star_column).alias("star_value")])
    reducers = {
        StarReduction.MEAN: pl.col(mapping.star_column).mean(),
        StarReduction.MIN: pl.col(mapping.star_column).min(),
        StarReduction.MAX: pl.col(mapping.star_column).max(),
        StarReduction.SUM: pl.col(mapping.star_column).sum(),
        StarReduction.COUNT: pl.len().cast(pl.Float64),
    }
    return frame.group_by(list(keys)).agg(reducers[mapping.star_reduction].alias("star_value"))


def _legacy_value_frame(artifacts: BenchmarkArtifacts, mapping: MetricMapping) -> pl.DataFrame | None:
    names = _schema_names(artifacts.legacy)
    required_keys = [_LEGACY_RUN_COLUMN, _LEGACY_EPISODE_COLUMN]
    if mapping.star_table in _ROBOT_KEYED_TABLES:
        required_keys.append(_LEGACY_ROBOT_COLUMN)
    missing_keys = [column for column in required_keys if column not in names]
    if missing_keys:
        raise ParitySchemaError(f"combined_metrics: missing join keys {missing_keys}")
    if any(column not in names for column in mapping.legacy_columns):
        return None
    frame = artifacts.legacy.select([*_legacy_key_expressions(mapping.star_table), _legacy_value_expression(mapping)])
    if mapping.star_table in _ROBOT_KEYED_TABLES:
        return frame
    # A legacy episode has one row per robot, but an episode-grain fact has one row per episode.
    # Evaluated robots carry the value and the others are null, so reduce the duplicates the way
    # the star side reduces its rows: the numeric extreme, else the first non-null.
    keys = _star_keys(mapping.star_table)
    dtype = frame.collect_schema()["legacy_value"]
    aggregator = pl.col("legacy_value").max() if dtype.is_numeric() else pl.col("legacy_value").drop_nulls().first()
    return frame.group_by(list(keys)).agg(aggregator.alias("legacy_value"))


def _null_reason_frame(artifacts: BenchmarkArtifacts, star_table: str) -> pl.DataFrame:
    """One provenance row per joined key, so a reduced star frame is not expanded by the join."""
    frame = artifacts.star[star_table]
    names = _schema_names(frame)
    keys = _star_keys(star_table)
    if "null_reason" not in names:
        return frame.select([*keys]).with_columns(pl.lit(None, dtype=pl.String).alias("star_null_reason"))
    return frame.select([*keys, pl.col("null_reason").alias("star_null_reason")]).group_by(list(keys)).agg(pl.col("star_null_reason").drop_nulls().first())


def _join_comparison_frame(*, legacy: pl.DataFrame, star: pl.DataFrame, keys: tuple[str, ...]) -> pl.DataFrame:
    return legacy.join(star, on=list(keys), how="inner")


# ---------------------------------------------------------------------------
# One comparison
# ---------------------------------------------------------------------------


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _example_text(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return ""
    shown = frame.head(_EXAMPLE_LIMIT)
    names = shown.collect_schema().names()
    robot = pl.col(_STAR_ROBOT_COLUMN) if _STAR_ROBOT_COLUMN in names else pl.lit("-")
    rendered = shown.select(
        pl.format(
            "{} ep={} robot={} legacy={} star={}",
            pl.col("legacy_column"),
            pl.col(_STAR_EPISODE_COLUMN),
            robot,
            pl.col("legacy_value"),
            pl.col("star_value"),
        )
    ).to_series()
    return "; ".join(str(value) for value in rendered.to_list())


def _numeric_stats(joined: pl.DataFrame, *, expected: ExpectedDiff | None, rel_tol: float) -> ComparisonStats:
    factor = expected.factor if expected is not None and expected.factor is not None else 1.0
    legacy_value = pl.col("legacy_value").cast(pl.Float64)
    star_value = pl.col("star_value").cast(pl.Float64)
    target = legacy_value * pl.lit(factor)
    both_present = legacy_value.is_not_null() & star_value.is_not_null()
    nan_both = legacy_value.is_nan() & star_value.is_nan()
    exact_equal = nan_both | (both_present & (star_value == target))
    scale = pl.max_horizontal(target.abs(), star_value.abs())
    abs_diff = (star_value - target).abs()
    rel_diff = pl.when(scale > pl.lit(0.0)).then(abs_diff / scale).otherwise(pl.lit(0.0))
    tolerance_equal = both_present & (rel_diff <= pl.lit(rel_tol))
    false_literal = pl.lit(False, dtype=pl.Boolean)

    unexpected_value = both_present & ~tolerance_equal
    expected_observed = false_literal
    if expected is not None and expected.kind is DiffKind.UNIT_SCALE:
        bound = expected.max_rel if expected.max_rel is not None else _DEFAULT_UNIT_SCALE_TOL
        allowed_value = both_present & (rel_diff <= pl.lit(bound))
        unexpected_value = unexpected_value & ~allowed_value
        expected_observed = both_present & ~exact_equal & allowed_value
    elif expected is not None and expected.kind is DiffKind.DERIVED:
        bound = expected.max_rel if expected.max_rel is not None else 0.0
        allowed_value = both_present & (rel_diff <= pl.lit(bound))
        unexpected_value = unexpected_value & ~allowed_value
        expected_observed = both_present & ~tolerance_equal & allowed_value
    elif expected is not None and expected.kind is DiffKind.GROUND_TRUTH:
        expected_observed = both_present & ~tolerance_equal
        unexpected_value = unexpected_value & ~expected_observed

    legacy_zero_star_null = (legacy_value == pl.lit(0.0)) & star_value.is_null()
    unexpected_null = star_value.is_null() & legacy_value.is_not_null()
    unexpected_legacy_null = legacy_value.is_null() & star_value.is_not_null()
    null_reason_observed = false_literal
    if "star_null_reason" in joined.collect_schema().names():
        reason_present = pl.col("star_null_reason").is_not_null() & (pl.col("star_null_reason") != pl.lit(""))
        null_reason_observed = legacy_zero_star_null & reason_present
        unexpected_null = unexpected_null & ~null_reason_observed
    if expected is not None and (expected.allows_null or expected.kind is DiffKind.NULL_FILL):
        unexpected_null = false_literal

    enriched = joined.with_columns(
        (exact_equal | (legacy_value.is_null() & star_value.is_null())).alias("_exact"),
        tolerance_equal.alias("_tolerance"),
        expected_observed.alias("_expected"),
        null_reason_observed.alias("_null_reason"),
        unexpected_value.alias("_unexpected_value"),
        unexpected_null.alias("_unexpected_null"),
        unexpected_legacy_null.alias("_unexpected_legacy_null"),
        abs_diff.alias("_abs_diff"),
        rel_diff.alias("_rel_diff"),
    )
    mismatch = enriched.filter(pl.col("_unexpected_value") | pl.col("_unexpected_null") | pl.col("_unexpected_legacy_null"))
    non_null = enriched.filter(both_present)
    both_null = enriched.filter(legacy_value.is_null() & star_value.is_null())
    max_abs = None if non_null.is_empty() else float(non_null["_abs_diff"].max())
    max_rel = None if non_null.is_empty() else float(non_null["_rel_diff"].max())
    return ComparisonStats(
        episodes_compared=enriched.height,
        null_compared=both_null.height,
        nonnull_compared=non_null.height,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        hard_mismatch_count=mismatch.height,
        expected_diff_count=enriched.filter(pl.col("_expected")).height,
        null_reason_count=enriched.filter(pl.col("_null_reason")).height,
        tolerance_count=enriched.filter(pl.col("_tolerance") & ~pl.col("_exact")).height,
        example=_example_text(mismatch),
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
            hard_mismatch_count=joined.height,
            expected_diff_count=0,
            null_reason_count=0,
            tolerance_count=0,
            example=f"dtype change {legacy_dtype} -> {star_dtype} has no reviewed dtype entry",
        )
    both_null = joined["legacy_value"].is_null() & joined["star_value"].is_null()
    equal = both_null | (joined["legacy_value"].cast(pl.String) == joined["star_value"].cast(pl.String))
    unexpected_null = joined["star_value"].is_null() & joined["legacy_value"].is_not_null()
    unexpected_legacy_null = joined["legacy_value"].is_null() & joined["star_value"].is_not_null()
    if expected is not None and (expected.allows_null or expected.kind is DiffKind.NULL_FILL):
        unexpected_null = _false_series(joined.height)
    difference_allowed = expected is not None and expected.kind is DiffKind.GROUND_TRUTH
    expected_observed = (~equal) if difference_allowed else _false_series(joined.height)
    mismatch = ~(equal | expected_observed) & ~unexpected_null & ~unexpected_legacy_null
    enriched = joined.with_columns(
        equal.alias("_equal"),
        expected_observed.alias("_expected"),
        unexpected_null.alias("_unexpected_null"),
        unexpected_legacy_null.alias("_unexpected_legacy_null"),
        mismatch.alias("_mismatch"),
    )
    mismatch_frame = enriched.filter(pl.col("_mismatch"))
    both_null_frame = enriched.filter(both_null)
    return ComparisonStats(
        episodes_compared=enriched.height,
        null_compared=both_null_frame.height,
        nonnull_compared=enriched.height - both_null_frame.height - enriched.filter(unexpected_null | unexpected_legacy_null).height,
        max_abs_diff=None,
        max_rel_diff=None,
        hard_mismatch_count=mismatch_frame.height,
        expected_diff_count=enriched.filter(pl.col("_expected")).height,
        null_reason_count=0,
        tolerance_count=0,
        example=_example_text(mismatch_frame),
    )


def _null_only_stats(joined: pl.DataFrame) -> ComparisonStats:
    """Compare presence only: one side has a Null dtype, so every one of its values is null."""
    legacy_is_null = joined["legacy_value"].is_null()
    star_is_null = joined["star_value"].is_null()
    mismatch = joined.filter(legacy_is_null != star_is_null)
    both_null = joined.filter(legacy_is_null & star_is_null)
    return ComparisonStats(
        episodes_compared=joined.height,
        null_compared=both_null.height,
        nonnull_compared=0,
        max_abs_diff=None,
        max_rel_diff=None,
        hard_mismatch_count=mismatch.height,
        expected_diff_count=0,
        null_reason_count=0,
        tolerance_count=0,
        example=_example_text(mismatch),
    )


def _comparable_stats(joined: pl.DataFrame, *, expected: ExpectedDiff | None, rel_tol: float) -> ComparisonStats:
    legacy_dtype = joined.collect_schema()["legacy_value"]
    star_dtype = joined.collect_schema()["star_value"]
    if legacy_dtype == pl.Null or star_dtype == pl.Null:
        return _null_only_stats(joined)
    if _is_numeric(legacy_dtype) and _is_numeric(star_dtype):
        return _numeric_stats(joined, expected=expected, rel_tol=rel_tol)
    return _non_numeric_stats(joined, expected=expected)


def _status_for(stats: ComparisonStats, *, min_nonnull_episodes: int) -> ParityStatus:
    """Verdict of one comparison; nulls alone never earn a green status."""
    if stats.hard_mismatch_count > 0:
        return ParityStatus.MISMATCH
    if stats.episodes_compared > 0 and stats.null_compared == stats.episodes_compared:
        return ParityStatus.CERT_NULL_ONLY
    if stats.expected_diff_count > 0:
        if stats.nonnull_compared < min_nonnull_episodes and stats.nonnull_compared > 0:
            return ParityStatus.INSUFFICIENT_EVIDENCE
        return ParityStatus.EXPECTED_DIFF
    if stats.null_reason_count > 0:
        return ParityStatus.NULL_REASON
    if stats.nonnull_compared < min_nonnull_episodes:
        return ParityStatus.INSUFFICIENT_EVIDENCE
    if stats.tolerance_count > 0:
        return ParityStatus.WITHIN_TOL
    return ParityStatus.EXACT


def _compare_mapping(
    *,
    artifacts: BenchmarkArtifacts,
    mapping: MetricMapping,
    expected: ExpectedDiff | None,
    rel_tol: float,
    min_nonnull_episodes: int,
) -> CertificateRow:
    legacy_columns = ", ".join(mapping.legacy_columns)
    legacy = _legacy_value_frame(artifacts, mapping)
    if legacy is None:
        return CertificateRow(
            metric_key=mapping.metric_key,
            legacy_column=legacy_columns,
            star_table=mapping.star_table,
            star_column=mapping.star_column,
            status=ParityStatus.MISSING_LEGACY,
            episodes_compared=0,
            null_compared=0,
            nonnull_compared=0,
            max_abs_diff=None,
            max_rel_diff=None,
            detail=f"legacy column absent from combined_metrics: {legacy_columns}",
        )
    star = _star_value_frame(artifacts, mapping)
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
            detail=f"mapped star column absent from {mapping.star_table}",
        )
    keys = _star_keys(mapping.star_table)
    star = star.join(_null_reason_frame(artifacts, mapping.star_table), on=list(keys), how="left")
    joined = _join_comparison_frame(legacy=legacy, star=star, keys=keys)
    if joined.is_empty():
        return CertificateRow(
            metric_key=mapping.metric_key,
            legacy_column=legacy_columns,
            star_table=mapping.star_table,
            star_column=mapping.star_column,
            status=ParityStatus.MISMATCH,
            episodes_compared=0,
            null_compared=0,
            nonnull_compared=0,
            max_abs_diff=None,
            max_rel_diff=None,
            detail="join on (run_id, episode_id[, robot_id]) produced no episodes",
        )
    joined = joined.with_columns(pl.lit(legacy_columns).alias("legacy_column"))
    stats = _comparable_stats(joined, expected=expected, rel_tol=rel_tol)
    status = _status_for(stats, min_nonnull_episodes=min_nonnull_episodes)
    if status is ParityStatus.MISMATCH:
        detail = f"{stats.hard_mismatch_count} unexplained episode(s)"
        if stats.example:
            detail = f"{detail}: {stats.example}"
    elif status is ParityStatus.CERT_NULL_ONLY:
        detail = f"both sides null for all {stats.episodes_compared} compared episode(s): no real comparison"
    elif status is ParityStatus.INSUFFICIENT_EVIDENCE:
        detail = f"{stats.nonnull_compared} non-null comparison(s), need at least {min_nonnull_episodes}; {stats.null_compared} episode(s) were null on both sides"
    elif status is ParityStatus.EXPECTED_DIFF and expected is not None:
        detail = f"{stats.expected_diff_count} reviewed episode(s): {expected.reason}"
    elif status is ParityStatus.NULL_REASON:
        detail = f"{stats.null_reason_count} episode(s): legacy 0.0 replaced by null with a reason"
    elif expected is not None:
        detail = f"reviewed entry not exercised ({expected.kind.value})"
    else:
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


def _expected_index(expected: tuple[ExpectedDiff, ...]) -> ExpectedIndex:
    exact: dict[tuple[str, str, str], ExpectedDiff] = {}
    by_key: dict[str, ExpectedDiff] = {}
    for entry in expected:
        if entry.star_table and entry.star_column:
            exact[(entry.metric_key, entry.star_table, entry.star_column)] = entry
        elif not entry.star_table and not entry.star_column:
            by_key[entry.metric_key] = entry
        else:
            raise ParitySchemaError(f"expected diff {entry.metric_key}: star_table and star_column must both be set or both empty")
    return ExpectedIndex(exact=exact, by_key=by_key)


def _legacy_dtype(artifacts: BenchmarkArtifacts, metric_key: str) -> pl.DataType | None:
    if metric_key not in _schema_names(artifacts.legacy):
        return None
    return artifacts.legacy.collect_schema()[metric_key]


def _triaged_owner(row: InventoryRow, *, dtype: pl.DataType | None) -> str:
    if dtype is not None and dtype.is_nested():
        return _G1_OWNER
    return _TEAM_OWNER_BY_CATEGORY.get(row.category, _G1_OWNER)


def _gap_rows(
    *,
    artifacts: BenchmarkArtifacts,
    inventory: tuple[InventoryRow, ...],
    no_producer_keys: frozenset[str],
    list_violations: Mapping[str, str],
) -> tuple[GapRow, ...]:
    gaps: list[GapRow] = []
    for row in inventory:
        if row.metric_key not in no_producer_keys:
            continue
        dtype = _legacy_dtype(artifacts, row.metric_key)
        note = "no reviewed star mapping and no produced star column: a real porting gap"
        if row.metric_key in list_violations:
            note = f"known remaining 1NF list violation: {list_violations[row.metric_key]}"
        gaps.append(
            GapRow(
                metric_key=row.metric_key,
                calculator=row.calculator,
                category=row.category,
                inventory_owner=row.inventory_owner,
                owner=_triaged_owner(row, dtype=dtype),
                target_scope=row.target_scope,
                target_table=row.target_table,
                legacy_dtype=str(dtype) if dtype is not None else "absent_in_run",
                note=note,
            )
        )
    return tuple(gaps)


def _star_column_index(artifacts: BenchmarkArtifacts) -> Mapping[str, tuple[tuple[str, str], ...]]:
    """Produced column name -> (star table, dtype) pairs across the dimension and all fact tables."""
    index: dict[str, list[tuple[str, str]]] = {}
    for table in ("dim_episodes", *_FACT_TABLES):
        if table not in artifacts.star:
            continue
        for column, dtype in artifacts.star[table].collect_schema().items():
            index.setdefault(column, []).append((table, str(dtype)))
    return {name: tuple(locations) for name, locations in index.items()}


def _locate_star_column(*, index: Mapping[str, tuple[tuple[str, str], ...]], row: InventoryRow) -> tuple[str, str] | None:
    """(star table, column) of a produced column matching the inventory key, preferring its intended table."""
    locations = index.get(row.metric_key)
    if not locations:
        return None
    for table, _ in locations:
        if table == row.target_table:
            return table, row.metric_key
    table, _ = locations[0]
    return table, row.metric_key


def _unmapped_star_columns(*, artifacts: BenchmarkArtifacts, mappings: tuple[MetricMapping, ...]) -> tuple[UnmappedStarColumn, ...]:
    mapped: set[tuple[str, str]] = {(mapping.star_table, mapping.star_column) for mapping in mappings}
    rows: list[UnmappedStarColumn] = []
    for table in _FACT_TABLES:
        if table not in artifacts.star:
            continue
        schema = artifacts.star[table].collect_schema()
        for column, dtype in schema.items():
            if column in _KEY_COLUMNS or column.endswith(("_ns", "_id")) or (table, column) in mapped or not _is_numeric(dtype):
                continue
            rows.append(UnmappedStarColumn(star_table=table, star_column=column, dtype=str(dtype), note="numeric star column not claimed by any reviewed inventory mapping"))
    return tuple(rows)


def _validate_review_sections(
    *,
    inventory_keys: frozenset[str],
    mapped_keys: frozenset[str],
    relocated_series: tuple[SeriesRelocation, ...],
    list_violations: tuple[ListViolation, ...],
    structural_replacements: tuple[StructuralReplacement, ...],
) -> None:
    """The reviewed routing sections must agree with the inventory and the map."""
    relocated_keys = [entry.metric_key for entry in relocated_series]
    unknown_relocated = sorted(set(relocated_keys) - inventory_keys)
    if unknown_relocated:
        raise ParitySchemaError(f"relocated_series references keys absent from the inventory: {unknown_relocated}")
    overlap = sorted(set(relocated_keys) & mapped_keys)
    if overlap:
        raise ParitySchemaError(f"keys cannot be both mapped and relocated to a series table: {overlap}")
    unknown_violations = sorted(entry.metric_key for entry in list_violations if entry.metric_key not in inventory_keys)
    if unknown_violations:
        raise ParitySchemaError(f"list_violations references keys absent from the inventory: {unknown_violations}")
    inventory_structural = sorted(entry.metric_key for entry in structural_replacements if entry.metric_key in inventory_keys)
    if inventory_structural:
        raise ParitySchemaError(f"structural_replacements must not be inventory keys: {inventory_structural}")


def certify(
    *,
    artifacts: BenchmarkArtifacts,
    inventory: tuple[InventoryRow, ...],
    mappings: tuple[MetricMapping, ...],
    expected_diffs: tuple[ExpectedDiff, ...] = (),
    relocated_series: tuple[SeriesRelocation, ...] = (),
    structural_replacements: tuple[StructuralReplacement, ...] = (),
    list_violations: tuple[ListViolation, ...] = (),
    rel_tol: float = REL_TOL,
    min_nonnull_episodes: int = 3,
) -> Certificate:
    """Compare every inventory key and every mapping, returning the certificate bundle."""
    inventory_keys = frozenset(row.metric_key for row in inventory)
    unknown = sorted({mapping.metric_key for mapping in mappings if mapping.metric_key not in inventory_keys})
    if unknown:
        raise ParitySchemaError(f"mappings reference keys absent from the inventory: {unknown}")
    mapped_keys = frozenset(mapping.metric_key for mapping in mappings)
    _validate_review_sections(
        inventory_keys=inventory_keys,
        mapped_keys=mapped_keys,
        relocated_series=relocated_series,
        list_violations=list_violations,
        structural_replacements=structural_replacements,
    )
    expected = _expected_index(expected_diffs)
    violation_reasons = {entry.metric_key: entry.reason for entry in list_violations}
    column_index = _star_column_index(artifacts)
    rows: list[CertificateRow] = []
    for mapping in mappings:
        rows.append(
            _compare_mapping(
                artifacts=artifacts,
                mapping=mapping,
                expected=expected.for_mapping(mapping),
                rel_tol=rel_tol,
                min_nonnull_episodes=min_nonnull_episodes,
            )
        )
    relocated_by_key = {entry.metric_key: entry for entry in relocated_series}
    unmapped: list[UnmappedKey] = []
    no_producer: list[str] = []
    for row in inventory:
        if row.metric_key in mapped_keys:
            continue
        dtype = _legacy_dtype(artifacts, row.metric_key)
        legacy_dtype = str(dtype) if dtype is not None else "absent_in_run"
        if row.metric_key in relocated_by_key:
            relocation = relocated_by_key[row.metric_key]
            rows.append(
                CertificateRow(
                    metric_key=row.metric_key,
                    legacy_column=row.metric_key if dtype is not None else "",
                    star_table=row.target_table,
                    star_column="",
                    status=ParityStatus.RELOCATED_SERIES,
                    episodes_compared=0,
                    null_compared=0,
                    nonnull_compared=0,
                    max_abs_diff=None,
                    max_rel_diff=None,
                    detail=f"per-sample data moved to a 1NF series table: {relocation.reason}",
                )
            )
            continue
        located = _locate_star_column(index=column_index, row=row)
        if located is not None:
            star_table, star_column = located
            note = "star column exists but has no reviewed mapping: manifest review, not a port"
            unmapped.append(
                UnmappedKey(
                    metric_key=row.metric_key,
                    calculator=row.calculator,
                    category=row.category,
                    inventory_owner=row.inventory_owner,
                    target_table=row.target_table,
                    star_table=star_table,
                    star_column=star_column,
                    legacy_dtype=legacy_dtype,
                    note=note,
                )
            )
            rows.append(
                CertificateRow(
                    metric_key=row.metric_key,
                    legacy_column=row.metric_key if dtype is not None else "",
                    star_table=star_table,
                    star_column=star_column,
                    status=ParityStatus.UNMAPPED,
                    episodes_compared=0,
                    null_compared=0,
                    nonnull_compared=0,
                    max_abs_diff=None,
                    max_rel_diff=None,
                    detail=note,
                )
            )
            continue
        no_producer.append(row.metric_key)
        detail = "no reviewed star mapping and no produced star column: a real porting gap"
        if row.metric_key in violation_reasons:
            detail = f"known remaining 1NF list violation: {violation_reasons[row.metric_key]}"
        rows.append(
            CertificateRow(
                metric_key=row.metric_key,
                legacy_column="",
                star_table=row.target_table,
                star_column="",
                status=ParityStatus.NO_PRODUCER,
                episodes_compared=0,
                null_compared=0,
                nonnull_compared=0,
                max_abs_diff=None,
                max_rel_diff=None,
                detail=detail,
            )
        )
    referenced = {entry.metric_key for entry in expected_diffs if entry.metric_key in mapped_keys}
    pending = sum(1 for entry in expected_diffs if entry.metric_key not in referenced)
    dim = artifacts.star["dim_episodes"]
    dim_names = _schema_names(dim)
    run_ids = tuple(sorted(str(value) for value in dim["run_id"].unique().to_list())) if "run_id" in dim_names else ()
    return Certificate(
        rows=tuple(sorted(rows, key=lambda row: (row.metric_key, row.star_table, row.star_column))),
        gaps=_gap_rows(
            artifacts=artifacts,
            inventory=inventory,
            no_producer_keys=frozenset(no_producer),
            list_violations=violation_reasons,
        ),
        unmapped=tuple(sorted(unmapped, key=lambda entry: entry.metric_key)),
        unmapped_star_columns=_unmapped_star_columns(artifacts=artifacts, mappings=mappings),
        relocated_series=relocated_series,
        structural_replacements=structural_replacements,
        list_violations=list_violations,
        run_ids=run_ids,
        inventory_keys=len(inventory),
        comparison_rows=len(mappings),
        mapped_keys=len(mapped_keys),
        expected_diff_count=len(expected_diffs),
        expected_diff_pending=pending,
        min_nonnull_episodes=min_nonnull_episodes,
        legacy_episodes=artifacts.legacy.height,
        star_episodes=dim.height,
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def status_counts(certificate: Certificate) -> Mapping[ParityStatus, int]:
    """Count certificate rows per status, including zero counts for stable output."""
    counts: dict[ParityStatus, int] = {status: 0 for status in ParityStatus}
    for row in certificate.rows:
        counts[row.status] += 1
    return counts


def write_certificate_csv(path: pathlib.Path, certificate: Certificate) -> None:
    """Write the flat certificate table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric_key", "legacy_column", "star_table", "star_column", "status", "episodes_compared", "null_compared", "nonnull_compared", "max_abs_diff", "max_rel_diff", "detail"])
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


def write_gaps_csv(path: pathlib.Path, certificate: Certificate) -> None:
    """Write the inventory keys that still have no scoped star producer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric_key", "calculator", "category", "inventory_owner", "owner", "target_scope", "target_table", "legacy_dtype", "note"])
        for row in certificate.gaps:
            writer.writerow([row.metric_key, row.calculator, row.category, row.inventory_owner, row.owner, row.target_scope, row.target_table, row.legacy_dtype, row.note])


def write_unmapped_csv(path: pathlib.Path, certificate: Certificate) -> None:
    """Write keys whose star column exists but whose legacy correspondence is unreviewed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric_key", "calculator", "category", "inventory_owner", "target_table", "star_table", "star_column", "legacy_dtype", "note"])
        for row in certificate.unmapped:
            writer.writerow([row.metric_key, row.calculator, row.category, row.inventory_owner, row.target_table, row.star_table, row.star_column, row.legacy_dtype, row.note])


def write_star_coverage_csv(path: pathlib.Path, certificate: Certificate) -> None:
    """Write numeric star columns the reviewed map does not claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["star_table", "star_column", "dtype", "note"])
        for row in certificate.unmapped_star_columns:
            writer.writerow([row.star_table, row.star_column, row.dtype, row.note])


def write_summary_yaml(path: pathlib.Path, certificate: Certificate, *, benchmark_dir: pathlib.Path) -> None:
    """Write the machine-readable certificate summary."""
    import yaml

    counts = status_counts(certificate)
    owner_counts: dict[str, int] = {}
    calculator_counts: dict[str, int] = {}
    for gap in certificate.gaps:
        owner_counts[gap.owner] = owner_counts.get(gap.owner, 0) + 1
        calculator_counts[gap.calculator] = calculator_counts.get(gap.calculator, 0) + 1
    document = {
        "benchmark_dir": str(benchmark_dir),
        "run_ids": list(certificate.run_ids),
        "inventory_keys": certificate.inventory_keys,
        "compared_metric_mappings": certificate.comparison_rows,
        "mapped_keys": certificate.mapped_keys,
        "expected_diff_entries": certificate.expected_diff_count,
        "expected_diff_pending": certificate.expected_diff_pending,
        "relocated_series_keys": len(certificate.relocated_series),
        "structural_replacements": len(certificate.structural_replacements),
        "list_violation_keys": len(certificate.list_violations),
        "min_nonnull_episodes": certificate.min_nonnull_episodes,
        "legacy_episode_rows": certificate.legacy_episodes,
        "star_episode_rows": certificate.star_episodes,
        "status_counts": {status.value: count for status, count in counts.items()},
        "unmapped_count": len(certificate.unmapped),
        "no_producer_count": len(certificate.gaps),
        "gap_owner_counts": dict(sorted(owner_counts.items())),
        "gap_calculator_counts": dict(sorted(calculator_counts.items(), key=lambda item: (-item[1], item[0]))),
        "unmapped_star_columns": len(certificate.unmapped_star_columns),
        "accepted_statuses": sorted(status.value for status in _ACCEPTED_STATUSES),
        "failing_statuses": sorted(status.value for status in _FAILING_STATUSES),
        "hard_failure_count": sum(counts[status] for status in _FAILING_STATUSES),
        "gate_passed": all(counts[status] == 0 for status in _FAILING_STATUSES),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def write_review_csvs(path: pathlib.Path, certificate: Certificate) -> None:
    """Write the relocation, structural-replacement and remaining-violation lists."""
    path.mkdir(parents=True, exist_ok=True)
    with (path / "relocated_series.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric_key", "reason"])
        for entry in certificate.relocated_series:
            writer.writerow([entry.metric_key, entry.reason])
    with (path / "structural_replacements.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric_key", "reason"])
        for entry in certificate.structural_replacements:
            writer.writerow([entry.metric_key, entry.reason])
    with (path / "list_violations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric_key", "reason"])
        for entry in certificate.list_violations:
            writer.writerow([entry.metric_key, entry.reason])


def write_gap_summary_markdown(path: pathlib.Path, certificate: Certificate, *, benchmark_dir: pathlib.Path) -> None:
    """Write the human-readable grouped gap list (by calculator and by owner)."""
    by_calculator: dict[str, int] = {}
    by_owner: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for gap in certificate.gaps:
        by_calculator[gap.calculator] = by_calculator.get(gap.calculator, 0) + 1
        by_owner[gap.owner] = by_owner.get(gap.owner, 0) + 1
        by_category[gap.category] = by_category.get(gap.category, 0) + 1
    lines = [
        "# G1 parity gap list",
        "",
        f"- benchmark: `{benchmark_dir}`",
        f"- run ids: {', '.join(f'`{run_id}`' for run_id in certificate.run_ids) or '(none)'}",
        f"- inventory keys: {certificate.inventory_keys}",
        f"- compared metric mappings: {certificate.comparison_rows}",
        f"- keys with a reviewed star mapping: {certificate.mapped_keys}",
        f"- keys without a reviewed mapping but with a produced star column (UNMAPPED, manifest review): {len(certificate.unmapped)}",
        f"- keys without a produced star column (NO_PRODUCER, real porting gaps): {len(certificate.gaps)}",
        f"- relocated series keys (not gaps): {len(certificate.relocated_series)}",
        f"- known remaining 1NF list violations: {len(certificate.list_violations)}",
        f"- structural replacement keys (not parity keys): {len(certificate.structural_replacements)}",
        f"- numeric star columns not claimed by any reviewed inventory mapping: {len(certificate.unmapped_star_columns)}",
        f"- minimum non-null comparisons per key: {certificate.min_nonnull_episodes}",
        "",
        "## UNMAPPED (star column exists, manifest review needed)",
        "",
        "| calculator | keys |",
        "| --- | ---: |",
    ]
    unmapped_by_calculator: dict[str, int] = {}
    for entry in certificate.unmapped:
        unmapped_by_calculator[entry.calculator] = unmapped_by_calculator.get(entry.calculator, 0) + 1
    lines.extend(f"| {calculator} | {count} |" for calculator, count in sorted(unmapped_by_calculator.items(), key=lambda item: (-item[1], item[0])))
    lines.extend(
        [
            "",
            "## NO_PRODUCER port gaps by owner",
            "",
            "| owner | keys |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {owner} | {count} |" for owner, count in sorted(by_owner.items()))
    lines.extend(["", "## Port gaps by calculator", "", "| calculator | keys |", "| --- | ---: |"])
    lines.extend(f"| {calculator} | {count} |" for calculator, count in sorted(by_calculator.items(), key=lambda item: (-item[1], item[0])))
    lines.extend(["", "## Port gaps by category", "", "| category | keys |", "| --- | ---: |"])
    lines.extend(f"| {category} | {count} |" for category, count in sorted(by_category.items(), key=lambda item: (-item[1], item[0])))
    if certificate.list_violations:
        lines.extend(["", "## Known remaining 1NF list violations", "", "| metric_key | reason |", "| --- | --- |"])
        lines.extend(f"| {entry.metric_key} | {entry.reason} |" for entry in certificate.list_violations)
    if certificate.relocated_series:
        lines.extend(["", "## Relocated series (no longer scalar columns)", "", "| metric_key | reason |", "| --- | --- |"])
        lines.extend(f"| {entry.metric_key} | {entry.reason} |" for entry in certificate.relocated_series)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(path: pathlib.Path, certificate: Certificate, *, benchmark_dir: pathlib.Path) -> None:
    """Write every certificate artefact below ``path``."""
    write_certificate_csv(path / "certificate.csv", certificate)
    write_gaps_csv(path / "gaps.csv", certificate)
    write_unmapped_csv(path / "unmapped.csv", certificate)
    write_star_coverage_csv(path / "star_coverage.csv", certificate)
    write_review_csvs(path, certificate)
    write_summary_yaml(path / "summary.yaml", certificate, benchmark_dir=benchmark_dir)
    write_gap_summary_markdown(path / "gap_summary.md", certificate, benchmark_dir=benchmark_dir)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _synthetic_artifacts() -> BenchmarkArtifacts:
    fact = pl.DataFrame(
        {
            "run_id": ["run"] * 4,
            "episode_id": [0, 1, 2, 3],
            "robot_id": ["env_0_jackal"] * 4,
            "value": [1.0, 2.0, None, 4.0],
            "mean": [2.0, 2.0, None, 5.0],
            "static_total": [0, 1, None, 0],
            "zero_star": [1.0, 2.0, None, 4.0],
            "scaled": [2.0, 4.0, None, 8.0],
            "derived_star": [1.02, 2.04, None, 4.08],
            "sparse": [1.0, 2.0, None, None],
            "nullcol": pl.Series([None, None, None, None], dtype=pl.Float64),
            "null_reason": [None, None, "value:no_samples", None],
        }
    )
    legacy = pl.DataFrame(
        {
            "benchmark_id": ["run"] * 4,
            "episode": [0, 1, 2, 3],
            "robot": ["env_0_jackal"] * 4,
            "value": [1.0, 2.0, None, 4.0],
            "listy": [[1.0, 3.0], [2.0, 2.0], None, [5.0, 5.0]],
            "wall": [0, 0, 0, 0],
            "static": [0, 1, None, 0],
            "zero": [1.0, 2.0, 0.0, 4.0],
            "scaled": [1.0, 2.0, None, 4.0],
            "derived": [1.0, 2.0, None, 4.0],
            "sparse": [1.0, 2.0, None, None],
            "nullcol": [None, None, None, None],
        }
    )
    star = {
        "dim_episodes": fact.select("run_id", "episode_id"),
        "fact_episodes": fact,
        "fact_encounters": fact,
        "fact_contacts": fact,
        "fact_pedestrians": fact,
        "fact_robots": fact,
    }
    return BenchmarkArtifacts(benchmark_dir=pathlib.Path("synthetic"), legacy=legacy, star=star)


def self_check() -> int:
    """Exercise the comparison engine on synthetic frames without touching a benchmark."""
    artifacts = _synthetic_artifacts()
    inventory = (
        InventoryRow(metric_key="value", calculator="c", category="social", inventory_owner="Agent 4 (parity)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="listy", calculator="c", category="performance", inventory_owner="Agent 7 (perf)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="wall", calculator="c", category="performance", inventory_owner="Agent 7 (perf)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="absent", calculator="c", category="social", inventory_owner="Agent 4 (parity)", target_scope="contacts", target_table="fact_contacts", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="series", calculator="c", category="ecological", inventory_owner="Agent 7 (perf)", target_scope="robots", target_table="fact_robots", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="mean", calculator="c", category="social", inventory_owner="Agent 4 (parity)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="sparse", calculator="c", category="performance", inventory_owner="Agent 7 (perf)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="nullcol", calculator="c", category="performance", inventory_owner="Agent 7 (perf)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="zero", calculator="c", category="social", inventory_owner="Agent 4 (parity)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="derived", calculator="c", category="performance", inventory_owner="Agent 7 (perf)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
        InventoryRow(metric_key="scaled", calculator="c", category="performance", inventory_owner="Agent 7 (perf)", target_scope="episodes", target_table="fact_episodes", test_file="t", port_status="PENDING_PORT"),
    )
    mappings = (
        MetricMapping(metric_key="value", star_table="fact_episodes", star_column="value", legacy_columns=("value",)),
        MetricMapping(metric_key="listy", star_table="fact_episodes", star_column="mean", legacy_columns=("listy",), legacy_reduction=LegacyReduction.LIST_MEAN),
        MetricMapping(metric_key="wall", star_table="fact_episodes", star_column="static_total", legacy_columns=("wall", "static"), legacy_combine=LegacyCombine.SUM),
        MetricMapping(metric_key="zero", star_table="fact_episodes", star_column="zero_star", legacy_columns=("zero",)),
        MetricMapping(metric_key="derived", star_table="fact_episodes", star_column="derived_star", legacy_columns=("derived",)),
        MetricMapping(metric_key="scaled", star_table="fact_episodes", star_column="scaled", legacy_columns=("scaled",)),
        MetricMapping(metric_key="sparse", star_table="fact_episodes", star_column="sparse", legacy_columns=("sparse",)),
        MetricMapping(metric_key="nullcol", star_table="fact_episodes", star_column="nullcol", legacy_columns=("nullcol",)),
    )
    expected = (
        ExpectedDiff(metric_key="zero", star_table="fact_episodes", star_column="zero_star", kind=DiffKind.NULL_FILL, reason="synthetic null fill"),
        ExpectedDiff(metric_key="derived", star_table="fact_episodes", star_column="derived_star", kind=DiffKind.DERIVED, max_rel=0.10, reason="synthetic derived"),
        ExpectedDiff(metric_key="scaled", star_table="fact_episodes", star_column="scaled", kind=DiffKind.UNIT_SCALE, factor=2.0, max_rel=1.0e-9, reason="synthetic scale"),
    )
    certificate = certify(
        artifacts=artifacts,
        inventory=inventory,
        mappings=mappings,
        expected_diffs=expected,
        relocated_series=(SeriesRelocation(metric_key="series", reason="synthetic relocation"),),
        rel_tol=REL_TOL,
    )
    rows = {row.metric_key: row for row in certificate.rows}
    failures: list[str] = []
    expected_status = {
        "value": ParityStatus.EXACT,
        "listy": ParityStatus.EXACT,
        "wall": ParityStatus.EXACT,
        "zero": ParityStatus.NULL_REASON,
        "derived": ParityStatus.EXPECTED_DIFF,
        "scaled": ParityStatus.EXACT,
        "absent": ParityStatus.NO_PRODUCER,
        "mean": ParityStatus.UNMAPPED,
        "sparse": ParityStatus.INSUFFICIENT_EVIDENCE,
        "nullcol": ParityStatus.CERT_NULL_ONLY,
        "series": ParityStatus.RELOCATED_SERIES,
    }
    for metric_key, status in expected_status.items():
        if rows[metric_key].status is not status:
            failures.append(f"{metric_key}: expected {status}, got {rows[metric_key].status}")
    if len(certificate.gaps) != 1:
        failures.append(f"gap list must hold exactly the no-producer inventory key, got {len(certificate.gaps)}")
    if len(certificate.unmapped) != 1:
        failures.append(f"unmapped list must hold exactly the produced-but-unreviewed key, got {len(certificate.unmapped)}")
    if rows["value"].nonnull_compared != 3:
        failures.append(f"value: expected 3 non-null comparisons, got {rows['value'].nonnull_compared}")
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
    parser = argparse.ArgumentParser(description="G1 parity certificate between combined_metrics.parquet and the star schema.")
    parser.add_argument("--benchmark-dir", type=pathlib.Path, help="Benchmark directory holding both dual-path artifacts")
    parser.add_argument("--output-dir", type=pathlib.Path, default=_RESULTS_DIR, help="Where to write the certificate artefacts")
    parser.add_argument("--inventory", type=pathlib.Path, default=_HERE / "inventory_254.csv", help="Reviewed metric inventory CSV")
    parser.add_argument("--mappings", type=pathlib.Path, default=_HERE / "metric_map.yaml", help="Reviewed metric_key -> star column map")
    parser.add_argument("--expected-diffs", type=pathlib.Path, default=_HERE / "expected_diff.yaml", help="Reviewed expected-difference manifest")
    parser.add_argument("--rel-tol", type=float, default=REL_TOL, help="Relative tolerance for WITHIN_TOL")
    parser.add_argument("--min-nonnull-episodes", type=int, default=3, help="Minimum non-null comparisons for a green status")
    parser.add_argument("--self-check", action="store_true", help="Exercise the comparison engine on synthetic frames and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    if args.self_check:
        return self_check()
    if args.benchmark_dir is None:
        print("error: --benchmark-dir is required unless --self-check is set", file=sys.stderr)
        return 2
    try:
        inventory = load_inventory(args.inventory)
        mappings = load_mappings(args.mappings)
        manifest = load_expected_manifest(args.expected_diffs)
        artifacts = load_benchmark_artifacts(args.benchmark_dir)
        certificate = certify(
            artifacts=artifacts,
            inventory=inventory,
            mappings=mappings,
            expected_diffs=manifest.entries,
            relocated_series=manifest.relocated_series,
            structural_replacements=manifest.structural_replacements,
            list_violations=manifest.list_violations,
            rel_tol=args.rel_tol,
            min_nonnull_episodes=args.min_nonnull_episodes,
        )
    except ParityInputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except ParitySchemaError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    write_outputs(args.output_dir, certificate, benchmark_dir=args.benchmark_dir)
    counts = status_counts(certificate)
    for status in ParityStatus:
        print(f"{status.value}: {counts[status]}")
    print(f"unmapped: {len(certificate.unmapped)}  no_producer: {len(certificate.gaps)}")
    print(f"certificate written to {args.output_dir}")
    return 1 if any(counts[status] for status in _FAILING_STATUSES) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
