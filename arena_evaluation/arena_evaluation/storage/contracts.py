"""Publish-time contract checks for one benchmark's stored star schema (08 H1.6).

Facts live once per episode, dimensions once at the benchmark root (06 D6/D7). This module
reads that canonical layout lazily through :func:`scan_star_schema` and reports every
violation with the table and column it belongs to; it never repairs data. The check itself
writes nothing: :func:`enforce_benchmark_contract` persists the report and stops processing.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Final

import polars as pl
import yaml

from arena_evaluation.storage.declarations import (
    FACT_KEY_COLUMNS,
    FACT_SCHEMAS,
    FACT_TABLE_NAMES,
    FACT_TABLES_WITHOUT_ROW_KEY,
    FactTable,
    is_known_null_reason,
    nullable_columns,
)
from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.star_schema import (
    DIMENSION_SCHEMAS,
    StarSchemaTables,
    episode_directories,
    scan_star_schema,
)

__all__ = [
    "ContractReport",
    "ContractViolation",
    "check_benchmark",
    "enforce_benchmark_contract",
    "write_contract_report",
]

#: Fixed-size ``Array`` is nested too; the 1NF rule bans it alongside List/Struct/Object.
_NESTED_DTYPES: Final[tuple[type, ...]] = (pl.List, pl.Struct, pl.Object, pl.Array)
_IDENTITY_KEYS: Final[tuple[str, ...]] = ("run_id", "episode_id", "robot_id")

#: The dimension a fact row must resolve against, and the columns the join uses.
_FACT_PARENT: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    FactTable.EPISODES.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.ENCOUNTERS.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.CONTACTS.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.PEDESTRIANS.value: ("dim_pedestrians", ("run_id", "episode_id", "env_id", "ped_id")),
    FactTable.ROBOTS.value: ("dim_robots", ("run_id", "episode_id", "env_id", "robot_id")),
    FactTable.EDGE_CASES.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.FORMATIONS.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.PHASE_OUTCOMES.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.JUDGE_DISAGREEMENTS.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.MECHANISM_STATES.value: ("dim_episodes", _IDENTITY_KEYS),
    FactTable.COLLISIONS.value: ("dim_episodes", _IDENTITY_KEYS),
}

#: Child fact table -> (counter on ``fact_episodes``, child table). One row per replayed phase.
_CHILD_COUNT_CHECKS: Final[dict[str, tuple[str, str]]] = {
    FactTable.PHASE_OUTCOMES.value: ("phases_total", "fact_phase_outcomes"),
}

#: (fact table, start column, end column) of every windowed fact.
_WINDOW_COLUMNS: Final[tuple[tuple[str, str, str], ...]] = (
    (FactTable.ENCOUNTERS.value, "t_start_ns", "t_end_ns"),
    (FactTable.CONTACTS.value, "t_onset_ns", "t_end_ns"),
)


@dataclasses.dataclass(frozen=True, slots=True)
class ContractViolation:
    """One broken contract rule, named by table and column."""

    table: str
    column: str
    check: str
    detail: str

    def message(self) -> str:
        """``table.column: check: detail``, the form the raised error and the report use."""
        return f"{self.table}.{self.column}: {self.check}: {self.detail}"


@dataclasses.dataclass(frozen=True, slots=True)
class ContractReport:
    """Every contract violation found in one benchmark, empty when the benchmark is clean."""

    benchmark_dir: pathlib.Path
    violations: tuple[ContractViolation, ...]
    checked_tables: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when no contract rule was broken."""
        return not self.violations

    def summary(self) -> str:
        """One line per violation, prefixed by the benchmark path."""
        head = f"benchmark contract failed for {self.benchmark_dir}: {len(self.violations)} violation(s)"
        return "\n".join([head, *(f"- {violation.message()}" for violation in self.violations)])

    def as_dict(self) -> dict[str, object]:
        """The report as plain data, for the YAML artifact."""
        return {
            "benchmark_dir": str(self.benchmark_dir),
            "ok": self.ok,
            "checked_tables": list(self.checked_tables),
            "violations": [{"table": violation.table, "column": violation.column, "check": violation.check, "detail": violation.detail} for violation in self.violations],
        }


def _nested_columns(schema: Mapping[str, pl.DataType]) -> tuple[str, ...]:
    """Columns whose dtype is List, Struct, Object or Array."""
    return tuple(name for name, dtype in schema.items() if isinstance(dtype, _NESTED_DTYPES))


def _schema_violations(table: str, actual: pl.Schema, expected: pl.Schema) -> list[ContractViolation]:
    """Columns the stored schema adds, drops or types differently from its declaration."""
    checked = [(name, "schema_missing") for name in expected.names() if name not in actual]
    checked += [(name, "schema_unexpected") for name in actual.names() if name not in expected]
    checked += [(name, "schema_dtype") for name in expected.names() if name in actual and actual[name] != expected[name]]
    return [
        ContractViolation(
            table=table,
            column=column,
            check=kind,
            detail=f"expected {expected[column]!r}, stored {actual[column]!r}" if kind == "schema_dtype" else f"declared columns: {list(expected.names())}",
        )
        for column, kind in checked
    ]


def _file_schema_violations(table: str, path: pathlib.Path, expected: pl.Schema) -> list[ContractViolation]:
    """The schema and flatness checks of one stored Parquet file."""
    actual = pl.scan_parquet(path).collect_schema()
    violations = _schema_violations(table, actual, expected)
    violations.extend(ContractViolation(table=table, column=column, check="nested_dtype", detail=f"stored {actual[column]!r} in {path.name}") for column in _nested_columns(actual))
    return violations


def _check_stored_schemas(benchmark_dir: pathlib.Path, *, check_dimension_files: bool) -> list[ContractViolation]:
    """Every stored fact file, and every stored dimension when it is read from disk, is declared and flat."""
    violations: list[ContractViolation] = []
    if check_dimension_files:
        for name in DIMENSION_SCHEMAS:
            path = benchmark_dir / f"{name}.parquet"
            if path.is_file():
                violations.extend(_file_schema_violations(name, path, DIMENSION_SCHEMAS[name]))
    episode_dirs = episode_directories(benchmark_dir)
    for name, schema in FACT_SCHEMAS.items():
        table_name = f"fact_{name}"
        for episode_dir in episode_dirs:
            path = episode_dir / "metrics" / f"{table_name}.parquet"
            if path.is_file():
                violations.extend(_file_schema_violations(table_name, path, schema))
    return violations


def _check_dimension_frames(dimensions: StarSchemaTables) -> list[ContractViolation]:
    """In-memory dimensions are written only after they pass their declared flat schema."""
    violations: list[ContractViolation] = []
    for name, frame in dimensions.dimension_tables().items():
        schema = frame.schema
        violations.extend(_schema_violations(name, schema, DIMENSION_SCHEMAS[name]))
        violations.extend(ContractViolation(table=name, column=column, check="nested_dtype", detail=f"in-memory {schema[column]!r}") for column in _nested_columns(schema))
    return violations


def _check_missing_dimensions(benchmark_dir: pathlib.Path, views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """A benchmark that stores facts must also store the dimensions they resolve against."""
    has_facts = any(views[f"fact_{name}"].select(pl.len()).collect().item() > 0 for name in FACT_TABLE_NAMES)
    if not has_facts:
        return []
    return [ContractViolation(table=name, column="run_id", check="dimension_missing", detail=f"{benchmark_dir / f'{name}.parquet'} is absent") for name in DIMENSION_SCHEMAS if not (benchmark_dir / f"{name}.parquet").is_file()]


def _check_keys(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every fact row key is non-null and, where the table has one, unique."""
    violations: list[ContractViolation] = []
    for name, keys in FACT_KEY_COLUMNS.items():
        table_name = f"fact_{name}"
        frame = views[table_name]
        null_counts = frame.select([pl.col(key).is_null().sum().alias(key) for key in keys]).collect().row(0, named=True)
        for key, count in null_counts.items():
            if count:
                violations.append(ContractViolation(table=table_name, column=key, check="key_null", detail=f"{count} rows"))
        if name in FACT_TABLES_WITHOUT_ROW_KEY:
            continue
        duplicates = frame.group_by(list(keys)).agg(pl.len().alias("rows")).filter(pl.col("rows") > 1).collect()
        if duplicates.height:
            violations.append(ContractViolation(table=table_name, column=",".join(keys), check="key_duplicate", detail=f"{duplicates.height} duplicated keys, first {duplicates.row(0, named=True)}"))
    return violations


def _check_orphans(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every fact row resolves to a row of its dimension."""
    violations: list[ContractViolation] = []
    for name, (dimension, keys) in _FACT_PARENT.items():
        table_name = f"fact_{name}"
        dimension_keys = views[dimension].select(list(keys)).unique()
        orphan_count = views[table_name].select(list(keys)).join(dimension_keys, on=list(keys), how="anti").select(pl.len()).collect().item()
        if orphan_count:
            violations.append(ContractViolation(table=table_name, column=",".join(keys), check="orphan_fact", detail=f"{orphan_count} rows have no {dimension} row"))
    return violations


def _check_windows(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """A windowed fact row starts strictly before it ends."""
    violations: list[ContractViolation] = []
    for name, start_column, end_column in _WINDOW_COLUMNS:
        table_name = f"fact_{name}"
        invalid = views[table_name].filter(pl.col(start_column).is_not_null() & pl.col(end_column).is_not_null() & (pl.col(start_column) >= pl.col(end_column)))
        count = invalid.select(pl.len()).collect().item()
        if count:
            violations.append(ContractViolation(table=table_name, column=f"{start_column},{end_column}", check="window_order", detail=f"{count} rows with {start_column} >= {end_column}"))
    return violations


def _check_actor_pairs(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """An encounter never pairs an actor with itself."""
    invalid = views["fact_encounters"].filter(pl.col("actor_1").is_not_null() & (pl.col("actor_1") == pl.col("actor_2")))
    count = invalid.select(pl.len()).collect().item()
    if not count:
        return []
    return [ContractViolation(table="fact_encounters", column="actor_1,actor_2", check="actor_self_pair", detail=f"{count} encounters pair an actor with itself")]


def _check_child_counts(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """A parent counter equals the number of child rows it claims."""
    violations: list[ContractViolation] = []
    for _child, (counter, child_table) in _CHILD_COUNT_CHECKS.items():
        parent = views["fact_episodes"].select([*_IDENTITY_KEYS, counter])
        child_counts = views[child_table].group_by(list(_IDENTITY_KEYS)).agg(pl.len().alias("child_rows"))
        mismatched = parent.filter(pl.col(counter).is_not_null()).join(child_counts, on=list(_IDENTITY_KEYS), how="left").with_columns(pl.col("child_rows").fill_null(0)).filter(pl.col("child_rows") != pl.col(counter))
        count = mismatched.select(pl.len()).collect().item()
        if count:
            violations.append(ContractViolation(table="fact_episodes", column=f"{counter},{child_table}", check="child_count", detail=f"{count} rows where {counter} != {child_table} rows"))
    return violations


def _check_null_reasons(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every NULL metric cell on a fact row is explained by that row's ``null_reason``."""
    violations: list[ContractViolation] = []
    for name in FACT_TABLE_NAMES:
        columns = nullable_columns(FactTable(name))
        if not columns:
            continue
        table_name = f"fact_{name}"
        frame = views[table_name]
        unexplained = frame.filter(pl.any_horizontal([pl.col(column).is_null() for column in columns]) & (pl.col("null_reason").is_null() | (pl.col("null_reason") == "")))
        count = unexplained.select(pl.len()).collect().item()
        if not count:
            continue
        null_flags = unexplained.select([pl.col(column).is_null().any().alias(column) for column in columns]).collect().row(0, named=True)
        null_columns = [column for column, is_null in null_flags.items() if is_null]
        violations.append(ContractViolation(table=table_name, column=",".join(null_columns), check="null_reason_missing", detail=f"{count} rows with NULL metric cells and no null_reason"))
    return violations


def _check_null_reason_vocabulary(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """Every emitted ``null_reason`` is a code of the closed vocabulary, parameter included."""
    violations: list[ContractViolation] = []
    for name in FACT_TABLE_NAMES:
        table_name = f"fact_{name}"
        emitted = views[table_name].select(pl.col("null_reason").drop_nulls().unique()).collect().get_column("null_reason")
        unknown = emitted.filter(~emitted.map_elements(is_known_null_reason, return_dtype=pl.Boolean))
        if not unknown.is_empty():
            violations.append(ContractViolation(table=table_name, column="null_reason", check="null_reason_unknown", detail=f"{unknown.len()} code(s) outside the vocabulary, first {unknown[0]!r}"))
    return violations


def _check_ranking_validity(views: Mapping[str, pl.LazyFrame]) -> list[ContractViolation]:
    """A published fact row always states whether it is valid for ranking; NULL is not valid."""
    violations: list[ContractViolation] = []
    for name in FACT_TABLE_NAMES:
        table_name = f"fact_{name}"
        frame = views[table_name]
        if "is_valid_for_ranking" not in frame.collect_schema().names():
            continue
        count = frame.filter(pl.col("is_valid_for_ranking").is_null()).select(pl.len()).collect().item()
        if count:
            violations.append(ContractViolation(table=table_name, column="is_valid_for_ranking", check="ranking_validity_null", detail=f"{count} rows without a ranking verdict"))
    return violations


def check_benchmark(path: pathlib.Path, *, dimensions: StarSchemaTables | None = None) -> ContractReport:
    """Run every 08 H1.6 publish-time check over the benchmark stored at ``path``.

    ``dimensions`` lets the publish boundary validate the dimensions it is about to write,
    before any benchmark-level file exists; without it the stored ``dim_*.parquet`` are read.
    """
    benchmark_dir = pathlib.Path(path)
    views = scan_star_schema(benchmark_dir)
    if dimensions is not None:
        views.update({name: frame.lazy() for name, frame in dimensions.dimension_tables().items()})
    violations = [
        *_check_stored_schemas(benchmark_dir, check_dimension_files=dimensions is None),
        *(_check_dimension_frames(dimensions) if dimensions is not None else []),
        *(_check_missing_dimensions(benchmark_dir, views) if dimensions is None else []),
        *_check_keys(views),
        *_check_orphans(views),
        *_check_windows(views),
        *_check_actor_pairs(views),
        *_check_child_counts(views),
        *_check_null_reasons(views),
        *_check_null_reason_vocabulary(views),
        *_check_ranking_validity(views),
    ]
    return ContractReport(benchmark_dir=benchmark_dir, violations=tuple(violations), checked_tables=tuple(views))


def write_contract_report(report: ContractReport, path: pathlib.Path) -> pathlib.Path:
    """Write the report as YAML, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(report.as_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def enforce_benchmark_contract(benchmark_dir: pathlib.Path, *, dimensions: StarSchemaTables | None = None) -> ContractReport:
    """Check the benchmark, write its report and raise when a rule is broken.

    Called at the publish boundary with the dimensions still in memory: a failed check stops
    processing before any benchmark-level table is published, and the YAML report keeps every
    table and column that broke.
    """
    report = check_benchmark(benchmark_dir, dimensions=dimensions)
    if report.ok:
        return report
    write_contract_report(report, benchmark_dir / "contract_report.yaml")
    raise ContractViolationError(report.summary())
