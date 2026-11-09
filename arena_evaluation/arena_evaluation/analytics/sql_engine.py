"""Polars OLAP SQL Engine with Read-Only Query Guard (Layer 4).

Executes guarded analytical SQL queries over :class:`BenchmarkCatalog` using
:class:`polars.SQLContext` (zero DuckDB, zero pandas).
"""

from __future__ import annotations

import pathlib
import re
from typing import Final

import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.star_schema import StarSchemaTables

__all__ = [
    "FORBIDDEN_SQL_KEYWORDS",
    "PolarsSQLEngine",
    "SQLGuardError",
    "normalize_sql_scientific_notation",
    "validate_readonly_sql",
]

FORBIDDEN_SQL_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "COPY",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "VACUUM",
    }
)

_ALLOWED_START_KEYWORDS: Final[frozenset[str]] = frozenset({"SELECT", "WITH"})
_BLOCK_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"--[^\r\n]*")
_STRING_LITERAL_RE: Final[re.Pattern[str]] = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\""
)
_WORD_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_SCIENTIFIC_NOTATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![a-zA-Z0-9_])([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+)(?![a-zA-Z0-9_])"
)


def normalize_sql_scientific_notation(sql: str) -> str:
    """Replace scientific notation numbers outside string literals with standard float strings."""

    def _repl(match: re.Match[str]) -> str:
        val = float(match.group(1))
        s = str(val)
        if "e" in s or "E" in s:
            s = f"{val:.12f}".rstrip("0")
            if s.endswith("."):
                s += "0"
        return s

    parts: list[str] = []
    last_end = 0
    for match in _STRING_LITERAL_RE.finditer(sql):
        code_part = sql[last_end : match.start()]
        parts.append(_SCIENTIFIC_NOTATION_RE.sub(_repl, code_part))
        parts.append(match.group(0))
        last_end = match.end()

    code_part = sql[last_end:]
    parts.append(_SCIENTIFIC_NOTATION_RE.sub(_repl, code_part))
    return "".join(parts)


class SQLGuardError(ValueError):
    """Raised when a SQL query violates the read-only analytical query guard."""


def validate_readonly_sql(sql: str) -> str:
    """Validate that ``sql`` is a single read-only ``SELECT`` or ``WITH ... SELECT`` statement."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLGuardError("SQL query must be a non-empty string.")

    without_block_comments = _BLOCK_COMMENT_RE.sub(" ", sql)
    without_comments = _LINE_COMMENT_RE.sub(" ", without_block_comments).strip()
    if not without_comments:
        raise SQLGuardError("SQL query contains only comments or whitespace.")

    masked = _STRING_LITERAL_RE.sub("''", without_comments).strip()
    if masked.endswith(";"):
        masked = masked[:-1].strip()
        without_comments = without_comments.rstrip()
        if without_comments.endswith(";"):
            without_comments = without_comments[:-1].rstrip()

    if ";" in masked:
        raise SQLGuardError("Multi-statement SQL queries (';') are forbidden.")

    tokens = [tok.upper() for tok in _WORD_TOKEN_RE.findall(masked)]
    if not tokens:
        raise SQLGuardError("SQL query contains no valid statement tokens.")

    forbidden_found = [tok for tok in tokens if tok in FORBIDDEN_SQL_KEYWORDS]
    if forbidden_found:
        raise SQLGuardError(
            f"Forbidden SQL keyword(s) detected: {', '.join(dict.fromkeys(forbidden_found))}"
        )

    first_token = tokens[0]
    if first_token not in _ALLOWED_START_KEYWORDS:
        raise SQLGuardError(
            f"Only SELECT or WITH ... SELECT queries are permitted; got '{first_token}'."
        )

    if first_token == "WITH" and "SELECT" not in tokens:
        raise SQLGuardError("CTE (WITH) query must contain a SELECT clause.")

    return without_comments


class PolarsSQLEngine:
    """Guarded read-only OLAP SQL engine backed by ``polars.SQLContext`` and ``BenchmarkCatalog``."""

    def __init__(
        self,
        catalog: BenchmarkCatalog | StarSchemaTables | pathlib.Path,
    ) -> None:
        if isinstance(catalog, BenchmarkCatalog):
            self.catalog = catalog
        else:
            self.catalog = BenchmarkCatalog(catalog)

    def register_table(self, name: str, frame: pl.LazyFrame | pl.DataFrame) -> None:
        """Register a derived table or view in the underlying catalog."""
        self.catalog.register_table(name, frame)

    def query_lazy(self, sql: str) -> pl.LazyFrame:
        """Validate ``sql`` and return a lazy Polars execution plan."""
        cleaned_sql = validate_readonly_sql(sql)
        preprocessed_sql = normalize_sql_scientific_notation(cleaned_sql)
        ctx = pl.SQLContext(frames=self.catalog.tables())
        return ctx.execute(preprocessed_sql, eager=False)

    def query(
        self,
        sql: str,
        *,
        limit: int | None = None,
        save_as: str | None = None,
    ) -> pl.DataFrame:
        """Execute a validated read-only SQL query and materialize the resulting ``DataFrame``."""
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")

        lf = self.query_lazy(sql)
        if limit is not None:
            lf = lf.limit(limit)
        if save_as is not None:
            self.catalog.register_table(save_as, lf)
        return lf.collect()
