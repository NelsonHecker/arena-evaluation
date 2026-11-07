"""SQL Query Store and Declarations for Benchmark OLAP Analytics (Layer 4).

Provides declarative, parameterized analytical SQL queries, catalog lookup,
safe literal substitution, and execution over :class:`PolarsSQLEngine`.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl
import yaml

from arena_evaluation.analytics.sql_engine import PolarsSQLEngine, validate_readonly_sql

__all__ = [
    "QueryCatalog",
    "QueryDeclaration",
    "format_sql_literal",
]

_PARAM_COLON_RE: Final[re.Pattern[str]] = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")
_PARAM_BRACE_RE: Final[re.Pattern[str]] = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

_DEFAULT_QUERY_SEARCH_DIRS: Final[tuple[pathlib.Path, ...]] = (
    pathlib.Path(__file__).resolve().parents[1] / "query_declarations",
    pathlib.Path(__file__).resolve().parents[2] / "query_declarations",
    pathlib.Path.cwd() / "query_declarations",
)


def format_sql_literal(value: object) -> str:
    """Format a Python value as a safe, canonical SQL literal string."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, (list, tuple, set)):
        items = [format_sql_literal(v) for v in value]
        return f"({', '.join(items)})"
    escaped_obj = str(value).replace("'", "''")
    return f"'{escaped_obj}'"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QueryDeclaration:
    """Immutable declaration of a parameterized analytical SQL query."""

    name: str
    description: str
    sql: str
    parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    title: str = ""

    def default_parameters(self) -> dict[str, Any]:
        """Extract default parameter values from the declaration."""
        defaults: dict[str, Any] = {}
        for key, spec in self.parameters.items():
            if isinstance(spec, Mapping) and "default" in spec:
                defaults[key] = spec["default"]
            elif not isinstance(spec, Mapping):
                defaults[key] = spec
        return defaults

    def render_sql(self, params: Mapping[str, Any] | None = None) -> str:
        """Substitute parameters into the query SQL and validate read-only compliance."""
        combined_params = self.default_parameters()
        if params is not None:
            combined_params.update(params)

        def replace_param(match: re.Match[str]) -> str:
            param_name = match.group(1)
            if param_name not in combined_params:
                raise ValueError(f"Missing required parameter '{param_name}' for query declaration '{self.name}'.")
            return format_sql_literal(combined_params[param_name])

        rendered = _PARAM_COLON_RE.sub(replace_param, self.sql)
        rendered = _PARAM_BRACE_RE.sub(replace_param, rendered)
        return validate_readonly_sql(rendered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize query declaration to a dictionary."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "parameters": dict(self.parameters),
            "sql": self.sql,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QueryDeclaration:
        """Deserialize a QueryDeclaration from a dictionary."""
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description") or ""),
            sql=str(payload["sql"]),
            parameters=dict(payload.get("parameters") or {}),
            title=str(payload.get("title") or ""),
        )


class QueryCatalog:
    """Catalog of stored reusable SQL query declarations with discovery and execution."""

    def __init__(self, search_dirs: Sequence[pathlib.Path] | None = None) -> None:
        candidates = list(search_dirs or []) + list(_DEFAULT_QUERY_SEARCH_DIRS)
        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = pathlib.Path(get_package_share_directory("arena_evaluation")) / "query_declarations"
            candidates.append(share_dir)
        except Exception:
            pass

        self._search_dirs = tuple(candidates)
        self._cache: dict[str, QueryDeclaration] = {}
        self._load_all()

    @property
    def search_dirs(self) -> tuple[pathlib.Path, ...]:
        """Configured directory search paths for query declarations."""
        return self._search_dirs

    def _load_all(self) -> None:
        """Scan search directories and populate cached query declarations."""
        for directory in self._search_dirs:
            if not directory.is_dir():
                continue
            for file_path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
                name = file_path.stem
                if name in self._cache:
                    continue
                try:
                    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
                    if isinstance(payload, Mapping) and "sql" in payload:
                        decl = QueryDeclaration(
                            name=str(payload.get("name") or name),
                            description=str(payload.get("description") or ""),
                            sql=str(payload["sql"]),
                            parameters=dict(payload.get("parameters") or {}),
                            title=str(payload.get("title") or name),
                        )
                        self._cache[name] = decl
                        self._cache[decl.name] = decl
                except Exception:
                    continue

    def list_queries(self) -> list[QueryDeclaration]:
        """Return all discovered query declarations sorted by name."""
        unique = {decl.name: decl for decl in self._cache.values()}
        return [unique[k] for k in sorted(unique.keys())]

    def has_query(self, name: str) -> bool:
        """Check whether a query declaration with ``name`` is available."""
        stem = name.removesuffix(".yaml").removesuffix(".yml")
        return stem in self._cache or name in self._cache

    def get_query(self, name: str) -> QueryDeclaration:
        """Retrieve a ``QueryDeclaration`` by name or stem."""
        stem = name.removesuffix(".yaml").removesuffix(".yml")
        if stem in self._cache:
            return self._cache[stem]
        if name in self._cache:
            return self._cache[name]
        raise KeyError(f"Query declaration '{name}' not found. Available: {sorted(set(self._cache))}")

    def save_query(
        self,
        *,
        name: str,
        description: str,
        sql: str,
        parameters: Mapping[str, Any] | None = None,
        title: str | None = None,
        directory: pathlib.Path | None = None,
    ) -> QueryDeclaration:
        """Persist a new query declaration to a YAML file in the catalog."""
        clean_name = name.removesuffix(".yaml").removesuffix(".yml").strip()
        if not clean_name:
            raise ValueError("Query name must be a non-empty string.")

        target_dir = directory
        if target_dir is None:
            for d in self._search_dirs:
                if d.is_dir():
                    target_dir = d
                    break
            if target_dir is None:
                target_dir = _DEFAULT_QUERY_SEARCH_DIRS[0]

        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_name}.yaml"

        decl = QueryDeclaration(
            name=clean_name,
            description=description.strip(),
            sql=sql.strip(),
            parameters=dict(parameters or {}),
            title=(title or clean_name).strip(),
        )

        yaml_dict = {
            "name": decl.name,
            "title": decl.title,
            "description": decl.description,
            "parameters": dict(decl.parameters),
            "sql": decl.sql,
        }
        file_path.write_text(yaml.safe_dump(yaml_dict, sort_keys=False), encoding="utf-8")

        self._cache[clean_name] = decl
        return decl

    def execute(
        self,
        name_or_declaration: str | QueryDeclaration,
        engine: PolarsSQLEngine,
        *,
        params: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Render and execute a stored query over :class:`PolarsSQLEngine`."""
        decl = name_or_declaration if isinstance(name_or_declaration, QueryDeclaration) else self.get_query(name_or_declaration)
        rendered_sql = decl.render_sql(params)
        return engine.query(rendered_sql, limit=limit)
