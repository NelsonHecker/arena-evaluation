"""Data and recipe resolver for the declarative presentation engine (Layer 5).

Binds PlotSpec definitions to Star Schema lakehouse tables, Polars SQL queries,
content-addressed EvidenceStore artifacts, and reusable plot recipes.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import polars as pl
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.presentation.checks import DiagnosticInspector, PlotDiagnostics
from arena_evaluation.presentation.predicate import compile_where_predicate
from arena_evaluation.storage.schemas import PlotSpec

__all__ = [
    "DataResolver",
    "find_recipe_path",
    "load_recipe",
]

_DEFAULT_RECIPE_SEARCH_DIRS = (
    pathlib.Path(__file__).resolve().parents[1] / "configs" / "declarations" / "plot_declarations",
    pathlib.Path(__file__).resolve().parents[2] / "configs" / "declarations" / "plot_declarations",
    pathlib.Path.cwd() / "configs" / "declarations" / "plot_declarations",
)


def find_recipe_path(name: str, extra_dirs: Sequence[pathlib.Path] | None = None) -> pathlib.Path | None:
    """Locate a YAML recipe by name in the configured search directories."""
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    candidates = list(extra_dirs or []) + list(_DEFAULT_RECIPE_SEARCH_DIRS)

    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = pathlib.Path(get_package_share_directory("arena_evaluation")) / "configs" / "declarations" / "plot_declarations"
        candidates.append(share_dir)
    except Exception:
        pass

    for directory in candidates:
        if not directory.is_dir():
            continue
        direct = directory / filename
        if direct.is_file():
            return direct
        alt = directory / f"{name}.yml"
        if alt.is_file():
            return alt

    return None


def load_recipe(name: str, extra_dirs: Sequence[pathlib.Path] | None = None) -> dict[str, Any]:
    """Load recipe dictionary from YAML declaration."""
    path = find_recipe_path(name, extra_dirs=extra_dirs)
    if path is None:
        raise FileNotFoundError(f"Plot recipe '{name}' not found in search paths.")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Recipe '{name}' must be a YAML mapping, got {type(data).__name__}.")
    return data


def _deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into base dictionary."""
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        elif value is not None:
            result[key] = value
    return result


class DataResolver:
    """Resolves PlotSpec data bindings against Star Schema, SQL queries, or EvidenceStore."""

    def __init__(
        self,
        *,
        catalog: BenchmarkCatalog | None = None,
        sql_engine: PolarsSQLEngine | None = None,
        evidence_store: EvidenceStore | None = None,
        query_catalog: QueryCatalog | None = None,
        benchmark_dir: pathlib.Path | None = None,
        recipe_dirs: Sequence[pathlib.Path] | None = None,
    ) -> None:
        self.benchmark_dir = pathlib.Path(benchmark_dir) if benchmark_dir is not None else None
        self.recipe_dirs = list(recipe_dirs) if recipe_dirs is not None else []
        if self.benchmark_dir is not None:
            self.recipe_dirs.append(self.benchmark_dir / "plot_declarations")

        if catalog is not None:
            self.catalog = catalog
        elif self.benchmark_dir is not None:
            self.catalog = BenchmarkCatalog.from_directory(self.benchmark_dir)
        else:
            self.catalog = None

        if sql_engine is not None:
            self.sql_engine = sql_engine
        elif self.catalog is not None:
            self.sql_engine = PolarsSQLEngine(self.catalog)
        else:
            self.sql_engine = None

        if evidence_store is not None:
            self.evidence_store = evidence_store
        elif self.benchmark_dir is not None and (self.benchmark_dir / "evidence").is_dir():
            self.evidence_store = EvidenceStore(self.benchmark_dir / "evidence")
        else:
            self.evidence_store = None

        if query_catalog is not None:
            self.query_catalog = query_catalog
        else:
            extra_query_dirs = [self.benchmark_dir / "query_declarations"] if self.benchmark_dir is not None else []
            self.query_catalog = QueryCatalog(extra_query_dirs)

    def resolve_spec(self, spec: PlotSpec) -> PlotSpec:
        """Resolve `uses: <recipe>` reference and return merged PlotSpec."""
        if not spec.uses:
            return spec

        recipe_raw = load_recipe(spec.uses, extra_dirs=self.recipe_dirs)
        user_overrides = spec.model_dump(exclude_unset=True)
        user_overrides.pop("uses", None)

        merged = _deep_merge_dict(recipe_raw, user_overrides)
        if not merged.get("id"):
            merged["id"] = spec.id or spec.uses
        return PlotSpec.model_validate(merged)

    def resolve(
        self,
        spec: PlotSpec,
        fallback_df: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, PlotDiagnostics]:
        """Resolve data frame and collect diagnostics for given PlotSpec."""
        resolved_spec = self.resolve_spec(spec)
        df = self._fetch_frame(resolved_spec, fallback_df=fallback_df)
        df = self._apply_slice(resolved_spec, df)
        df = self._apply_filters(resolved_spec, df)
        diagnostics = DiagnosticInspector.inspect(df, resolved_spec)
        return df, diagnostics

    def _fetch_frame(self, spec: PlotSpec, fallback_df: pl.DataFrame | None) -> pl.DataFrame:
        data_spec = spec.data

        # 1. Stored Query execution
        if data_spec is not None and data_spec.stored_query:
            if self.sql_engine is None:
                raise RuntimeError(f"Cannot execute stored query for plot '{spec.id}': SQL engine not initialized.")
            return self.query_catalog.execute(data_spec.stored_query, self.sql_engine, params=data_spec.params)

        # 2. SQL Query execution
        if data_spec is not None and data_spec.query:
            if self.sql_engine is None:
                raise RuntimeError(f"Cannot execute SQL query for plot '{spec.id}': SQL engine not initialized.")
            return self.sql_engine.query(data_spec.query)

        # 3. Evidence Store retrieval
        if data_spec is not None and data_spec.evidence_id:
            if self.evidence_store is None:
                raise RuntimeError(f"Cannot fetch evidence '{data_spec.evidence_id}' for plot '{spec.id}': EvidenceStore not initialized.")
            return self.evidence_store.read_frame(data_spec.evidence_id)

        # 4. Catalog table or view
        table_source = data_spec.source if (data_spec is not None and data_spec.source) else spec.data_source
        if table_source:
            if self.catalog is None:
                raise RuntimeError(f"Cannot fetch table '{table_source}' for plot '{spec.id}': catalog not initialized.")
            lf = self.catalog.get_table(table_source)
            if data_spec is not None and data_spec.where:
                lf = lf.filter(compile_where_predicate(data_spec.where, lf.collect_schema()))
            return lf.collect()

        # 5. Fallback in-memory DataFrame
        if fallback_df is not None:
            return fallback_df

        raise ValueError(f"Cannot resolve data for plot '{spec.id}': no data source, stored_query, query, evidence_id, or fallback provided.")

    @staticmethod
    def _apply_slice(spec: PlotSpec, df: pl.DataFrame) -> pl.DataFrame:
        if spec.slice is None:
            return df

        res_df = df
        schema = res_df.schema
        if spec.slice.episode_id is not None and schema.get("episode_id") is not None:
            res_df = res_df.filter(pl.col("episode_id") == spec.slice.episode_id)
        if spec.slice.encounter_id is not None and schema.get("encounter_id") is not None:
            res_df = res_df.filter(pl.col("encounter_id") == spec.slice.encounter_id)
        if spec.slice.contact_id is not None and schema.get("contact_id") is not None:
            res_df = res_df.filter(pl.col("contact_id") == spec.slice.contact_id)
        return res_df

    @staticmethod
    def _apply_filters(spec: PlotSpec, df: pl.DataFrame) -> pl.DataFrame:
        if not spec.filter:
            return df

        res_df = df
        schema = res_df.schema
        for key, value in spec.filter.items():
            if schema.get(key) is None:
                continue
            if schema[key] == pl.List:
                if isinstance(value, (list, tuple, set)):
                    v_list = list(value)
                    if v_list:
                        res_df = res_df.filter(pl.any_horizontal([pl.col(key).list.contains(x) for x in v_list]))
                    else:
                        res_df = res_df.filter(pl.lit(False))
                else:
                    res_df = res_df.filter(pl.col(key).list.contains(value))
            elif isinstance(value, (list, tuple, set)):
                res_df = res_df.filter(pl.col(key).is_in(list(value)))
            else:
                res_df = res_df.filter(pl.col(key) == value)
        return res_df
