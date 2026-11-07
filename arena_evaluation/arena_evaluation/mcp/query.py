"""MCP Query Gateway and Evidence Store Tools (Layer 4).

Exposes OLAP SQL query execution, transient vs. pinned evidence persistence,
stored query declaration management, and evidence index querying.
"""

from __future__ import annotations

from typing import Annotated, Any

import pydantic

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.analytics.queries import QueryCatalog, QueryDeclaration
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine, validate_readonly_sql
from arena_mcp.core import ToolError

from .base import OptionalRunId, component, domain, run_path

__all__ = [
    "evaluation_evidence",
    "evaluation_queries",
    "evaluation_query",
    "evaluation_query_save",
]


@domain.tool(read_only=False)
def evaluation_query(
    run_id: OptionalRunId = None,
    query: Annotated[str | None, pydantic.Field(description="Ad-hoc read-only SQL query to execute")] = None,
    stored_query: Annotated[str | None, pydantic.Field(description="Name of stored query declaration to execute")] = None,
    params: Annotated[dict[str, Any] | None, pydantic.Field(description="Parameters for stored query or SQL placeholders")] = None,
    pin_evidence: Annotated[bool, pydantic.Field(description="Persist query result as content-addressed evidence")] = False,
    purpose: Annotated[str, pydantic.Field(description="Hypothesis or purpose for pinning evidence")] = "",
) -> dict[str, Any]:
    """Execute an ad-hoc or stored analytical SQL query over the lakehouse catalog with transient or pinned evidence output."""
    if query is None and stored_query is None:
        raise ToolError("Must provide either 'query' or 'stored_query'", hint="evaluation_queries lists stored queries")
    if query is not None and stored_query is not None:
        raise ToolError("Cannot provide both 'query' and 'stored_query'")

    benchmark_path = run_path(run_id)
    catalog = BenchmarkCatalog(benchmark_path)
    engine = PolarsSQLEngine(catalog)

    if stored_query is not None:
        q_catalog = QueryCatalog()
        try:
            decl = q_catalog.get_query(stored_query)
        except KeyError as exc:
            raise ToolError(f"Stored query {stored_query!r} not found", hint="evaluation_queries lists available queries") from exc
        try:
            sql_to_run = decl.render_sql(params)
        except ValueError as exc:
            raise ToolError(f"Invalid parameters for {stored_query!r}: {exc}") from exc
        operation = f"stored_query:{stored_query}"
    else:
        clean_query = query.strip() if query is not None else ""
        if params:
            decl = QueryDeclaration(name="ad_hoc", description="", sql=clean_query, parameters=params)
            try:
                sql_to_run = decl.render_sql(params)
            except ValueError as exc:
                raise ToolError(f"Invalid query parameters: {exc}") from exc
        else:
            try:
                sql_to_run = validate_readonly_sql(clean_query)
            except Exception as exc:
                raise ToolError(f"Invalid SQL query: {exc}") from exc
        operation = f"sql:{clean_query}"

    try:
        df = engine.query(sql_to_run)
    except Exception as exc:
        raise ToolError(f"Query execution failed: {exc}") from exc

    should_pin = pin_evidence or bool(purpose.strip())
    if should_pin:
        ev_dir = benchmark_path / "evidence"
        store = EvidenceStore(evidence_dir=ev_dir)
        rec = store.record(
            df,
            kind="sql",
            operation=operation,
            purpose=purpose.strip(),
            params=params or {},
        )
        return {
            "run_id": benchmark_path.name,
            "pin_status": "pinned",
            "evidence_id": rec.evidence_id,
            "purpose": rec.purpose,
            "row_count": df.height,
            "columns": list(df.columns),
            "artifact_path": str(rec.artifact_path) if rec.artifact_path is not None else None,
            "data": df.to_dicts(),
        }

    return {
        "run_id": benchmark_path.name,
        "pin_status": "transient",
        "evidence_id": None,
        "purpose": "",
        "row_count": df.height,
        "columns": list(df.columns),
        "artifact_path": None,
        "data": df.to_dicts(),
    }


@domain.tool(read_only=True)
def evaluation_queries() -> list[dict[str, Any]]:
    """List all stored query declarations in the SQL Query Store."""
    q_catalog = QueryCatalog()
    return [q.to_dict() for q in q_catalog.list_queries()]


@domain.tool(read_only=False)
def evaluation_query_save(
    name: Annotated[str, pydantic.Field(description="Unique identifier for the query declaration")],
    description: Annotated[str, pydantic.Field(description="Description of what this analytical query computes")],
    sql: Annotated[str, pydantic.Field(description="Read-only SQL statement with optional :param placeholders")],
    parameters: Annotated[dict[str, Any] | None, pydantic.Field(description="Parameter declarations with types and defaults")] = None,
    title: Annotated[str | None, pydantic.Field(description="Human-readable title")] = None,
) -> dict[str, Any]:
    """Save a new reusable query declaration to the SQL Query Store."""
    safe_name = component(name.removesuffix(".yaml").removesuffix(".yml"), "name")
    decl = QueryDeclaration(
        name=safe_name,
        description=description.strip(),
        sql=sql.strip(),
        parameters=parameters or {},
        title=title or safe_name,
    )
    try:
        _ = decl.render_sql(decl.default_parameters())
    except Exception as exc:
        raise ToolError(f"Query declaration validation failed: {exc}") from exc

    q_catalog = QueryCatalog()
    saved = q_catalog.save_query(
        name=safe_name,
        description=decl.description,
        sql=decl.sql,
        parameters=decl.parameters,
        title=decl.title,
    )
    return {
        "status": "saved",
        "name": saved.name,
        "title": saved.title,
        "description": saved.description,
    }


@domain.tool(read_only=True)
def evaluation_evidence(
    run_id: OptionalRunId = None,
    evidence_id: Annotated[str | None, pydantic.Field(description="Specific evidence_id to load data for")] = None,
    purpose_filter: Annotated[str | None, pydantic.Field(description="Substring filter for evidence purpose")] = None,
) -> dict[str, Any]:
    """Query evidence index or load specific pinned evidence DataFrame for a benchmark run."""
    import polars as pl

    benchmark_path = run_path(run_id)
    ev_dir = benchmark_path / "evidence"
    store = EvidenceStore(evidence_dir=ev_dir)

    if evidence_id is not None:
        clean_ev_id = component(evidence_id, "evidence_id")
        try:
            rec = store.get(clean_ev_id)
            df = store.read_frame(clean_ev_id)
        except KeyError as exc:
            raise ToolError(f"Evidence {clean_ev_id!r} not found in run {benchmark_path.name!r}") from exc
        return {
            "run_id": benchmark_path.name,
            "evidence_id": rec.evidence_id,
            "kind": rec.kind,
            "operation": rec.operation,
            "purpose": rec.purpose,
            "row_count": df.height,
            "columns": list(df.columns),
            "created_at_utc": rec.created_at_utc,
            "artifact_path": str(rec.artifact_path) if rec.artifact_path is not None else None,
            "data": df.to_dicts(),
        }

    index_df = store.load_index()
    if purpose_filter is not None and purpose_filter.strip():
        pat = purpose_filter.strip().lower()
        index_df = index_df.filter(pl.col("purpose").str.to_lowercase().str.contains(pat))

    return {
        "run_id": benchmark_path.name,
        "total_records": index_df.height,
        "records": index_df.to_dicts(),
    }
