"""MCP Presentation Gateway Tools (Layer 5).

Exposes declarative plot recipe discovery, schema inspection, plot recipe persistence,
and standalone visual artifact rendering (Plotly HTML, Seaborn PNG/SVG/PDF).
"""

from __future__ import annotations

import pathlib
from typing import Annotated, Any, Literal

import pydantic
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.resolver import (
    _DEFAULT_RECIPE_SEARCH_DIRS,
    DataResolver,
    _deep_merge_dict,
    find_recipe_path,
)
from arena_evaluation.presentation.seaborn_renderer import SeabornRenderer
from arena_evaluation.storage.schemas import PlotSpec
from arena_mcp.core import ToolError

from .base import OptionalRunId, YamlText, component, domain, parse_yaml, run_path

__all__ = [
    "evaluation_plot_declarations",
    "evaluation_read_plot_declaration",
    "evaluation_render_plot",
    "evaluation_save_plot_declaration",
]


@domain.tool(read_only=True)
def evaluation_plot_declarations() -> list[dict[str, Any]]:
    """List all registered declarative plot declarations and recipes with encodings and data sources."""
    results: dict[str, dict[str, Any]] = {}

    candidates: list[pathlib.Path] = list(_DEFAULT_RECIPE_SEARCH_DIRS)
    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = pathlib.Path(get_package_share_directory("arena_evaluation")) / "plot_declarations"
        candidates.append(share_dir)
    except Exception:
        pass

    for directory in candidates:
        if not directory.is_dir():
            continue
        for file_path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
            name = file_path.stem
            if name in results:
                continue
            try:
                raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict) or "type" not in raw:
                    continue
                spec = PlotSpec.model_validate(raw)
                source_val = spec.data.source if (spec.data and spec.data.source) else spec.data_source
                stored_query_val = spec.data.stored_query if (spec.data and spec.data.stored_query) else None
                results[name] = {
                    "id": spec.id or name,
                    "title": spec.title,
                    "type": spec.type,
                    "source": source_val,
                    "stored_query": stored_query_val,
                    "encodings": spec.channels.model_dump(exclude_unset=True),
                    "options": dict(spec.options),
                }
            except Exception:
                continue

    return [results[k] for k in sorted(results.keys())]


@domain.tool(read_only=True)
def evaluation_read_plot_declaration(
    name: Annotated[str, pydantic.Field(description="Name or stem of registered plot declaration")],
) -> dict[str, Any]:
    """Read a registered plot declaration YAML document and its validated specification."""
    safe_name = component(name.removesuffix(".yaml").removesuffix(".yml"), "name")
    path = find_recipe_path(safe_name)
    if path is None:
        raise ToolError(f"Plot declaration {name!r} not found", hint="evaluation_plot_declarations lists available declarations")

    text = path.read_text(encoding="utf-8")
    parsed = parse_yaml(text, f"plot declaration '{safe_name}'")
    if not isinstance(parsed, dict):
        raise ToolError(f"Plot declaration '{safe_name}' must be a YAML mapping")

    try:
        spec = PlotSpec.model_validate(parsed)
    except Exception as exc:
        raise ToolError(f"Invalid PlotSpec schema in declaration '{safe_name}': {exc}") from exc

    return {
        "name": safe_name,
        "yaml": text,
        "spec": spec.model_dump(),
    }


@domain.tool(read_only=False)
def evaluation_save_plot_declaration(
    name: Annotated[str, pydantic.Field(description="Unique identifier for the plot declaration")],
    yaml_text: YamlText,
) -> dict[str, Any]:
    """Validate against PlotSpec and save a plot declaration recipe to the plot declarations catalog."""
    safe_name = component(name.removesuffix(".yaml").removesuffix(".yml"), "name")
    parsed = parse_yaml(yaml_text, "yaml_text")
    if not isinstance(parsed, dict):
        raise ToolError("Plot declaration YAML must be a mapping")

    try:
        spec = PlotSpec.model_validate(parsed)
    except Exception as exc:
        raise ToolError(f"Invalid PlotSpec schema: {exc}") from exc

    target_dir: pathlib.Path | None = None
    for d in _DEFAULT_RECIPE_SEARCH_DIRS:
        if d.is_dir():
            target_dir = d
            break
    if target_dir is None:
        target_dir = _DEFAULT_RECIPE_SEARCH_DIRS[0]

    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{safe_name}.yaml"
    file_path.write_text(yaml_text.strip() + "\n", encoding="utf-8")

    return {
        "status": "saved",
        "name": safe_name,
        "id": spec.id or safe_name,
        "type": spec.type,
        "title": spec.title,
    }


@domain.tool(read_only=False)
def evaluation_render_plot(
    run_id: OptionalRunId = None,
    declaration: Annotated[str | None, pydantic.Field(description="Name of registered plot recipe/declaration to render")] = None,
    spec_yaml: Annotated[str | None, pydantic.Field(description="Inline YAML string defining a PlotSpec to render")] = None,
    overrides: Annotated[dict[str, Any] | None, pydantic.Field(description="Optional dictionary to deep-merge over declaration")] = None,
    output_format: Annotated[Literal["png", "svg", "pdf", "html"], pydantic.Field(description="Desired output format")] = "png",
) -> dict[str, Any]:
    """Render a plot recipe or inline spec directly into a standalone artifact file with quality diagnostics."""
    if declaration is None and spec_yaml is None:
        raise ToolError("Must provide either 'declaration' or 'spec_yaml'", hint="evaluation_plot_declarations lists available recipes")
    if declaration is not None and spec_yaml is not None:
        raise ToolError("Cannot provide both 'declaration' and 'spec_yaml'")

    if declaration is not None:
        safe_decl = component(declaration.removesuffix(".yaml").removesuffix(".yml"), "declaration")
        recipe_path = find_recipe_path(safe_decl)
        if recipe_path is None:
            raise ToolError(f"Plot declaration {declaration!r} not found", hint="evaluation_plot_declarations lists available recipes")
        raw_recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        if not isinstance(raw_recipe, dict):
            raise ToolError(f"Recipe '{safe_decl}' is not a valid YAML mapping")
        merged = _deep_merge_dict(raw_recipe, overrides or {})
        try:
            spec = PlotSpec.model_validate(merged)
        except Exception as exc:
            raise ToolError(f"Invalid PlotSpec schema in recipe '{safe_decl}': {exc}") from exc
    else:
        parsed = parse_yaml(spec_yaml or "", "spec_yaml")
        if not isinstance(parsed, dict):
            raise ToolError("spec_yaml must be a YAML mapping")
        merged = _deep_merge_dict(parsed, overrides or {})
        try:
            spec = PlotSpec.model_validate(merged)
        except Exception as exc:
            raise ToolError(f"Invalid PlotSpec schema: {exc}") from exc

    benchmark_path = run_path(run_id)
    catalog = BenchmarkCatalog(benchmark_path)
    sql_engine = PolarsSQLEngine(catalog)
    ev_dir = benchmark_path / "evidence"
    evidence_store = EvidenceStore(ev_dir) if ev_dir.is_dir() else None
    q_catalog = QueryCatalog([benchmark_path / "query_declarations"])

    resolver = DataResolver(
        catalog=catalog,
        sql_engine=sql_engine,
        evidence_store=evidence_store,
        query_catalog=q_catalog,
        benchmark_dir=benchmark_path,
    )

    try:
        df, diagnostics = resolver.resolve(spec)
    except Exception as exc:
        raise ToolError(f"Failed to resolve data for plot '{spec.id}': {exc}") from exc

    if df.is_empty():
        plot_id = spec.id or declaration or "plot"
        raise ToolError(
            f"Data source or query for plot '{plot_id}' returned 0 rows; cannot render plot with an empty dataset",
            hint="Verify filter conditions or run 'evaluation_query' to check row counts.",
        )

    plots_dir = benchmark_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    clean_stem = component(spec.id or declaration or "plot", "plot_name")
    out_file = plots_dir / f"{clean_stem}.{output_format}"

    if output_format == "html":
        plotly_renderer = PlotlyRenderer(units={})
        html_body = plotly_renderer.render(spec, df, run_dir=benchmark_path)
        if html_body is None:
            raise ToolError(f"Plotly renderer failed to produce output for '{spec.id}'")
        if isinstance(html_body, list):
            html_body = "\n".join(html_body)

        full_html = (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            f"<title>{spec.title or clean_stem}</title>\n"
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
            "</head>\n<body style='margin: 0; padding: 12px; font-family: sans-serif;'>\n"
            f"{html_body}\n"
            "</body>\n</html>\n"
        )
        out_file.write_text(full_html, encoding="utf-8")
    else:
        seaborn_renderer = SeabornRenderer(generate_gifs=False, units={})
        seaborn_renderer.render(spec, df, out_path=out_file, run_dir=benchmark_path)
        if not out_file.is_file() or out_file.stat().st_size == 0:
            raise ToolError(f"Static renderer failed to produce output at {out_file}")

    return {
        "plot_path": str(out_file),
        "format": output_format,
        "spec_id": spec.id or clean_stem,
        "diagnostics": diagnostics.to_dict(),
    }
