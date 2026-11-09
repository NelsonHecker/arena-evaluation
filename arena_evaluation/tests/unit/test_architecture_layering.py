from __future__ import annotations

import ast
import pathlib

import polars as pl

from arena_evaluation.storage.declarations import (
    CONTACT_FACT_SCHEMA,
    ENCOUNTER_FACT_SCHEMA,
    EPISODE_FACT_SCHEMA,
    FACT_SCHEMAS,
    FACT_TABLE_NAMES,
    PEDESTRIAN_FACT_SCHEMA,
    ROBOT_FACT_SCHEMA,
    MetricDeclaration,
    MetricScope,
    ScopedFactTables,
    coerce_scope,
)


def _collect_imports(file_path: pathlib.Path) -> list[str]:
    """Parse a Python source file and return all imported module paths."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_storage_layer_never_imports_processing_or_presentation() -> None:
    """Storage (Layer 3) must never import from processing (Layer 2) or presentation (Layer 5)."""
    storage_dir = (
        pathlib.Path(__file__).resolve().parents[2] / "arena_evaluation" / "storage"
    )
    assert storage_dir.is_dir(), f"Storage directory not found at {storage_dir}"

    forbidden_prefixes = (
        "arena_evaluation.processing",
        "arena_evaluation.presentation",
        "..processing",
        "..presentation",
    )

    violations: list[tuple[str, str]] = []
    for py_file in storage_dir.glob("*.py"):
        for imported in _collect_imports(py_file):
            if any(imported.startswith(prefix) for prefix in forbidden_prefixes):
                violations.append((py_file.name, imported))

    assert not violations, (
        f"Layering violation: storage modules import from higher layers: {violations}"
    )


def test_analytics_layer_never_imports_processing_or_presentation() -> None:
    """Analytics (Layer 4) must never import from processing (Layer 2) or presentation (Layer 5)."""
    analytics_dir = (
        pathlib.Path(__file__).resolve().parents[2] / "arena_evaluation" / "analytics"
    )
    assert analytics_dir.is_dir(), f"Analytics directory not found at {analytics_dir}"

    forbidden_prefixes = (
        "arena_evaluation.processing",
        "arena_evaluation.presentation",
        "..processing",
        "..presentation",
    )

    violations: list[tuple[str, str]] = []
    for py_file in analytics_dir.glob("*.py"):
        for imported in _collect_imports(py_file):
            if any(imported.startswith(prefix) for prefix in forbidden_prefixes):
                violations.append((py_file.name, imported))

    assert not violations, (
        f"Layering violation: analytics modules import from higher layers: {violations}"
    )


def test_metric_declarations_neutral_module() -> None:
    """Metric declarations and fact schemas must be accessible from storage.declarations."""
    decl = MetricDeclaration(
        name="detour_time_s",
        unit="s",
        scopes=(MetricScope.ENCOUNTER,),
        dtype=pl.Float64,
        description="Detour time during an encounter",
        direction="lower",
    )
    assert decl.name == "detour_time_s"
    assert decl.unit == "s"
    assert decl.scopes == (MetricScope.ENCOUNTER,)
    assert decl.direction == "lower"

    assert coerce_scope("episodes") == MetricScope.EPISODE
    assert coerce_scope("encounters") == MetricScope.ENCOUNTER
    assert coerce_scope(MetricScope.CONTACT) == MetricScope.CONTACT

    for name in ("episodes", "encounters", "contacts", "pedestrians", "robots"):
        assert name in FACT_TABLE_NAMES
        assert name in FACT_SCHEMAS
        assert isinstance(FACT_SCHEMAS[name], pl.Schema)

    assert "run_id" in EPISODE_FACT_SCHEMA
    assert "encounter_id" in ENCOUNTER_FACT_SCHEMA
    assert "contact_id" in CONTACT_FACT_SCHEMA
    assert "ped_id" in PEDESTRIAN_FACT_SCHEMA
    assert "robot_id" in ROBOT_FACT_SCHEMA

    empty_ep = pl.DataFrame(schema=EPISODE_FACT_SCHEMA)
    tables = ScopedFactTables(
        episodes=empty_ep,
        encounters=pl.DataFrame(schema=ENCOUNTER_FACT_SCHEMA),
        contacts=pl.DataFrame(schema=CONTACT_FACT_SCHEMA),
        pedestrians=pl.DataFrame(schema=PEDESTRIAN_FACT_SCHEMA),
        robots=pl.DataFrame(schema=ROBOT_FACT_SCHEMA),
    )
    assert len(tables.by_name()) == 5


if __name__ == "__main__":
    test_storage_layer_never_imports_processing_or_presentation()
    test_analytics_layer_never_imports_processing_or_presentation()
    test_metric_declarations_neutral_module()
    print("All layering tests passed!")
