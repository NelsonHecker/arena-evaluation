"""Declared report data sources: aliases, Star Schema tables, canonical views and series views.

A report manifest names the table or view its plots read. ``metrics`` is the historic alias for
the canonical episode summary; a series-backed report names the catalog view of the 1NF table the
pipeline writes (``v_series_<stem>``). This module is the single declaration of those names so a
manifest that references a view nothing publishes fails at load time instead of rendering nothing.
"""

from __future__ import annotations

import functools
from typing import Final

from arena_evaluation.analytics.catalog import CANONICAL_VIEW_NAMES, SERIES_TABLE_PREFIX
from arena_evaluation.storage.star_schema import STAR_TABLE_NAMES

__all__ = [
    "DATA_SOURCE_ALIASES",
    "EPISODE_SUMMARY_VIEW",
    "data_source_table",
    "declared_data_sources",
    "declared_series_stems",
    "declared_view_names",
    "series_view_name",
    "validate_data_source",
]

EPISODE_SUMMARY_VIEW: Final[str] = "v_episode_summary"

# Historic manifest source names and the canonical view each reads.
DATA_SOURCE_ALIASES: Final[dict[str, str]] = {
    "metrics": EPISODE_SUMMARY_VIEW,
    "episode_summary": EPISODE_SUMMARY_VIEW,
}


def data_source_table(data_source: str | None) -> str:
    """Map a manifest data source to the Star Schema catalog table or view it reads."""
    if not data_source:
        return EPISODE_SUMMARY_VIEW
    return DATA_SOURCE_ALIASES.get(data_source, data_source)


def series_view_name(stem: str) -> str:
    """The catalog view name of one published 1NF series table."""
    return f"v_{SERIES_TABLE_PREFIX}{stem}"


@functools.lru_cache(maxsize=1)
def declared_series_stems() -> tuple[str, ...]:
    """Stems of the 1NF series tables the metric registry declares, the pipeline's series writer."""
    from arena_evaluation.processing.metrics.registry import MetricRegistry

    return MetricRegistry.series_table_stems()


@functools.lru_cache(maxsize=1)
def declared_view_names() -> frozenset[str]:
    """Every catalog view a manifest may name: the canonical analytical views and the series views."""
    return frozenset({*CANONICAL_VIEW_NAMES, *(series_view_name(stem) for stem in declared_series_stems())})


@functools.lru_cache(maxsize=1)
def declared_data_sources() -> frozenset[str]:
    """Every data source a manifest may declare: the aliases, the Star Schema tables and the views."""
    return frozenset({*DATA_SOURCE_ALIASES, *STAR_TABLE_NAMES, *declared_view_names()})


def validate_data_source(data_source: str) -> str:
    """The table or view a manifest names, a ValueError when it names an undeclared view.

    A view-shaped name (``v_*``) declares a catalog view, so it must resolve against the views the
    package builds; free-form names stay data-dependent and are resolved against the benchmark
    catalog at render time.
    """
    resolved = data_source_table(data_source)
    if (data_source.startswith("v_") or resolved.startswith("v_")) and resolved not in declared_view_names():
        available = ", ".join(sorted(declared_view_names()))
        raise ValueError(f"unknown report data source {data_source!r}: no declared catalog view {resolved!r}. Declared views: {available}")
    return resolved
