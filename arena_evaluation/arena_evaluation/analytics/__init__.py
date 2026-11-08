"""Polars OLAP Analytics Core (Layer 4).

Exposes :class:`BenchmarkCatalog` (:class:`StarCatalog`), :class:`PolarsSQLEngine`,
statistical procedures (:mod:`arena_evaluation.analytics.stats`), and
:class:`EvidenceStore`.
"""

from __future__ import annotations

from .catalog import (
    CANONICAL_VIEW_NAMES,
    TELEMETRY_TABLE_NAMES,
    TELEMETRY_TABLE_SCHEMAS,
    BenchmarkCatalog,
    StarCatalog,
    synthesize_star_schema_from_combined,
)
from .evidence import (
    EVIDENCE_INDEX_SCHEMA,
    VALID_EVIDENCE_KINDS,
    EvidenceRecord,
    EvidenceStore,
    compute_frame_hash,
    compute_schema_signature,
)
from .queries import (
    QueryCatalog,
    QueryDeclaration,
    format_sql_literal,
)
from .ranking import (
    CONTEXTUAL_STABILITY_SCHEMA,
    RANK_INTERVAL_SCHEMA,
    RANK_SPLITS_SCHEMA,
    RANK_STABILITY_SCHEMA,
    ContextualStabilityResult,
    RankInterval,
    RankStabilityResult,
    contextual_rank_stability,
    paired_cluster_bootstrap_rank_intervals,
    seed_split_rank_stability,
    split_discovery_verification_cohorts,
)
from .sql_engine import (
    FORBIDDEN_SQL_KEYWORDS,
    PolarsSQLEngine,
    SQLGuardError,
    normalize_sql_scientific_notation,
    validate_readonly_sql,
)
from .stats import (
    CLUSTER_BOOTSTRAP_SCHEMA,
    KENDALL_RANKINGS_SCHEMA,
    PAIRED_DIFF_PAIRS_SCHEMA,
    PAIRED_DIFF_SUMMARY_SCHEMA,
    RANK_SWAP_MATRIX_SCHEMA,
    KendallWResult,
    PairedDiffResult,
    cluster_bootstrap_ci,
    kendalls_w,
    paired_treatment_diff,
    rank_swap_matrix,
    split_discovery_verification_seeds,
)

__all__ = [
    "BenchmarkCatalog",
    "CANONICAL_VIEW_NAMES",
    "CLUSTER_BOOTSTRAP_SCHEMA",
    "CONTEXTUAL_STABILITY_SCHEMA",
    "ContextualStabilityResult",
    "EVIDENCE_INDEX_SCHEMA",
    "EvidenceRecord",
    "EvidenceStore",
    "FORBIDDEN_SQL_KEYWORDS",
    "KENDALL_RANKINGS_SCHEMA",
    "KendallWResult",
    "PAIRED_DIFF_PAIRS_SCHEMA",
    "PAIRED_DIFF_SUMMARY_SCHEMA",
    "PairedDiffResult",
    "PolarsSQLEngine",
    "QueryCatalog",
    "QueryDeclaration",
    "RANK_INTERVAL_SCHEMA",
    "RANK_SPLITS_SCHEMA",
    "RANK_STABILITY_SCHEMA",
    "RANK_SWAP_MATRIX_SCHEMA",
    "RankInterval",
    "RankStabilityResult",
    "SQLGuardError",
    "StarCatalog",
    "TELEMETRY_TABLE_NAMES",
    "TELEMETRY_TABLE_SCHEMAS",
    "VALID_EVIDENCE_KINDS",
    "cluster_bootstrap_ci",
    "compute_frame_hash",
    "compute_schema_signature",
    "contextual_rank_stability",
    "format_sql_literal",
    "kendalls_w",
    "normalize_sql_scientific_notation",
    "paired_cluster_bootstrap_rank_intervals",
    "paired_treatment_diff",
    "rank_swap_matrix",
    "seed_split_rank_stability",
    "split_discovery_verification_cohorts",
    "split_discovery_verification_seeds",
    "synthesize_star_schema_from_combined",
    "validate_readonly_sql",
]
