"""Polars OLAP Analytics Core (Layer 4).

Exposes :class:`BenchmarkCatalog`, :class:`PolarsSQLEngine`, statistical
procedures (:mod:`arena_evaluation.analytics.stats`), sensitivity sweeps
(:mod:`arena_evaluation.analytics.what_if`), and :class:`EvidenceStore`.
"""

from __future__ import annotations

from .catalog import CANONICAL_VIEW_NAMES, BenchmarkCatalog
from .evidence import (
    VALID_EVIDENCE_KINDS,
    EvidenceRecord,
    EvidenceStore,
    compute_frame_hash,
    compute_schema_signature,
)
from .sql_engine import (
    FORBIDDEN_SQL_KEYWORDS,
    PolarsSQLEngine,
    SQLGuardError,
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
from .what_if import (
    CLOSER_SWEEP_METRIC_SCHEMA,
    DEFAULT_CLEARANCE_THRESHOLDS_M,
    DEFAULT_DOMINANT_SHARE_THRESHOLDS,
    ENCOUNTER_SWEEP_METRIC_SCHEMA,
    sweep_closer_share_thresholds,
    sweep_encounter_clearance_thresholds,
)

__all__ = [
    "BenchmarkCatalog",
    "CANONICAL_VIEW_NAMES",
    "CLOSER_SWEEP_METRIC_SCHEMA",
    "CLUSTER_BOOTSTRAP_SCHEMA",
    "DEFAULT_CLEARANCE_THRESHOLDS_M",
    "DEFAULT_DOMINANT_SHARE_THRESHOLDS",
    "ENCOUNTER_SWEEP_METRIC_SCHEMA",
    "EvidenceRecord",
    "EvidenceStore",
    "FORBIDDEN_SQL_KEYWORDS",
    "KENDALL_RANKINGS_SCHEMA",
    "KendallWResult",
    "PAIRED_DIFF_PAIRS_SCHEMA",
    "PAIRED_DIFF_SUMMARY_SCHEMA",
    "PairedDiffResult",
    "PolarsSQLEngine",
    "RANK_SWAP_MATRIX_SCHEMA",
    "SQLGuardError",
    "VALID_EVIDENCE_KINDS",
    "cluster_bootstrap_ci",
    "compute_frame_hash",
    "compute_schema_signature",
    "kendalls_w",
    "paired_treatment_diff",
    "rank_swap_matrix",
    "split_discovery_verification_seeds",
    "sweep_closer_share_thresholds",
    "sweep_encounter_clearance_thresholds",
    "validate_readonly_sql",
]
