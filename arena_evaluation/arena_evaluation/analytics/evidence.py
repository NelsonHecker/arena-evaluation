"""Content-Addressed Evidence Store for Benchmark OLAP Analytics (Layer 4).

Records analytical SQL results and statistical outputs with deterministic
``ev-<sha256[:12]>`` identifiers and optional Parquet + JSON persistence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import time
from collections.abc import Mapping
from typing import Any, Final

import polars as pl

__all__ = [
    "VALID_EVIDENCE_KINDS",
    "EvidenceRecord",
    "EvidenceStore",
    "compute_frame_hash",
    "compute_schema_signature",
]

VALID_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "sql",
        "bootstrap_ci",
        "paired_diff",
        "rank_stability",
        "what_if",
    }
)


def compute_schema_signature(frame: pl.DataFrame | pl.LazyFrame) -> str:
    """Compute a canonical string signature ``col1:Type1,col2:Type2,...`` for a frame schema."""
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return ",".join(f"{name}:{dtype}" for name, dtype in schema.items())


def compute_frame_hash(frame: pl.DataFrame) -> str:
    """Compute a deterministic SHA-256 hex digest of a Polars DataFrame's contents and schema."""
    hasher = hashlib.sha256()
    hasher.update(compute_schema_signature(frame).encode())
    hasher.update(f"|rows:{frame.height}|".encode())
    if frame.height > 0 and frame.width > 0:
        row_hashes = frame.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3).to_numpy()
        hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceRecord:
    """Immutable metadata record for a content-addressed analytical evidence artifact."""

    evidence_id: str
    kind: str
    operation: str
    params: Mapping[str, Any]
    row_count: int
    columns: tuple[str, ...]
    schema_signature: str
    created_at_ns: int
    artifact_path: pathlib.Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize record metadata to a JSON-compatible dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "operation": self.operation,
            "params": dict(self.params),
            "row_count": self.row_count,
            "columns": list(self.columns),
            "schema_signature": self.schema_signature,
            "created_at_ns": self.created_at_ns,
            "artifact_path": str(self.artifact_path) if self.artifact_path is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceRecord:
        """Deserialize an EvidenceRecord from a dictionary."""
        raw_artifact = payload.get("artifact_path")
        return cls(
            evidence_id=str(payload["evidence_id"]),
            kind=str(payload["kind"]),
            operation=str(payload["operation"]),
            params=dict(payload.get("params") or {}),
            row_count=int(payload["row_count"]),
            columns=tuple(str(c) for c in (payload.get("columns") or ())),
            schema_signature=str(payload["schema_signature"]),
            created_at_ns=int(payload["created_at_ns"]),
            artifact_path=pathlib.Path(str(raw_artifact)) if raw_artifact is not None else None,
        )


class EvidenceStore:
    """Content-addressed store for analytical query results and statistical evidence frames."""

    def __init__(self, evidence_dir: pathlib.Path | None = None) -> None:
        self.evidence_dir = evidence_dir
        if self.evidence_dir is not None:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, EvidenceRecord] = {}
        self._frames: dict[str, pl.DataFrame] = {}

    def record(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        *,
        kind: str,
        operation: str,
        params: Mapping[str, Any] | None = None,
        created_at_ns: int | None = None,
    ) -> EvidenceRecord:
        """Record a DataFrame result and return its content-addressed ``EvidenceRecord``."""
        if kind not in VALID_EVIDENCE_KINDS:
            raise ValueError(f"Unsupported evidence kind {kind!r}. Valid kinds: {sorted(VALID_EVIDENCE_KINDS)}")
        if not operation or not operation.strip():
            raise ValueError("operation must be a non-empty string.")

        df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        normalized_params = dict(params or {})
        schema_sig = compute_schema_signature(df)
        data_hash = compute_frame_hash(df)

        fingerprint_payload = {
            "kind": kind,
            "operation": operation.strip(),
            "params": normalized_params,
            "schema_signature": schema_sig,
            "data_hash": data_hash,
        }
        digest = hashlib.sha256(_canonical_json(fingerprint_payload).encode()).hexdigest()
        evidence_id = f"ev-{digest[:12]}"

        artifact_path: pathlib.Path | None = None
        if self.evidence_dir is not None:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.evidence_dir / f"{evidence_id}.parquet"
            df.write_parquet(artifact_path)

        stamp_ns = int(created_at_ns) if created_at_ns is not None else time.time_ns()
        rec = EvidenceRecord(
            evidence_id=evidence_id,
            kind=kind,
            operation=operation.strip(),
            params=normalized_params,
            row_count=df.height,
            columns=tuple(df.schema.names()),
            schema_signature=schema_sig,
            created_at_ns=stamp_ns,
            artifact_path=artifact_path,
        )

        if self.evidence_dir is not None:
            meta_path = self.evidence_dir / f"{evidence_id}.json"
            meta_path.write_text(json.dumps(rec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

        self._records[evidence_id] = rec
        self._frames[evidence_id] = df
        return rec

    def get(self, evidence_id: str) -> EvidenceRecord:
        """Retrieve an ``EvidenceRecord`` by ID from memory or ``evidence_dir``."""
        if evidence_id in self._records:
            return self._records[evidence_id]
        if self.evidence_dir is not None:
            meta_path = self.evidence_dir / f"{evidence_id}.json"
            if meta_path.is_file():
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                rec = EvidenceRecord.from_dict(payload)
                self._records[evidence_id] = rec
                return rec
        raise KeyError(f"Evidence record '{evidence_id}' not found.")

    def read_frame(self, evidence_id: str) -> pl.DataFrame:
        """Retrieve the materialized ``pl.DataFrame`` for ``evidence_id``."""
        if evidence_id in self._frames:
            return self._frames[evidence_id]
        if self.evidence_dir is not None:
            parquet_path = self.evidence_dir / f"{evidence_id}.parquet"
            if parquet_path.is_file():
                df = pl.read_parquet(parquet_path)
                self._frames[evidence_id] = df
                return df
        raise KeyError(f"Evidence frame '{evidence_id}' not found.")

    def list_records(self) -> tuple[EvidenceRecord, ...]:
        """Return all registered ``EvidenceRecord`` entries."""
        return tuple(self._records.values())
