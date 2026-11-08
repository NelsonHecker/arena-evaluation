"""Content-Addressed Evidence Store for Benchmark OLAP Analytics (Layer 4).

Records analytical SQL results and statistical outputs with deterministic
``ev-<sha256[:12]>`` identifiers, authoritative Parquet index, and optional Parquet + JSON persistence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Final

import polars as pl

__all__ = [
    "EVIDENCE_INDEX_SCHEMA",
    "VALID_EVIDENCE_KINDS",
    "EvidenceRecord",
    "EvidenceStore",
    "compute_frame_hash",
    "compute_schema_signature",
]

VALID_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "sql",
        "derived_table",
        "slice",
        "bootstrap_ci",
        "paired_diff",
        "rank_stability",
        "what_if",
    }
)

EVIDENCE_INDEX_SCHEMA: Final[dict[str, pl.DataType]] = {
    "evidence_id": pl.String,
    "kind": pl.String,
    "operation": pl.String,
    "purpose": pl.String,
    "params_json": pl.String,
    "row_count": pl.Int64,
    "columns": pl.List(pl.String),
    "schema_signature": pl.String,
    "created_at_utc": pl.String,
    "created_at_ns": pl.Int64,
    "artifact_path": pl.String,
}


def _ns_to_iso_utc(timestamp_ns: int) -> str:
    """Format nanosecond timestamp as ISO-8601 UTC string."""
    dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC)
    return dt.isoformat()


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
    purpose: str = ""
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize record metadata to a JSON-compatible dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "operation": self.operation,
            "purpose": self.purpose,
            "params": dict(self.params),
            "params_json": _canonical_json(self.params),
            "row_count": self.row_count,
            "columns": list(self.columns),
            "schema_signature": self.schema_signature,
            "created_at_ns": self.created_at_ns,
            "created_at_utc": self.created_at_utc,
            "artifact_path": str(self.artifact_path) if self.artifact_path is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceRecord:
        """Deserialize an EvidenceRecord from a dictionary."""
        raw_artifact = payload.get("artifact_path")
        stamp_ns = int(payload["created_at_ns"])
        utc_str = str(payload.get("created_at_utc") or "")
        if not utc_str:
            utc_str = _ns_to_iso_utc(stamp_ns)
        return cls(
            evidence_id=str(payload["evidence_id"]),
            kind=str(payload["kind"]),
            operation=str(payload["operation"]),
            purpose=str(payload.get("purpose") or ""),
            params=dict(payload.get("params") or {}),
            row_count=int(payload["row_count"]),
            columns=tuple(str(c) for c in (payload.get("columns") or ())),
            schema_signature=str(payload["schema_signature"]),
            created_at_ns=stamp_ns,
            created_at_utc=utc_str,
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

    @property
    def index_path(self) -> pathlib.Path | None:
        """Path to the authoritative evidence_index.parquet file if directory backed."""
        if self.evidence_dir is None:
            return None
        return self.evidence_dir / "evidence_index.parquet"

    def record(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        *,
        kind: str,
        operation: str,
        purpose: str = "",
        params: Mapping[str, Any] | None = None,
        created_at_ns: int | None = None,
        created_at_utc: str | None = None,
    ) -> EvidenceRecord:
        """Record a DataFrame result and return its content-addressed ``EvidenceRecord``."""
        if kind not in VALID_EVIDENCE_KINDS:
            raise ValueError(f"Unsupported evidence kind {kind!r}. Valid kinds: {sorted(VALID_EVIDENCE_KINDS)}")
        if not operation or not operation.strip():
            raise ValueError("operation must be a non-empty string.")

        df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        if df.is_empty():
            raise ValueError("Cannot record empty evidence artifact (DataFrame has 0 rows).")
        normalized_params = dict(params or {})
        schema_sig = compute_schema_signature(df)
        data_hash = compute_frame_hash(df)

        fingerprint_payload: dict[str, Any] = {
            "kind": kind,
            "operation": operation.strip(),
            "params": normalized_params,
            "schema_signature": schema_sig,
            "data_hash": data_hash,
        }
        if purpose and purpose.strip():
            fingerprint_payload["purpose"] = purpose.strip()

        digest = hashlib.sha256(_canonical_json(fingerprint_payload).encode()).hexdigest()
        evidence_id = f"ev-{digest[:12]}"

        artifact_path: pathlib.Path | None = None
        if self.evidence_dir is not None:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.evidence_dir / f"{evidence_id}.parquet"
            df.write_parquet(artifact_path)

        stamp_ns = int(created_at_ns) if created_at_ns is not None else time.time_ns()
        iso_utc = str(created_at_utc) if created_at_utc is not None else _ns_to_iso_utc(stamp_ns)

        rec = EvidenceRecord(
            evidence_id=evidence_id,
            kind=kind,
            operation=operation.strip(),
            purpose=purpose.strip(),
            params=normalized_params,
            row_count=df.height,
            columns=tuple(df.schema.names()),
            schema_signature=schema_sig,
            created_at_ns=stamp_ns,
            created_at_utc=iso_utc,
            artifact_path=artifact_path,
        )

        if self.evidence_dir is not None:
            meta_path = self.evidence_dir / f"{evidence_id}.json"
            meta_path.write_text(json.dumps(rec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            self._upsert_index(rec)

        self._records[evidence_id] = rec
        self._frames[evidence_id] = df
        return rec

    def _upsert_index(self, record: EvidenceRecord) -> None:
        """Upsert a single record into the authoritative evidence_index.parquet file."""
        if self.index_path is None:
            return

        row = pl.DataFrame(
            {
                "evidence_id": [record.evidence_id],
                "kind": [record.kind],
                "operation": [record.operation],
                "purpose": [record.purpose],
                "params_json": [_canonical_json(record.params)],
                "row_count": [record.row_count],
                "columns": [list(record.columns)],
                "schema_signature": [record.schema_signature],
                "created_at_utc": [record.created_at_utc],
                "created_at_ns": [record.created_at_ns],
                "artifact_path": [str(record.artifact_path) if record.artifact_path is not None else ""],
            },
            schema=EVIDENCE_INDEX_SCHEMA,
        )

        if self.index_path.is_file():
            existing = pl.read_parquet(self.index_path)
            filtered = existing.filter(pl.col("evidence_id") != record.evidence_id)
            updated = pl.concat([filtered, row], how="vertical_relaxed")
        else:
            updated = row

        updated.write_parquet(self.index_path)

    def load_index(self) -> pl.DataFrame:
        """Load the evidence index as a DataFrame, or an empty index frame if none exists."""
        if self.index_path is not None and self.index_path.is_file():
            return pl.read_parquet(self.index_path)

        if not self._records:
            return pl.DataFrame(schema=EVIDENCE_INDEX_SCHEMA)

        rows = [
            {
                "evidence_id": r.evidence_id,
                "kind": r.kind,
                "operation": r.operation,
                "purpose": r.purpose,
                "params_json": _canonical_json(r.params),
                "row_count": r.row_count,
                "columns": list(r.columns),
                "schema_signature": r.schema_signature,
                "created_at_utc": r.created_at_utc,
                "created_at_ns": r.created_at_ns,
                "artifact_path": str(r.artifact_path) if r.artifact_path is not None else "",
            }
            for r in self._records.values()
        ]
        return pl.DataFrame(rows, schema=EVIDENCE_INDEX_SCHEMA)

    def scan_index(self) -> pl.LazyFrame:
        """Scan the evidence index as a LazyFrame."""
        if self.index_path is not None and self.index_path.is_file():
            return pl.scan_parquet(self.index_path)
        return self.load_index().lazy()

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
