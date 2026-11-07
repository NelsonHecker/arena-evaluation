from __future__ import annotations

import json
import logging
import pathlib

import polars as pl

from arena_evaluation.processing.mcap_reader import load_bundles
from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.schemas import RunMetadata, TopicBundle

_log = logging.getLogger(__name__)


def _without_empty_structs(value: object) -> object:
    """Empty dicts infer as zero-field structs, which Parquet cannot represent."""
    if isinstance(value, dict):
        if not value:
            return None
        return {k: _without_empty_structs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_empty_structs(v) for v in value]
    return value


def _writable_rows(rows: list[dict]) -> list[dict]:
    """Metric rows with empty dicts replaced by None, so an episode with no doors writes."""
    return [{k: _without_empty_structs(v) for k, v in row.items()} for row in rows]


class ParquetStore:
    """Reads and writes metric DataFrames to Parquet format with embedded metadata."""

    METADATA_KEY = "arena_evaluation_metadata"

    @staticmethod
    def write(df: pl.DataFrame, dest: pathlib.Path, metadata: RunMetadata | None = None) -> None:
        """Write DataFrame to Parquet, embedding optional RunMetadata."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        table = df.to_arrow()

        if metadata is not None:
            meta_dict = table.schema.metadata or {}
            meta_dict[ParquetStore.METADATA_KEY.encode()] = json.dumps(metadata.model_dump(exclude_none=True)).encode()
            table = table.replace_schema_metadata(meta_dict)

        import pyarrow.parquet as pq

        pq.write_table(table, dest)

    @staticmethod
    def write_rows(
        rows: list[dict],
        dest: pathlib.Path,
        *,
        schema_overrides: dict | None = None,
        metadata: RunMetadata | None = None,
    ) -> None:
        """Build a frame from metric rows and write it, dropping empty dicts on the way."""
        df = pl.DataFrame(_writable_rows(rows), schema_overrides=schema_overrides, infer_schema_length=None)
        ParquetStore.write(df, dest, metadata)

    @staticmethod
    def read(source: pathlib.Path) -> tuple[pl.DataFrame, dict | None]:
        """Read Parquet file and return (DataFrame, metadata_dict)."""
        if not source.exists():
            raise FileNotFoundError(f"Parquet file not found: {source}")

        import pyarrow.parquet as pq

        table = pq.read_table(source)

        metadata_dict = None
        if table.schema.metadata:
            meta_bytes = table.schema.metadata.get(ParquetStore.METADATA_KEY.encode())
            if meta_bytes:
                metadata_dict = json.loads(meta_bytes.decode())

        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)
        return df, metadata_dict

    @staticmethod
    def combine(sources: list[pathlib.Path], dest: pathlib.Path) -> None:
        """Combine multiple metrics.parquet files into a single combined file."""
        if not sources:
            return

        dfs = []
        for src in sources:
            try:
                df, _ = ParquetStore.read(src)
                dfs.append(df)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Failed to read {src} for combining: {e}")

        if not dfs:
            return

        try:
            combined = pl.concat(dfs, how="diagonal_relaxed")
        except Exception as e:
            raise SchemaViolationError(f"Failed to combine parquet files due to schema mismatch: {e}") from e

        ParquetStore.write(combined, dest)


class TopicParquetStore:
    """Caches one episode's flat telemetry tables, written by the MCAP reader, and loads them back as TopicBundles."""

    VERSION = 3
    VERSION_FILE = ".cache_version"

    @staticmethod
    def is_current(source_dir: pathlib.Path) -> bool:
        """True when source_dir holds a finished extraction written by this layout."""
        try:
            stamp = (source_dir / TopicParquetStore.VERSION_FILE).read_text().strip()
        except OSError:
            return False
        return stamp == str(TopicParquetStore.VERSION)

    @staticmethod
    def mark_complete(source_dir: pathlib.Path) -> None:
        """Stamp a finished extraction so later runs read it instead of extracting again."""
        (source_dir / TopicParquetStore.VERSION_FILE).write_text(f"{TopicParquetStore.VERSION}\n")

    @staticmethod
    def read(source_dir: pathlib.Path) -> dict[str, TopicBundle] | None:
        """TopicBundles over the flat tables of source_dir; None when it holds no robot or a table cannot be read."""
        if not source_dir.is_dir():
            return None
        try:
            return load_bundles(source_dir, {}) or None
        except (OSError, pl.exceptions.PolarsError) as error:
            _log.warning(f"{source_dir}: cached telemetry unreadable, extracting again: {error}")
            return None
