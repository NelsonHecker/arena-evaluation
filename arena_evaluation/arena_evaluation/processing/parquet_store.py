from __future__ import annotations

import pathlib
import json
import polars as pl

from ..storage.schemas import RunMetadata, TopicBundle
from ..storage.exceptions import SchemaViolationError


class ParquetStore:
    """
    Reads and writes metric DataFrames to Parquet format, embedding metadata in the footer.
    """
    METADATA_KEY = "arena_evaluation_metadata"
    
    @staticmethod
    def write(df: pl.DataFrame, dest: pathlib.Path, metadata: RunMetadata | None = None) -> None:
        """
        Write a DataFrame to Parquet, optionally embedding RunMetadata as JSON.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # We need PyArrow to embed custom metadata
        table = df.to_arrow()
        
        if metadata is not None:
            # Add custom metadata to existing schema metadata
            meta_dict = table.schema.metadata or {}
            meta_dict[ParquetStore.METADATA_KEY.encode()] = json.dumps(
                metadata.model_dump(exclude_none=True)
            ).encode()
            
            # Replace schema with new metadata
            table = table.replace_schema_metadata(meta_dict)
            
        import pyarrow.parquet as pq
        pq.write_table(table, dest)

    @staticmethod
    def read(source: pathlib.Path) -> tuple[pl.DataFrame, dict | None]:
        """
        Read a Parquet file and extract its embedded metadata if any.
        Returns (DataFrame, metadata_dict).
        """
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
        return df, metadata_dict

    @staticmethod
    def combine(sources: list[pathlib.Path], dest: pathlib.Path) -> None:
        """
        Combine multiple metrics.parquet files into a single combined_metrics.parquet.
        """
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
            raise SchemaViolationError(f"Failed to combine parquet files due to schema mismatch: {e}")
            
        ParquetStore.write(combined, dest)


class TopicParquetStore:
    """
    Reads and writes a TopicBundle to individual Parquet files per topic.
    """
    @staticmethod
    def write(bundle: TopicBundle, dest_dir: pathlib.Path) -> None:
        """
        Write non-None DataFrames/LazyFrames in a TopicBundle to Parquet files using zstd compression.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate through the fields of the dataclass
        import dataclasses
        for field in dataclasses.fields(bundle):
            df = getattr(bundle, field.name)
            if df is not None:
                final_path = dest_dir / f"{field.name}.parquet"
                
                # If it is a LazyFrame, check if the file is already written
                if isinstance(df, pl.LazyFrame):
                    if final_path.exists():
                        # Already written by MCAPReader or cached
                        continue
                    df = df.collect()
                
                if not df.is_empty():
                    # Write to a temporary file first for atomic writes
                    temp_path = dest_dir / f"{field.name}.parquet.tmp"
                    df.write_parquet(temp_path, compression="zstd")
                    temp_path.rename(final_path)

    @staticmethod
    def read(source_dir: pathlib.Path) -> TopicBundle | None:
        """
        Read Parquet files from source_dir to reconstruct a TopicBundle.
        Returns None if no parquet files exist.
        """
        if not source_dir.exists() or not source_dir.is_dir():
            return None
            
        parquet_files = list(source_dir.glob("*.parquet"))
        if not parquet_files:
            return None
            
        kwargs = {}
        for p in parquet_files:
            topic_name = p.stem
            try:
                lf = pl.scan_parquet(p)
                if "time_ns" in lf.columns:
                    lf = lf.sort("time_ns")
                kwargs[topic_name] = lf
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to read extracted parquet {p}: {e}")
                
        if not kwargs:
            return None
            
        return TopicBundle(**kwargs)

