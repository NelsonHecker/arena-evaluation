from __future__ import annotations

import pathlib
import yaml

from .schemas import RunMetadata
from .exceptions import ManifestGenerationError


class MetadataWriter:
    """
    Helper for reading and writing metadata.yaml files.
    """
    
    @staticmethod
    def write(metadata: RunMetadata, dest: pathlib.Path) -> None:
        """Write RunMetadata to a YAML file."""
        try:
            data = metadata.model_dump(exclude_none=True)
            with open(dest, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            try:
                dest.chmod(0o666)
            except Exception:
                pass
        except Exception as e:
            raise ManifestGenerationError(f"Failed to write metadata to {dest}: {e}")

    @staticmethod
    def read(source: pathlib.Path) -> RunMetadata:
        """Read RunMetadata from a YAML file."""
        if not source.exists():
            raise ManifestGenerationError(f"Metadata file not found: {source}")
            
        try:
            with open(source, "r") as f:
                data = yaml.safe_load(f)
            return RunMetadata.model_validate(data)
        except Exception as e:
            raise ManifestGenerationError(f"Failed to read metadata from {source}: {e}")

    @staticmethod
    def update(source: pathlib.Path, **kwargs) -> RunMetadata:
        """Update existing metadata with new fields and save."""
        metadata = MetadataWriter.read(source)
        
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
            else:
                raise ManifestGenerationError(f"Invalid metadata field: {key}")
                
        MetadataWriter.write(metadata, source)
        return metadata
