from __future__ import annotations

import os
import pathlib
from ament_index_python.packages import get_package_share_directory

from .schemas import RunDescriptor
from .manifest import MetadataWriter


class FolderManager:
    """
    Manages the directory structure and paths for the Arena Evaluation Pipeline.
    Ensures all resolved paths stay within the designated data_root.
    """
    def __init__(self, data_root: pathlib.Path | None = None):
        if data_root is None:
            # Default to the data/ directory inside the arena_evaluation package
            self.data_root = pathlib.Path(
                get_package_share_directory("arena_evaluation")
            ) / "data"
        else:
            self.data_root = pathlib.Path(data_root).resolve()

        if not self.data_root.exists():
            self.data_root.mkdir(parents=True, exist_ok=True)

    def _safe_resolve(self, target: pathlib.Path) -> pathlib.Path:
        """Resolve path and ensure it's within data_root to prevent traversal."""
        resolved = target.resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError:
            raise ValueError(f"Path {resolved} is outside data_root {self.data_root}")
        return resolved

    def run_dir(self, benchmark_id: str, planner: str, stage: str) -> pathlib.Path:
        """
        Get the directory for a specific contestant + stage run.
        Format: data_root / benchmark_id / "recordings" / planner / stage /
        """
        target = self.data_root / benchmark_id / "recordings" / planner / stage
        return self._safe_resolve(target)

    def mcap_path(self, run_dir: pathlib.Path) -> pathlib.Path:
        """Get the path for the step-level MCAP file."""
        # rosbag2 pattern: run_dir/recording/<name>_0.mcap (or _1.mcap, ...)
        recording_subdir = run_dir / "recording"
        if recording_subdir.exists() and recording_subdir.is_dir():
            # Skip zero-byte files — rosbag2 creates the file even when nothing is recorded
            candidates = sorted(p for p in recording_subdir.glob("*.mcap") if p.stat().st_size > 0)
            if candidates:
                return self._safe_resolve(candidates[0])

        # Legacy / flat file fallback
        return self._safe_resolve(run_dir / "recording.mcap")
    
    def legacy_csv_dir(self, run_dir: pathlib.Path) -> pathlib.Path:
        """Get the path for the legacy CSV directory (if applicable)."""
        return self._safe_resolve(run_dir)

    def extracted_topics_path(self, run_dir: pathlib.Path) -> pathlib.Path:
        """Get the path for the extracted topic parquet files directory."""
        return self._safe_resolve(run_dir / "topics")

    def metrics_path(self, run_dir: pathlib.Path) -> pathlib.Path:
        """Get the path for the per-planner metrics Parquet file."""
        return self._safe_resolve(run_dir / "metrics.parquet")

    def combined_metrics_path(self, benchmark_id: str) -> pathlib.Path:
        """Get the path for the combined benchmark metrics Parquet file."""
        target = self.data_root / benchmark_id / "combined_metrics.parquet"
        return self._safe_resolve(target)

    def discover_runs(self, benchmark_id: str) -> list[RunDescriptor]:
        """
        Discover all valid runs within a benchmark directory.
        A valid run contains a metadata.yaml file.
        """
        benchmark_dir = self.data_root / benchmark_id
        if not benchmark_dir.exists() or not benchmark_dir.is_dir():
            return []

        recordings_dir = benchmark_dir / "recordings"
        if not recordings_dir.exists() or not recordings_dir.is_dir():
            recordings_dir = benchmark_dir

        runs = []
        for planner_dir in recordings_dir.iterdir():
            if not planner_dir.is_dir() or planner_dir.name == "plots":
                continue
            
            for stage_dir in planner_dir.iterdir():
                if not stage_dir.is_dir():
                    continue
                
                metadata_path = stage_dir / "metadata.yaml"
                if metadata_path.exists() and metadata_path.is_file():
                    runs.append(
                        RunDescriptor(
                            run_dir=str(stage_dir),
                            benchmark_id=benchmark_id,
                            planner=planner_dir.name,
                            stage=stage_dir.name,
                        )
                    )
        
        return runs
