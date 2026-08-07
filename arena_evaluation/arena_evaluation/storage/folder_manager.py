from __future__ import annotations

import os
import pathlib
from ament_index_python.packages import get_package_share_directory

from .schemas import RunDescriptor, EpisodeDescriptor
from .manifest import MetadataWriter


class FolderManager:
    """
    Manages the directory structure and paths for the Arena Evaluation Pipeline.
    Ensures all resolved paths stay within the designated data_root.

    Supports two structures:
    - Legacy: data_root / benchmark_id / recordings / planner / stage /
    - New (flat): data_root / benchmark_id / episodes / episode_XXX /
    """
    def __init__(self, data_root: pathlib.Path | None = None):
        if data_root is None:
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

    # ── New flat-episode API ───────────────────────────────────────────────────

    def episodes_dir(self, benchmark_id: str) -> pathlib.Path:
        """Return the episodes/ root for a benchmark."""
        return self._safe_resolve(self.data_root / benchmark_id / "episodes")

    def episode_dir(self, benchmark_id: str, episode_id: int) -> pathlib.Path:
        """Return the directory for a specific episode."""
        return self._safe_resolve(
            self.data_root / benchmark_id / "episodes" / f"episode_{episode_id:03d}"
        )

    def discover_episodes(self, benchmark_id: str) -> list[EpisodeDescriptor]:
        """
        Discover all valid episodes within the flat episodes/ directory.
        A valid episode has a directory named episode_XXX/ containing episode_XXX.yaml.
        """
        eps_dir = self.data_root / benchmark_id / "episodes"
        if not eps_dir.exists() or not eps_dir.is_dir():
            return []

        episodes: list[EpisodeDescriptor] = []
        for ep_dir in sorted(eps_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            try:
                ep_id = int(ep_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue

            # Find the yaml sidecar: episode_XXX.yaml
            yaml_path = ep_dir / f"{ep_dir.name}.yaml"
            if not yaml_path.exists():
                # Fall back to legacy metadata.yaml if present
                yaml_path = ep_dir / "metadata.yaml"
            if not yaml_path.exists():
                continue

            try:
                meta = MetadataWriter.read(yaml_path)
                episodes.append(EpisodeDescriptor(
                    episode_dir=str(ep_dir),
                    benchmark_id=benchmark_id,
                    episode_id=ep_id,
                    planner=meta.planner,
                    stage=meta.stage,
                    map=meta.map,
                    is_reference=meta.is_reference,
                    reference_type=meta.reference_type,
                ))
            except Exception:
                continue

        return episodes

    def mcap_path_for_episode(self, episode_dir: pathlib.Path) -> pathlib.Path:
        """Get the MCAP file path for a given episode directory."""
        # Prefer the canonically named file (e.g. episode_001.mcap)
        canonical = episode_dir / f"{episode_dir.name}.mcap"
        if canonical.exists():
            return self._safe_resolve(canonical)
        # Fall back to any .mcap file present
        candidates = sorted(p for p in episode_dir.glob("*.mcap") if p.stat().st_size > 0)
        if candidates:
            return self._safe_resolve(candidates[0])
        return self._safe_resolve(canonical)  # return expected path even if missing

    def extracted_topics_path_for_episode(self, episode_dir: pathlib.Path) -> pathlib.Path:
        """Return topics/ directory for a given episode directory."""
        return self._safe_resolve(episode_dir / "topics")

    def combined_metrics_path(self, benchmark_id: str) -> pathlib.Path:
        """Get the path for the combined benchmark metrics Parquet file."""
        target = self.data_root / benchmark_id / "combined_metrics.parquet"
        return self._safe_resolve(target)

    def characterization_summary_path(self, benchmark_id: str) -> pathlib.Path:
        """Get the path for the open-loop characterization summary (per working point)."""
        return self._safe_resolve(self.data_root / benchmark_id / "characterization_summary.parquet")

    def characterization_samples_path(self, benchmark_id: str) -> pathlib.Path:
        """Get the path for the per-sample characterization data (Layer 5 plotting)."""
        return self._safe_resolve(self.data_root / benchmark_id / "characterization_samples.parquet")

    # ── Legacy API (kept for backward-compatibility) ───────────────────────────

    def run_dir(self, benchmark_id: str, planner: str, stage: str) -> pathlib.Path:
        """
        Get the directory for a specific contestant + stage run (legacy structure).
        Format: data_root / benchmark_id / "recordings" / planner / stage /
        """
        target = self.data_root / benchmark_id / "recordings" / planner / stage
        return self._safe_resolve(target)

    def mcap_path(self, run_dir: pathlib.Path) -> pathlib.Path:
        """Get the path for the step-level MCAP file or directory containing MCAPs (legacy)."""
        if run_dir.exists() and run_dir.is_dir():
            flat_mcaps = sorted(p for p in run_dir.glob("*.mcap") if p.stat().st_size > 0)
            if flat_mcaps:
                return self._safe_resolve(run_dir)
        recording_subdir = run_dir / "recording"
        if recording_subdir.exists() and recording_subdir.is_dir():
            candidates = sorted(p for p in recording_subdir.glob("*.mcap") if p.stat().st_size > 0)
            if candidates:
                return self._safe_resolve(candidates[0])

        return self._safe_resolve(run_dir / "recording.mcap")

    def extracted_topics_path(self, run_dir: pathlib.Path) -> pathlib.Path:
        """Get the path for the extracted topic parquet files directory (legacy)."""
        return self._safe_resolve(run_dir / "topics")

    def metrics_path(self, run_dir: pathlib.Path) -> pathlib.Path:
        """Get the path for the per-planner metrics Parquet file (legacy)."""
        return self._safe_resolve(run_dir / "metrics.parquet")

    def discover_runs(self, benchmark_id: str) -> list[RunDescriptor]:
        """
        Discover all valid runs within a benchmark directory (legacy nested structure).
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
            if not planner_dir.is_dir() or planner_dir.name in ("plots", "episodes"):
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
