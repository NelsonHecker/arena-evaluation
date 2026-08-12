"""Declarative report-manifest resolution, mirroring the suite/contest pattern."""

from __future__ import annotations

import pathlib
import typing

import yaml

from .viz_manifest import VizManifest

if typing.TYPE_CHECKING:
    pass

MANIFESTS_SUBDIR = "configs/benchmark/manifests"


def is_inline(ref: str) -> bool:
    stripped = ref.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def share_dir() -> pathlib.Path | None:
    """Share directory of the arena_evaluation package, or None (no ROS install)."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return pathlib.Path(get_package_share_directory("arena_evaluation"))
    except Exception:
        return None


def source_tree_dir() -> pathlib.Path | None:
    """The package root in the source checkout."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "configs" / "benchmark" / "manifests"
        if cand.is_dir():
            return parent
    return None


def find_manifest_file(stem: str) -> pathlib.Path | None:
    """Resolve a manifest name to its YAML file (share dir → source tree)."""
    for base in (share_dir(), source_tree_dir()):
        if base is None:
            continue
        cand = base / MANIFESTS_SUBDIR / f"{stem}.yaml"
        if cand.is_file():
            return cand
    return None


def available_manifests() -> list[str]:
    """Sorted stems of all bundled manifests."""
    found: set[str] = set()
    for base in (share_dir(), source_tree_dir()):
        if base is None:
            continue
        d = base / MANIFESTS_SUBDIR
        if d.is_dir():
            found.update(p.stem for p in d.glob("*.yaml"))
    return sorted(found)


class ManifestNotFoundError(FileNotFoundError):
    """Raised when a named manifest cannot be resolved anywhere."""

    def __init__(self, name: str, message: str | None = None) -> None:
        self.name = name
        available = ", ".join(available_manifests()) or "(none bundled)"
        super().__init__(
            message
            or f"Report manifest '{name}' not found. Available: {available}. "
            f"Pass a name, a path to a YAML file, or inline {{...}} YAML."
        )


def _load_note_manifest(benchmark_dir: pathlib.Path) -> VizManifest | None:
    """Read the report_manifest.yaml note file written after a prior report."""
    note = benchmark_dir / "report_manifest.yaml"
    if not note.is_file():
        return None
    try:
        data = yaml.safe_load(note.read_text())
        name = (data or {}).get("name")
        if not name:
            return None
        return resolve_manifest(str(name), benchmark_dir, _allow_note=False)
    except Exception:
        return None


def resolve_manifest(
    ref: str | None,
    benchmark_dir: pathlib.Path | None = None,
    *,
    _allow_note: bool = True,
) -> VizManifest:
    """Resolve a manifest reference to a :class:`VizManifest`."""
    if ref is None:
        if benchmark_dir is not None:
            legacy = benchmark_dir / "viz_manifest.yaml"
            if legacy.is_file():
                return VizManifest.load(legacy)
            if _allow_note:
                noted = _load_note_manifest(benchmark_dir)
                if noted is not None:
                    return noted
        return VizManifest.load_default()

    ref = ref.strip()
    if is_inline(ref):
        data = yaml.safe_load(ref)
        if not isinstance(data, dict):
            raise ValueError(f"Inline manifest must be a YAML mapping, got {type(data).__name__}")
        return VizManifest.model_validate(data)

    p = pathlib.Path(ref)
    if p.exists():
        return VizManifest.load(p)

    p = find_manifest_file(ref.removesuffix(".yaml"))
    if p is None:
        raise ManifestNotFoundError(ref)
    return VizManifest.load(p)
