from __future__ import annotations

import os
import pathlib

from ament_index_python.packages import get_package_share_directory


def benchmarks_root() -> pathlib.Path:
    """$ARENA_DATA_DIR/benchmarks, or the package share data dir."""
    env = os.environ.get("ARENA_DATA_DIR")
    if env:
        return pathlib.Path(env) / "benchmarks"
    return pathlib.Path(get_package_share_directory("arena_evaluation")) / "data"


def latest_benchmark(root: pathlib.Path | None = None) -> pathlib.Path | None:
    """Most recent benchmark dir under the benchmarks root, or None."""
    root = root or benchmarks_root()
    if not root.is_dir():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    return runs[0] if runs else None
