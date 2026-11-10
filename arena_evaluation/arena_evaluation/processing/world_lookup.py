"""The copy of each world that the installed arena_simulation_setup package ships."""

from __future__ import annotations

import pathlib


def installed_world_dir(map_name: str) -> pathlib.Path | None:
    """The installed copy of a world, or None when the arena_simulation_setup package cannot be found.

    The package is located through ament. PackageNotFoundError is a KeyError, and OSError is what ament raises when
    AMENT_PREFIX_PATH is unset.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        share = pathlib.Path(get_package_share_directory("arena_simulation_setup"))
    except (ImportError, OSError, KeyError):
        return None
    return share / "worlds" / map_name
