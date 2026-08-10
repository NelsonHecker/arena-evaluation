from __future__ import annotations


def split_planner_name(planner_name: str | None) -> tuple[str, str]:
    """Split the contestant/planner name into local_planner and inter_planner."""
    
    if not planner_name:
        return "unknown", "unknown"

    parts = str(planner_name).split("-")

    if len(parts) >= 3:
        return parts[1], "-".join(parts[2:])
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        return parts[0], "none"
