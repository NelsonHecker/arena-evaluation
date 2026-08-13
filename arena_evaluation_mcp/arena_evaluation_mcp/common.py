from __future__ import annotations

import re

_COMPONENT_RE = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")

_TERMINAL_STEP_STATUSES = ("ok", "failed", "partial", "skipped")


def validate_path_component(name: str) -> str:
    """Validate a client-supplied name used as a single path component."""
    if not isinstance(name, str) or not _COMPONENT_RE.fullmatch(name):
        raise ValueError(f"invalid name {name!r}: must be a single path component of [A-Za-z0-9._-] without a leading dot")
    return name


def run_status(statuses: list) -> str:
    """Derive a benchmark run status from its per-step statuses."""
    if not statuses:
        return "unknown"
    if all(s in _TERMINAL_STEP_STATUSES for s in statuses):
        return "completed"
    return "in_progress"
