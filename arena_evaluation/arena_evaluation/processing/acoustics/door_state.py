"""Door state timeline constructed from SemanticSnapshot messages."""

from __future__ import annotations

import bisect

import numpy as np
import polars as pl

OPEN_PROGRESS_THRESHOLD = 0.5


class DoorStateTimeline:
    """Sorted (time, open-doors frozenset) lookup with binary search."""

    def __init__(self, times_ns: np.ndarray, open_sets: list[frozenset[str]]):
        self.times_ns = times_ns
        self.open_sets = open_sets

    @classmethod
    def from_semantic_frame(cls, semantic: pl.DataFrame | None) -> DoorStateTimeline | None:
        """Build the timeline from the flattened semantic snapshot table."""
        if semantic is None:
            return None
        if isinstance(semantic, pl.LazyFrame):
            semantic = semantic.collect()
        if len(semantic) == 0:
            return None

        df = semantic
        missing = {"t_ns", "entity_name", "kind", "field_name"} - set(df.columns)
        if missing:
            raise ValueError(f"semantic table lacks columns {sorted(missing)}; re-extract world_semantics for the episode (has {df.columns})")
        df = df.filter(pl.col("kind") == "door")
        if len(df) == 0:
            return None

        is_open_expr = (
            ((pl.col("field_name") == "open") & (pl.col("value_bool") == True))  # noqa: E712
            | ((pl.col("field_name") == "state") & (pl.col("value_str") == "open"))
            | ((pl.col("field_name") == "progress") & pl.col("value_num").is_not_nan() & (pl.col("value_num") > OPEN_PROGRESS_THRESHOLD))
        )
        all_timestamps = df.select("t_ns").unique().sort("t_ns")
        open_by_ts = df.filter(is_open_expr).group_by("t_ns").agg(pl.col("entity_name").unique().alias("open_doors"))
        timeline_df = all_timestamps.join(open_by_ts, on="t_ns", how="left").sort("t_ns")
        timestamps = timeline_df["t_ns"].to_numpy()
        open_sets = [frozenset(doors) if doors is not None else frozenset() for doors in timeline_df["open_doors"]]
        return cls(timestamps, open_sets)

    def open_doors_at(self, time_ns: int) -> frozenset[str]:
        """Backward-asof: door state at or before *time_ns* (empty before first)."""
        idx = bisect.bisect_right(self.times_ns, time_ns) - 1
        if idx < 0:
            return frozenset()
        return self.open_sets[idx]
