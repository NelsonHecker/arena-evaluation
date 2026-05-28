from __future__ import annotations

import polars as pl
import typing

if typing.TYPE_CHECKING:
    from ..storage.schemas import TopicBundle, AlignedEpisodeBundle


class TopicAligner:
    """
    Aligns multiple asynchronous topics onto a single time axis using ASOF joins.
    """
    def __init__(self, tolerance_ns: int = 100_000_000):
        # Default tolerance is 100ms
        self.tolerance_ns = tolerance_ns

    def align(
        self,
        bundle: TopicBundle,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> pl.DataFrame | None:
        """
        Align all available topics in the bundle onto the odom time axis.
        Optionally filter by start and end times (inclusive).
        
        Uses join_asof with strategy="backward" (match exact or previous within tolerance).
        """
        if bundle.odom is None or len(bundle.odom) == 0:
            return None

        # Base DataFrame is odom
        df = bundle.odom
        
        # Apply time window if provided
        if start_time_ns is not None:
            df = df.filter(pl.col("time_ns") >= start_time_ns)
        if end_time_ns is not None:
            df = df.filter(pl.col("time_ns") <= end_time_ns)
            
        if len(df) == 0:
            return None

        # Sort just to be absolutely sure for join_asof
        df = df.sort("time_ns")
        
        # Helper to join a secondary topic
        def join_topic(primary: pl.DataFrame, secondary: pl.DataFrame | None, prefix: str) -> pl.DataFrame:
            if secondary is None or len(secondary) == 0:
                return primary
                
            # Filter secondary to rough time bounds for performance
            sec_df = secondary
            if start_time_ns is not None:
                sec_df = sec_df.filter(pl.col("time_ns") >= start_time_ns - self.tolerance_ns)
            if end_time_ns is not None:
                sec_df = sec_df.filter(pl.col("time_ns") <= end_time_ns + self.tolerance_ns)
                
            if len(sec_df) == 0:
                return primary
                
            # Perform asof join
            return primary.join_asof(
                sec_df.sort("time_ns"),
                on="time_ns",
                strategy="backward",
                tolerance=self.tolerance_ns
            )

        # Join all available secondary topics
        df = join_topic(df, bundle.scan, "scan")
        df = join_topic(df, bundle.cmd_vel, "cmd")
        df = join_topic(df, bundle.joint_states, "joint")
        df = join_topic(df, bundle.peds, "peds")
        df = join_topic(df, bundle.collision_events, "col")

        return df
