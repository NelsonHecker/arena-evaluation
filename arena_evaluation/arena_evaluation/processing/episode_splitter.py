from __future__ import annotations

import typing
import yaml
import polars as pl

from ..storage.schemas import TopicBundle, AlignedEpisodeBundle
from .topic_aligner import TopicAligner

class EpisodeSplitter:
    """
    Splits continuous topic data into discrete episodes using EpisodeRecord messages.
    """
    def __init__(self, aligner: TopicAligner, min_episode_frames: int = 5):
        self.aligner = aligner
        self.min_episode_frames = min_episode_frames

    def split(self, bundle: TopicBundle) -> list[AlignedEpisodeBundle]:
        """
        Split the raw bundle into aligned episodes.
        """
        if bundle.odom is None or len(bundle.odom) == 0:
            return []
            
        episodes = []
        
        # If there are no EpisodeRecords, treat the whole file as one episode
        if bundle.episode_record is None or len(bundle.episode_record) == 0:
            aligned_df = self.aligner.align(bundle)
            if aligned_df is not None and len(aligned_df) >= self.min_episode_frames:
                episodes.append(
                    AlignedEpisodeBundle(
                        episode_id=0,
                        data=aligned_df,
                        start_pos=[],
                        goal_pos=[],
                        num_pedestrians=self._estimate_peds(aligned_df)
                    )
                )
            return episodes

        record_df = bundle.episode_record
        
        # Group records by episode_id while preserving order
        # Usually, an episode has a start record and an end record.
        # We can detect this by seeing if the next record has the same episode_id.
        
        rows = list(record_df.iter_rows(named=True))
        
        i = 0
        while i < len(rows):
            row = rows[i]
            start_time = row["time_ns"]
            
            # Check if the next record is the end of this episode
            if i + 1 < len(rows) and rows[i+1]["episode_id"] == row["episode_id"]:
                # The next record is the 'end' record for the same episode
                end_time = rows[i+1]["time_ns"]
                i += 2 # Skip the end record for the next iteration
            else:
                # Only 1 record for this episode, or next record is a new episode.
                if i + 1 < len(rows):
                    end_time = rows[i + 1]["time_ns"] - 1
                else:
                    end_time = bundle.odom.select(pl.col("time_ns").max()).item()
                i += 1
                
            aligned_df = self.aligner.align(bundle, start_time, end_time)
            
            if aligned_df is None or len(aligned_df) < self.min_episode_frames:
                continue
                
            start_pos = []
            goal_pos = []
            
            try:
                params_str = row["robots_params"]
                if params_str:
                    params_dict = yaml.safe_load(params_str)
                    for robot_params in params_dict.values():
                        if "start" in robot_params:
                            start_pos = robot_params["start"]
                        if "goal" in robot_params:
                            goal_pos = robot_params["goal"]
                        break
            except Exception:
                pass
                
            episodes.append(
                AlignedEpisodeBundle(
                    episode_id=row["episode_id"],
                    data=aligned_df,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    num_pedestrians=self._estimate_peds(aligned_df)
                )
            )
            
        return episodes

    def _estimate_peds(self, df: pl.DataFrame) -> int:
        if "num_pedestrians" in df.columns:
            peds = df["num_pedestrians"].drop_nulls()
            if len(peds) > 0:
                return int(peds.max())
        return 0
