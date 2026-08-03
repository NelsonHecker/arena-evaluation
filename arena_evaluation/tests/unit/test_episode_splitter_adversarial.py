import polars as pl
import pytest
from arena_evaluation.storage.schemas import TopicBundle
from arena_evaluation.processing.topic_aligner import TopicAligner
from arena_evaluation.processing.episode_splitter import EpisodeSplitter

def test_overlapping_episodes():
    odom_df = pl.DataFrame({"time_ns": [10, 20, 30, 40, 50], "pos_x": [1, 2, 3, 4, 5]})
    record_df = pl.DataFrame({
        "time_ns": [10, 40, 20, 50],  # Overlapping start/end: Ep 1 starts at 10, Ep 1 ends at 40. Ep 2 starts at 20, Ep 2 ends at 50.
        "episode_id": [1, 1, 2, 2],
        "robots_params": ["", "", "", ""]
    })
    
    bundle = TopicBundle(odom=odom_df, episode_record=record_df)
    aligner = TopicAligner(tolerance_ns=5)
    splitter = EpisodeSplitter(aligner, min_episode_frames=1)
    
    episodes = splitter.split(bundle)
    assert len(episodes) == 2
    assert len(episodes[0].data) == 4
    assert len(episodes[1].data) == 4

def test_zero_duration_episode():
    odom_df = pl.DataFrame({"time_ns": [10, 20, 30], "pos_x": [1, 2, 3]})
    record_df = pl.DataFrame({
        "time_ns": [20, 20],  # Zero duration
        "episode_id": [1, 1],
        "robots_params": ["", ""]
    })
    
    bundle = TopicBundle(odom=odom_df, episode_record=record_df)
    aligner = TopicAligner(tolerance_ns=5)
    splitter = EpisodeSplitter(aligner, min_episode_frames=1)
    
    episodes = splitter.split(bundle)
    assert len(episodes) == 1
    assert len(episodes[0].data) == 1
    assert episodes[0].data["time_ns"][0] == 20

def test_backward_jumping_timestamps():
    odom_df = pl.DataFrame({"time_ns": [10, 20, 30], "pos_x": [1, 2, 3]})
    record_df = pl.DataFrame({
        "time_ns": [30, 10],  # End before start
        "episode_id": [1, 1],
        "robots_params": ["", ""]
    })
    
    bundle = TopicBundle(odom=odom_df, episode_record=record_df)
    aligner = TopicAligner(tolerance_ns=5)
    splitter = EpisodeSplitter(aligner, min_episode_frames=1)
    
    episodes = splitter.split(bundle)
    assert len(episodes) == 0  # Should be empty since start > end
