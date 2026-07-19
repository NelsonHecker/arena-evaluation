import polars as pl
from arena_evaluation.storage.schemas import TopicBundle
from arena_evaluation.processing.topic_aligner import TopicAligner
from arena_evaluation.processing.episode_splitter import EpisodeSplitter

def test_split_episodes():
    odom_df = pl.DataFrame({
        "time_ns": [10, 20, 30, 40, 50, 60, 70, 80],
        "pos_x": [1, 2, 3, 4, 5, 6, 7, 8]
    })
    
    record_df = pl.DataFrame({
        "time_ns": [15, 45],
        "episode_id": [1, 2],
        "robots_params": ["", ""]
    })
    
    bundle = TopicBundle(odom=odom_df, episode_record=record_df)
    
    aligner = TopicAligner(tolerance_ns=5)
    splitter = EpisodeSplitter(aligner, min_episode_frames=2)
    
    episodes = splitter.split(bundle)
    
    assert len(episodes) == 2
    
    ep1 = episodes[0]
    assert ep1.episode_id == 1
    # ep1 should be between time_ns 15 and 44
    assert len(ep1.data) == 3 # times 20, 30, 40
    
    ep2 = episodes[1]
    assert ep2.episode_id == 2
    # ep2 should be >= 45
    assert len(ep2.data) == 4 # times 50, 60, 70, 80


def test_env_offset_from_tf_static():
    from arena_evaluation.processing.episode_splitter import _env_offset

    tf_static = pl.DataFrame({
        "frame_id": ["map", "env_0/jackal/base_link"],
        "child_frame_id": ["env_0/map", "env_0/jackal/chassis_link"],
        "trans_x": [5.0, 0.0],
        "trans_y": [5.0, 0.0],
    })
    assert _env_offset(tf_static) == (5.0, 5.0)
    assert _env_offset(None) == (0.0, 0.0)
    assert _env_offset(tf_static.head(0)) == (0.0, 0.0)

    two_envs = pl.DataFrame({
        "frame_id": ["map", "map"],
        "child_frame_id": ["env_0/map", "env_1/map"],
        "trans_x": [5.0, 45.0],
        "trans_y": [5.0, 5.0],
    })
    assert _env_offset(two_envs) is None
