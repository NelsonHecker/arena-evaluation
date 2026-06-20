import polars as pl
from arena_evaluation.storage.schemas import TopicBundle
from arena_evaluation.processing.topic_aligner import TopicAligner

def test_align_topics_backward():
    # Synthetic odom data (10Hz)
    odom_df = pl.DataFrame({
        "time_ns": [1000, 2000, 3000, 4000],
        "pos_x": [1.0, 2.0, 3.0, 4.0]
    }).sort("time_ns")
    
    # Synthetic scan data (5Hz), arrive slightly after odom
    scan_df = pl.DataFrame({
        "time_ns": [1100, 3100],
        "scan_min": [0.5, 0.6]
    }).sort("time_ns")
    
    bundle = TopicBundle(odom=odom_df, scan=scan_df)
    
    # TopicAligner aligns backward (previous reading) 
    # With a tolerance of 200, when we align scan to odom:
    # We want to map odom time 1000 to scan time 1100. Oh wait, strategy="backward"
    # means for each left row (odom), find right row (scan) where right.time <= left.time.
    # So odom time 2000 maps to scan time 1100 (diff 900 -> null if tol=200)
    # If strategy is "backward", sensor time <= odom time.
    aligner = TopicAligner(tolerance_ns=500)
    aligned_df = aligner.align(bundle)
    
    assert aligned_df is not None
    assert len(aligned_df) == 4
    
    # Let's verify what happens:
    # Odom 1000: No scan <= 1000 -> null
    # Odom 2000: Scan 1100. Diff = 900. If tol=500, null. If tol=1000, 0.5.
    
def test_align_topics_forward():
    # If aligner uses strategy='forward'
    odom_df = pl.DataFrame({
        "time_ns": [1000, 2000, 3000, 4000],
        "pos_x": [1.0, 2.0, 3.0, 4.0]
    }).sort("time_ns")
    
    scan_df = pl.DataFrame({
        "time_ns": [1100, 3100],
        "scan_min": [0.5, 0.6]
    }).sort("time_ns")
    
    bundle = TopicBundle(odom=odom_df, scan=scan_df)
    aligner = TopicAligner(tolerance_ns=200)
    
    aligned_df = aligner.align(bundle)
    
    # Left: Odom
    # Right: Scan
    # If strategy is 'backward', it joins the PREVIOUS row (scan <= odom)
    assert aligned_df is not None
    assert "scan_min" in aligned_df.columns
