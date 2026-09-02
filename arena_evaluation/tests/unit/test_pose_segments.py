import numpy as np
import pytest

pytest.importorskip("polars")

from arena_evaluation.processing.pose_segments import teleport_jumps


def test_sparse_ground_truth_is_one_segment():
    # 1 m/s robot sampled every 1.5 s: 1.5 m steps, no teleport
    t = np.arange(0, 30, 1.5)
    x = t * 1.0
    y = np.zeros_like(x)
    assert teleport_jumps(x, y, (t * 1e9).astype(np.int64)).size == 0


def test_reset_teleport_is_a_jump():
    t = np.arange(0, 10, 0.05)
    x = t * 1.0
    x[100:] += 20.0  # robot restaged 20 m away within one sample
    y = np.zeros_like(x)
    assert list(teleport_jumps(x, y, (t * 1e9).astype(np.int64))) == [99]


def test_distance_rule_without_time():
    x = np.array([0.0, 0.2, 0.4, 3.0, 3.2])
    y = np.zeros_like(x)
    assert list(teleport_jumps(x, y)) == [2]
