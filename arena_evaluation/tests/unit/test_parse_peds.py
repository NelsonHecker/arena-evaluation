import numpy as np
import pytest

pytest.importorskip("polars")

from arena_evaluation.processing.metrics.base import BaseMetricCalculator


def test_flat_positions_reshape_with_a_float_hint():
    # After the asof join num_pedestrians is a float column, numpy refuses float shapes.
    arr = BaseMetricCalculator._parse_peds([1.0, 2.0, 0.0, 3.0, 4.0, 0.0], np.float64(2.0))
    assert arr.shape == (2, 3)
    assert arr[1, 0] == 3.0


def test_nan_hint_falls_back_to_the_flat_layout():
    arr = BaseMetricCalculator._parse_peds([1.0, 2.0, 0.0, 3.0, 4.0, 0.0], float("nan"))
    assert arr.shape[0] == 2


def test_missing_hint_still_parses():
    arr = BaseMetricCalculator._parse_peds([1.0, 2.0, 3.0, 4.0], None)
    assert arr.shape[0] == 2
