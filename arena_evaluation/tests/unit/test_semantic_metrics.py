import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.ecological.semantic_metrics import SemanticInteractionMetricsCalculator


def _calc():
    return SemanticInteractionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))


def _episode(events_df, end_time_ns=6_000_000_000):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"time_ns": [0, end_time_ns]}),
        start_pos=[],
        goal_pos=[],
        semantic_events=events_df,
    )


def test_time_waiting_at_doors_counts_triggered_not_open():
    events_df = pl.DataFrame({
        "time_ns": [1_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000],
        "entity": ["env_0/door_1"] * 4,
        "kind": ["door"] * 4,
        "field": ["triggered", "state", "state", "triggered"],
        "current": ["True", "opening", "open", "False"],
    })

    results = _calc().calculate(_episode(events_df), {})

    # triggered+not-open holds [1s,3s) closed and [3s,4s) opening: 2s + 1s
    assert results["time_waiting_at_doors"] == pytest.approx(3.0)
    assert results["elevator_rides"] == 0


def test_time_waiting_at_doors_sums_across_doors():
    events_df = pl.DataFrame({
        "time_ns": [
            1_000_000_000, 2_000_000_000,
            1_000_000_000, 3_000_000_000,
        ],
        "entity": [
            "env_0/door_1", "env_0/door_1",
            "env_0/door_2", "env_0/door_2",
        ],
        "kind": ["door"] * 4,
        "field": ["triggered", "state", "triggered", "state"],
        "current": ["True", "open", "True", "open"],
    })

    results = _calc().calculate(_episode(events_df), {})

    # door_1: triggered [1s,2s) not open -> 1s. door_2: triggered [1s,3s) not open -> 2s.
    assert results["time_waiting_at_doors"] == pytest.approx(3.0)


def test_elevator_rides_requires_nonzero_occupants():
    events_df = pl.DataFrame({
        "time_ns": [
            1_000_000_000, 2_000_000_000, 3_000_000_000,
            4_000_000_000, 5_000_000_000,
        ],
        "entity": ["env_0/1_elevator"] * 5,
        "kind": ["elevator"] * 5,
        "field": ["occupants", "just_arrived", "just_arrived", "occupants", "just_arrived"],
        "current": ["1.0", "True", "False", "0.0", "True"],
    })

    results = _calc().calculate(_episode(events_df), {})

    # Only the first just_arrived=True edge has occupants > 0.
    assert results["elevator_rides"] == 1
    assert results["time_waiting_at_doors"] == pytest.approx(0.0)


def test_no_semantic_events_returns_zero_defaults():
    results = _calc().calculate(_episode(None), {})

    assert results["time_waiting_at_doors"] == 0.0
    assert results["elevator_rides"] == 0


def test_empty_semantic_events_returns_zero_defaults():
    events_df = pl.DataFrame({
        "time_ns": [],
        "entity": [],
        "kind": [],
        "field": [],
        "current": [],
    })

    results = _calc().calculate(_episode(events_df), {})

    assert results["time_waiting_at_doors"] == 0.0
    assert results["elevator_rides"] == 0
