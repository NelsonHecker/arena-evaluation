import math

import polars as pl

from arena_evaluation.processing.metrics.performance.pedestrian_path_metrics import PedestrianPathMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
from flat_peds import flat_peds

S = 1_000_000_000


def _episode(times: list[int], frames: list[tuple[int, list[dict]]] | None = None) -> AlignedEpisodeBundle:
    """Robot samples at the given times, with the pedestrian table of the given frames."""
    data = pl.DataFrame({"time_ns": times}, schema={"time_ns": pl.Int64})
    if frames is None:
        return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[])
    peds = flat_peds(frames)
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[], peds=peds, topics={"pedestrian_tracks": peds})


def _paths(times: list[int], frames: list[tuple[int, list[dict]]]) -> list:
    return PedestrianPathMetricsCalculator(RobotParams()).calculate(_episode(times, frames), {})["pedestrian_path"]


def _is_nan_point(point: list) -> bool:
    return len(point) == 3 and all(math.isnan(v) for v in point)


def _at(x: float, y: float) -> dict:
    return {"x": x, "y": y}


def test_output_keys_and_unit():
    assert PedestrianPathMetricsCalculator.output_keys() == ["pedestrian_path"]
    assert PedestrianPathMetricsCalculator.UNITS == {"pedestrian_path": "m"}


def test_missing_frame_yields_no_paths():
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    assert PedestrianPathMetricsCalculator(RobotParams()).calculate(episode, {}) == {"pedestrian_path": []}


def test_robot_samples_without_a_pedestrian_table_yield_no_paths():
    assert PedestrianPathMetricsCalculator(RobotParams()).calculate(_episode([0, S]), {}) == {"pedestrian_path": []}


def test_a_pedestrian_table_without_rows_yields_no_paths():
    assert _paths([0, S], []) == []


def test_two_pedestrians_are_split_into_time_aligned_paths():
    frames = [(0, [{"ped_id": "a", **_at(0.0, 0.0)}, {"ped_id": "b", **_at(5.0, 5.0)}]), (S, [{"ped_id": "a", **_at(1.0, 0.0)}, {"ped_id": "b", **_at(5.0, 6.0)}])]

    assert _paths([0, S], frames) == [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[5.0, 5.0, 0.0], [5.0, 6.0, 0.0]],
    ]


def test_a_pedestrian_missing_from_a_frame_repeats_its_previous_position():
    frames = [
        (0, [{"ped_id": "a", **_at(0.0, 0.0)}, {"ped_id": "b", **_at(3.0, 3.0)}]),
        (S, [{"ped_id": "b", **_at(3.0, 3.0)}]),
        (2 * S, [{"ped_id": "a", **_at(2.0, 0.0)}, {"ped_id": "b", **_at(3.0, 4.0)}]),
    ]

    assert _paths([0, S, 2 * S], frames) == [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[3.0, 3.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0]],
    ]


def test_a_pedestrian_dropping_out_holds_its_last_position():
    frames = [
        (0, [{"ped_id": "a", **_at(5.0, 5.0)}, {"ped_id": "b", **_at(0.0, 0.0)}]),
        (S, [{"ped_id": "b", **_at(1.0, 0.0)}]),
        (2 * S, [{"ped_id": "b", **_at(2.0, 0.0)}]),
    ]

    assert _paths([0, S, 2 * S], frames) == [
        [[5.0, 5.0, 0.0], [5.0, 5.0, 0.0], [5.0, 5.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    ]


def test_leading_samples_before_the_first_frame_stay_nan():
    (path,) = _paths([0, S], [(S, [{"ped_id": "a", **_at(1.0, 2.0)}])])

    assert len(path) == 2
    assert _is_nan_point(path[0])
    assert path[1] == [1.0, 2.0, 0.0]


def test_a_pedestrian_appearing_late_is_nan_before_its_first_sample():
    frames = [(0, [{"ped_id": "a", **_at(0.0, 0.0)}]), (S, [{"ped_id": "a", **_at(1.0, 0.0)}, {"ped_id": "b", **_at(7.0, 7.0)}])]

    first, second = _paths([0, S], frames)

    assert first == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert _is_nan_point(second[0])
    assert second[1] == [7.0, 7.0, 0.0]


def test_a_pedestrian_with_nan_x_throughout_is_dropped():
    frames = [
        (0, [{"ped_id": "a", **_at(0.0, 0.0)}, {"ped_id": "b", **_at(float("nan"), 1.0)}]),
        (S, [{"ped_id": "a", **_at(1.0, 0.0)}, {"ped_id": "b", **_at(float("nan"), 2.0)}]),
    ]

    assert _paths([0, S], frames) == [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]


def test_a_single_sample_episode_yields_one_point_paths():
    frames = [(0, [{"ped_id": "a", "x": 4.0, "y": 5.0}, {"ped_id": "b", "x": 7.0, "y": 8.0}])]

    assert _paths([0], frames) == [[[4.0, 5.0, 0.0]], [[7.0, 8.0, 0.0]]]
