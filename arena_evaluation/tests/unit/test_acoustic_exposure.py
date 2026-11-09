import dataclasses
import logging
import math
import pathlib
import uuid

import numpy as np
import polars as pl
import pytest
import yaml
from PIL import Image
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations
from arena_evaluation.processing.metrics.base import MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.ecological.acoustic_exposure import AcousticExposureCalculator
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, EpisodeDescriptor, RobotParams


def test_acoustics_free_space():
    grid = np.zeros((10, 10), dtype=np.uint8)
    att = compute_attenuations(grid, 1.0, 0.0, 0.0, np.array([3.0], dtype=np.float32), np.array([4.0], dtype=np.float32), wall_tl=47.0, mic_distance=1.0)
    assert np.isclose(att[0], 20.0 * np.log10(5.0), atol=0.1)


def test_acoustics_one_wall():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 2] = 255
    att = compute_attenuations(grid, 1.0, 0.0, 0.0, np.array([3.0], dtype=np.float32), np.array([4.0], dtype=np.float32), wall_tl=47.0, mic_distance=1.0)
    assert np.isclose(att[0], 20.0 * np.log10(5.0) + 47.0, atol=0.5)


def test_acoustics_pruning_dominance():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[0:7, 2] = 255
    att = compute_attenuations(grid, 1.0, 0.0, 3.0, np.array([4.0], dtype=np.float32), np.array([3.0], dtype=np.float32), wall_tl=47.0, mic_distance=1.0)
    assert att[0] < 40.0


def _field(grid, resolution, sx, sy, mic_distance, wall_tl=47.0):
    h, w = grid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tx = np.ascontiguousarray(xx.ravel().astype(np.float32))
    ty = np.ascontiguousarray(yy.ravel().astype(np.float32))
    return compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=wall_tl, mic_distance=mic_distance).reshape(h, w)


def test_receiver_inside_mic_distance_costs_the_floor():
    att = compute_attenuations(np.zeros((40, 40), dtype=np.uint8), 0.1, 20.0, 20.0, np.array([25.0, 23.0, 20.0], dtype=np.float32), np.array([20.0] * 3, dtype=np.float32), wall_tl=47.0, mic_distance=2.0)
    assert np.allclose(att, 20.0 * np.log10(2.0), atol=1e-4)


def test_wall_inside_mic_distance_adds_only_its_tl():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[:, 20] = 1
    att = compute_attenuations(grid, 0.1, 17.0, 20.0, np.array([23.0], dtype=np.float32), np.array([20.0], dtype=np.float32), wall_tl=47.0, mic_distance=1.0)
    assert np.isclose(att[0], 20.0 * np.log10(1.0) + 47.0, atol=1e-4)


def test_tied_costs_inside_floor_keep_shortest_paths_beyond_it():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[18:23, 21] = 1
    floored = _field(grid, 0.1, 20.0, 20.0, mic_distance=1.0, wall_tl=1000.0)
    reference = _field(grid, 0.1, 20.0, 20.0, mic_distance=1e-3, wall_tl=1000.0)
    beyond = reference > 20.0 * np.log10(1.05)
    beyond &= reference < 100.0
    assert beyond.sum() > 1000
    assert np.abs(floored - reference)[beyond].max() < 0.25


SECOND_NS = 1_000_000_000
NULLS = dict.fromkeys(AcousticExposureCalculator.output_keys())


def _att(distance_m: float) -> float:
    return 20.0 * math.log10(max(distance_m, 1.0))


def _leq(values: list[float]) -> float:
    return 10.0 * math.log10(sum(10.0 ** (value / 10.0) for value in values) / len(values))


def _white(size: int = 50) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


def _walled(size: int = 50, column: int = 25) -> np.ndarray:
    pixels = _white(size)
    pixels[:, column] = 0
    return pixels


@pytest.fixture
def calc(tmp_path: pathlib.Path) -> AcousticExposureCalculator:
    calculator = AcousticExposureCalculator(RobotParams())
    calculator.maps_cache = tmp_path / "maps_cache"
    calculator.maps_cache.mkdir()
    return calculator


@pytest.fixture
def cached_map(calc: AcousticExposureCalculator):
    cache = calc.maps_cache

    def make(pixels: np.ndarray | None = None, *, resolution: float = 0.1, origin: tuple[float, float, float] = (0.0, 0.0, 0.0), png_bytes: bytes | None = None, with_png_path: bool = True) -> str:
        name = f"acx-{uuid.uuid4().hex}"
        png = cache / f"{name}.png"
        meta_path = cache / f"{name}.yaml"
        if png_bytes is not None:
            png.write_bytes(png_bytes)
        else:
            Image.fromarray(_white() if pixels is None else pixels).save(png)
        meta = {"resolution": resolution, "origin": list(origin)}
        if with_png_path:
            meta["png_path"] = str(png)
        meta_path.write_text(yaml.dump(meta))
        return name

    return make


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def module_log():
    logger = logging.getLogger("arena_evaluation.processing.metrics.ecological.acoustic_exposure")
    handler = _Records()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler.messages
    logger.removeHandler(handler)
    logger.setLevel(level)


def _door_run(tmp_path: pathlib.Path, map_name: str) -> tuple[FolderManager, EpisodeDescriptor]:
    world_dir = tmp_path / "bench" / "worlds" / map_name
    world_dir.mkdir(parents=True)
    door = {"name": "front_door", "span": [[2.5, 2.0], [2.5, 3.0]], "width": 0.1}
    (world_dir / "world.yaml").write_text(yaml.dump({"format": "arena-world/2", "levels": [{"id": "0", "doors": [door]}]}))
    run = EpisodeDescriptor(episode_dir=str(tmp_path / "bench" / "ep"), benchmark_id="bench", episode_id=7, planner="p", stage="s", map=map_name)
    return FolderManager(tmp_path), run


def _semantic(rows: list[tuple[int, str, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [row[0] for row in rows],
            "kind": ["door"] * len(rows),
            "entity_name": [row[1] for row in rows],
            "field_name": ["open"] * len(rows),
            "value_bool": [row[2] for row in rows],
            "value_str": [None] * len(rows),
            "value_num": [None] * len(rows),
        },
        schema={"t_ns": pl.Int64, "kind": pl.Utf8, "entity_name": pl.Utf8, "field_name": pl.Utf8, "value_bool": pl.Boolean, "value_str": pl.Utf8, "value_num": pl.Float64},
    )


def _frames(robot: list[tuple[float | None, float | None]], peds: list[list[tuple[float, float]]], *, source: list[float | None] | None = None, times: list[int] | None = None) -> pl.DataFrame:
    columns = {
        "time_ns": pl.Series(times if times is not None else [index * SECOND_NS for index in range(len(robot))], dtype=pl.Int64),
        "pos_x_gt": pl.Series([position[0] for position in robot], dtype=pl.Float64),
        "pos_y_gt": pl.Series([position[1] for position in robot], dtype=pl.Float64),
        "peds_positions": pl.Series([[list(point) for point in frame] for frame in peds], dtype=pl.List(pl.List(pl.Float64))),
    }
    if source is not None:
        columns["total_level_af_dba"] = pl.Series(source, dtype=pl.Float64)
    return pl.DataFrame(columns)


def _episode(data: pl.DataFrame | None, map_name: str | None, **kwargs) -> AlignedEpisodeBundle:
    topics = dict(kwargs.pop("topics", {}) or {})
    if data is not None and "time_ns" in data.columns and "pedestrian_tracks" not in topics and "peds_positions" in data.columns:
        peds_rows = []
        seen = set()
        for stamp, frame in zip(data["time_ns"].to_list(), data["peds_positions"].to_list()):
            if frame:
                for index, point in enumerate(frame):
                    key = (int(stamp), f"ped_{index}")
                    if key not in seen and len(point) >= 2 and not math.isnan(point[0]) and not math.isnan(point[1]):
                        seen.add(key)
                        peds_rows.append({"t_ns": int(stamp), "ped_id": f"ped_{index}", "x": float(point[0]), "y": float(point[1])})
        schema = {"t_ns": pl.Int64, "ped_id": pl.Utf8, "x": pl.Float64, "y": pl.Float64}
        topics["pedestrian_tracks"] = pl.DataFrame(peds_rows, schema=schema) if peds_rows else pl.DataFrame(schema=schema)
    if data is not None and "time_ns" in data.columns and "robot_tracks" not in topics:
        tracks = pl.DataFrame(
            {
                "t_ns": data["time_ns"].cast(pl.Int64),
                "x": data["pos_x_gt"].cast(pl.Float64),
                "y": data["pos_y_gt"].cast(pl.Float64),
                "yaw": [0.0] * len(data),
            }
        ).drop_nulls(["x", "y"])
        topics["robot_tracks"] = tracks
    return AlignedEpisodeBundle(episode_id=7, data=data, start_pos=[0.0, 0.0], goal_pos=[0.0, 0.0], map=map_name, topics=topics, **kwargs)


def test_map_occupancy_marks_dark_pixels_and_flips_rows(calc, cached_map):
    pixels = np.full((3, 4), 255, dtype=np.uint8)
    pixels[0, 0] = 0
    pixels[1, 2] = 199
    pixels[2, 1] = 200
    name = cached_map(pixels, resolution=0.25, origin=(-1.0, 2.0, 0.0))
    grid, resolution, origin = calc._get_map_occupancy(name)
    assert grid.dtype == np.uint8
    assert grid.flags["C_CONTIGUOUS"]
    assert grid.tolist() == [[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    assert resolution == 0.25
    assert list(origin) == [-1.0, 2.0, 0.0]


def test_map_occupancy_is_none_for_unknown_map():
    assert AcousticExposureCalculator(RobotParams())._get_map_occupancy("unknown") is None


def test_map_occupancy_is_none_without_png_path(calc, cached_map):
    assert calc._get_map_occupancy(cached_map(with_png_path=False)) is None


def test_map_occupancy_is_none_for_unreadable_image(calc, cached_map, module_log):
    assert calc._get_map_occupancy(cached_map(png_bytes=b"not a png")) is None
    assert any(message.startswith("Failed to load map image for acoustics") for message in module_log)


def test_reference_run_yields_nulls(calc, cached_map):
    run = EpisodeDescriptor(episode_dir="ep", benchmark_id="bench", episode_id=7, planner="p", stage="s", is_reference=True)
    assert calc.calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), cached_map(), run=run), {}) == NULLS


@pytest.mark.parametrize("data", [None, pl.DataFrame()], ids=["none", "empty"])
def test_missing_episode_data_yields_nulls(data, calc, cached_map):
    assert calc.calculate(_episode(data, cached_map()), {}) == NULLS


def test_missing_map_name_yields_nulls():
    assert AcousticExposureCalculator(RobotParams()).calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), None), {}) == NULLS


def test_unloadable_map_yields_nulls():
    assert AcousticExposureCalculator(RobotParams()).calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), "unknown"), {}) == NULLS


@pytest.mark.parametrize("dropped", ["robot_tracks", "pedestrian_tracks"])
def test_missing_required_topic_yields_nulls(dropped, calc, cached_map):
    episode = _episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), cached_map())
    topics = {name: table for name, table in episode.topics.items() if name != dropped}
    episode = AlignedEpisodeBundle(episode_id=7, data=episode.data, start_pos=[0.0, 0.0], goal_pos=[0.0, 0.0], map=episode.map, topics=topics)
    assert calc.calculate(episode, {}) == NULLS


def test_world_fallback_supplies_the_map(calc, cached_map):
    calc.world = cached_map()
    result = calc.calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), None), {})
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)


def test_prior_map_overrides_episode_map(calc, cached_map):
    result = calc.calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), "unknown"), {"map": cached_map()})
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)


def test_scalars_follow_the_exposure_series(calc, cached_map):
    frames = _frames([(1.0, 1.0)] * 4, [[(4.0, 1.0)], [(4.0, 1.0)], [(4.0, 1.0), (1.0, 4.0)], [(4.0, 1.0), (1.0, 4.0)]], source=[42.0, 60.0, 60.0, 60.0])
    episode = _episode(frames, cached_map())
    result = calc.calculate(episode, {})
    table = calc.series_table(episode, run_id="run", robot_id="robot")
    assert table.schema == pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "ped_id": pl.String,
            "exposure_dba": pl.Float64,
            "attenuation_db": pl.Float64,
        }
    )
    values = table["exposure_dba"].drop_nulls()
    assert result["ped_max_exposure_dba"] == pytest.approx(float(values.max()))
    assert result["ped_leq_exposure_dba"] == pytest.approx(_leq(values.to_list()))
    # One row per (sample, reachable pedestrian).
    assert len(table) == 6
    assert result["ped_max_startle_factor"] == pytest.approx(18.0, abs=1e-3)
    worst = result["worst_case_acoustic_frame"]
    assert worst["robot_x"] == 1.0
    assert worst["robot_y"] == 1.0
    assert worst["source_dba"] == 60.0
    assert worst["pedestrians"] == [[4.0, 1.0]]
    assert worst["door_states"] == {}


def test_frames_without_ground_truth_produce_no_exposure(calc, cached_map):
    frames = _frames([(None, None), (0.0, 0.0), (0.0, 0.0)], [[], [(3.0, 4.0)], [(3.0, 4.0)]])
    episode = _episode(frames, cached_map())
    result = calc.calculate(episode, {})
    expected = 42.0 - _att(5.0)
    assert result["ped_max_exposure_dba"] == pytest.approx(expected, abs=1e-3)
    assert result["ped_leq_exposure_dba"] == pytest.approx(expected, abs=1e-3)
    assert result["ped_max_startle_factor"] == 0.0
    assert result["worst_case_acoustic_frame"]["robot_x"] == 0.0
    assert result["worst_case_acoustic_frame"]["robot_y"] == 0.0
    assert result["worst_case_acoustic_frame"]["pedestrians"] == [[3.0, 4.0]]
    table = calc.series_table(episode)
    assert table["t_ns"].unique().to_list() == [SECOND_NS, 2 * SECOND_NS]


def test_repeated_timestamp_contributes_no_startle(calc, cached_map):
    frames = _frames([(1.0, 2.0), (1.0, 1.0)], [[(4.0, 1.0)], [(4.0, 1.0)]], times=[0, 0])
    result = calc.calculate(_episode(frames, cached_map()), {})
    assert result["ped_max_startle_factor"] == 0.0
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)


def test_unreachable_pedestrians_are_dropped(calc, cached_map):
    frames = _frames([(1.0, 1.0), (1.0, 1.0)], [[(4.0, 1.0), (9.0, 9.0)], [(9.0, 9.0)]])
    episode = _episode(frames, cached_map())
    result = calc.calculate(episode, {})
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)
    assert result["worst_case_acoustic_frame"]["pedestrians"] == [[4.0, 1.0], [9.0, 9.0]]
    assert calc.series_table(episode)["ped_id"].unique().to_list() == ["ped_0"]


def test_robot_off_map_yields_nulls(calc, cached_map):
    assert calc.calculate(_episode(_frames([(9.0, 9.0), (9.0, 9.0)], [[(4.0, 1.0)], []]), cached_map()), {}) == NULLS


def test_static_scene_logs_progress_every_fifty_evaluations(calc, cached_map, module_log):
    episode = _episode(_frames([(1.0, 1.0)] * 50, [[(4.0, 1.0)]] * 50), cached_map())
    result = calc.calculate(episode, {})
    assert [message for message in module_log if "Evaluated" in message] == ["AcousticExposureCalculator: Evaluated 50 frames... (50/50 total frames processed)"]
    assert "AcousticExposureCalculator: Finished episode 7 with 2 unique solver evaluations." in module_log
    assert result["ped_max_startle_factor"] == 0.0
    assert len(calc.series_table(episode)) == 50


def test_door_opening_switches_attenuation_and_marks_worst_frame_open(tmp_path, calc, cached_map):
    name = cached_map(_walled())
    folder_manager, run = _door_run(tmp_path, name)
    frames = _frames([(1.0, 2.5)] * 3, [[(4.0, 2.5)]] * 3, source=[60.0] * 3)
    semantic = _semantic([(SECOND_NS, "env_0/front_door/0", True)])
    result = calc.calculate(_episode(frames, name, run=run, folder_manager=folder_manager, semantic_snapshot=semantic), {})
    closed, opened = _att(3.0) + 25.0, _att(3.0)
    assert result["ped_max_exposure_dba"] == pytest.approx(60.0 - opened, abs=1e-3)
    assert result["ped_leq_exposure_dba"] == pytest.approx(_leq([60.0 - closed, 60.0 - opened, 60.0 - opened]), abs=1e-3)
    assert result["ped_max_startle_factor"] == pytest.approx(25.0, abs=1e-3)
    assert result["worst_case_acoustic_frame"]["door_states"] == {"world/front_door": "open"}


def test_the_pipeline_run_dir_reads_the_doors_of_the_run_snapshot(tmp_path, calc, cached_map):
    name = cached_map(_walled())
    _door_run(tmp_path, name)
    result = calc.calculate(_episode(_frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0]), name, run_dir=tmp_path / "bench"), {})
    assert result["worst_case_acoustic_frame"]["door_states"] == {"world/front_door": "closed"}
    assert calc.series_table(_episode(_frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0]), name, run_dir=tmp_path / "bench"))["attenuation_db"][0] == pytest.approx(_att(3.0) + 25.0, abs=1e-3)


def test_doors_without_semantic_timeline_stay_closed(tmp_path, calc, cached_map):
    name = cached_map(_walled())
    folder_manager, run = _door_run(tmp_path, name)
    frames = _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0])
    episode = _episode(frames, name, run=run, folder_manager=folder_manager)
    assert calc.calculate(episode, {})["worst_case_acoustic_frame"]["door_states"] == {"world/front_door": "closed"}
    assert calc.series_table(episode)["attenuation_db"][0] == pytest.approx(_att(3.0) + 25.0, abs=1e-3)


def test_semantic_timeline_for_another_door_keeps_door_closed(tmp_path, calc, cached_map):
    name = cached_map(_walled())
    folder_manager, run = _door_run(tmp_path, name)
    semantic = _semantic([(0, "env_0/back_door/0", True)])
    frames = _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0])
    episode = _episode(frames, name, run=run, folder_manager=folder_manager, semantic_snapshot=semantic)
    assert calc.calculate(episode, {})["worst_case_acoustic_frame"]["door_states"] == {"world/front_door": "closed"}
    assert calc.series_table(episode)["attenuation_db"][0] == pytest.approx(_att(3.0) + 25.0, abs=1e-3)


@pytest.mark.parametrize("benchmark_id", [None, ""], ids=["no_run", "no_benchmark_id"])
def test_run_worlds_need_a_benchmark_id(benchmark_id, tmp_path, calc, cached_map):
    name = cached_map(_walled())
    folder_manager, _ = _door_run(tmp_path, name)
    run = None if benchmark_id is None else EpisodeDescriptor(episode_dir="ep", benchmark_id=benchmark_id, episode_id=7, planner="p", stage="s")
    frames = _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0])
    episode = _episode(frames, name, run=run, folder_manager=folder_manager)
    assert calc.series_table(episode)["attenuation_db"][0] == pytest.approx(_att(3.0) + 47.0, abs=1e-3)

# --- PEDESTRIAN scope --------------------------------------------------------

_PEDESTRIAN_RESULT_SCHEMA = {
    "ped_max_exposure_dba": pl.Float64,
    "ped_leq_exposure_dba": pl.Float64,
    "ped_max_startle_factor": pl.Float64,
}


def _pedestrian_slice(episode: AlignedEpisodeBundle, ped_id: str) -> ScopeSlice:
    topics = dict(episode.topics or {})
    topics["pedestrian_tracks"] = topics["pedestrian_tracks"].filter(pl.col("ped_id") == ped_id)
    window = dataclasses.replace(episode, data=pl.DataFrame(), topics=topics, fleet={})
    time_ns = episode.data["time_ns"].to_numpy()
    return ScopeSlice(
        scope=MetricScope.PEDESTRIAN,
        slice_id=ped_id,
        parent_id=None,
        t_start_ns=int(time_ns[0]),
        t_end_ns=int(time_ns[-1]),
        identity={},
        window_bundle=window,
        full_episode=episode,
    )


def _pedestrian_result_frame(result: dict) -> pl.DataFrame:
    return pl.DataFrame({key: [result[key]] for key in _PEDESTRIAN_RESULT_SCHEMA}, schema=_PEDESTRIAN_RESULT_SCHEMA)


def test_pedestrian_scope_reports_each_pedestrians_own_exposure(calc, cached_map):
    frames = _frames([(1.0, 1.0)] * 3, [[(4.0, 1.0), (3.0, 4.0)]] * 3)
    episode = _episode(frames, cached_map())
    near_slice = _pedestrian_slice(episode, "ped_0")
    far_slice = _pedestrian_slice(episode, "ped_1")
    near = calc.calculate_for_scope(near_slice.window_bundle, {}, near_slice)
    far = calc.calculate_for_scope(far_slice.window_bundle, {}, far_slice)

    near_exposure = 42.0 - _att(3.0)
    far_exposure = 42.0 - _att(math.hypot(2.0, 3.0))
    assert set(near) == set(AcousticExposureCalculator.output_keys_for_scope(MetricScope.PEDESTRIAN))
    assert_frame_equal(
        _pedestrian_result_frame(near),
        pl.DataFrame(
            {
                "ped_max_exposure_dba": [near_exposure],
                "ped_leq_exposure_dba": [near_exposure],
                "ped_max_startle_factor": [0.0],
            },
            schema=_PEDESTRIAN_RESULT_SCHEMA,
        ),
        check_exact=False,
        rel_tol=1e-9,
    )
    assert_frame_equal(
        _pedestrian_result_frame(far),
        pl.DataFrame(
            {
                "ped_max_exposure_dba": [far_exposure],
                "ped_leq_exposure_dba": [far_exposure],
                "ped_max_startle_factor": [0.0],
            },
            schema=_PEDESTRIAN_RESULT_SCHEMA,
        ),
        check_exact=False,
        rel_tol=1e-9,
    )
    episode_result = calc.calculate(episode, {})
    assert episode_result["ped_max_exposure_dba"] == pytest.approx(
        max(near["ped_max_exposure_dba"], far["ped_max_exposure_dba"])
    )
    assert episode_result["ped_leq_exposure_dba"] == pytest.approx(_leq([near_exposure, far_exposure]))


def test_pedestrian_startle_uses_only_that_pedestrians_own_series(calc, cached_map):
    frames = _frames(
        [(1.0, 1.0), (1.0, 1.0)],
        [[(4.0, 1.0), (3.0, 4.0)], [(2.0, 1.0), (3.0, 4.0)]],
        source=[60.0, 60.0],
    )
    episode = _episode(frames, cached_map())
    near_slice = _pedestrian_slice(episode, "ped_0")
    far_slice = _pedestrian_slice(episode, "ped_1")
    near = calc.calculate_for_scope(near_slice.window_bundle, {}, near_slice)
    far = calc.calculate_for_scope(far_slice.window_bundle, {}, far_slice)

    expected_near_startle = _att(3.0) - _att(1.0)
    assert near["ped_max_startle_factor"] == pytest.approx(expected_near_startle, abs=1e-3)
    assert far["ped_max_startle_factor"] == 0.0


def test_pedestrian_scope_without_a_map_yields_nulls():
    calculator = AcousticExposureCalculator(RobotParams())
    episode = _episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), None)
    pedestrian_slice = _pedestrian_slice(episode, "ped_0")
    result = calculator.calculate_for_scope(pedestrian_slice.window_bundle, {}, pedestrian_slice)
    assert all(value is None for value in result.values())


def test_pedestrian_scope_publishes_the_three_scalars():
    assert AcousticExposureCalculator.output_keys_for_scope(MetricScope.PEDESTRIAN) == (
        "ped_max_exposure_dba",
        "ped_leq_exposure_dba",
        "ped_max_startle_factor",
    )
