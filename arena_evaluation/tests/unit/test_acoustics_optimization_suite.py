"""Unit tests for the spatial-indexed acoustic solver and exposure calculator."""

from __future__ import annotations

import logging
import math
import pathlib
from collections.abc import Iterator

import numpy as np
import polars as pl
import pytest
import yaml
from PIL import Image

from arena_evaluation.processing.acoustics.door_state import (
    ACOUSTIC_OPEN_PROGRESS_THRESHOLD,
    OPEN_PROGRESS_THRESHOLD,
    DoorStateTimeline,
)
from arena_evaluation.processing.acoustics.impedance_grid import (
    compute_acoustic_field,
    compute_attenuations,
)
from arena_evaluation.processing.metrics.ecological.acoustic_exposure import (
    AcousticExposureCalculator,
)
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import (
    AlignedEpisodeBundle,
    EpisodeDescriptor,
    RobotParams,
)


def _write_cached_map(
    cache_dir: pathlib.Path,
    map_name: str,
    *,
    size: int,
    resolution: float,
    wall_columns: tuple[int, ...] = (),
) -> None:
    """Write a map PNG and registry metadata file into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pixels = np.full((size, size), 254, dtype=np.uint8)
    for column in wall_columns:
        pixels[:, column] = 0
    png_path = cache_dir / f"{map_name}.png"
    Image.fromarray(pixels).save(png_path)
    meta = {
        "png_path": str(png_path),
        "resolution": resolution,
        "origin": [0.0, 0.0, 0.0],
        "width": size,
        "height": size,
    }
    (cache_dir / f"{map_name}.yaml").write_text(yaml.safe_dump(meta))


def _write_door_world(run_dir: pathlib.Path, map_name: str, *, door_name: str = "door_1") -> None:
    """Write a single-level world with a door in a wall at x = 2.5 m."""
    world_dir = run_dir / "worlds" / map_name
    world_dir.mkdir(parents=True)
    world = {
        "format": "arena-world/2",
        "levels": [
            {
                "id": "0",
                "zones": [{"name": "room", "polygon": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]]}],
                "walls": [{"path": [[2.5, 0.0], [2.5, 2.0]]}, {"path": [[2.5, 3.0], [2.5, 5.0]]}],
                "doors": [{"name": door_name, "span": [[2.5, 2.0], [2.5, 3.0]]}],
            }
        ],
    }
    (world_dir / "world.yaml").write_text(yaml.safe_dump(world, sort_keys=False))


def _make_calculator(
    tmp_path: pathlib.Path,
    map_name: str,
    *,
    size: int = 100,
    resolution: float = 0.1,
    wall_columns: tuple[int, ...] = (),
) -> AcousticExposureCalculator:
    """Build an AcousticExposureCalculator backed by a cached map PNG in tmp_path."""
    cache_dir = tmp_path / "maps_cache"
    _write_cached_map(cache_dir, map_name, size=size, resolution=resolution, wall_columns=wall_columns)
    calc = AcousticExposureCalculator(RobotParams(0.25, 0.0, 30.0))
    calc.world = map_name
    calc.maps_cache = cache_dir
    return calc


class _LogCollector(logging.Handler):
    """Collect formatted log messages emitted during a test."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def module_log() -> Iterator[list[str]]:
    """Capture INFO log messages from the acoustic_exposure module."""
    target_logger = logging.getLogger("arena_evaluation.processing.metrics.ecological.acoustic_exposure")
    handler = _LogCollector()
    prev_level = target_logger.level
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.INFO)
    try:
        yield handler.messages
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(prev_level)


class TestAcousticGridSolverParity:
    """Verify compute_acoustic_field matches compute_attenuations across obstacle layouts."""

    def test_empty_grid_parity(self) -> None:
        """In an unobstructed grid, field values match point queries."""
        h, w = 40, 40
        grid = np.zeros((h, w), dtype=np.uint8)
        res = 0.05
        sx, sy = 20.0, 20.0

        field = compute_acoustic_field(grid, res, sx, sy, wall_tl=47.0, mic_distance=1.0)
        assert field.shape == (h, w)

        test_pts = [
            (20.0, 20.0),
            (25.0, 20.0),
            (20.0, 15.0),
            (28.0, 28.0),
            (10.0, 32.0),
            (5.0, 5.0),
        ]
        tx = np.array([p[0] for p in test_pts], dtype=np.float32)
        ty = np.array([p[1] for p in test_pts], dtype=np.float32)

        point_atts = compute_attenuations(grid, res, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

        for i, (px, py) in enumerate(test_pts):
            field_val = field[int(py), int(px)]
            point_val = point_atts[i]
            assert np.isclose(field_val, point_val, atol=1e-4)

    def test_wall_and_detour_parity(self) -> None:
        """Field and point solvers match when sound travels through and around barriers."""
        h, w = 50, 50
        grid = np.zeros((h, w), dtype=np.uint8)
        grid[10:45, 25] = 1
        grid[28:32, 25] = 0
        res = 0.1
        sx, sy = 15.0, 25.0

        field = compute_acoustic_field(grid, res, sx, sy, wall_tl=47.0, mic_distance=1.0)

        test_pts = [
            (20.0, 25.0),
            (35.0, 30.0),
            (35.0, 15.0),
            (25.0, 20.0),
        ]
        tx = np.array([p[0] for p in test_pts], dtype=np.float32)
        ty = np.array([p[1] for p in test_pts], dtype=np.float32)

        point_atts = compute_attenuations(grid, res, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

        for i, (px, py) in enumerate(test_pts):
            field_val = field[int(py), int(px)]
            point_val = point_atts[i]
            assert np.isclose(field_val, point_val, atol=1e-4)

    def test_pixel_tl_doors_parity(self) -> None:
        """Field and point solvers match when given explicit per-pixel transmission loss maps."""
        h, w = 30, 30
        grid = np.zeros((h, w), dtype=np.uint8)
        pixel_tl = np.zeros((h, w), dtype=np.float32)

        grid[:, 15] = 1
        pixel_tl[:, 15] = 25.0
        pixel_tl[12:18, 15] = 0.0

        res = 0.2
        sx, sy = 5.0, 15.0

        field = compute_acoustic_field(grid, res, sx, sy, pixel_tl=pixel_tl)

        test_pts = [(22.0, 15.0), (22.0, 5.0)]
        tx = np.array([p[0] for p in test_pts], dtype=np.float32)
        ty = np.array([p[1] for p in test_pts], dtype=np.float32)

        point_atts = compute_attenuations(grid, res, sx, sy, tx, ty, pixel_tl=pixel_tl)

        for i, (px, py) in enumerate(test_pts):
            field_val = field[int(py), int(px)]
            point_val = point_atts[i]
            assert np.isclose(field_val, point_val, atol=1e-4)


class TestSparseEarlyExitCorrectness:
    """Verify that early exit in Dijkstra returns exact attenuation values."""

    def test_early_exit_matches_exhaustive(self) -> None:
        """Sparse queries yield identical values to the full field."""
        h, w = 60, 60
        grid = np.zeros((h, w), dtype=np.uint8)
        grid[20:40, 30] = 1
        res = 0.05
        sx, sy = 15.0, 30.0

        field = compute_acoustic_field(grid, res, sx, sy)

        tx = np.array([20.0, 25.0], dtype=np.float32)
        ty = np.array([30.0, 25.0], dtype=np.float32)

        sparse_atts = compute_attenuations(grid, res, sx, sy, tx, ty)

        assert np.isclose(sparse_atts[0], field[30, 20], atol=1e-4)
        assert np.isclose(sparse_atts[1], field[25, 25], atol=1e-4)

    def test_coincident_target_at_source(self) -> None:
        """Target at the source returns 20*log10(mic_distance)."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        res = 0.1
        mic = 1.0
        sx, sy = 10.0, 10.0

        tx = np.array([10.0], dtype=np.float32)
        ty = np.array([10.0], dtype=np.float32)

        att = compute_attenuations(grid, res, sx, sy, tx, ty, mic_distance=mic)
        expected = 20.0 * np.log10(mic)
        assert np.isclose(att[0], expected, atol=1e-4)

    def test_duplicate_target_pixels(self) -> None:
        """Multiple target indices pointing to the same pixel receive identical attenuations."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        res = 0.1
        sx, sy = 5.0, 5.0

        tx = np.array([12.0, 12.0, 12.0], dtype=np.float32)
        ty = np.array([8.0, 8.0, 8.0], dtype=np.float32)

        att = compute_attenuations(grid, res, sx, sy, tx, ty)
        assert len(att) == 3
        assert np.isclose(att[0], att[1], atol=1e-5)
        assert np.isclose(att[1], att[2], atol=1e-5)

    def test_out_of_bounds_targets(self) -> None:
        """Out-of-bounds targets return infinity without crashing."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        res = 0.1
        sx, sy = 10.0, 10.0

        tx = np.array([-5.0, 25.0, 10.0], dtype=np.float32)
        ty = np.array([10.0, 10.0, -2.0], dtype=np.float32)

        att = compute_attenuations(grid, res, sx, sy, tx, ty)
        assert np.isinf(att[0])
        assert np.isinf(att[1])
        assert np.isinf(att[2])

    def test_empty_target_array(self) -> None:
        """Zero targets return an empty array without error."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        tx = np.array([], dtype=np.float32)
        ty = np.array([], dtype=np.float32)

        att = compute_attenuations(grid, 0.1, 5.0, 5.0, tx, ty)
        assert len(att) == 0


class TestWallBudgetReachability:
    """Verify that barriers follow the TL map and the wall budget stays RAM-bounded."""

    def test_four_closed_walls_exceed_the_budget(self) -> None:
        """Four solid walls are past the four-barrier budget and report unreachable."""
        grid = np.zeros((30, 40), dtype=np.uint8)
        for x in (5, 10, 15, 20):
            grid[:, x] = 1
        tx = np.array([30.0], dtype=np.float32)
        ty = np.array([15.0], dtype=np.float32)

        att = compute_attenuations(grid, 0.1, 1.0, 15.0, tx, ty)
        assert np.isinf(att[0])

    def test_four_open_doorways_are_reachable(self) -> None:
        """A target behind four zero-TL doorway pixels is scored as open space."""
        grid = np.zeros((30, 40), dtype=np.uint8)
        for x in (5, 10, 15, 20):
            grid[:, x] = 1
        pixel_tl = np.zeros(grid.shape, dtype=np.float32)
        tx = np.array([30.0], dtype=np.float32)
        ty = np.array([15.0], dtype=np.float32)

        att = compute_attenuations(grid, 0.1, 1.0, 15.0, tx, ty, pixel_tl=pixel_tl)
        assert np.isfinite(att[0])
        assert np.isclose(att[0], 20.0 * np.log10(2.9), atol=0.2)


class TestWorkspaceMemorySafety:
    """Verify that the thread-local workspace resets cleanly across calls."""

    def test_alternating_source_positions(self) -> None:
        """Consecutive calls from opposite corners do not retain stale costs."""
        grid = np.zeros((30, 30), dtype=np.uint8)
        res = 0.1

        att1 = compute_attenuations(
            grid,
            res,
            0.0,
            0.0,
            np.array([5.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
        )
        att2 = compute_attenuations(
            grid,
            res,
            29.0,
            29.0,
            np.array([5.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
        )
        att3 = compute_attenuations(
            grid,
            res,
            0.0,
            0.0,
            np.array([5.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
        )

        assert att2[0] > att1[0]
        assert np.isclose(att1[0], att3[0], atol=1e-5)

    def test_grid_capacity_growth(self) -> None:
        """Workspace expands and contracts safely across grid dimensions."""
        res = 0.1
        g_small = np.zeros((15, 15), dtype=np.uint8)
        f_small = compute_acoustic_field(g_small, res, 5.0, 5.0)
        assert f_small.shape == (15, 15)

        g_large = np.zeros((80, 80), dtype=np.uint8)
        f_large = compute_acoustic_field(g_large, res, 40.0, 40.0)
        assert f_large.shape == (80, 80)

        g_med = np.zeros((40, 40), dtype=np.uint8)
        f_med = compute_acoustic_field(g_med, res, 20.0, 20.0)
        assert f_med.shape == (40, 40)

    def test_non_contiguous_inputs_are_accepted(self) -> None:
        """Strided array views with matching dtypes are converted to C-contiguous buffers."""
        base = np.zeros((20, 40), dtype=np.uint8)
        non_contig_grid = base[:, ::2]
        assert not non_contig_grid.flags["C_CONTIGUOUS"]

        field = compute_acoustic_field(non_contig_grid, 0.1, 0.0, 0.0)
        assert field.shape == (20, 20)
        assert np.isclose(field[5, 5], 20.0 * np.log10(1.0), atol=1e-4)


class TestProximityAdaptiveThresholding:
    """Verify adaptive displacement thresholds and caching in AcousticExposureCalculator."""

    @pytest.fixture
    def setup_calculator_env(self, tmp_path: pathlib.Path) -> AcousticExposureCalculator:
        """Create a calculator backed by a 100x100 free-space map."""
        return _make_calculator(tmp_path, "test_acoustic_map")

    def test_pedestrian_sub_threshold_movement_does_not_recompute(
        self,
        setup_calculator_env: AcousticExposureCalculator,
        module_log: list[str],
    ) -> None:
        """Sub-threshold pedestrian movement reuses cached attenuations without invoking the solver."""
        calc = setup_calculator_env
        df = pl.DataFrame(
            {
                "time_ns": [0, 100_000_000, 200_000_000],
                "pos_x_gt": [5.0, 5.0, 5.0],
                "pos_y_gt": [5.0, 5.0, 5.0],
                "total_level_af_dba": [70.0, 70.0, 70.0],
                "peds_positions": [
                    [[6.00, 5.0]],
                    [[6.02, 5.0]],
                    [[6.08, 5.0]],
                ],
            }
        )

        bundle = AlignedEpisodeBundle(
            episode_id="test_ep_stat_robot",
            data=df,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[10.0, 10.0, 0.0],
            map="test_acoustic_map",
        )

        res = calc.calculate(bundle, {"map": "test_acoustic_map"})

        assert "AcousticExposureCalculator: Finished episode test_ep_stat_robot with 2 unique solver evaluations." in module_log
        atts = [frame[0] for frame in res["timeseries_acoustic_attenuation_db"]]
        assert len(atts) == 3
        assert atts[0] == atts[1]

    def test_close_proximity_fine_granularity(
        self,
        setup_calculator_env: AcousticExposureCalculator,
        module_log: list[str],
    ) -> None:
        """Within 2 m of a pedestrian, robot displacement above 0.05 m triggers recomputation."""
        calc = setup_calculator_env
        df = pl.DataFrame(
            {
                "time_ns": [0, 100_000_000, 200_000_000],
                "pos_x_gt": [5.00, 5.03, 5.07],
                "pos_y_gt": [5.00, 5.00, 5.00],
                "total_level_af_dba": [70.0, 70.0, 70.0],
                "peds_positions": [
                    [[6.0, 5.0]],
                    [[6.0, 5.0]],
                    [[6.0, 5.0]],
                ],
            }
        )

        bundle = AlignedEpisodeBundle(
            episode_id="test_ep_near_tier",
            data=df,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[10.0, 10.0, 0.0],
            map="test_acoustic_map",
        )

        calc.calculate(bundle, {"map": "test_acoustic_map"})

        assert "AcousticExposureCalculator: Finished episode test_ep_near_tier with 2 unique solver evaluations." in module_log


class TestGranularDoorTransition:
    """Verify that door openings breach the acoustic seal at 15% progress."""

    def test_acoustic_progress_threshold_constant(self) -> None:
        """Acoustic seal breach threshold is 0.15 while general open threshold is 0.5."""
        assert ACOUSTIC_OPEN_PROGRESS_THRESHOLD == 0.15
        assert OPEN_PROGRESS_THRESHOLD == 0.5

    def test_door_breach_at_fifteen_percent(self) -> None:
        """Door with progress 0.16 is treated as open while progress 0.10 stays closed."""
        df_closed = pl.DataFrame(
            {
                "time_ns": [100],
                "kind": ["door"],
                "entity": ["door_a"],
                "field": ["progress"],
                "value_num": [0.10],
                "value_bool": [None],
                "value_str": [None],
            }
        )
        tl_closed = DoorStateTimeline.from_semantic_frame(df_closed, progress_threshold=ACOUSTIC_OPEN_PROGRESS_THRESHOLD)
        assert tl_closed is not None
        assert tl_closed.open_doors_at(100) == frozenset()

        df_open = pl.DataFrame(
            {
                "time_ns": [100],
                "kind": ["door"],
                "entity": ["door_a"],
                "field": ["progress"],
                "value_num": [0.16],
                "value_bool": [None],
                "value_str": [None],
            }
        )
        tl_open = DoorStateTimeline.from_semantic_frame(df_open, progress_threshold=ACOUSTIC_OPEN_PROGRESS_THRESHOLD)
        assert tl_open is not None
        assert tl_open.open_doors_at(100) == frozenset({"door_a"})

    def test_door_opening_triggers_field_update(
        self,
        tmp_path: pathlib.Path,
        module_log: list[str],
    ) -> None:
        """A door opening event triggers an acoustic field recomputation even when stationary."""
        map_name = "test_door_map"
        calc = _make_calculator(tmp_path, map_name, size=50, wall_columns=(25,))

        run_root = tmp_path / "run"
        _write_door_world(run_root / "bench", map_name)

        semantic_df = pl.DataFrame(
            {
                "time_ns": [0, 100_000_000],
                "kind": ["door", "door"],
                "entity": ["world/door_1", "world/door_1"],
                "field": ["progress", "progress"],
                "value_num": [0.05, 0.20],
                "value_bool": [None, None],
                "value_str": [None, None],
            }
        )

        telemetry_df = pl.DataFrame(
            {
                "time_ns": [0, 100_000_000],
                "pos_x_gt": [2.0, 2.0],
                "pos_y_gt": [2.5, 2.5],
                "total_level_af_dba": [70.0, 70.0],
                "peds_positions": [
                    [[3.5, 2.5]],
                    [[3.5, 2.5]],
                ],
            }
        )

        bundle = AlignedEpisodeBundle(
            episode_id="test_ep_door_transition",
            data=telemetry_df,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[10.0, 10.0, 0.0],
            semantic_snapshot=semantic_df,
            map=map_name,
            folder_manager=FolderManager(run_root),
            run=EpisodeDescriptor(
                episode_dir=str(run_root / "bench" / "ep"),
                benchmark_id="bench",
                episode_id=1,
                planner="p",
                stage="s",
                map=map_name,
            ),
        )

        res = calc.calculate(bundle, {"map": map_name})

        assert "AcousticExposureCalculator: Finished episode test_ep_door_transition with 2 unique solver evaluations." in module_log
        assert res["timeseries_acoustic_attenuation_db"][0][0] > res["timeseries_acoustic_attenuation_db"][1][0]

    def test_acoustic_exposure_skips_reference_episodes(self, module_log: list[str]) -> None:
        """Reference episodes return null metrics without running the solver."""
        calc = AcousticExposureCalculator(RobotParams(0.25, 0.0, 30.0))
        calc.world = "test_map"

        bundle_ref = AlignedEpisodeBundle(
            episode_id=99,
            data=pl.DataFrame({"pos_x": [1.0], "pos_y": [1.0]}),
            start_pos=[0.0, 0.0],
            goal_pos=[1.0, 1.0],
            is_reference=True,
        )

        res = calc.calculate(bundle_ref, {"map": "test_map"})
        assert "Skipping acoustic calculation for reference episode 99" in module_log
        assert res["ped_max_exposure_dba"] is None
        assert res["ped_leq_exposure_dba"] is None

    def test_per_pedestrian_displacement_decoupling(self, tmp_path: pathlib.Path) -> None:
        """Moving past the near threshold updates the close pedestrian while keeping the mid-tier one cached."""
        map_name = "test_decoupling_map"
        calc = _make_calculator(tmp_path, map_name, size=100, resolution=0.1)

        df = pl.DataFrame(
            {
                "time_ns": [100, 200],
                "pos_x_gt": [1.0, 1.1],
                "pos_y_gt": [1.0, 1.0],
                "total_level_af_dba": [70.0, 70.0],
                "peds_positions": [
                    [[2.5, 1.0], [1.0, 4.0]],
                    [[2.5, 1.0], [1.0, 4.2]],
                ],
            }
        )

        bundle = AlignedEpisodeBundle(
            episode_id="test_ep_decoupling",
            data=df,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[10.0, 10.0, 0.0],
            map=map_name,
        )

        res = calc.calculate(bundle, {"map": map_name})
        atts = res["timeseries_acoustic_attenuation_db"]
        assert atts[0][0] == pytest.approx(20.0 * math.log10(1.5), abs=1e-3)
        assert atts[0][1] == pytest.approx(20.0 * math.log10(3.0), abs=1e-3)
        assert atts[1][0] == pytest.approx(20.0 * math.log10(1.4), abs=1e-3)
        assert atts[1][1] == pytest.approx(20.0 * math.log10(3.0), abs=1e-3)


class TestExposureMetricCorrections:
    """Verify positive-only startle rate and duration-weighted Leq aggregation."""

    @staticmethod
    def _calculator(tmp_path: pathlib.Path) -> AcousticExposureCalculator:
        """Build an AcousticExposureCalculator over an all-free 100x100 map."""
        return _make_calculator(tmp_path, "test_corrections_map", size=100, resolution=0.1)

    @staticmethod
    def _bundle(time_ns: list[int], source_dba: list[float]) -> AlignedEpisodeBundle:
        """Build an episode bundle with a pedestrian 1 m from the stationary robot."""
        n = len(time_ns)
        df = pl.DataFrame(
            {
                "time_ns": time_ns,
                "pos_x_gt": [5.0] * n,
                "pos_y_gt": [5.0] * n,
                "total_level_af_dba": source_dba,
                "peds_positions": [[[6.0, 5.0]]] * n,
            }
        )
        return AlignedEpisodeBundle(
            episode_id=1,
            data=df,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[10.0, 10.0, 0.0],
            map="test_corrections_map",
        )

    def test_startle_factor_ignores_a_falling_exposure(self, tmp_path: pathlib.Path) -> None:
        """A monotonic decrease yields 0.0 rather than a negative startle rate."""
        calc = self._calculator(tmp_path)
        bundle = self._bundle(
            time_ns=[0, 100_000_000, 200_000_000],
            source_dba=[70.0, 60.0, 50.0],
        )
        res = calc.calculate(bundle, {"map": "test_corrections_map"})
        assert res["ped_max_startle_factor"] == 0.0

    def test_startle_factor_reports_the_largest_rise(self, tmp_path: pathlib.Path) -> None:
        """When a positive exposure rise is present, the largest positive rate is reported."""
        calc = self._calculator(tmp_path)
        bundle = self._bundle(
            time_ns=[0, 100_000_000, 200_000_000],
            source_dba=[50.0, 55.0, 40.0],
        )
        res = calc.calculate(bundle, {"map": "test_corrections_map"})
        assert res["ped_max_startle_factor"] == pytest.approx(50.0, abs=1e-3)

    def test_leq_is_duration_weighted(self, tmp_path: pathlib.Path) -> None:
        """Non-uniform frame spacing weights each frame's energy by its duration."""
        calc = self._calculator(tmp_path)
        bundle = self._bundle(
            time_ns=[0, 100_000_000, 1_100_000_000],
            source_dba=[40.0, 40.0, 70.0],
        )
        res = calc.calculate(bundle, {"map": "test_corrections_map"})

        energy = 10 ** (np.array([40.0, 40.0, 70.0]) / 10.0)
        weights = np.array([0.1, 1.0, 1.0])
        expected = 10.0 * np.log10(np.sum(weights * energy) / np.sum(weights))
        unweighted = 10.0 * np.log10(np.mean(energy))

        assert abs(expected - unweighted) > 1.0
        assert res["ped_leq_exposure_dba"] == pytest.approx(expected, abs=1e-3)
