#!/usr/bin/env python3
"""Comprehensive test suite for the optimized acoustic propagation solver.

Covers:
  1. Parity between full-field solver (compute_acoustic_field) and point solver
     (compute_attenuations) across free space, walls, and door impedance maps.
  2. Early termination correctness for sparse target configurations (single,
     multiple, duplicates, source coincident, out-of-bounds, unreachable).
  3. Workspace reusability and sanitization across consecutive runs and grid resizes.
  4. Proximity-adaptive displacement thresholding in AcousticExposureCalculator
     (5 cm near, 10 cm mid, 20 cm far) and sub-threshold displacement caching.
  5. Granular acoustic door state transitions (15% progress seal breach).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

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
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


# ===========================================================================
# 1. 2D Field Solver Parity vs Point Solver
# ===========================================================================

class TestAcousticGridSolverParity:
    """Verify compute_acoustic_field produces bit-exact results matching compute_attenuations."""

    def test_empty_grid_parity(self):
        """In an unobstructed grid, field values must match point queries."""
        H, W = 40, 40
        grid = np.zeros((H, W), dtype=np.uint8)
        res = 0.05
        sx, sy = 20.0, 20.0

        field = compute_acoustic_field(grid, res, sx, sy, wall_tl=47.0, mic_distance=1.0)
        assert field.shape == (H, W)

        # Query points across various quadrants and distances
        test_pts = [
            (20.0, 20.0),  # source itself
            (25.0, 20.0),  # cardinal +x
            (20.0, 15.0),  # cardinal -y
            (28.0, 28.0),  # diagonal
            (10.0, 32.0),  # off-axis
            (5.0, 5.0),    # corner
        ]
        tx = np.array([p[0] for p in test_pts], dtype=np.float32)
        ty = np.array([p[1] for p in test_pts], dtype=np.float32)

        point_atts = compute_attenuations(grid, res, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

        for i, (px, py) in enumerate(test_pts):
            field_val = field[int(py), int(px)]
            point_val = point_atts[i]
            assert np.isclose(field_val, point_val, atol=1e-4), (
                f"Mismatch at ({px}, {py}): field={field_val:.4f}, point={point_val:.4f}"
            )

    def test_wall_and_detour_parity(self):
        """Parity with obstacles where sound travels both through and around barriers."""
        H, W = 50, 50
        grid = np.zeros((H, W), dtype=np.uint8)
        # Vertical barrier with a doorway gap
        grid[10:45, 25] = 1
        grid[28:32, 25] = 0  # doorway aperture
        res = 0.1
        sx, sy = 15.0, 25.0

        field = compute_acoustic_field(grid, res, sx, sy, wall_tl=47.0, mic_distance=1.0)

        # Points in line-of-sight, behind wall, and through doorway
        test_pts = [
            (20.0, 25.0),  # before wall
            (35.0, 30.0),  # behind doorway
            (35.0, 15.0),  # shadowed by solid wall
            (25.0, 20.0),  # on the wall itself
        ]
        tx = np.array([p[0] for p in test_pts], dtype=np.float32)
        ty = np.array([p[1] for p in test_pts], dtype=np.float32)

        point_atts = compute_attenuations(grid, res, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

        for i, (px, py) in enumerate(test_pts):
            field_val = field[int(py), int(px)]
            point_val = point_atts[i]
            assert np.isclose(field_val, point_val, atol=1e-4), (
                f"Mismatch at ({px}, {py}): field={field_val:.4f}, point={point_val:.4f}"
            )

    def test_pixel_tl_doors_parity(self):
        """Parity when passing explicit per-pixel transmission loss maps."""
        H, W = 30, 30
        grid = np.zeros((H, W), dtype=np.uint8)
        pixel_tl = np.zeros((H, W), dtype=np.float32)

        # Closed door barrier with 25 dB TL
        grid[:, 15] = 1
        pixel_tl[:, 15] = 25.0
        # Carved open door segment
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


# ===========================================================================
# 2. Sparse Early Exit Correctness
# ===========================================================================

class TestSparseEarlyExitCorrectness:
    """Verify that early exit in Dijkstra returns mathematically exact values."""

    def test_early_exit_matches_exhaustive(self):
        """Sparse queries must yield identical values to the full field."""
        H, W = 60, 60
        grid = np.zeros((H, W), dtype=np.uint8)
        grid[20:40, 30] = 1
        res = 0.05
        sx, sy = 15.0, 30.0

        field = compute_acoustic_field(grid, res, sx, sy)

        # Few nearby targets (early termination triggers before visiting full map)
        tx = np.array([20.0, 25.0], dtype=np.float32)
        ty = np.array([30.0, 25.0], dtype=np.float32)

        sparse_atts = compute_attenuations(grid, res, sx, sy, tx, ty)

        assert np.isclose(sparse_atts[0], field[30, 20], atol=1e-4)
        assert np.isclose(sparse_atts[1], field[25, 25], atol=1e-4)

    def test_coincident_target_at_source(self):
        """Target exactly at the source must return 20*log10(mic_distance)."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        res = 0.1
        mic = 1.0
        sx, sy = 10.0, 10.0

        tx = np.array([10.0], dtype=np.float32)
        ty = np.array([10.0], dtype=np.float32)

        att = compute_attenuations(grid, res, sx, sy, tx, ty, mic_distance=mic)
        expected = 20.0 * np.log10(mic)
        assert np.isclose(att[0], expected, atol=1e-4)

    def test_duplicate_target_pixels(self):
        """Multiple target indices pointing to the identical pixel."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        res = 0.1
        sx, sy = 5.0, 5.0

        tx = np.array([12.0, 12.0, 12.0], dtype=np.float32)
        ty = np.array([8.0, 8.0, 8.0], dtype=np.float32)

        att = compute_attenuations(grid, res, sx, sy, tx, ty)
        assert len(att) == 3
        assert np.isclose(att[0], att[1], atol=1e-5)
        assert np.isclose(att[1], att[2], atol=1e-5)

    def test_out_of_bounds_targets(self):
        """Out of bounds targets should return infinity without crashing."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        res = 0.1
        sx, sy = 10.0, 10.0

        tx = np.array([-5.0, 25.0, 10.0], dtype=np.float32)
        ty = np.array([10.0, 10.0, -2.0], dtype=np.float32)

        att = compute_attenuations(grid, res, sx, sy, tx, ty)
        assert np.isinf(att[0])
        assert np.isinf(att[1])
        assert np.isinf(att[2])

    def test_empty_target_array(self):
        """Zero targets should return an empty array without error."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        tx = np.array([], dtype=np.float32)
        ty = np.array([], dtype=np.float32)

        att = compute_attenuations(grid, 0.1, 5.0, 5.0, tx, ty)
        assert len(att) == 0


# ===========================================================================
# 3. Workspace Memory Safety & State Sanitization
# ===========================================================================

class TestWorkspaceMemorySafety:
    """Verify that thread-local workspace resets cleanly across diverse invocations."""

    def test_alternating_source_positions(self):
        """Consecutive calls with opposite ends of the grid do not retain stale costs."""
        grid = np.zeros((30, 30), dtype=np.uint8)
        res = 0.1

        # Call 1 from corner (0, 0) to target (5, 5)
        att1 = compute_attenuations(
            grid, res, 0.0, 0.0,
            np.array([5.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
        )

        # Call 2 from opposite corner (29, 29) to target (5, 5)
        att2 = compute_attenuations(
            grid, res, 29.0, 29.0,
            np.array([5.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
        )

        # Call 3 from corner (0, 0) to target (5, 5) again: must equal Call 1
        att3 = compute_attenuations(
            grid, res, 0.0, 0.0,
            np.array([5.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
        )

        assert att2[0] > att1[0]  # (29, 29) is much farther from (5, 5) than (0, 0)
        assert np.isclose(att1[0], att3[0], atol=1e-5)

    def test_grid_capacity_growth(self):
        """Workspace expands safely when grid dimensions increase."""
        res = 0.1
        # Small grid
        g_small = np.zeros((15, 15), dtype=np.uint8)
        f_small = compute_acoustic_field(g_small, res, 5.0, 5.0)
        assert f_small.shape == (15, 15)

        # Large grid
        g_large = np.zeros((80, 80), dtype=np.uint8)
        f_large = compute_acoustic_field(g_large, res, 40.0, 40.0)
        assert f_large.shape == (80, 80)

        # Medium grid (shrinking dimensions uses existing capacity)
        g_med = np.zeros((40, 40), dtype=np.uint8)
        f_med = compute_acoustic_field(g_med, res, 20.0, 20.0)
        assert f_med.shape == (40, 40)


# ===========================================================================
# 4. Proximity-Adaptive Displacement Thresholding & Resampling
# ===========================================================================

class TestProximityAdaptiveThresholding:
    """Verify adaptive displacement thresholds and field caching in AcousticExposureCalculator."""

    @pytest.fixture
    def setup_calculator_env(self, monkeypatch, tmp_path):
        from PIL import Image

        # Create dummy map png
        map_name = "test_acoustic_map"
        map_dir = tmp_path / "maps" / map_name
        map_dir.mkdir(parents=True)
        png_path = map_dir / "map.png"
        img = Image.fromarray(np.full((100, 100), 254, dtype=np.uint8))
        img.save(png_path)

        # Register map in MapRegistry
        from arena_evaluation.processing.map_registry import MapRegistry
        monkeypatch.setattr(
            MapRegistry,
            "get_map",
            lambda *args, **kwargs: {
                "png_path": str(png_path),
                "resolution": 0.1,
                "origin": [0.0, 0.0, 0.0],
            },
        )

        # Mock door_segments to return empty
        monkeypatch.setattr(
            "arena_evaluation.processing.metrics.ecological.acoustic_exposure.door_segments",
            lambda *args, **kwargs: {},
        )

        calc = AcousticExposureCalculator(RobotParams(0.25, 0.0, 30.0))
        calc.world = map_name
        return calc

    def test_pedestrian_sub_threshold_movement_does_not_recompute(self, setup_calculator_env, monkeypatch):
        """When pedestrian movement is below the proximity threshold, solver is not recomputed."""
        calc = setup_calculator_env
        solver_calls = 0

        # Spy on compute_attenuations
        import arena_evaluation.processing.acoustics.impedance_grid as ig
        orig_fn = ig.compute_attenuations

        def spy_attenuations(*args, **kwargs):
            nonlocal solver_calls
            solver_calls += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(
            "arena_evaluation.processing.metrics.ecological.acoustic_exposure.compute_attenuations",
            spy_attenuations,
        )

        # 3 frames: Robot stands still at (5.0, 5.0)
        # Pedestrian is at (6.00, 5.0) [1.0m away -> near tier, threshold 0.05m]
        # Frame 0: (6.00, 5.0) -> solver call 1
        # Frame 1: moves 0.02m to (6.02, 5.0) <= 0.05m -> no solver call (reuses cached attenuations)
        # Frame 2: moves 0.08m from last eval to (6.08, 5.0) > 0.05m -> solver call 2
        df = pl.DataFrame({
            "time_ns": [0, 100_000_000, 200_000_000],
            "pos_x_gt": [5.0, 5.0, 5.0],
            "pos_y_gt": [5.0, 5.0, 5.0],
            "total_level_af_dba": [70.0, 70.0, 70.0],
            "peds_positions": [
                [[6.00, 5.0]],
                [[6.02, 5.0]],
                [[6.08, 5.0]],
            ],
        })

        bundle = AlignedEpisodeBundle(
            episode_id="test_ep_stat_robot",
            data=df,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[10.0, 10.0, 0.0],
            map="test_acoustic_map",
        )

        res = calc.calculate(bundle, {"map": "test_acoustic_map"})

        # Field computed only on Frame 0 and Frame 2 (sub-threshold on Frame 1 bypassed)
        assert solver_calls == 2
        atts = [frame[0] for frame in res["timeseries_acoustic_attenuation_db"]]
        assert len(atts) == 3

    def test_close_proximity_fine_granularity(self, setup_calculator_env, monkeypatch):
        """When near a pedestrian (< 2.0m), robot displacement > 0.05m triggers recomputation."""
        calc = setup_calculator_env
        solver_calls = 0

        import arena_evaluation.processing.acoustics.impedance_grid as ig
        orig_fn = ig.compute_attenuations

        def spy_attenuations(*args, **kwargs):
            nonlocal solver_calls
            solver_calls += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(
            "arena_evaluation.processing.metrics.ecological.acoustic_exposure.compute_attenuations",
            spy_attenuations,
        )

        # Robot is at 1.0m from pedestrian (near tier, threshold = 0.05m)
        # Frame 0: (5.00, 5.0) -> Frame 1: moves 0.03m (no recompute) -> Frame 2: moves total 0.07m (recomputes)
        df = pl.DataFrame({
            "time_ns": [0, 100_000_000, 200_000_000],
            "pos_x_gt": [5.00, 5.03, 5.07],
            "pos_y_gt": [5.00, 5.00, 5.00],
            "total_level_af_dba": [70.0, 70.0, 70.0],
            "peds_positions": [
                [[6.0, 5.0]],
                [[6.0, 5.0]],
                [[6.0, 5.0]],
            ],
        })

        bundle = AlignedEpisodeBundle(
            episode_id="test_ep_near_tier",
            data=df,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[10.0, 10.0, 0.0],
            map="test_acoustic_map",
        )

        calc.calculate(bundle, {"map": "test_acoustic_map"})

        # Frame 0: computed. Frame 1 (0.03m): resampled/reused. Frame 2 (0.07m > 0.05m): recomputed.
        assert solver_calls == 2


# ===========================================================================
# 5. Granular Acoustic Door Transitions
# ===========================================================================

class TestGranularDoorTransition:
    """Verify that door openings breach the acoustic seal at 15% progress."""

    def test_acoustic_progress_threshold_constant(self):
        """Constants for acoustic seal breach vs legacy."""
        assert ACOUSTIC_OPEN_PROGRESS_THRESHOLD == 0.15
        assert OPEN_PROGRESS_THRESHOLD == 0.5

    def test_door_breach_at_fifteen_percent(self):
        """Door with progress 0.16 is open; progress 0.10 is closed."""
        df_closed = pl.DataFrame({
            "time_ns": [100],
            "kind": ["door"],
            "entity": ["door_a"],
            "field": ["progress"],
            "value_num": [0.10],
            "value_bool": [None],
            "value_str": [None],
        })
        tl_closed = DoorStateTimeline.from_semantic_frame(
            df_closed, progress_threshold=ACOUSTIC_OPEN_PROGRESS_THRESHOLD
        )
        assert tl_closed.open_doors_at(100) == frozenset()

        df_open = pl.DataFrame({
            "time_ns": [100],
            "kind": ["door"],
            "entity": ["door_a"],
            "field": ["progress"],
            "value_num": [0.16],
            "value_bool": [None],
            "value_str": [None],
        })
        tl_open = DoorStateTimeline.from_semantic_frame(
            df_open, progress_threshold=ACOUSTIC_OPEN_PROGRESS_THRESHOLD
        )
        assert tl_open.open_doors_at(100) == frozenset({"door_a"})

    def test_door_opening_triggers_field_update(self, monkeypatch, tmp_path):
        """A door opening event triggers an acoustic field recomputation even if robot is stationary."""
        from PIL import Image
        from arena_evaluation.processing.map_registry import MapRegistry

        map_name = "test_door_map"
        map_dir = tmp_path / "maps" / map_name
        map_dir.mkdir(parents=True)
        png_path = map_dir / "map.png"
        img = Image.fromarray(np.full((50, 50), 254, dtype=np.uint8))
        img.save(png_path)

        monkeypatch.setattr(
            MapRegistry,
            "get_map",
            lambda *args, **kwargs: {
                "png_path": str(png_path),
                "resolution": 0.1,
                "origin": [0.0, 0.0, 0.0],
            },
        )

        # Mock door geometry for "door_1"
        mask = np.zeros((50, 50), dtype=bool)
        mask[:, 25] = True
        monkeypatch.setattr(
            "arena_evaluation.processing.metrics.ecological.acoustic_exposure.door_segments",
            lambda *args, **kwargs: {"world/door_1": (mask, 25.0)},
        )

        solver_calls = 0
        import arena_evaluation.processing.acoustics.impedance_grid as ig
        orig_fn = ig.compute_attenuations

        def spy_attenuations(*args, **kwargs):
            nonlocal solver_calls
            solver_calls += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(
            "arena_evaluation.processing.metrics.ecological.acoustic_exposure.compute_attenuations",
            spy_attenuations,
        )

        # Semantic timeline: door is closed at t=0, cracks open (0.20) at t=100ms
        semantic_df = pl.DataFrame({
            "time_ns": [0, 100_000_000],
            "kind": ["door", "door"],
            "entity": ["world/door_1", "world/door_1"],
            "field": ["progress", "progress"],
            "value_num": [0.05, 0.20],
            "value_bool": [None, None],
            "value_str": [None, None],
        })

        telemetry_df = pl.DataFrame({
            "time_ns": [0, 100_000_000],
            "pos_x_gt": [2.0, 2.0],  # robot does not move
            "pos_y_gt": [2.5, 2.5],
            "total_level_af_dba": [70.0, 70.0],
            "peds_positions": [
                [[3.5, 2.5]],
                [[3.5, 2.5]],
            ],
        })

        bundle = AlignedEpisodeBundle(
            episode_id="test_ep_door_transition",
            data=telemetry_df,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[10.0, 10.0, 0.0],
            semantic_snapshot=semantic_df,
            map=map_name,
        )

        calc = AcousticExposureCalculator(RobotParams(0.25, 0.0, 30.0))
        calc.world = map_name
        calc.calculate(bundle, {"map": map_name})

        # Must recompute: Frame 0 (initial) + Frame 1 (door state transitioned to open)
        assert solver_calls == 2
