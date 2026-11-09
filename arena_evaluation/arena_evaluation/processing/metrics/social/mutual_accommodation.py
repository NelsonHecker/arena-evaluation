"""Mutual Accommodation Ratio and reference-twin metrics.

MAR, PFI, the robot detour and the kinematic cost of social compliance are defined against a
second episode (the unobstructed-robot and unhindered-pedestrian reference runs), so they are
reconciled across episodes by the benchmark pipeline (:meth:`MutualAccommodationCalculator.
reconcile_stage_references`), not inside one episode's scope hierarchy. The EPISODE scope
therefore returns null for every key it cannot know on its own, and the pipeline fills them
from the reference rows.
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

logger = logging.getLogger(__name__)

DTW_MAX_POINTS: Final[int] = 300
DEGENERATE_SEGMENT_M2: Final[float] = 1e-12
MIN_ROBOT_DEVIATION_M: Final[float] = 0.05
MIN_PATH_DEVIATION_M: Final[float] = 0.01


def _point_matrix(points: object) -> np.ndarray | None:
    """(N, 2) float matrix of a trajectory, None when it is not a rectangular planar point container."""
    if points is None:
        return None
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 2 or array.shape[0] == 0:
        return None
    return array[:, :2]


def _stack_paths(paths: object) -> np.ndarray | None:
    """Concatenated (N, 2) points of a recorded pedestrian path list, None when the paths are not uniform planar samples."""
    if not isinstance(paths, list) or not paths:
        return None
    matrices: list[np.ndarray] = []
    for path in paths:
        if not path:
            continue
        try:
            matrix = np.asarray(path, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            return None
        matrices.append(matrix)
    if not matrices or len({matrix.shape[1] for matrix in matrices}) != 1:
        return None
    return np.concatenate(matrices, axis=0)[:, :2]


class MutualAccommodationCalculator(BaseMetricCalculator):
    """Mutual Accommodation Ratio, Pedestrian Flow Interference and reference-twin cost metrics."""

    NAME = "mutual_accommodation"
    CATEGORY = "social"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["proxemics_extended", "path_metrics", "energy_extended"]
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_tracks"]

    UNITS = {
        "mar": "",
        "pfi": "m",
        "robot_path_deviation_m": "m",
        "relative_throughput": "",
        "kcsc_wh": "Wh",
    }

    PRIMARY_OUTPUTS = ["mar", "pfi", "relative_throughput", "kcsc_wh"]
    OUTPUT_DIRECTIONS = {
        "mar": "lower",
        "pfi": "lower",
        "robot_path_deviation_m": "lower",
        "relative_throughput": "higher",
        "kcsc_wh": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "mar",
            "pfi",
            "robot_path_deviation_m",
            "relative_throughput",
            "kcsc_wh",
        ]

    @staticmethod
    def compute_orthogonal_deviation(
        points: object, reference_polyline: object
    ) -> tuple[float, float]:
        """(max, mean) orthogonal deviation of planar points from a reference polyline, 0 when either is unusable."""
        point_matrix = _point_matrix(points)
        reference_matrix = _point_matrix(reference_polyline)
        if point_matrix is None or reference_matrix is None or reference_matrix.shape[0] < 2:
            return 0.0, 0.0

        segment_start = reference_matrix[:-1]
        segment_delta = reference_matrix[1:] - segment_start
        segment_length_m2 = np.einsum("ij,ij->i", segment_delta, segment_delta)
        usable = segment_length_m2 > DEGENERATE_SEGMENT_M2
        if not np.any(usable):
            distances_m = np.linalg.norm(
                point_matrix[:, None, :] - segment_start[None, 0, :], axis=2
            )
        else:
            start = segment_start[usable]
            delta = segment_delta[usable]
            length_m2 = segment_length_m2[usable]
            offset = point_matrix[:, None, :] - start[None, :, :]
            projection = np.clip(
                np.einsum("nmk,mk->nm", offset, delta) / length_m2, 0.0, 1.0
            )
            projected = start[None, :, :] + projection[..., None] * delta[None, :, :]
            distances_m = np.linalg.norm(point_matrix[:, None, :] - projected, axis=2)
        nearest_m = distances_m.min(axis=1)
        return float(np.max(nearest_m)), float(np.mean(nearest_m))

    @staticmethod
    def compute_pfi_dtw(actual_paths: object, reference_paths: object) -> float | None:
        """Pedestrian Flow Interference: mean normalised dynamic-time-warping distance of the paths."""
        actual = actual_paths if isinstance(actual_paths, Sequence) else ()
        reference = reference_paths if isinstance(reference_paths, Sequence) else ()
        if not actual or not reference:
            return None

        errors: list[float] = []
        for actual_path, reference_path in zip(actual, reference, strict=False):
            actual_matrix = _point_matrix(actual_path)
            reference_matrix = _point_matrix(reference_path)
            if actual_matrix is None or reference_matrix is None:
                continue
            if len(actual_matrix) < 2 or len(reference_matrix) < 2:
                continue
            actual_matrix = actual_matrix[np.isfinite(actual_matrix).all(axis=1)]
            reference_matrix = reference_matrix[np.isfinite(reference_matrix).all(axis=1)]
            if len(actual_matrix) < 2 or len(reference_matrix) < 2:
                continue
            if len(actual_matrix) > DTW_MAX_POINTS:
                actual_matrix = actual_matrix[
                    np.linspace(0, len(actual_matrix) - 1, DTW_MAX_POINTS, dtype=int)
                ]
            if len(reference_matrix) > DTW_MAX_POINTS:
                reference_matrix = reference_matrix[
                    np.linspace(0, len(reference_matrix) - 1, DTW_MAX_POINTS, dtype=int)
                ]
            distance_m = np.hypot(
                actual_matrix[:, None, 0] - reference_matrix[None, :, 0],
                actual_matrix[:, None, 1] - reference_matrix[None, :, 1],
            )
            rows, columns = distance_m.shape
            cost = np.full((rows, columns), np.inf)
            cost[0, 0] = distance_m[0, 0]
            for row in range(1, rows):
                cost[row, 0] = cost[row - 1, 0] + distance_m[row, 0]
            for column in range(1, columns):
                cost[0, column] = cost[0, column - 1] + distance_m[0, column]
            for row in range(1, rows):
                for column in range(1, columns):
                    cost[row, column] = distance_m[row, column] + min(
                        cost[row - 1, column],
                        cost[row, column - 1],
                        cost[row - 1, column - 1],
                    )
            errors.append(float(cost[-1, -1] / max(rows, columns)))

        return float(np.mean(errors)) if errors else None

    @classmethod
    def reconcile_stage_references(
        cls,
        dynamic_row: Mapping[str, typing.Any],
        ref_robot_row: Mapping[str, typing.Any] | None,
        ref_ped_row: Mapping[str, typing.Any] | None,
    ) -> dict[str, typing.Any]:
        """Reconcile one episode's metrics against its stage's unobstructed-robot and unhindered-pedestrian rows."""
        from arena_evaluation.processing.metrics.social.pedestrian_disturbance import (
            PedestrianDisturbanceCalculator,
        )

        results: dict[str, typing.Any] = dict.fromkeys(cls.output_keys())

        actual_ped_paths = dynamic_row.get("pedestrian_path")
        actual_robot_path = dynamic_row.get("path")
        reference_ped_paths = None if ref_ped_row is None else ref_ped_row.get("pedestrian_path")

        interference: float | None = None
        if actual_ped_paths and reference_ped_paths:
            interference = cls.compute_pfi_dtw(actual_ped_paths, reference_ped_paths)
            if interference is not None:
                results["pfi"] = round(float(interference), 4)
            actual_ped_points = _stack_paths(actual_ped_paths)
            reference_ped_points = _stack_paths(reference_ped_paths)
            if actual_ped_points is not None and reference_ped_points is not None:
                deflection = PedestrianDisturbanceCalculator._compute_trajectory_deflection(
                    actual_ped_points, reference_ped_points
                )
                results["ped_path_deflection_m"] = round(float(deflection), 3)

        if ref_robot_row is None:
            return results

        reference_robot_path = ref_robot_row.get("path")
        robot_deviation_m: float | None = None
        actual_robot_points = _point_matrix(actual_robot_path)
        reference_robot_points = _point_matrix(reference_robot_path)
        if actual_robot_points is not None and reference_robot_points is not None:
            maximum_m, _ = cls.compute_orthogonal_deviation(
                actual_robot_points, reference_robot_points
            )
            robot_deviation_m = float(maximum_m)
            results["robot_path_deviation_m"] = round(robot_deviation_m, 3)

        if interference is not None:
            if robot_deviation_m is not None and robot_deviation_m > 0.0:
                results["mar"] = round(
                    float(interference / max(robot_deviation_m, MIN_ROBOT_DEVIATION_M)), 4
                )
            elif (
                dynamic_row.get("path_length") is not None
                and ref_robot_row.get("path_length") is not None
            ):
                path_deviation_m = max(
                    float(dynamic_row["path_length"])
                    - float(ref_robot_row["path_length"]),
                    MIN_PATH_DEVIATION_M,
                )
                results["mar"] = round(float(interference / path_deviation_m), 4)

        reference_time_s = ref_robot_row.get("time_to_goal")
        episode_time_s = dynamic_row.get("time_to_goal")
        if (
            reference_time_s is not None
            and episode_time_s is not None
            and float(episode_time_s) > 0.0
        ):
            results["relative_throughput"] = round(
                float(reference_time_s) / float(episode_time_s), 4
            )

        reference_energy_wh = ref_robot_row.get("energy_total_wh")
        episode_energy_wh = dynamic_row.get("energy_total_wh")
        if reference_energy_wh is not None and episode_energy_wh is not None:
            results["kcsc_wh"] = round(
                max(float(episode_energy_wh) - float(reference_energy_wh), 0.0), 4
            )
        return results

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """EPISODE scope: every reference-twin output is null until the pipeline reconciles the stage's references."""
        del episode, prior_results
        return dict.fromkeys(self.output_keys())

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE scope only: the twin baselines are separate episodes, reconciled after the run."""
        if scope_slice is not None and scope_slice.scope != MetricScope.EPISODE:
            return dict.fromkeys(self.output_keys())
        return self.calculate(episode, prior_results)
