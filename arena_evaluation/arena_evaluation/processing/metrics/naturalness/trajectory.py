from __future__ import annotations

import typing

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.path.theta_star import (
    ThetaStarResult,
    compute_theta_star_for_episode,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

_TWO_PI: typing.Final[float] = 2.0 * np.pi


def _discrete_frechet(path: np.ndarray, reference: np.ndarray) -> float:
    """Discrete Frechet distance via dynamic programming with <=500 point decimation."""
    n, m = len(path), len(reference)
    if n == 0 or m == 0:
        return 0.0
    if n > 500:
        path = path[np.linspace(0, n - 1, 500, dtype=int)]
        n = len(path)
    if m > 500:
        reference = reference[np.linspace(0, m - 1, 500, dtype=int)]
        m = len(reference)

    ca = np.full((n, m), -1.0)
    ca[0, 0] = np.linalg.norm(path[0] - reference[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(path[i] - reference[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(path[0] - reference[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                np.linalg.norm(path[i] - reference[j]),
            )
    return float(ca[n - 1, m - 1])


def _path_irregularity(pos_x: np.ndarray, pos_y: np.ndarray, yaw: np.ndarray, goal_pt: tuple[float, float] | None) -> float | None:
    """Heading change beyond the minimum turn to the goal, per metre travelled; None without a usable track."""
    if len(pos_x) < 2 or goal_pt is None:
        return None
    length_m = float(np.sum(np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2)))
    if length_m <= 0.1:
        return None
    delta_yaw = np.diff(yaw)
    delta_yaw = (delta_yaw + np.pi) % _TWO_PI - np.pi
    target_angle = np.arctan2(goal_pt[1] - pos_y[0], goal_pt[0] - pos_x[0])
    rotation_to_target = (target_angle - yaw[0] + np.pi) % _TWO_PI - np.pi
    return float((np.sum(np.abs(delta_yaw)) - np.abs(rotation_to_target)) / length_m)


def _topological_complexity(peds_df: pl.DataFrame | None, pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray) -> float:
    """Total absolute net winding of every pedestrian around the robot's path."""
    if peds_df is None or len(peds_df) == 0 or len(pos_x) < 2 or len(time_ns) < 2:
        return 0.0
    required = {"t_ns", "ped_id", "x", "y"}
    if not required <= set(peds_df.columns):
        return 0.0
    robot = pl.LazyFrame({"t_ns": time_ns, "rx": pos_x, "ry": pos_y}).sort("t_ns")
    pedestrians = peds_df.lazy().select("t_ns", "ped_id", "x", "y").drop_nulls(["t_ns", "ped_id", "x", "y"]).sort("t_ns")
    rel_x = pl.col("rx") - pl.col("x")
    rel_y = pl.col("ry") - pl.col("y")
    angle = pl.arctan2(rel_y, rel_x)
    delta_angle = angle.diff().over("ped_id")
    wrapped = ((delta_angle + pl.lit(np.pi)) % pl.lit(_TWO_PI)) - pl.lit(np.pi)
    winding = pedestrians.join_asof(robot, on="t_ns", strategy="backward").with_columns(wrapped_d_angle=wrapped).group_by("ped_id").agg(net_winding=pl.col("wrapped_d_angle").drop_nulls().sum().abs() / pl.lit(_TWO_PI)).select(total_winding=pl.col("net_winding").sum()).collect()
    if len(winding) == 0 or winding["total_winding"][0] is None:
        return 0.0
    return float(winding["total_winding"][0])


class TrajectoryMetricsCalculator(BaseMetricCalculator):
    NAME = "trajectory_naturalness"
    CATEGORY = "naturalness"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom")]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})

    UNITS = {
        "ade": "m",
        "fde": "m",
        "adtw": "m",
        "mhd": "m",
        "frechet_distance": "m",
        "path_irregularity": "rad/m",
        "topological_complexity": "",
        "theta_star_length": "m",
    }

    PRIMARY_OUTPUTS = ["ade", "fde", "mhd", "frechet_distance"]
    OUTPUT_DIRECTIONS = {
        "ade": "lower",
        "fde": "lower",
        "adtw": "lower",
        "mhd": "lower",
        "frechet_distance": "lower",
        "path_irregularity": "lower",
        "topological_complexity": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ade",
            "fde",
            "adtw",
            "mhd",
            "frechet_distance",
            "path_irregularity",
            "topological_complexity",
            "theta_star_length",
        ]

    @classmethod
    def output_keys_for_scope(cls, scope: MetricScope) -> tuple[str, ...]:
        if scope == MetricScope.ENCOUNTER:
            return ("path_irregularity", "topological_complexity")
        return tuple(cls.output_keys())

    def _episode_pose(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ground-truth pose of the episode on the native /tf clock, falling back to the aligned ground-truth segment."""
        pos_x, pos_y, yaw, time_ns = self.resolve_native_pose(episode)
        if len(pos_x) > 0:
            return pos_x, pos_y, yaw, time_ns
        if episode.data is None or len(episode.data) == 0:
            empty = np.array([])
            return empty, empty, empty, np.array([], dtype=np.int64)
        pos_x, pos_y, yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        time_ns = episode.data["time_ns"].to_numpy().astype(np.int64) if "time_ns" in episode.data.columns else np.arange(len(pos_x), dtype=np.int64)
        return pos_x, pos_y, yaw, time_ns

    @staticmethod
    def _goal_point(episode: AlignedEpisodeBundle) -> tuple[float, float] | None:
        return (float(episode.goal_pos[0]), float(episode.goal_pos[1])) if episode.goal_pos and len(episode.goal_pos) >= 2 else None

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        pos_x, pos_y, yaw, time_ns = self._episode_pose(episode)
        if len(pos_x) == 0:
            return {key: None for key in self.output_keys()}

        goal_pt = self._goal_point(episode)
        results: dict[str, typing.Any] = {key: None for key in self.output_keys()}
        results["path_irregularity"] = _path_irregularity(pos_x, pos_y, yaw, goal_pt)
        results["topological_complexity"] = _topological_complexity(self.native_ped_frame(episode), pos_x, pos_y, time_ns)

        reference = None
        start_pt = (float(pos_x[0]), float(pos_y[0]))
        if goal_pt is not None and episode.map:
            try:
                robot_radius = self.robot_params.robot_radius if self.robot_params else 0.25
                reference = compute_theta_star_for_episode(
                    episode.map,
                    start_pt,
                    goal_pt,
                    robot_radius=robot_radius,
                    run_dir=episode.run_dir,
                )
            except (OSError, ValueError, RuntimeError, OverflowError) as error:
                import logging

                logging.getLogger(__name__).info(f"theta* reference unavailable for episode {episode.episode_id}: {error!r}")
                reference = None

        if reference is not None and reference.success and len(reference.path_x) > 0:
            results.update(self._reference_metrics(pos_x, pos_y, goal_pt, reference))
        elif goal_pt is not None:
            results["fde"] = float(np.sqrt((pos_x[-1] - goal_pt[0]) ** 2 + (pos_y[-1] - goal_pt[1]) ** 2))
            results["theta_star_length"] = float(np.sqrt((goal_pt[0] - start_pt[0]) ** 2 + (goal_pt[1] - start_pt[1]) ** 2))
        return results

    @staticmethod
    def _reference_metrics(pos_x: np.ndarray, pos_y: np.ndarray, goal_pt: tuple[float, float] | None, reference: ThetaStarResult) -> dict[str, float | None]:
        """ADE/FDE/MHD/Frechet/ADTW against the geometric Theta* reference path."""
        ref_x, ref_y = reference.path_x, reference.path_y
        metrics: dict[str, float | None] = {
            "theta_star_length": float(reference.geodesic_length),
            "fde": None,
            "ade": None,
            "mhd": None,
            "frechet_distance": None,
            "adtw": None,
        }
        if goal_pt is not None:
            metrics["fde"] = float(np.sqrt((pos_x[-1] - goal_pt[0]) ** 2 + (pos_y[-1] - goal_pt[1]) ** 2))
        else:
            metrics["fde"] = float(np.sqrt((pos_x[-1] - ref_x[-1]) ** 2 + (pos_y[-1] - ref_y[-1]) ** 2))

        current_distance = np.insert(np.cumsum(np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2)), 0, 0)
        reference_distance = np.insert(np.cumsum(np.sqrt(np.diff(ref_x) ** 2 + np.diff(ref_y) ** 2)), 0, 0)
        if current_distance[-1] > 0 and reference_distance[-1] > 0:
            current_fraction = current_distance / current_distance[-1]
            reference_fraction = reference_distance / reference_distance[-1]
            interpolated_x = np.interp(current_fraction, reference_fraction, ref_x)
            interpolated_y = np.interp(current_fraction, reference_fraction, ref_y)
            metrics["ade"] = float(np.mean(np.sqrt((pos_x - interpolated_x) ** 2 + (pos_y - interpolated_y) ** 2)))

        robot_points = np.column_stack((pos_x, pos_y))
        reference_points = np.column_stack((ref_x, ref_y))
        metrics["mhd"] = TrajectoryMetricsCalculator._modified_hausdorff(robot_points, reference_points)

        robot_sample = robot_points
        reference_sample = reference_points
        if len(robot_points) > 1000:
            robot_sample = robot_points[np.linspace(0, len(robot_points) - 1, 1000, dtype=int)]
        if len(reference_points) > 1000:
            reference_sample = reference_points[np.linspace(0, len(reference_points) - 1, 1000, dtype=int)]
        metrics["frechet_distance"] = _discrete_frechet(robot_sample, reference_sample)
        metrics["adtw"] = TrajectoryMetricsCalculator._asymmetric_dtw(robot_sample, reference_sample)
        return metrics

    @staticmethod
    def _modified_hausdorff(robot_points: np.ndarray, reference_points: np.ndarray) -> float:
        """Modified Hausdorff distance between two paths."""
        try:
            from scipy.spatial import cKDTree

            robot_tree = cKDTree(robot_points)
            reference_tree = cKDTree(reference_points)
            robot_to_reference, _ = reference_tree.query(robot_points, k=1)
            reference_to_robot, _ = robot_tree.query(reference_points, k=1)
        except (ImportError, ValueError):
            distances = cdist(robot_points, reference_points)
            return float(max(np.mean(np.min(distances, axis=1)), np.mean(np.min(distances, axis=0))))
        return float(max(np.mean(robot_to_reference), np.mean(reference_to_robot)))

    @staticmethod
    def _asymmetric_dtw(robot_points: np.ndarray, reference_points: np.ndarray) -> float | None:
        """Asymmetric DTW distance via dtaidistance when installed, else the normalized cost matrix."""
        try:
            from dtaidistance import dtw_ndim
        except ImportError:
            distances = cdist(robot_points, reference_points)
            rows, columns = distances.shape
            cost = np.full((rows, columns), np.inf)
            cost[0, 0] = distances[0, 0]
            for i in range(1, rows):
                cost[i, 0] = cost[i - 1, 0] + distances[i, 0]
            for j in range(1, columns):
                cost[0, j] = cost[0, j - 1] + distances[0, j]
            for i in range(1, rows):
                for j in range(1, columns):
                    cost[i, j] = distances[i, j] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            return float(cost[-1, -1] / (rows + columns))
        return float(dtw_ndim.distance(robot_points, reference_points))

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            pos_x, pos_y, yaw, time_ns = self.resolve_native_pose(episode)
            if len(pos_x) == 0:
                return {key: None for key in self.output_keys_for_scope(MetricScope.ENCOUNTER)}
            return {
                "path_irregularity": _path_irregularity(pos_x, pos_y, yaw, self._goal_point(episode)),
                "topological_complexity": _topological_complexity(self.native_ped_frame(episode), pos_x, pos_y, time_ns),
            }
        raise ValueError(f"trajectory_naturalness does not serve the {scope_slice.scope.value} scope")
