from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.naturalness.trajectory import (
    _route_geodesic_length,
    _route_waypoints,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

logger = logging.getLogger(__name__)

COLLISION_EVENTS_TOPIC: Final[str] = "collision_events"
COLLISION_CONTACTS_TOPIC: Final[str] = "collision_contacts"

# arena_runtime_msgs/msg/EpisodeRecord outcome_state values.
_OUTCOME_SUCCESS: Final[int] = 2
_OUTCOME_RESULTS: Final[dict[int, str]] = {3: "FAILED", 4: "CANCELLED", 5: "FATAL"}
# QUEUED / RUNNING: never resolved by the runtime.
_OUTCOME_UNRESOLVED: Final[tuple[int, ...]] = (0, 1)

# Native collision_events counter per canonical count key.
_NATIVE_COUNT_COLUMNS: Final[dict[str, str]] = {
    "collision_event": "event_count",
    "collision_wall": "wall_count",
    "collision_static": "static_count",
    "collision_pedestrian": "ped_count",
}

_CONTACT_OUTPUT_KEYS: Final[tuple[str, ...]] = (
    "collision_amount",
    "collision_amount_wall",
    "collision_obstacles",
    "collisions",
    "spl",
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeSplInputs:
    """The episode-level quantities a contact fact row needs to carry its parent's success-weighted path length."""

    path_length_m: float
    theta_star_length_m: float | None
    time_to_goal_s: float | None
    collision_amount: int


class CollisionMetricsCalculator(BaseMetricCalculator):
    """Calculates collision counts, determines episode success status, and computes SPL."""

    NAME = "collision_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["time_metrics", "path_metrics", "trajectory_naturalness"]
    REQUIRED_TOPICS = ["collision_events", ("robot_tracks", "robot_odom")]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER, MetricScope.CONTACT})

    UNITS = {
        "collision_amount": "",
        "collision_amount_wall": "",
        "collision_amount_static": "",
        "collision_amount_pedestrian": "",
        "collision_obstacles": "",
        "collisions": "",
        "result": "",
        "success": "",
        "spl": "",
    }

    PRIMARY_OUTPUTS = ["success", "collision_amount", "spl"]
    OUTPUT_DIRECTIONS = {"success": "higher", "spl": "higher", "collision_amount": "lower"}

    TIMEOUT_THRESHOLD_S = 180.0
    MAX_COLLISIONS = 3

    @classmethod
    def required_topics_for_scope(cls, scope: MetricScope) -> tuple[str | tuple[str, ...], ...]:
        # A window may legitimately lack an optional sensor stream; a missing stream is then per-metric
        # `no_samples`, not a row-level `topic_missing` that would block the whole encounter or contact.
        return () if scope in (MetricScope.ENCOUNTER, MetricScope.CONTACT) else super().required_topics_for_scope(scope)

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "collision_amount",
            "collision_amount_wall",
            "collision_amount_static",
            "collision_amount_pedestrian",
            "collision_obstacles",
            "collisions",
            "result",
            "success",
            "spl",
        ]

    @staticmethod
    def _count_rising_edges(is_active: np.ndarray) -> int:
        count = 0
        if len(is_active) > 0 and is_active[0]:
            count += 1
        for i in range(1, len(is_active)):
            if is_active[i] and not is_active[i - 1]:
                count += 1
        return count

    @staticmethod
    def _collision_frame(episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """The native collision_events stream when present, else the aligned data."""
        if episode.topics is not None and COLLISION_EVENTS_TOPIC in episode.topics:
            return episode.topics[COLLISION_EVENTS_TOPIC].sort("t_ns")
        return episode.data

    @classmethod
    def _column_rising_edges(cls, episode: AlignedEpisodeBundle, column: str) -> int | None:
        frame = cls._collision_frame(episode)
        if frame is None:
            return None
        name = _NATIVE_COUNT_COLUMNS[column]
        if name not in frame.columns:
            return None
        counts = frame[name].to_numpy().astype(float)
        if np.isnan(counts).all():
            return None
        return cls._count_rising_edges(np.where(np.isnan(counts), 0.0, counts) > 0)

    @staticmethod
    def _hit_obstacles(episode: AlignedEpisodeBundle) -> list[str]:
        contacts = episode.topics.get(COLLISION_CONTACTS_TOPIC) if episode.topics else None
        if contacts is None or "obstacle_id" not in contacts.columns:
            return []
        return sorted({obstacle for obstacle in contacts["obstacle_id"].drop_nulls().to_list() if obstacle})

    @classmethod
    def output_keys_for_scope(cls, scope: MetricScope) -> tuple[str, ...]:
        if scope == MetricScope.ENCOUNTER:
            return (
                "collision_amount",
                "collision_amount_wall",
                "collision_amount_static",
                "collision_amount_pedestrian",
                "collision_obstacles",
            )
        if scope == MetricScope.CONTACT:
            return _CONTACT_OUTPUT_KEYS
        return tuple(cls.output_keys())

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            return {
                "collision_amount": self._column_rising_edges(episode, "collision_event"),
                "collision_amount_wall": self._column_rising_edges(episode, "collision_wall"),
                "collision_amount_static": self._column_rising_edges(episode, "collision_static"),
                "collision_amount_pedestrian": self._column_rising_edges(episode, "collision_pedestrian"),
                "collision_obstacles": self._hit_obstacles(episode),
            }
        if scope_slice.scope == MetricScope.CONTACT:
            return self._contact_result(episode, scope_slice)
        raise ValueError(f"collision_metrics does not serve the {scope_slice.scope.value} scope")

    def _contact_result(self, episode: AlignedEpisodeBundle, scope_slice: ScopeSlice) -> dict[str, typing.Any]:
        """One contact window's own collision counts and subject actor, plus its parent episode's SPL."""
        return {
            "collision_amount": self._window_onset_count(episode, scope_slice, "collision_event"),
            "collision_amount_wall": self._window_onset_count(episode, scope_slice, "collision_wall"),
            "collision_obstacles": [str(scope_slice.identity["obstacle_id"])],
            # The legacy producer hardcodes an empty list for every episode: the key carries no value.
            "collisions": [],
            "spl": self._episode_spl(scope_slice.full_episode),
        }

    def _window_onset_count(
        self,
        episode: AlignedEpisodeBundle,
        scope_slice: ScopeSlice,
        column: str,
    ) -> int | None:
        """Collision onsets of the window's own contact interval.

        The rising edge is detected on the padded window the slice carries, then counted only when its
        timestamp falls inside ``[t_start_ns, t_end_ns]``. The padding supplies the preceding state, and
        the interval filter keeps a continued bout from being counted again by the next contact window of
        the same collision (the detector splits one bout into several contact windows on the raw stream).
        """
        events = self.native_topics(episode).get(COLLISION_EVENTS_TOPIC)
        if events is None:
            return None
        active = (pl.col(_NATIVE_COUNT_COLUMNS[column]).cast(pl.Float64).fill_null(0.0) > 0.0).alias("active")
        onsets = events.select(pl.col("t_ns").cast(pl.Int64), active).sort("t_ns").filter(pl.col("active") & ~pl.col("active").shift(1, fill_value=False)).get_column("t_ns")
        inside = (onsets >= scope_slice.t_start_ns) & (onsets <= scope_slice.t_end_ns)
        return int(inside.sum())

    def _episode_spl(self, episode: AlignedEpisodeBundle) -> float | None:
        """The episode's success-weighted path length on the whole-mission route, reproduced for a contact row."""
        inputs = self._episode_spl_inputs(episode)
        if not self._episode_success(episode, inputs):
            return 0.0
        theta_star_length_m = inputs.theta_star_length_m
        if theta_star_length_m is None or theta_star_length_m <= 0.0:
            theta_star_length_m = self._straight_line_start_goal_m(episode)
        denominator_m = max(inputs.path_length_m, theta_star_length_m)
        if denominator_m <= 0.0:
            return 1.0
        return min(max(theta_star_length_m / denominator_m, 0.0), 1.0)

    def _episode_spl_inputs(self, episode: AlignedEpisodeBundle) -> EpisodeSplInputs:
        return EpisodeSplInputs(
            path_length_m=self._ground_truth_path_length_m(episode),
            theta_star_length_m=self._theta_star_length_m(episode),
            time_to_goal_s=self._aligned_duration_s(episode),
            collision_amount=self._column_rising_edges(episode, "collision_event") or 0,
        )

    def _episode_success(self, episode: AlignedEpisodeBundle, inputs: EpisodeSplInputs) -> bool:
        max_collisions = self.metrics_config.get("max_collisions", self.MAX_COLLISIONS)
        success = True
        if inputs.time_to_goal_s is not None and float(inputs.time_to_goal_s) >= self.TIMEOUT_THRESHOLD_S:
            success = False
        elif inputs.collision_amount >= max_collisions:
            success = False
        # The runtime's verdict wins over anything derived from the trace.
        if episode.outcome_info == "collision":
            return False
        if episode.outcome_state in _OUTCOME_UNRESOLVED:
            return False
        if episode.outcome_state is not None and episode.outcome_state != _OUTCOME_SUCCESS:
            return False
        return success

    @staticmethod
    def _aligned_duration_s(episode: AlignedEpisodeBundle) -> float | None:
        """time_metrics' time_to_goal: the span of the aligned episode clock."""
        frame = episode.data
        if frame is None or "time_ns" not in frame.columns:
            return None
        time_ns = frame["time_ns"].to_numpy()
        if len(time_ns) < 2:
            return 0.0
        return float(time_ns[-1] - time_ns[0]) / 1e9

    def _ground_truth_path_length_m(self, episode: AlignedEpisodeBundle) -> float:
        """path_metrics' path_length on the same teleport-free ground-truth segment, in metres."""
        if episode.data is None or "pos_x" not in episode.data.columns:
            return 0.0
        pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        if len(pos_x) < 2:
            return 0.0
        path_2d = np.column_stack((pos_x, pos_y))
        return float(np.sum(np.linalg.norm(path_2d[1:] - path_2d[:-1], axis=1)))

    def _theta_star_length_m(self, episode: AlignedEpisodeBundle) -> float | None:
        """Whole-mission route geodesic from trajectory_naturalness, the shared SPL/path-efficiency reference."""
        start_point = self._episode_start_point_m(episode)
        goal = episode.goal_pos
        if start_point is None or not goal or len(goal) < 2:
            return None
        goal_point = (float(goal[0]), float(goal[1]))
        if episode.map:
            pos_x, pos_y, _yaw, _time_ns = self.resolve_native_pose(episode)
            if len(pos_x) == 0 and episode.data is not None and len(episode.data) > 0:
                pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
            if len(pos_x) > 0:
                waypoints = _route_waypoints(
                    start_pt=start_point,
                    goal_pt=goal_point,
                    phases=episode.phases,
                    topics=episode.topics,
                    track_x=pos_x,
                    track_y=pos_y,
                )
                try:
                    route_length = _route_geodesic_length(episode=episode, waypoints=waypoints, robot_radius=float(self.robot_params.robot_radius))
                except (OSError, ValueError, RuntimeError, OverflowError) as error:
                    logger.info("theta* mission route unavailable for episode %s: %r", episode.episode_id, error)
                else:
                    if route_length is not None:
                        return route_length
        return float(np.sqrt((goal_point[0] - start_point[0]) ** 2 + (goal_point[1] - start_point[1]) ** 2))

    def _episode_start_point_m(self, episode: AlignedEpisodeBundle) -> tuple[float, float] | None:
        """trajectory_naturalness' episode start: the first native /tf pose, else the aligned ground-truth one."""
        pos_x, pos_y, _yaw, _time_ns = self.resolve_native_pose(episode)
        if len(pos_x) == 0:
            if episode.data is None or len(episode.data) == 0:
                return None
            pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        if len(pos_x) == 0:
            return None
        return (float(pos_x[0]), float(pos_y[0]))

    @staticmethod
    def _straight_line_start_goal_m(episode: AlignedEpisodeBundle) -> float:
        start, goal = episode.start_pos, episode.goal_pos
        if start and goal and len(start) >= 2 and len(goal) >= 2:
            return float(np.linalg.norm(np.array(goal[:2]) - np.array(start[:2])))
        return 0.0

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:

        collisions = []
        result = "GOAL_REACHED"
        success = True

        collision_amount = self._column_rising_edges(episode, "collision_event") or 0
        collision_amount_wall = self._column_rising_edges(episode, "collision_wall")
        collision_amount_static = self._column_rising_edges(episode, "collision_static")
        collision_amount_pedestrian = self._column_rising_edges(episode, "collision_pedestrian")
        collision_obstacles = self._hit_obstacles(episode)

        time_to_goal = prior_results.get("time_to_goal")

        max_collisions = self.metrics_config.get("max_collisions", self.MAX_COLLISIONS)

        if time_to_goal is not None and float(time_to_goal) >= self.TIMEOUT_THRESHOLD_S:
            result = "TIMEOUT"
            success = False
        elif collision_amount >= max_collisions:
            result = "COLLISION"
            success = False
        else:
            result = "GOAL_REACHED"
            success = True

        # The runtime's verdict wins over anything derived from the trace.
        if episode.outcome_info == "collision":
            result = "COLLISION"
            success = False
        elif episode.outcome_state in _OUTCOME_UNRESOLVED:
            result = "UNRESOLVED"
            success = False
        elif episode.outcome_state is not None and episode.outcome_state != _OUTCOME_SUCCESS:
            success = False
            if result == "GOAL_REACHED":
                result = _OUTCOME_RESULTS.get(episode.outcome_state, "FAILED")

        # Calculate Success weighted by Path Length (SPL)
        # SPL = Success * (L_0 / max(L_actual, L_0))
        actual_length = prior_results.get("path_length", 0.0) or 0.0
        l0 = prior_results.get("theta_star_length")
        if l0 is None or l0 <= 0:
            if episode.start_pos and episode.goal_pos and len(episode.start_pos) >= 2 and len(episode.goal_pos) >= 2:
                l0 = float(np.linalg.norm(np.array(episode.goal_pos[:2]) - np.array(episode.start_pos[:2])))
            else:
                l0 = 0.0

        if success:
            denom = max(float(actual_length), float(l0))
            if denom > 0:
                spl = float(l0 / denom)
                spl = min(max(spl, 0.0), 1.0)
            else:
                spl = 1.0
        else:
            spl = 0.0

        return {
            "collision_amount": collision_amount,
            "collision_amount_wall": collision_amount_wall,
            "collision_amount_static": collision_amount_static,
            "collision_amount_pedestrian": collision_amount_pedestrian,
            "collision_obstacles": collision_obstacles,
            "collisions": collisions,
            "result": result,
            "success": success,
            "spl": float(spl),
        }
