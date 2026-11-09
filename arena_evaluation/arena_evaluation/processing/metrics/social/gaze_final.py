"""Mutual gaze cone metrics on the robot sample axis, with the per-sample counts as a 1NF series.

The pedestrian state at each robot sample comes from one backward asof join
(:func:`~arena_evaluation.processing.pedestrian_frames.pedestrians_at`) matched with the
robot state at the same timestamps; the cone tests are Polars expressions over that frame
x pedestrian table, reduced per robot sample.
"""

from __future__ import annotations

import typing
from typing import Final

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.pedestrian_frames import (
    ROBOT_TRACK_SCHEMA,
    pedestrians_at,
    robot_state_at,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

GAZE_CONE_RAD: Final[float] = 0.08726646259971647
FRAME_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "t_ns": pl.Int64,
        "looking_at_peds": pl.Int64,
        "looked_at_by_peds": pl.Int64,
    }
)

SERIES_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "time_s": pl.Float64,
        "looking_at_peds": pl.Int64,
        "looked_at_by_peds": pl.Int64,
    }
)


def _angle_difference(left_rad: pl.Expr, right_rad: pl.Expr) -> pl.Expr:
    """Absolute angular distance of two headings in [0, pi]."""
    return pl.math.pi() - ((left_rad - right_rad).abs() - pl.math.pi()).abs()


def gaze_counts(robot_times_ns: pl.Series, peds: pl.DataFrame | None, robot_tracks: pl.DataFrame | None) -> pl.DataFrame:
    """Per robot sample: how many pedestrians it looks at, and how many look back."""
    if peds is None or peds.height == 0 or robot_tracks is None or robot_tracks.height == 0:
        return pl.DataFrame(schema=FRAME_SCHEMA)
    robot_state = robot_state_at(robot_times_ns, robot_tracks)
    matched = pedestrians_at(robot_times_ns, peds).join(
        robot_state, on="time_ns", how="inner"
    )
    if matched.height == 0:
        return pl.DataFrame(schema=FRAME_SCHEMA)
    directions = matched.with_columns(
        dx_m=pl.col("x").cast(pl.Float64) - pl.col("robot_x_m"),
        dy_m=pl.col("y").cast(pl.Float64) - pl.col("robot_y_m"),
    ).with_columns(
        angle_to_ped_rad=pl.arctan2(pl.col("dy_m"), pl.col("dx_m")),
        angle_to_robot_rad=pl.arctan2(-pl.col("dy_m"), -pl.col("dx_m")),
    )
    looking = _angle_difference(
        pl.col("robot_yaw_rad"), pl.col("angle_to_ped_rad")
    ) <= GAZE_CONE_RAD
    looked_at = (
        pl.col("yaw").is_not_null()
        & (_angle_difference(pl.col("yaw").cast(pl.Float64), pl.col("angle_to_robot_rad")) <= GAZE_CONE_RAD)
    )
    return (
        directions.with_columns(looking_at=looking, looked_at=looked_at)
        .group_by("time_ns")
        .agg(
            looking_at_peds=pl.col("looking_at").sum().cast(pl.Int64),
            looked_at_by_peds=pl.col("looked_at").sum().cast(pl.Int64),
        )
        .select(FRAME_SCHEMA)
    )


def time_in_cone_s(samples: pl.DataFrame, column: str) -> float | None:
    """Seconds of robot time whose count in ``column`` is positive, from the forward sample steps."""
    if samples.height == 0:
        return None
    steps = samples.sort("t_ns").with_columns(
        step_s=(pl.col("t_ns").shift(-1) - pl.col("t_ns")).cast(pl.Float64) / 1e9
    )
    return float(
        steps.select(
            pl.when((pl.col(column) > 0) & pl.col("step_s").is_not_null())
            .then(pl.col("step_s"))
            .otherwise(0.0)
            .sum()
        ).item()
    )


class GazeMetricsCalculator(BaseMetricCalculator):
    """Robot-to-pedestrian and pedestrian-to-robot gaze cone dwell."""

    NAME = "gaze_metrics"
    CATEGORY = "social"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_tracks"]

    UNITS = {
        "total_time_looking_at_pedestrians": "s",
        "total_time_looked_at_by_pedestrians": "s",
    }

    PRIMARY_OUTPUTS = [
        "total_time_looking_at_pedestrians",
        "total_time_looked_at_by_pedestrians",
    ]

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "total_time_looking_at_pedestrians",
            "total_time_looked_at_by_pedestrians",
        ]

    def _robot_times_ns(self, episode: AlignedEpisodeBundle) -> pl.Series | None:
        """Robot sample timestamps of the episode view, none without an aligned data table."""
        if episode.data is None or episode.data.height == 0:
            return None
        if ROBOT_TRACK_SCHEMA is None:  # pragma: no cover - keeps the import contract explicit
            raise AssertionError("unreachable")
        return episode.data.get_column("time_ns").cast(pl.Int64)

    def _samples(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Per robot sample gaze counts, empty with the frame schema when nothing is resolvable."""
        times = self._robot_times_ns(episode)
        if times is None:
            return pl.DataFrame(schema=FRAME_SCHEMA)
        return gaze_counts(
            times,
            self.native_ped_frame(episode),
            self.native_topics(episode).get("robot_tracks"),
        )

    def _episode_metrics(self, episode: AlignedEpisodeBundle) -> dict[str, typing.Any]:
        """Total gaze dwell of both directions over the given episode view."""
        samples = self._samples(episode)
        if samples.height == 0:
            return dict.fromkeys(self.output_keys())
        return {
            "total_time_looking_at_pedestrians": time_in_cone_s(samples, "looking_at_peds"),
            "total_time_looked_at_by_pedestrians": time_in_cone_s(
                samples, "looked_at_by_peds"
            ),
        }

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Compute the episode-scope gaze dwell over the bundle's own pedestrian frames."""
        del prior_results
        return self._episode_metrics(episode)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE scope only: the slice's own window and pedestrian set are the population."""
        del prior_results
        if scope_slice is not None and scope_slice.scope != MetricScope.EPISODE:
            return dict.fromkeys(self.output_keys())
        return self._episode_metrics(episode)

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF per-sample gaze counts, zero where the robot had no pedestrian frame."""
        times = self._robot_times_ns(episode)
        if times is None:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        counts = self._samples(episode)
        samples = pl.DataFrame({"t_ns": times}).unique().sort("t_ns").join(
            counts, on="t_ns", how="left"
        )
        first_t_ns = int(samples.select(pl.col("t_ns").min()).item())
        resolved_robot = robot_id or (episode.robot_name or "robot")
        return samples.select(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
            robot_id=pl.lit(resolved_robot, dtype=pl.String),
            t_ns=pl.col("t_ns"),
            time_s=(pl.col("t_ns") - first_t_ns).cast(pl.Float64) / 1e9,
            looking_at_peds=pl.col("looking_at_peds").fill_null(0).cast(pl.Int64),
            looked_at_by_peds=pl.col("looked_at_by_peds").fill_null(0).cast(pl.Int64),
        )
