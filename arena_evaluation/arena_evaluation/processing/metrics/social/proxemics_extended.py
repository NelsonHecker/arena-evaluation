"""Hall proxemic-zone dwell, zone-entry events, closing-time and personal-space intrusion integral.

One vectorized plan: the robot track is sampled onto the pedestrian frame axis by a single
backward asof join (:func:`~arena_evaluation.processing.pedestrian_frames.frame_geometry`),
the closest pedestrian of every frame is picked with a per-frame rank, and every aggregate
is a Polars expression over that frame table. There is no per-frame Python loop and no
nested list column: the per-frame clearance series is emitted as a 1NF table by
:meth:`ProxemicsExtendedCalculator.series_table`.
"""

from __future__ import annotations

import typing
from typing import Final

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.social.proxemics import (
    INTIMATE_CLEARANCE_M,
    PERSONAL_CLEARANCE_M,
)
from arena_evaluation.processing.pedestrian_frames import (
    ALIGN_TOLERANCE_NS,
    NS_PER_S,
    FrameGeometry,
    frame_geometry,
    nearest_pedestrian_frames,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SOCIAL_CLEARANCE_M: Final[float] = 3.6
PEDESTRIAN_RADIUS_M: Final[float] = 0.3
TTI_LOOKAHEAD_S: Final[float] = 5.0
TTI_MIN_REL_SPEED_MPS: Final[float] = 0.05
ZONE_EVENT_MERGE_GAP_NS: Final[int] = 2 * 10**9
ZONES: Final[tuple[str, ...]] = ("intimate", "personal", "social", "public")

SERIES_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "time_s": pl.Float64,
        "min_ped_clearance_m": pl.Float64,
    }
)


def _zone_expression() -> pl.Expr:
    """Hall zone of an edge-to-edge clearance, null while the clearance is unknown."""
    clearance_m = pl.col("clearance_m")
    return (
        pl.when(clearance_m.is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(clearance_m < INTIMATE_CLEARANCE_M)
        .then(pl.lit("intimate"))
        .when(clearance_m < PERSONAL_CLEARANCE_M)
        .then(pl.lit("personal"))
        .when(clearance_m < SOCIAL_CLEARANCE_M)
        .then(pl.lit("social"))
        .otherwise(pl.lit("public"))
    )


def _dwell_expression(zone: str) -> pl.Expr:
    """Seconds whose closest pedestrian sat in ``zone``."""
    return (
        pl.when(pl.col("zone") == zone)
        .then(pl.col("dt_s"))
        .otherwise(0.0)
        .sum()
        .alias(f"time_in_{zone}_zone_s")
    )


def _zone_top_speed_expression(zone: str) -> pl.Expr:
    """Fastest robot speed while the closest pedestrian sat in ``zone``, null when never."""
    return (
        pl.when(pl.col("zone") == zone)
        .then(pl.col("robot_speed_mps"))
        .otherwise(None)
        .max()
        .alias(f"max_speed_{zone}_zone_mps")
    )


def zone_entry_events(
    cleared: pl.DataFrame,
    zone: str,
    merge_gap_ns: int = ZONE_EVENT_MERGE_GAP_NS,
) -> int:
    """Number of contiguous zone stays, bridging stays that resume within ``merge_gap_ns`` of the previous one."""
    runs = (
        cleared.with_columns(in_zone=pl.col("zone") == zone)
        .select("t_ns", "in_zone")
        .with_columns(run_id=pl.col("in_zone").rle_id())
        .group_by("run_id")
        .agg(
            in_zone=pl.col("in_zone").first(),
            start_ns=pl.col("t_ns").min(),
            end_ns=pl.col("t_ns").max(),
        )
        .filter(pl.col("in_zone"))
        .sort("start_ns")
    )
    if runs.height == 0:
        return 0
    previous_end_ns = pl.col("end_ns").shift(1)
    starts_event = previous_end_ns.is_null() | (
        (pl.col("start_ns") - previous_end_ns) >= merge_gap_ns
    )
    return int(runs.select(starts_event.sum()).item())


class ProxemicsExtendedCalculator(BaseMetricCalculator):
    """Hall proxemic zone dwell, zone-entry events and closing-time metrics on the pedestrian frame axis."""

    NAME = "proxemics_extended"
    CATEGORY = "social"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics", "path_metrics"]
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_tracks"]

    UNITS = {
        "time_in_intimate_zone": "s",
        "time_in_personal_zone": "s",
        "time_in_social_zone": "s",
        "time_in_public_zone": "s",
        "psi_intimate_events": "",
        "psi_personal_events": "",
        "psi_social_events": "",
        "max_speed_intimate_zone": "m/s",
        "max_speed_personal_zone": "m/s",
        "max_speed_social_zone": "m/s",
        "max_speed_public_zone": "m/s",
        "movement_towards_peds_ratio": "",
        "tti_min": "s",
        "tti_mean": "s",
        "personal_space_intrusion_integral": "m·s",
    }

    PRIMARY_OUTPUTS = [
        "time_in_intimate_zone",
        "psi_intimate_events",
        "tti_min",
    ]
    OUTPUT_DIRECTIONS = {
        "time_in_intimate_zone": "lower",
        "time_in_personal_zone": "lower",
        "max_speed_intimate_zone": "lower",
        "max_speed_personal_zone": "lower",
        "tti_min": "higher",
        "personal_space_intrusion_integral": "lower",
        "psi_intimate_events": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "time_in_intimate_zone",
            "time_in_personal_zone",
            "time_in_social_zone",
            "time_in_public_zone",
            "psi_intimate_events",
            "psi_personal_events",
            "psi_social_events",
            "max_speed_intimate_zone",
            "max_speed_personal_zone",
            "max_speed_social_zone",
            "max_speed_public_zone",
            "movement_towards_peds_ratio",
            "tti_min",
            "tti_mean",
            "personal_space_intrusion_integral",
        ]

    @property
    def _combined_radius_m(self) -> float:
        """Robot-plus-pedestrian radius removed from the centre distance to reach the edge clearance."""
        return float(self.robot_params.robot_radius) + PEDESTRIAN_RADIUS_M

    def _geometry(self, episode: AlignedEpisodeBundle) -> FrameGeometry:
        """Robot state on the pedestrian frame axis, with every pedestrian's relative geometry."""
        return frame_geometry(
            self.native_ped_frame(episode),
            self.native_topics(episode).get("robot_tracks"),
            tolerance_ns=ALIGN_TOLERANCE_NS,
        )

    def _cleared_frames(self, geometry: FrameGeometry) -> pl.DataFrame:
        """The closest pedestrian of every resolved frame, its edge clearance, Hall zone and integration step."""
        return (
            nearest_pedestrian_frames(geometry)
            .with_columns(clearance_m=pl.col("distance_m") - self._combined_radius_m)
            .sort("t_ns")
            .with_columns(zone=_zone_expression())
        )

    def _closing_times(self, geometry: FrameGeometry) -> pl.DataFrame:
        """Telemetry-only time-to-collision of every pedestrian closing on the robot inside the lookahead."""
        clearance_m = pl.col("distance_m") - self._combined_radius_m
        closing_m2_s2 = pl.col("dx_m") * pl.col("rel_vx_mps") + pl.col("dy_m") * pl.col("rel_vy_mps")
        return (
            geometry.pedestrians.with_columns(clearance_m=clearance_m)
            .filter(
                (pl.col("rel_speed_mps") > TTI_MIN_REL_SPEED_MPS)
                & (closing_m2_s2 < 0.0)
                & (pl.col("clearance_m") > 0.0)
            )
            .with_columns(tti_s=pl.col("clearance_m") / pl.col("rel_speed_mps"))
            .filter((pl.col("tti_s") > 0.0) & (pl.col("tti_s") < TTI_LOOKAHEAD_S))
            .select("tti_s")
        )

    def _episode_metrics(self, episode: AlignedEpisodeBundle) -> dict[str, typing.Any]:
        """Every proxemic output over the given episode view, all null without a resolved robot-pedestrian frame."""
        geometry = self._geometry(episode)
        cleared = self._cleared_frames(geometry)
        if cleared.height == 0:
            return dict.fromkeys(self.output_keys())

        summary = cleared.select(
            _dwell_expression("intimate"),
            _dwell_expression("personal"),
            _dwell_expression("social"),
            _dwell_expression("public"),
            _zone_top_speed_expression("intimate"),
            _zone_top_speed_expression("personal"),
            _zone_top_speed_expression("social"),
            _zone_top_speed_expression("public"),
            personal_space_intrusion_integral_m_s=(
                pl.when(pl.col("clearance_m") < PERSONAL_CLEARANCE_M)
                .then(
                    (PERSONAL_CLEARANCE_M - pl.col("clearance_m").clip(lower_bound=0.0))
                    * pl.col("dt_s")
                )
                .otherwise(0.0)
                .sum()
            ),
            approaching_frames=(
                (
                    pl.col("robot_vx_mps") * pl.col("dx_m")
                    + pl.col("robot_vy_mps") * pl.col("dy_m")
                )
                > 0.0
            ).sum(),
            sampled_frames=pl.len(),
        ).row(0, named=True)

        closing = self._closing_times(geometry)
        tti_min = closing.select(pl.col("tti_s").min()).item()
        tti_mean = closing.select(pl.col("tti_s").mean()).item()
        sampled_frames = int(summary["sampled_frames"])
        approaching_frames = int(summary["approaching_frames"])

        return {
            "time_in_intimate_zone": float(summary["time_in_intimate_zone_s"]),
            "time_in_personal_zone": float(summary["time_in_personal_zone_s"]),
            "time_in_social_zone": float(summary["time_in_social_zone_s"]),
            "time_in_public_zone": float(summary["time_in_public_zone_s"]),
            "psi_intimate_events": zone_entry_events(cleared, "intimate"),
            "psi_personal_events": zone_entry_events(cleared, "personal"),
            "psi_social_events": zone_entry_events(cleared, "social"),
            "max_speed_intimate_zone": summary["max_speed_intimate_zone_mps"],
            "max_speed_personal_zone": summary["max_speed_personal_zone_mps"],
            "max_speed_social_zone": summary["max_speed_social_zone_mps"],
            "max_speed_public_zone": summary["max_speed_public_zone_mps"],
            "movement_towards_peds_ratio": approaching_frames / sampled_frames
            if sampled_frames > 0
            else None,
            "tti_min": tti_min,
            "tti_mean": tti_mean,
            "personal_space_intrusion_integral": float(
                summary["personal_space_intrusion_integral_m_s"]
            ),
        }

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Compute the episode-scope proxemics set over the bundle's own pedestrian frames."""
        del prior_results
        return self._episode_metrics(episode)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE keeps the aligned population; an ENCOUNTER slice measures its own window telemetry."""
        del prior_results
        scope = MetricScope.EPISODE if scope_slice is None else scope_slice.scope
        match scope:
            case MetricScope.EPISODE:
                return self._episode_metrics(episode)
            case MetricScope.ENCOUNTER:
                return self._episode_metrics(scope_slice.window_bundle)
            case _:
                return dict.fromkeys(self.output_keys())

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF per-frame clearance series: one row per pedestrian frame, null clearance where unresolved."""
        geometry = self._geometry(episode)
        if geometry.frames.height == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        first_t_ns = int(geometry.frames.select(pl.col("t_ns").min()).item())
        clearances = nearest_pedestrian_frames(geometry).select(
            "frame_ix", clearance_m=pl.col("distance_m") - self._combined_radius_m
        )
        resolved_robot = robot_id or (episode.robot_name or "robot")
        return (
            geometry.frames.join(clearances, on="frame_ix", how="left")
            .sort("t_ns")
            .select(
                run_id=pl.lit(run_id, dtype=pl.String),
                episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
                robot_id=pl.lit(resolved_robot, dtype=pl.String),
                t_ns=pl.col("t_ns"),
                time_s=(pl.col("t_ns") - first_t_ns).cast(pl.Float64) / NS_PER_S,
                min_ped_clearance_m=pl.col("clearance_m"),
            )
        )
