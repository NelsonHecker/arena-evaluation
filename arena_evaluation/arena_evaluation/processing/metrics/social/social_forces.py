"""Helbing-Molnar social force and personal-space index on the pedestrian frame axis.

One vectorized plan: the robot track is sampled onto the pedestrian frame axis by a single
backward asof join (:func:`~arena_evaluation.processing.pedestrian_frames.frame_geometry`),
every repulsive force, anisotropy weight, personal-space index and per-frame total is a
Polars expression over the resulting frame x pedestrian table. The per-frame force and index
series are emitted as a 1NF table by :meth:`SocialForcesCalculator.series_table`.
"""

from __future__ import annotations

import typing
from typing import Final

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.pedestrian_frames import (
    ALIGN_TOLERANCE_NS,
    NS_PER_S,
    FrameGeometry,
    frame_geometry,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SFM_INTERACTION_STRENGTH_N: Final[float] = 2.1
SFM_INTERACTION_RANGE_M: Final[float] = 0.3
PEDESTRIAN_RADIUS_M: Final[float] = 0.3
SFM_CUTOFF_M: Final[float] = 5.0
SFM_MAX_FORCE_N: Final[float] = 100.0
ESFM_ANISOTROPY: Final[float] = 0.5
CI_SIGMA_FRONT_M: Final[float] = 1.0
CI_SIGMA_SIDE_M: Final[float] = 0.5
CI_SIGMA_X_M: Final[float] = 0.28
CI_SIGMA_Y_M: Final[float] = 0.28

SERIES_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "time_s": pl.Float64,
        "sfm_force_n": pl.Float64,
        "ci": pl.Float64,
    }
)


def _anisotropy(angle_rad: pl.Expr) -> pl.Expr:
    """Extended-SFM anisotropy weight in [0, 1] of a heading angle difference."""
    return (ESFM_ANISOTROPY + (1.0 - ESFM_ANISOTROPY) * (1.0 + angle_rad.cos()) / 2.0).clip(
        0.0, 1.0
    )


def _force_expression(combined_radius_m: float) -> pl.Expr:
    """Helbing repulsive magnitude of one pedestrian, clamped, zero beyond the cutoff, null while unresolved."""
    distance_m = pl.col("distance_m")
    return (
        pl.when(distance_m.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(distance_m <= SFM_CUTOFF_M)
        .then(
            (
                SFM_INTERACTION_STRENGTH_N
                * ((combined_radius_m - distance_m) / SFM_INTERACTION_RANGE_M).exp()
            ).clip(upper_bound=SFM_MAX_FORCE_N)
        )
        .otherwise(0.0)
    )


def _personal_space_index_expression() -> pl.Expr:
    """Gaussian personal-space index of one pedestrian, elliptical along its heading when that heading is recorded."""
    heading_cos = pl.col("ped_yaw_rad").cos()
    heading_sin = pl.col("ped_yaw_rad").sin()
    d_front_m = pl.col("dx_m") * heading_cos + pl.col("dy_m") * heading_sin
    d_side_m = -pl.col("dx_m") * heading_sin + pl.col("dy_m") * heading_cos
    positional_exponent = -(
        pl.col("dx_m").pow(2) / (2.0 * CI_SIGMA_X_M**2)
        + pl.col("dy_m").pow(2) / (2.0 * CI_SIGMA_Y_M**2)
    )
    heading_exponent = -(
        d_front_m.pow(2) / (2.0 * CI_SIGMA_FRONT_M**2)
        + d_side_m.pow(2) / (2.0 * CI_SIGMA_SIDE_M**2)
    )
    return (
        pl.when(pl.col("ped_yaw_rad").is_null())
        .then(positional_exponent)
        .otherwise(heading_exponent)
        .exp()
    )


def _frame_value(column: str, resolved: str) -> pl.Expr:
    """A frame's pedestrian-derived value: zero without pedestrians, null while the robot state is unresolved."""
    return (
        pl.when(~pl.col("has_pedestrians"))
        .then(pl.lit(0.0))
        .when(~pl.col(resolved))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(column))
    )


def _integral_expression(value: str) -> pl.Expr:
    """Time integral of a per-frame value over the integration steps, null frames skipped."""
    return (pl.col(value) * pl.col("dt_s")).drop_nulls().sum()


def _summary_expression(value: str) -> pl.Expr:
    """Aggregate of a per-frame value over the frames that observed it, null when no frame did."""
    return pl.col(value).drop_nulls()


class SocialForcesCalculator(BaseMetricCalculator):
    """Helbing-Molnar social force, extended-SFM anisotropy and personal-space index metrics."""

    NAME = "social_forces"
    CATEGORY = "social"
    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.ENCOUNTER})
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_tracks"]

    UNITS = {
        "sfm_cumulative_force": "NÂ·s",
        "sfm_peak_force": "N",
        "sfm_mean_force": "N",
        "esfm_cumulative_force": "NÂ·s",
        "esfm_peak_force": "N",
        "esfm_mean_force": "N",
        "esfm_ped_cumulative_force": "NÂ·s",
        "esfm_ped_peak_force": "N",
        "esfm_ped_mean_force": "N",
        "social_force_max": "N",
        "social_force_mean": "N",
        "social_force_ped_max": "N",
        "social_force_ped_mean": "N",
        "ci_max": "",
        "ci_mean": "",
    }

    PRIMARY_OUTPUTS = ["sfm_mean_force", "ci_mean", "social_force_ped_max"]
    OUTPUT_DIRECTIONS = {
        "sfm_mean_force": "lower",
        "ci_mean": "lower",
        "social_force_ped_max": "lower",
        "social_force_ped_mean": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "sfm_cumulative_force",
            "sfm_peak_force",
            "sfm_mean_force",
            "esfm_cumulative_force",
            "esfm_peak_force",
            "esfm_mean_force",
            "esfm_ped_cumulative_force",
            "esfm_ped_peak_force",
            "esfm_ped_mean_force",
            "social_force_max",
            "social_force_mean",
            "social_force_ped_max",
            "social_force_ped_mean",
            "ci_max",
            "ci_mean",
        ]

    @property
    def _combined_radius_m(self) -> float:
        """Robot-plus-pedestrian radius of the repulsive contact distance."""
        return float(self.robot_params.robot_radius) + PEDESTRIAN_RADIUS_M

    def _geometry(self, episode: AlignedEpisodeBundle) -> FrameGeometry:
        """Robot state on the pedestrian frame axis, with every pedestrian's relative geometry."""
        return frame_geometry(
            self.native_ped_frame(episode),
            self.native_topics(episode).get("robot_tracks"),
            tolerance_ns=ALIGN_TOLERANCE_NS,
        )

    def _frame_forces(self, geometry: FrameGeometry) -> pl.DataFrame:
        """Per-frame social force totals and personal-space index, one row per pedestrian frame."""
        force_n = _force_expression(self._combined_radius_m)
        robot_anisotropy = _anisotropy(
            pl.col("robot_yaw_rad") - pl.arctan2(pl.col("dy_m"), pl.col("dx_m"))
        )
        pedestrian_anisotropy = (
            pl.when(pl.col("ped_yaw_rad").is_null())
            .then(pl.lit(1.0))
            .otherwise(
                _anisotropy(
                    pl.col("ped_yaw_rad") - pl.arctan2(-pl.col("dy_m"), -pl.col("dx_m"))
                )
            )
        )
        contributions = geometry.pedestrians.with_columns(
            force_n=force_n,
            esfm_force_n=force_n * robot_anisotropy,
            esfm_ped_force_n=force_n * pedestrian_anisotropy,
            ci=_personal_space_index_expression(),
        )
        totals = contributions.group_by("frame_ix").agg(
            has_pedestrians=(pl.len() > 0),
            resolved=pl.col("distance_m").is_not_null().any(),
            esfm_resolved=pl.col("esfm_force_n").is_not_null().any(),
            sfm_force_n=pl.col("force_n").drop_nulls().sum(),
            esfm_force_n=pl.col("esfm_force_n").drop_nulls().sum(),
            esfm_ped_force_n=pl.col("esfm_ped_force_n").drop_nulls().sum(),
            ci=pl.col("ci").drop_nulls().max(),
        )
        return (
            geometry.frames.select("frame_ix", "t_ns", "dt_s")
            .join(totals, on="frame_ix", how="left")
            .with_columns(
                has_pedestrians=pl.col("has_pedestrians").fill_null(False),
                resolved=pl.col("resolved").fill_null(False),
                esfm_resolved=pl.col("esfm_resolved").fill_null(False),
                sfm_force_n=_frame_value("sfm_force_n", "resolved"),
                esfm_force_n=_frame_value("esfm_force_n", "esfm_resolved"),
                esfm_ped_force_n=_frame_value("esfm_ped_force_n", "resolved"),
                ci=_frame_value("ci", "resolved"),
            )
            .sort("t_ns")
        )

    @staticmethod
    def _aggregate(frame_forces: pl.DataFrame) -> dict[str, float | None]:
        """Force and index summaries over the frames that observed a resolved robot-pedestrian pair."""
        keys: Final[tuple[str, ...]] = (
            "sfm_cumulative_force",
            "esfm_cumulative_force",
            "esfm_ped_cumulative_force",
            "sfm_peak_force",
            "sfm_mean_force",
            "esfm_peak_force",
            "esfm_mean_force",
            "esfm_ped_peak_force",
            "esfm_ped_mean_force",
            "ci_max",
            "ci_mean",
        )
        if frame_forces.height == 0 or frame_forces.select(
            pl.col("sfm_force_n").is_not_null().any()
        ).item() is not True:
            return dict.fromkeys(keys)
        row = frame_forces.select(
            sfm_cumulative_force=_integral_expression("sfm_force_n"),
            esfm_cumulative_force=_integral_expression("esfm_force_n"),
            esfm_ped_cumulative_force=_integral_expression("esfm_ped_force_n"),
            sfm_peak_force=_summary_expression("sfm_force_n").max(),
            sfm_mean_force=_summary_expression("sfm_force_n").mean(),
            esfm_peak_force=_summary_expression("esfm_force_n").max(),
            esfm_mean_force=_summary_expression("esfm_force_n").mean(),
            esfm_ped_peak_force=_summary_expression("esfm_ped_force_n").max(),
            esfm_ped_mean_force=_summary_expression("esfm_ped_force_n").mean(),
            ci_max=pl.col("ci").drop_nulls().max(),
            ci_mean=pl.col("ci").drop_nulls().mean(),
        ).row(0, named=True)
        return {
            key: None if row[key] is None else float(row[key])
            for key in keys
        }

    def _episode_metrics(self, episode: AlignedEpisodeBundle) -> dict[str, float | None]:
        """Every social-force output over the given episode view."""
        geometry = self._geometry(episode)
        if not geometry.observed:
            return dict.fromkeys(self.output_keys())
        aggregate = self._aggregate(self._frame_forces(geometry))
        return {
            **aggregate,
            "social_force_max": aggregate["sfm_peak_force"],
            "social_force_mean": aggregate["sfm_mean_force"],
            "social_force_ped_max": aggregate["esfm_ped_peak_force"],
            "social_force_ped_mean": aggregate["esfm_ped_mean_force"],
        }

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Compute the episode-scope social forces over the bundle's own pedestrian frames."""
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
        """1NF per-frame force and index series: one row per pedestrian frame, null where the state is unresolved."""
        geometry = self._geometry(episode)
        if geometry.frames.height == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        first_t_ns = int(geometry.frames.select(pl.col("t_ns").min()).item())
        resolved_robot = robot_id or (episode.robot_name or "robot")
        return self._frame_forces(geometry).select(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
            robot_id=pl.lit(resolved_robot, dtype=pl.String),
            t_ns=pl.col("t_ns"),
            time_s=(pl.col("t_ns") - first_t_ns).cast(pl.Float64) / NS_PER_S,
            sfm_force_n=pl.col("sfm_force_n"),
            ci=pl.col("ci"),
        )
