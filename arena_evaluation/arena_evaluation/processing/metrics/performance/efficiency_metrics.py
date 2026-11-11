from __future__ import annotations

import math
import typing

from arena_evaluation.processing.metrics.base import NULL_REASON_SUFFIX, BaseMetricCalculator
from arena_evaluation.storage.declarations import NullReason
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


def _finite_positive(value: object) -> float | None:
    """The value as a finite float when it is positive, else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(typing.cast(float, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


_REFERENCE_EXCEEDS_PATH: typing.Final[str] = f"{NullReason.INVALID_EPISODE.value}:reference_exceeds_path"


class PathEfficiencyCalculator(BaseMetricCalculator):
    """Whole-mission route geodesic over travelled path length.

    The reference is the mission route through every task goal the robot reached, not just the
    current start -> goal leg. A null result carries its reason: ``no_samples`` when the reference
    or the path is unavailable, ``invalid_episode:reference_exceeds_path`` when the travelled path
    is shorter than the route. The travelled length is sampled ground truth, so it can undercount
    by a few percent; a route longer than the path is that sampling anomaly, not an efficiency
    above one.
    """

    NAME = "path_efficiency"
    CATEGORY = "performance"
    DEPENDS_ON = ["path_metrics", "trajectory_naturalness"]
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom")]
    UNITS = {"path_efficiency": ""}

    PRIMARY_OUTPUTS = ["path_efficiency"]
    OUTPUT_DIRECTIONS = {"path_efficiency": "higher"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "path_efficiency",
        ]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Geodesic reference over travelled length; null when either side is unavailable, never a clamp."""
        del episode
        path_length = _finite_positive(prior_results.get("path_length"))
        reference_length = _finite_positive(prior_results.get("theta_star_length"))
        if path_length is None or reference_length is None:
            return {"path_efficiency": None}
        if reference_length > path_length:
            return {"path_efficiency": None, f"path_efficiency{NULL_REASON_SUFFIX}": _REFERENCE_EXCEEDS_PATH}
        efficiency = reference_length / path_length
        if not math.isfinite(efficiency):
            return {"path_efficiency": None}
        return {"path_efficiency": efficiency}
