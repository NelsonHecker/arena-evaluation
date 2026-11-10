"""Episode duration, sample interval and idling dwell.

The sample clock the legacy ``time`` key carried is the ``t_ns`` column every 1NF
series table already publishes, so it has no metric-row output of its own. The
remaining outputs are scalars derived from the motion series' own speed track.
"""

from __future__ import annotations

import typing

import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.performance.motion_metrics import episode_speed_mps
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
IDLE_SPEED_MPS: typing.Final[float] = 0.01

_SUMMARY_KEYS: typing.Final[tuple[str, ...]] = ("time_diff", "time_to_goal", "idling_time")


class TimeMetricsCalculator(BaseMetricCalculator):
    """Time-related metrics of the episode.

    Metrics:
    - time_to_goal: Total duration of the episode in seconds
    - time_diff: Total duration of the episode in nanoseconds
    - idling_time: Total time spent with velocity near zero (< 0.01 m/s)
    """

    NAME = "time_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = ["robot_tracks"]

    UNITS = {
        "time_diff": "ns",
        "time_to_goal": "s",
        "idling_time": "s",
    }

    PRIMARY_OUTPUTS = ["time_to_goal", "idling_time"]
    OUTPUT_DIRECTIONS = {"time_to_goal": "lower", "idling_time": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(_SUMMARY_KEYS)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        if episode.data is None or "t_ns" not in episode.data.columns:
            return dict.fromkeys(_SUMMARY_KEYS)

        t_ns = episode.data["t_ns"].to_numpy()
        if len(t_ns) < 2:
            return {"time_diff": 0, "time_to_goal": 0.0, "idling_time": 0.0}

        duration_ns = int(t_ns[-1] - t_ns[0])
        speed_mps, _speed_time_ns = episode_speed_mps(self, episode)
        idling_time_s = 0.0
        if len(speed_mps) == len(t_ns):
            dt_s = np.diff(t_ns) / NANOSECONDS_PER_SECOND
            idling_time_s = float(np.sum(dt_s[speed_mps[:-1] < IDLE_SPEED_MPS]))
        return {
            "time_diff": duration_ns,
            "time_to_goal": float(duration_ns) / NANOSECONDS_PER_SECOND,
            "idling_time": idling_time_s,
        }
