"""Reference-twin resolution: the robot-free pedestrian recording and the obstacle-free robot recording of a stage.

The benchmark runner records two reference episodes per stage next to the contestant episodes:
an ``unobstructed_robot`` twin (the same planner with no dynamic obstacles) and an
``unhindered_peds`` twin (a robot-free recording of the same pedestrian flow). Their lookup
keys are the ones the runner records them with: ``(stage, planner)`` for the robot twin and
``stage`` for the pedestrian twin.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import Final

import polars as pl

from arena_evaluation.storage.schemas import EpisodeDescriptor

UNHINDERED_PEDS: Final[str] = "unhindered_peds"
UNOBSTRUCTED_ROBOT: Final[str] = "unobstructed_robot"
PEDESTRIAN_TRACKS_TOPIC: Final[str] = "pedestrian_tracks"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TwinIndex:
    """Reference episodes of one benchmark, keyed by the identity the runner records them under."""

    peds_by_stage: dict[str, EpisodeDescriptor]
    robot_by_stage_planner: dict[tuple[str, str], EpisodeDescriptor]

    def pedestrian_twin(self, episode: EpisodeDescriptor) -> EpisodeDescriptor | None:
        """The robot-free pedestrian recording of the episode's stage, None when the stage has none."""
        if episode.is_reference:
            return None
        return self.peds_by_stage.get(episode.stage)

    def robot_twin(self, episode: EpisodeDescriptor) -> EpisodeDescriptor | None:
        """The obstacle-free recording of the episode's stage and planner, None when the run has none."""
        if episode.is_reference:
            return None
        return self.robot_by_stage_planner.get((episode.stage, episode.planner))

    def has_twin(self, episode: EpisodeDescriptor, reference_type: str) -> bool:
        """Whether the episode's stage has a twin of the given reference type."""
        if reference_type == UNHINDERED_PEDS:
            return self.pedestrian_twin(episode) is not None
        if reference_type == UNOBSTRUCTED_ROBOT:
            return self.robot_twin(episode) is not None
        return False


def build_twin_index(episodes: Sequence[EpisodeDescriptor]) -> TwinIndex:
    """Index the reference episodes of one benchmark, first recorded episode winning a duplicate key."""
    peds_by_stage: dict[str, EpisodeDescriptor] = {}
    robot_by_stage_planner: dict[tuple[str, str], EpisodeDescriptor] = {}
    for episode in sorted(episodes, key=lambda descriptor: descriptor.episode_id):
        if not episode.is_reference:
            continue
        if episode.reference_type == UNHINDERED_PEDS:
            peds_by_stage.setdefault(episode.stage, episode)
        elif episode.reference_type == UNOBSTRUCTED_ROBOT:
            robot_by_stage_planner.setdefault((episode.stage, episode.planner), episode)
    return TwinIndex(
        peds_by_stage=peds_by_stage,
        robot_by_stage_planner=robot_by_stage_planner,
    )


def episode_topic_path(episode: EpisodeDescriptor, topic: str) -> pathlib.Path:
    """Extracted flat-table path of one topic inside an episode directory."""
    return pathlib.Path(episode.episode_dir) / "topics" / f"{topic}.parquet"


def read_pedestrian_tracks(episode: EpisodeDescriptor | None) -> pl.DataFrame | None:
    """Flat pedestrian rows of a robot-free twin recording, None when it has not been extracted."""
    if episode is None:
        return None
    path = episode_topic_path(episode, PEDESTRIAN_TRACKS_TOPIC)
    if not path.is_file():
        return None
    return pl.read_parquet(path)

def read_pedestrian_tracks_from_dir(episode_dir: pathlib.Path | None) -> pl.DataFrame | None:
    """Flat pedestrian rows of a robot-free twin episode directory, None when it has not been extracted."""
    if episode_dir is None:
        return None
    path = episode_dir / "topics" / f"{PEDESTRIAN_TRACKS_TOPIC}.parquet"
    if not path.is_file():
        return None
    return pl.read_parquet(path)