from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import typing


class RunMetadata(BaseModel):
    """
    Metadata for a benchmark run, serialized to metadata.yaml.
    """
    benchmark_id: str
    planner: str
    robot_model: list[str] = Field(default_factory=list)
    map: str
    stage: str
    episodes_requested: int
    suite_name: str
    contest_name: str
    inter_planner: str
    agent_name: str
    recording_started_at: str
    arena_git_sha: str | None = None
    arena_git_dirty: bool = False
    python_version: str
    ros_distro: str

    tm_obstacles: str | None = None
    tm_robots: str | None = None
    tm_modules: list[str] | None = None
    obstacles_params: dict[str, typing.Any] | None = None
    robots_params: dict[str, typing.Any] | None = None

    recording_ended_at: str | None = None
    episodes_recorded: int | None = None
    pedsim_available: bool | None = None
    recorded_topics: list[str] | None = None

    processing_completed_at: str | None = None
    episodes_valid: int | None = None
    pipeline_version: str | None = None


@dataclass(frozen=True)
class RobotParams:
    """Robot physical parameters loaded at runtime."""
    robot_radius: float
    laser_min_range: float
    laser_max_range: float

    @classmethod
    def load(cls, model: str) -> "RobotParams":
        """Load parameters from arena_robots caps or fallback to safe defaults."""
        import os
        import yaml
        from ament_index_python.packages import get_package_share_directory

        radius = 0.25
        min_range = 0.0
        max_range = 30.0

        try:
            caps_file = os.path.join(
                get_package_share_directory("arena_robots"),
                "robots",
                model,
                "caps",
                "mobile.yaml"
            )
            with open(caps_file, "r") as file:
                caps_content = yaml.safe_load(file)
                radius = float(caps_content.get("radius", radius))
        except Exception:
            pass
        
        return cls(
            robot_radius=radius,
            laser_min_range=min_range,
            laser_max_range=max_range,
        )


@dataclass(frozen=True)
class RunDescriptor:
    """Describes a single run discovered by FolderManager."""
    run_dir: str
    benchmark_id: str
    planner: str
    stage: str


@dataclass
class TopicBundle:
    """Raw topics extracted from MCAP"""
    odom: pl.DataFrame | None = None
    scan: pl.DataFrame | None = None
    cmd_vel: pl.DataFrame | None = None
    joint_states: pl.DataFrame | None = None
    peds: pl.DataFrame | None = None
    episode_record: pl.DataFrame | None = None
    collision_events: pl.DataFrame | None = None
    collision_monitor_state: pl.DataFrame | None = None
    plan: pl.DataFrame | None = None
    initialpose: pl.DataFrame | None = None
    tf: pl.DataFrame | None = None
    tf_static: pl.DataFrame | None = None
    tf_gt: pl.DataFrame | None = None
    semantic_snapshot: pl.DataFrame | None = None
    semantic_events: pl.DataFrame | None = None


@dataclass
class AlignedEpisodeBundle:
    """Aligned data for a single episode"""
    episode_id: int
    data: pl.DataFrame
    start_pos: list[float]
    goal_pos: list[float]
    num_pedestrians: int = 0
    robot_name: str | None = None
    semantic_events: pl.DataFrame | None = None


class PlotSpec(BaseModel):
    """Specification for a single plot in viz_manifest.yaml."""
    id: str
    type: str
    title: str
    data_key: str
    group_by: list[str] | str | None = None
    differentiate: str | None = "planner"
    auto_differentiate: bool = True
    filter: dict[str, str] | None = None
    options: dict[str, typing.Any] = Field(default_factory=dict)
    layout_group: str | None = None
