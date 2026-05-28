from __future__ import annotations

import pathlib
import yaml
from pydantic import BaseModel, Field

from ..storage.schemas import PlotSpec

class VizManifest(BaseModel):
    """
    Manifest configuring which plots to generate for a benchmark.
    """
    manifest_version: str = "1.0"
    plots: list[PlotSpec] = Field(default_factory=list)

    @classmethod
    def load(cls, path: pathlib.Path) -> "VizManifest":
        """Load manifest from YAML."""
        if not path.exists():
            return cls._default_manifest()
            
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
        
    @classmethod
    def _default_manifest(cls) -> "VizManifest":
        """Generate a basic default manifest."""
        return cls(plots=[
            PlotSpec(
                id="violin_path_length",
                type="violin",
                title="Path Length Distribution",
                data_key="path_length",
                differentiate="planner"
            ),
            PlotSpec(
                id="violin_time_to_goal",
                type="violin",
                title="Time to Goal Distribution",
                data_key="time_to_goal",
                differentiate="planner"
            ),
            PlotSpec(
                id="bar_collision_amount",
                type="bar",
                title="Average Collisions",
                data_key="collision_amount",
                differentiate="planner"
            ),
            PlotSpec(
                id="trajectory_plot",
                type="trajectory",
                title="Robot Trajectories",
                data_key="path",
                differentiate="planner"
            ),
            PlotSpec(
                id="radar_performance",
                type="radar",
                title="Performance Overview",
                data_key="*",
                differentiate="planner"
            ),
        ])
