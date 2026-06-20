from __future__ import annotations

import os
import sys
import datetime
import subprocess

from ..storage.schemas import RunMetadata
from ..storage.manifest import MetadataWriter


class IngestionMetadata:
    """Helper to generate metadata during ingestion (recording)."""
    
    @staticmethod
    def get_git_sha(workspace_dir: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                cwd=workspace_dir, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None
            
    @staticmethod
    def is_git_dirty(workspace_dir: str) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"], 
                cwd=workspace_dir, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return len(result.stdout.strip()) > 0
        except Exception:
            return False
            
    @staticmethod
    def create_initial_metadata(
        benchmark_id: str,
        planner: str,
        stage: str,
        map_name: str,
        episodes_requested: int,
        robot_model: str,
        suite_name: str,
        contest_name: str,
        workspace_dir: str = "/opt/arena_ws"
    ) -> RunMetadata:
        
        return RunMetadata(
            benchmark_id=benchmark_id,
            planner=planner,
            robot_model=[robot_model],
            map=map_name,
            stage=stage,
            episodes_requested=episodes_requested,
            suite_name=suite_name,
            contest_name=contest_name,
            inter_planner="", # Add if known
            agent_name="", # Add if known
            recording_started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            arena_git_sha=IngestionMetadata.get_git_sha(workspace_dir),
            arena_git_dirty=IngestionMetadata.is_git_dirty(workspace_dir),
            python_version=sys.version.split()[0],
            ros_distro=os.environ.get("ROS_DISTRO", "unknown")
        )
