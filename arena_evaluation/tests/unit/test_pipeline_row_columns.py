"""Integration check: the crowd-visibility and pose-frame columns reach the episode rows.

The unit tests for `_crowd_visibility` and `_pose_frame` prove the helpers; this test proves the
row builder actually carries their output, which is what the star schema publishes.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _status_row
from arena_evaluation.storage.schemas import EpisodeDescriptor

_FIELDS = ("crowd_visible", "crowd_condition", "crowd_visibility_source", "pose_frame")


def _episode(tmp_path: pathlib.Path, manifest: str | None = None) -> EpisodeDescriptor:
    episode_dir = tmp_path / "benchmarks" / "run_a" / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    if manifest is not None:
        (tmp_path / "benchmarks" / "run_a" / "manifest.yaml").write_text(manifest, encoding="utf-8")
    return EpisodeDescriptor(
        episode_dir=str(episode_dir),
        benchmark_id="run_a",
        episode_id=0,
        planner="dwb",
        stage="stage_1",
    )


def _record(robots_visible: bool) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [100, 900],
            "episode_id": [0, 0],
            "outcome_state": [1, 2],
            "outcome_info": ["", ""],
            "human_params": ["{}", json.dumps({"humansim.robots_visible": robots_visible})],
        }
    )


def test_status_row_carries_every_crowd_and_pose_column(tmp_path: pathlib.Path) -> None:
    row = _status_row(_episode(tmp_path, "run_id: run_a\ncli_args: ['human.robots.visible:=false']\n"), None, "robot", "no_recording", "no MCAP")
    assert set(_FIELDS) <= set(row)
    assert row["crowd_condition"] == "blind"
    assert row["crowd_visibility_source"] == "launch_arg"
    assert row["pose_frame"] is None


def test_accepted_episode_param_wins_over_the_run_manifest(tmp_path: pathlib.Path) -> None:
    episode = _episode(tmp_path, "run_id: run_a\ncli_args: ['human.robots.visible:=false']\n")
    row = _status_row(episode, None, "robot", "no_trajectory", "no pose", _record(robots_visible=True))
    assert row["crowd_visible"] is True
    assert row["crowd_condition"] == "visible"
    assert row["crowd_visibility_source"] == "episode_param"


def test_row_without_manifest_or_record_reports_unknown_visibility(tmp_path: pathlib.Path) -> None:
    row = _status_row(_episode(tmp_path), None, "robot", "error", "processing failed")
    assert row["crowd_visible"] is None
    assert row["crowd_condition"] is None
    assert row["crowd_visibility_source"] is None
