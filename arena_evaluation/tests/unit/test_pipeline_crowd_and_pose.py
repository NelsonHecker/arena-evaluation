from __future__ import annotations

import pathlib

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _crowd_visibility, _pose_frame, _run_manifest


def _record(*human_params: str | None) -> pl.DataFrame:
    """This episode's records, RUNNING first and results after, each carrying the given human_params JSON."""
    n = len(human_params)
    return pl.DataFrame({"time_ns": list(range(100, 100 + n)), "episode_id": [5] * n, "outcome_state": [1] + [2] * (n - 1), "human_params": list(human_params)})


def test_crowd_visibility_comes_from_the_episodes_accepted_param() -> None:
    record = _record("{}", '{"humansim.robots_visible": false}')
    assert _crowd_visibility(record, {"cli_args": ["human.robots.visible:=true"]}) == {"crowd_visible": False, "crowd_condition": "blind", "crowd_visibility_source": "episode_param"}


def test_crowd_visibility_prefers_the_cli_launch_arg_over_the_suite_launch_block() -> None:
    manifest = {"cli_args": ["--suite", "s", "human.robots.visible:=false"], "suite": {"launch": {"human.robots.visible": "true"}}}
    assert _crowd_visibility(_record("{}"), manifest) == {"crowd_visible": False, "crowd_condition": "blind", "crowd_visibility_source": "launch_arg"}


def test_crowd_visibility_reads_the_suite_launch_block() -> None:
    manifest = {"cli_args": ["--suite", "s"], "suite": {"launch": {"human.robots.visible": "false"}}}
    assert _crowd_visibility(None, manifest)["crowd_condition"] == "blind"
    assert _crowd_visibility(None, manifest)["crowd_visibility_source"] == "launch_arg"


def test_crowd_visibility_defaults_to_visible_and_says_so() -> None:
    assert _crowd_visibility(_record(None), None) == {"crowd_visible": True, "crowd_condition": "visible", "crowd_visibility_source": "default"}


def test_crowd_visibility_ignores_the_previous_episodes_latched_param() -> None:
    record = pl.DataFrame({"time_ns": [50, 100, 900], "episode_id": [4, 5, 5], "outcome_state": [3, 1, 2], "human_params": ['{"humansim.robots_visible": false}', "{}", "{}"]})
    assert _crowd_visibility(record, None)["crowd_visibility_source"] == "default"


def test_run_manifest_is_the_one_above_the_episode_dir(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "benchmarks"
    episode = root / "run_a" / "episodes" / "episode_000"
    episode.mkdir(parents=True)
    (root / "run_a" / "manifest.yaml").write_text("run_id: run_a\ncli_args: ['human.robots.visible:=false']\n")
    manifest = _run_manifest(episode, root)
    assert manifest is not None
    assert manifest["run_id"] == "run_a"
    assert _run_manifest(root / "run_b" / "episodes" / "episode_000", root) is None


def test_pose_frame_is_the_most_frequent_parent_inside_the_window() -> None:
    tf_gt = pl.DataFrame({"time_ns": [1, 2, 3, 4], "frame_id": ["map", "env_0/jackal/odom", "env_0/jackal/odom", "map"]})
    assert _pose_frame(tf_gt, (2, 3)) == "env_0/jackal/odom"
    assert _pose_frame(tf_gt, (None, None)) == "env_0/jackal/odom"
    assert _pose_frame(tf_gt.lazy(), (4, None)) == "map"
    assert _pose_frame(None, (None, None)) is None


def test_crowd_visibility_takes_the_last_accepted_value_of_the_episode() -> None:
    record = _record('{"humansim.robots_visible": false}', '{"humansim.robots_visible": true}')
    assert _crowd_visibility(record, None) == {"crowd_visible": True, "crowd_condition": "visible", "crowd_visibility_source": "episode_param"}


def test_crowd_visibility_ignores_a_param_value_that_is_not_a_bool() -> None:
    record = _record('{"humansim.robots_visible": "false"}')
    assert _crowd_visibility(record, None) == {"crowd_visible": True, "crowd_condition": "visible", "crowd_visibility_source": "default"}


def test_crowd_visibility_is_null_without_a_record_when_the_stages_set_it() -> None:
    manifest = {"cli_args": [], "suite": {"stages": [{"name": "blind", "config": {"humansim": {"robots_visible": False}}}]}}
    assert _crowd_visibility(None, manifest) == {"crowd_visible": None, "crowd_condition": None, "crowd_visibility_source": None}


def test_run_manifest_is_none_when_the_file_is_not_text(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "benchmarks"
    episode = root / "run_a" / "episodes" / "episode_000"
    episode.mkdir(parents=True)
    (root / "run_a" / "manifest.yaml").write_bytes(b"\xff\xfe\x00")
    assert _run_manifest(episode, root) is None


def test_pose_frame_is_none_when_the_window_excludes_every_row() -> None:
    tf_gt = pl.DataFrame({"time_ns": [1, 2], "frame_id": ["map", "map"]})
    assert _pose_frame(tf_gt, (10, 20)) is None


def test_pose_frame_ignores_null_frame_ids() -> None:
    tf_gt = pl.DataFrame({"time_ns": [1, 2, 3], "frame_id": [None, None, "map"]})
    assert _pose_frame(tf_gt, (None, None)) == "map"
