import pytest
import yaml
from pydantic import ValidationError

from arena_evaluation.storage.exceptions import ManifestGenerationError
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import RunMetadata

_BASE = {
    "benchmark_id": "test_bench",
    "planner": "teb",
    "robot_model": ["jackal"],
    "map": "map1",
    "stage": "stage1",
    "episodes_requested": 10,
    "suite_name": "suite1",
    "contest_name": "contest1",
    "inter_planner": "",
    "agent_name": "",
    "recording_started_at": "2026-01-01T00:00:00+00:00",
    "python_version": "3.10",
    "ros_distro": "humble",
    "modes": {"robots": ["scenario"]},
    "params": {"task.scenario.file": "barn"},
}


def test_run_metadata_serialization():
    meta = RunMetadata(**_BASE)

    dumped = meta.model_dump(exclude_none=True)
    assert dumped["benchmark_id"] == "test_bench"
    assert dumped["modes"] == {"robots": ["scenario"]}
    assert dumped["params"] == {"task.scenario.file": "barn"}
    assert "recording_ended_at" not in dumped

    reloaded = RunMetadata.model_validate(dumped)
    assert reloaded.planner == "teb"
    assert reloaded.robot_model == ["jackal"]
    assert reloaded.recording_ended_at is None


@pytest.mark.parametrize("missing", ["modes", "params"])
def test_run_metadata_requires_modes_and_params(missing: str):
    payload = {key: value for key, value in _BASE.items() if key != missing}

    with pytest.raises(ValidationError, match=missing):
        RunMetadata(**payload)


def test_old_format_yaml_without_modes_and_params_raises(tmp_path):
    old = {
        "benchmark_id": "test_bench",
        "planner": "teb",
        "map": "map1",
        "stage": "stage1",
        "recording_started_at": "2026-01-01T00:00:00+00:00",
        "python_version": "3.10",
        "ros_distro": "humble",
        "tm_robots": "scenario",
        "tm_obstacles": "random",
        "tm_modules": [],
        "robots_params": {"file": "barn"},
        "obstacles_params": {"static": {"n": 1}},
    }
    path = tmp_path / "episode_000.yaml"
    path.write_text(yaml.safe_dump(old))

    with pytest.raises(ManifestGenerationError) as excinfo:
        MetadataWriter.read(path)

    message = str(excinfo.value)
    assert "modes" in message
    assert "params" in message


def test_new_format_yaml_loads(tmp_path):
    path = tmp_path / "episode_000.yaml"
    path.write_text(yaml.safe_dump(_BASE))

    meta = MetadataWriter.read(path)

    assert meta.modes == {"robots": ["scenario"]}
    assert meta.params == {"task.scenario.file": "barn"}
    assert meta.recording_started_at == _BASE["recording_started_at"]
