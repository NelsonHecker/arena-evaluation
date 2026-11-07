import json
import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.judge")

import shapely

from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry
from arena_evaluation.processing.metrics.ecological.condition_metrics import _wall_cache, _world_cache
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import EpisodeDescriptor

S = 1_000_000_000
WORLD = "synthetic_pipeline_world"
ENV = "env_0"
ROBOT = "env_0_jackal_0"
PHASES = {"jackal_0": {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}], "conditions": [], "map_poses": [[1.0, 1.0, 0.0]]}}


def _robot_frame(points: list[tuple[float, float, float]], columns: dict[str, str]) -> dict[str, list]:
    """Flat robot rows of a track: the time, identity and the columns named by the mapping."""
    return {"t_ns": [int(t * S) for t, _, _ in points], "env_id": [ENV] * len(points), "robot_id": [ROBOT] * len(points), **{column: [value(x, y) for _, x, y in points] for column, value in columns.items()}}


def _snapshot(rows: list[tuple[int, str, str]]) -> pl.DataFrame:
    n = len(rows)
    return pl.DataFrame(
        {
            "t_ns": [t for t, _, _ in rows],
            "env_id": [ENV] * n,
            "world": [WORLD] * n,
            "entity_name": ["env_0/jackal_0"] * n,
            "kind": ["robot"] * n,
            "field_name": [f for _, f, _ in rows],
            "field_kind": ["discrete"] * n,
            "value_str": [v for _, _, v in rows],
            "value_num": pl.Series([None] * n, dtype=pl.Float64),
            "value_bool": pl.Series([None] * n, dtype=pl.Boolean),
        }
    )


def _robot_rows(t: int, phase: str, met: str) -> list[tuple[int, str, str]]:
    return [(t, "phase", phase), (t, "met", met), (t, "failed", ""), (t, "dropped", ""), (t, "violated", "")]


def _episode(tmp_path: pathlib.Path) -> tuple[FolderManager, EpisodeDescriptor]:
    episode_dir = tmp_path / "run" / "episodes" / "nav2" / "episode_000"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_000.mcap").touch()
    track = [(t / 10, 5.0 - 0.4 * t, 5.0 - 0.4 * t) for t in range(11)] + [(t / 10, 1.0, 1.0) for t in range(11, 31)]
    topics = episode_dir / "topics"
    topics.mkdir()
    pl.DataFrame(_robot_frame(track, {"odom_x": lambda x, _: x, "odom_y": lambda _, y: y, "odom_yaw": lambda *_: 0.0, "odom_vx": lambda *_: 0.5, "odom_wz": lambda *_: 0.0, "drift_error_m": lambda *_: 0.0})).write_parquet(topics / "robot_odom.parquet")
    pl.DataFrame(_robot_frame(track, {"x": lambda x, _: x, "y": lambda _, y: y, "z": lambda *_: 0.0, "yaw": lambda *_: 0.0, "vx": lambda *_: 0.0, "vy": lambda *_: 0.0, "vz": lambda *_: 0.0, "omega_z": lambda *_: 0.0, "speed_mps": lambda *_: 0.0, "pose_source": lambda *_: "tf_gt"})).write_parquet(
        topics / "robot_tracks.parquet"
    )
    pl.DataFrame(_robot_frame(track, {"stamp_ns": lambda *_: 0, "pos_x": lambda x, _: x, "pos_y": lambda _, y: y, "yaw": lambda *_: 0.0})).write_parquet(topics / "robot_task_pose.parquet")
    pl.DataFrame(
        {
            "t_ns": [0, 3 * S],
            "env_id": [ENV, ENV],
            "episode_id": [1, 1],
            "outcome_state": [1, 2],
            "outcome_info": ["", "finished"],
            "goal_uuid": ["", ""],
            "phases_yaml": [json.dumps(PHASES), json.dumps(PHASES)],
            "params_yaml": ["", ""],
            "conditions_yaml": ["", ""],
            "human_params": ["", ""],
        }
    ).write_parquet(topics / "episode_lifecycle.parquet")
    _snapshot(_robot_rows(0, "0", "") + _robot_rows(int(1.5 * S), "", "0")).write_parquet(topics / "world_semantics.parquet")
    TopicParquetStore.mark_complete(topics)
    descriptor = EpisodeDescriptor(episode_dir=str(episode_dir), benchmark_id="run", episode_id=0, planner="nav2", stage="s0", map=WORLD)
    return FolderManager(data_root=tmp_path), descriptor


def test_process_episode_replays_the_task_on_the_episode_world(tmp_path: pathlib.Path) -> None:
    _world_cache[WORLD] = [_ZoneGeometry(name="kitchen", polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]), max_speed=None, quiet=False, restricted=False)]
    _wall_cache[WORLD] = None
    folders, descriptor = _episode(tmp_path)
    rows = ProcessingPipeline(folders, workers=1).process_episode(descriptor)
    assert [row["robot"] for row in rows] == [ROBOT]
    assert rows[0]["map"] == WORLD
    assert rows[0]["phase_outcomes"] == ["met"]
    assert rows[0]["judge_agrees"] is True
    assert rows[0]["judge_disagreements"] == []
    assert (tmp_path / "run" / "episodes" / "nav2" / "episode_000" / "metrics.parquet").exists()
