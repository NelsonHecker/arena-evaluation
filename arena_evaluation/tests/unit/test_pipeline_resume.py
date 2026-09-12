import pathlib
import polars as pl
import pytest
from unittest import mock

from arena_evaluation.processing.parquet_store import ParquetStore
from arena_evaluation.processing.pipeline import (
    ProcessingPipeline,
    _STATUS_EVALUATED,
    _process_worker,
    _METRIC_DTYPES,
)
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata
from arena_evaluation.storage.manifest import MetadataWriter


@pytest.fixture
def dummy_episode(tmp_path: pathlib.Path) -> tuple[pathlib.Path, EpisodeDescriptor]:
    bench_dir = tmp_path / 'benchmarks' / 'test_bench'
    ep_dir = bench_dir / 'episodes' / 'episode_001'
    ep_dir.mkdir(parents=True)

    meta = RunMetadata(
        benchmark_id='test_bench',
        planner='dwb',
        map='scene_01',
        stage='stage1',
        robot_model=['turtlebot3_burger'],
        recording_started_at='2026-01-01T00:00:00+00:00',
        python_version='3.12',
        ros_distro='jazzy',
    )
    MetadataWriter.write(meta, ep_dir / 'metadata.yaml')

    ep = EpisodeDescriptor(
        episode_dir=str(ep_dir),
        benchmark_id='test_bench',
        episode_id=1,
        planner='dwb',
        stage='stage1',
        map='scene_01',
    )
    return ep_dir, ep


def test_process_episode_uses_cached_parquet(dummy_episode: tuple[pathlib.Path, EpisodeDescriptor]):
    ep_dir, ep = dummy_episode
    metrics_path = ep_dir / 'metrics.parquet'

    cached_rows = [
        {
            'episode': 1,
            'planner': 'dwb',
            'stage': 'stage1',
            'status': _STATUS_EVALUATED,
            'success': True,
            'time_to_goal': 15.5,
        }
    ]
    ParquetStore.write_rows(cached_rows, metrics_path, schema_overrides=_METRIC_DTYPES)

    pipeline = ProcessingPipeline(folder_manager=mock.MagicMock())

    # When force_process=False, should return cached rows without calling extract_episode
    with mock.patch.object(pipeline, 'extract_episode') as mock_extract:
        result = pipeline.process_episode(ep, force_process=False)
        mock_extract.assert_not_called()
        assert len(result) == 1
        assert result[0]['episode'] == 1
        assert result[0]['status'] == _STATUS_EVALUATED
        assert result[0]['time_to_goal'] == 15.5


def test_process_worker_returns_cached_status(dummy_episode: tuple[pathlib.Path, EpisodeDescriptor]):
    ep_dir, ep = dummy_episode
    metrics_path = ep_dir / 'metrics.parquet'

    cached_rows = [
        {
            'episode': 1,
            'planner': 'dwb',
            'stage': 'stage1',
            'status': _STATUS_EVALUATED,
            'success': True,
            'time_to_goal': 12.0,
        }
    ]
    ParquetStore.write_rows(cached_rows, metrics_path, schema_overrides=_METRIC_DTYPES)

    ep_id, rows, elapsed, was_cached = _process_worker(
        data_root_str=str(ep_dir.parent.parent.parent),
        ep=ep,
        force_extract=False,
        status_dict=None,
        force_process=False,
    )

    assert ep_id == 1
    assert was_cached is True
    assert len(rows) == 1
    assert rows[0]['time_to_goal'] == 12.0


def test_process_worker_recalculates_when_force_process(dummy_episode: tuple[pathlib.Path, EpisodeDescriptor]):
    ep_dir, ep = dummy_episode
    metrics_path = ep_dir / 'metrics.parquet'

    cached_rows = [
        {
            'episode': 1,
            'planner': 'dwb',
            'stage': 'stage1',
            'status': _STATUS_EVALUATED,
            'success': True,
        }
    ]
    ParquetStore.write_rows(cached_rows, metrics_path, schema_overrides=_METRIC_DTYPES)

    # When force_process=True, it should not use the fast-path
    with mock.patch('arena_evaluation.processing.pipeline.ProcessingPipeline.process_episode') as mock_pe:
        mock_pe.return_value = [{'episode': 1, 'status': _STATUS_EVALUATED, 'recalculated': True}]
        ep_id, rows, elapsed, was_cached = _process_worker(
            data_root_str=str(ep_dir.parent.parent.parent),
            ep=ep,
            force_extract=False,
            status_dict=None,
            force_process=True,
        )
        assert was_cached is False
        mock_pe.assert_called_once()
