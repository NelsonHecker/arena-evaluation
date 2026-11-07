"""Unit tests for the Parquet layer (processing/parquet_store.py and the telemetry sink it reads).

Covers ParquetStore (metric frames with Pydantic metadata in the schema footer)
and TopicParquetStore (the flat telemetry cache: completion stamp, one bundle per
robot over shared global tables, unreadable caches read as misses).
"""

from __future__ import annotations

import json
import pathlib

import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st

from arena_evaluation.processing.parquet_store import ParquetStore, TopicParquetStore
from arena_evaluation.processing.telemetry.writer import TableSink
from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import RunMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DF_SCHEMA = pl.Schema(
    {
        "time_ns": pl.Int64,
        "value": pl.Float64,
        "label": pl.String,
    }
)

_META_BASE = dict(
    benchmark_id="bench1",
    planner="nav2",
    map="hospital_1",
    stage="final",
    recording_started_at="2026-01-01T00:00:00Z",
    python_version="3.12",
    ros_distro="humble",
)


def _df(n: int = 5, start: int = 1_000, step: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [start + i * step for i in range(n)],
            "value": [float(i) * 0.5 for i in range(n)],
            "label": [f"row{i}" for i in range(n)],
        },
        schema=_DF_SCHEMA,
    )


def _meta(**overrides) -> RunMetadata:
    kw = dict(_META_BASE)
    kw.update(overrides)
    return RunMetadata(**kw)


def test_metadata_of_a_recording_without_modes_reads_them_as_empty():
    assert "modes" not in _META_BASE
    assert _meta().modes == {}
    assert _meta(modes={"robots": ["explore"]}).modes == {"robots": ["explore"]}


def test_metadata_written_before_modes_reads_into_modes_and_params(tmp_path):
    old = {
        **_META_BASE,
        "tm_robots": "scenario",
        "tm_obstacles": "random",
        "tm_modules": ["benchmark"],
        "robots_params": {"file": "case_1", "start": {"x": 1.0}},
        "obstacles_params": {"static": {"n": [1, 2]}},
    }
    path = tmp_path / "episode.yaml"
    path.write_text(yaml.safe_dump(old))

    meta = MetadataWriter.read(path)

    assert meta.modes == {"robots": ["scenario"], "obstacles": ["random"], "modules": ["benchmark"]}
    assert meta.params == {"task.scenario.file": "case_1", "task.scenario.start.x": 1.0, "task.random.static.n": [1, 2]}
    dumped = meta.model_dump(exclude_none=True)
    assert not {"tm_robots", "tm_obstacles", "tm_modules", "robots_params", "obstacles_params"} & dumped.keys()


def test_metadata_with_modes_and_tm_aliases_keeps_its_modes():
    meta = RunMetadata.model_validate({**_META_BASE, "modes": {"robots": ["patrol"]}, "tm_robots": "explore"})
    assert meta.modes == {"robots": ["patrol"]}


def _write_src(tmp_path: pathlib.Path, name: str, df: pl.DataFrame) -> pathlib.Path:
    src = tmp_path / name
    ParquetStore.write(df, src)
    return src


# ---------------------------------------------------------------------------
# ParquetStore.write / read
# ---------------------------------------------------------------------------


def test_write_read_roundtrip_no_metadata(tmp_path):
    df = _df()
    dest = tmp_path / "metrics.parquet"
    ParquetStore.write(df, dest)
    assert dest.exists()

    back, metadata = ParquetStore.read(dest)
    assert metadata is None
    assert back.equals(df)
    assert back.schema == _DF_SCHEMA
    # footer carries no arena metadata key
    footer_meta = pq.read_schema(dest).metadata or {}
    assert ParquetStore.METADATA_KEY.encode() not in footer_meta


def test_write_creates_parent_directories(tmp_path):
    dest = tmp_path / "a" / "b" / "c" / "metrics.parquet"
    ParquetStore.write(_df(2), dest)
    assert dest.exists()


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ParquetStore.read(tmp_path / "nope.parquet")


def test_metadata_roundtrip_through_footer(tmp_path):
    """Pydantic metadata must round-trip through the parquet schema footer."""
    df = _df(4)
    meta = _meta(episode_id=7, outcome_state=2, params={"task.random.wall": True}, agent_name="ada")
    dest = tmp_path / "metrics.parquet"
    ParquetStore.write(df, dest, metadata=meta)

    back, metadata = ParquetStore.read(dest)
    assert back.equals(df)
    assert back.schema == _DF_SCHEMA
    assert metadata == meta.model_dump(exclude_none=True)
    assert metadata["episode_id"] == 7
    assert metadata["params"] == {"task.random.wall": True}

    # raw footer check: the JSON lives under the METADATA_KEY in the footer
    footer = pq.read_schema(dest)
    key = ParquetStore.METADATA_KEY.encode()
    assert footer.metadata is not None and key in footer.metadata
    assert json.loads(footer.metadata[key].decode()) == meta.model_dump(exclude_none=True)


def test_metadata_exclude_none():
    meta = _meta(episode_id=None, arena_git_sha=None, outcome_info="", is_reference=False)
    dumped = meta.model_dump(exclude_none=True)
    assert "episode_id" not in dumped
    assert "arena_git_sha" not in dumped
    # empty-string and False fields survive exclude_none
    assert dumped["outcome_info"] == ""
    assert dumped["is_reference"] is False


def test_read_metadata_none_when_key_absent(tmp_path):
    dest = _write_src(tmp_path, "m.parquet", _df(3))
    _, metadata = ParquetStore.read(dest)
    assert metadata is None


def test_write_none_metadata_keeps_footer_clean(tmp_path):
    dest = tmp_path / "m.parquet"
    ParquetStore.write(_df(2), dest, metadata=None)
    footer_meta = pq.read_schema(dest).metadata or {}
    assert ParquetStore.METADATA_KEY.encode() not in footer_meta


# ---------------------------------------------------------------------------
# ParquetStore.combine
# ---------------------------------------------------------------------------


def test_combine_empty_sources_noop(tmp_path):
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([], dest)
    assert not dest.exists()


def test_combine_single_source(tmp_path):
    src = _write_src(tmp_path, "a.parquet", _df(3))
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([src], dest)
    back, meta = ParquetStore.read(dest)
    assert meta is None
    assert back.equals(_df(3))


def test_combine_multiple_diagonal(tmp_path):
    df_a = _df(3)
    df_b = pl.DataFrame(
        {"time_ns": [10_000], "other": ["x"]},
        schema=pl.Schema({"time_ns": pl.Int64, "other": pl.String}),
    )
    src_a = _write_src(tmp_path, "a.parquet", df_a)
    src_b = _write_src(tmp_path, "b.parquet", df_b)
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([src_a, src_b], dest)

    back, _ = ParquetStore.read(dest)
    assert back.height == 4
    assert "label" in back.columns and "other" in back.columns
    assert back.filter(pl.col("label").is_not_null()).height == 3
    assert back.filter(pl.col("other").is_not_null()).height == 1


def test_combine_schema_mismatch_raises(tmp_path):
    df_a = pl.DataFrame({"a": [1, 2]}, schema=pl.Schema({"a": pl.Int64}))
    df_b = pl.DataFrame({"a": [[1], [2]]}, schema=pl.Schema({"a": pl.List(pl.Int64)}))
    src_a = _write_src(tmp_path, "a.parquet", df_a)
    src_b = _write_src(tmp_path, "b.parquet", df_b)
    dest = tmp_path / "combined.parquet"
    with pytest.raises(SchemaViolationError):
        ParquetStore.combine([src_a, src_b], dest)
    assert not dest.exists()


def test_combine_skips_unreadable_sources(tmp_path, capsys):
    # NOTE: the module logger is hijacked by the `launch` LaunchLogger
    # (propagate=False), so caplog cannot see the warning; assert behaviour
    # (and stderr output where the LaunchLogger handler writes it) instead.
    good = _write_src(tmp_path, "good.parquet", _df(2))
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([good, tmp_path / "missing.parquet"], dest)
    back, _ = ParquetStore.read(dest)
    assert back.equals(_df(2))


def test_combine_all_sources_unreadable(tmp_path):
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([tmp_path / "m1.parquet", tmp_path / "m2.parquet"], dest)
    assert not dest.exists()


@given(
    n=st.integers(min_value=0, max_value=20),
    with_str=st.booleans(),
    with_bool=st.booleans(),
)
@settings(
    max_examples=100,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_roundtrip_hypothesis(tmp_path, n, with_str, with_bool):
    cols = {
        "a": [float(i) for i in range(n)],
        "t": [i * 1000 for i in range(n)],
    }
    schema: dict = {"a": pl.Float64, "t": pl.Int64}
    if with_str:
        cols["s"] = [f"x{i}" for i in range(n)]
        schema["s"] = pl.String
    if with_bool:
        cols["b"] = [i % 2 == 0 for i in range(n)]
        schema["b"] = pl.Boolean
    df = pl.DataFrame(cols, schema=pl.Schema(schema))

    # repeated (n, flags) examples may share a file; writes are idempotent
    dest = tmp_path / f"h_{n}_{with_str}_{with_bool}.parquet"
    ParquetStore.write(df, dest, metadata=None)
    back, metadata = ParquetStore.read(dest)
    assert metadata is None
    assert back.equals(df)
    assert back.schema == df.schema


# ---------------------------------------------------------------------------
# TopicParquetStore: the flat telemetry cache
# ---------------------------------------------------------------------------


def _odometry(robot: str, stamps: list[int]) -> pl.DataFrame:
    n = len(stamps)
    return pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": ["env_0"] * n,
            "robot_id": [robot] * n,
            "odom_x": [float(i) for i in range(n)],
            "odom_y": [0.0] * n,
            "odom_yaw": [0.0] * n,
            "odom_vx": [0.0] * n,
            "odom_wz": [0.0] * n,
            "drift_error_m": [0.0] * n,
        }
    )


def _static_transform() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [0],
            "parent_frame": ["map"],
            "child_frame": ["env_0/map"],
            "tx": [0.0],
            "ty": [0.0],
            "tz": [0.0],
            "qx": [0.0],
            "qy": [0.0],
            "qz": [0.0],
            "qw": [1.0],
        }
    )


def test_each_robot_gets_its_own_rows_and_the_globals_are_shared(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    pl.concat([_odometry("env_0_r1", [1000, 2000]), _odometry("env_0_r2", [3000])]).write_parquet(dest / "robot_odom.parquet")
    _static_transform().write_parquet(dest / "tf_static.parquet")

    restored = TopicParquetStore.read(dest)

    assert set(restored) == {"env_0_r1", "env_0_r2"}
    assert restored["env_0_r1"].robot_odom.collect()["t_ns"].to_list() == [1000, 2000]
    assert restored["env_0_r2"].robot_odom.collect()["t_ns"].to_list() == [3000]
    assert restored["env_0_r1"].tf_static is restored["env_0_r2"].tf_static
    assert restored["env_0_r1"].robot_tracks is None


def test_tables_read_back_with_their_declared_types(tmp_path):
    dest = tmp_path / "telemetry"
    sink = TableSink(dest)
    sink.append("robot_odom", {"t_ns": 1000, "env_id": "env_0", "robot_id": "env_0_r1", "odom_x": 1.0, "odom_y": 2.0, "odom_yaw": 0.0, "odom_vx": 0.5, "odom_wz": 0.1, "drift_error_m": None})
    sink.close()

    frame = TopicParquetStore.read(dest)["env_0_r1"].robot_odom.collect()

    assert frame.schema["t_ns"] == pl.Int64
    assert frame.schema["robot_id"] == pl.String
    assert frame.schema["odom_x"] == pl.Float64


def test_the_sink_sorts_a_table_written_out_of_order(tmp_path):
    dest = tmp_path / "telemetry"
    sink = TableSink(dest)
    for stamp in (3000, 1000, 2000):
        sink.append("robot_odom", {"t_ns": stamp, "env_id": "env_0", "robot_id": "env_0_r1", "odom_x": float(stamp), "odom_y": 0.0, "odom_yaw": 0.0, "odom_vx": 0.0, "odom_wz": 0.0, "drift_error_m": None})
    sink.close()

    assert pl.read_parquet(dest / "robot_odom.parquet")["t_ns"].to_list() == [1000, 2000, 3000]


def test_a_table_without_rows_is_absent_from_its_bundle(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    _odometry("env_0_r1", [1000]).write_parquet(dest / "robot_odom.parquet")

    bundle = TopicParquetStore.read(dest)["env_0_r1"]

    assert bundle.robot_odom is not None
    assert bundle.robot_plans is None
    assert bundle.episode_lifecycle is None


def test_read_missing_or_file_dir(tmp_path):
    assert TopicParquetStore.read(tmp_path / "nope") is None
    f = tmp_path / "afile"
    f.write_text("x")
    assert TopicParquetStore.read(f) is None


def test_read_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert TopicParquetStore.read(empty) is None


def test_read_globals_only_returns_none(tmp_path):
    """Global tables with no robot rows produce no bundles -> None."""
    dest = tmp_path / "cache"
    dest.mkdir()
    _static_transform().write_parquet(dest / "tf_static.parquet")
    assert TopicParquetStore.read(dest) is None


def test_a_corrupt_table_reads_as_a_cache_miss(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / "robot_odom.parquet").write_bytes(b"not parquet data")
    assert TopicParquetStore.read(dest) is None


def test_cache_is_current_only_once_marked_complete(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    assert not TopicParquetStore.is_current(dest)

    TopicParquetStore.mark_complete(dest)

    assert (dest / TopicParquetStore.VERSION_FILE).read_text().strip() == str(TopicParquetStore.VERSION)
    assert TopicParquetStore.is_current(dest)


def test_cache_stamped_by_another_layout_is_not_current(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / TopicParquetStore.VERSION_FILE).write_text(f"{TopicParquetStore.VERSION + 1}\n")

    assert not TopicParquetStore.is_current(dest)


def test_write_rows_status_rows_first_keep_metric_columns(tmp_path: pathlib.Path) -> None:
    # status rows finish first under as_completed
    status = {"episode": 0, "status": "no_trajectory", "success": False}
    metric = {"episode": 1, "status": "evaluated", "success": True, "time": [1, 2, 3], "path_length": 4.2}
    dest = tmp_path / "combined.parquet"
    ParquetStore.write_rows([dict(status, episode=i) for i in range(150)] + [metric], dest, schema_overrides={"success": pl.Boolean})
    df, _ = ParquetStore.read(dest)
    assert df.columns[-2:] == ["time", "path_length"]
    assert df["time"].null_count() == 150
    assert df["path_length"].drop_nulls().to_list() == [4.2]
