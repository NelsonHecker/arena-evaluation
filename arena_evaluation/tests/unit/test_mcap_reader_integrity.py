from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

import polars as pl
from mcap_ros2.writer import Writer

from arena_evaluation.processing.mcap_reader import MCAPReader

SEC = 1_000_000_000
NAME = "arena_runtime_msgs/msg/EpisodeRecord"
CURRENT = "uint32 episode_id\nuint8 outcome_state\nstring outcome_info\nstring goal_uuid\nstring[] integrity"
BOOL_FIELD = "uint32 episode_id\nuint8 outcome_state\nstring outcome_info\nstring goal_uuid\nbool integrity"
NO_FIELD = "uint32 episode_id\nuint8 outcome_state\nstring outcome_info\nstring goal_uuid"


def _bag(path: pathlib.Path, records: list[tuple[str, str, dict, int]]) -> pathlib.Path:
    with path.open("wb") as fh:
        writer = Writer(fh)
        schemas = {}
        for topic, text, msg, log_time in records:
            if text not in schemas:
                schemas[text] = writer.register_msgdef(NAME, text)
            writer.write_message(topic, schemas[text], msg, log_time=log_time)
        writer.finish()
    return path


def _record(state: int, **fields: object) -> dict:
    return {"episode_id": 1, "outcome_state": state, "outcome_info": "", "goal_uuid": "", **fields}


@pytest.fixture
def extracted(tmp_path: pathlib.Path) -> pathlib.Path:
    bag = _bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/state/episode", CURRENT, _record(1, integrity=[]), 1 * SEC),
            ("/env_0/state/episode", CURRENT, _record(1, integrity=["robot jackal spawned by hand"]), 2 * SEC),
            ("/env_0/state/episode", CURRENT, _record(2, integrity=["robot jackal spawned by hand", "task parameters changed mid-episode"]), 3 * SEC),
            ("/env_1/state/episode", BOOL_FIELD, _record(2, integrity=True), 1 * SEC),
            ("/env_2/state/episode", NO_FIELD, _record(3), 1 * SEC),
        ],
    )
    out = tmp_path / "topics"
    MCAPReader(bag).read(out)
    return out


def test_each_flag_of_each_record_is_kept_as_it_accrues(extracted: pathlib.Path) -> None:
    flags = pl.read_parquet(extracted / "episode_integrity.parquet")
    assert flags.select("t_ns", "env_id", "flag_idx", "flag").rows() == [
        (2 * SEC, "env_0", 0, "robot jackal spawned by hand"),
        (3 * SEC, "env_0", 0, "robot jackal spawned by hand"),
        (3 * SEC, "env_0", 1, "task parameters changed mid-episode"),
    ]


def test_a_boolean_integrity_of_an_older_schema_carries_no_reasons(extracted: pathlib.Path) -> None:
    assert "env_1" not in pl.read_parquet(extracted / "episode_integrity.parquet")["env_id"].to_list()


def test_a_record_without_integrity_carries_no_reasons(extracted: pathlib.Path) -> None:
    assert "env_2" not in pl.read_parquet(extracted / "episode_integrity.parquet")["env_id"].to_list()
