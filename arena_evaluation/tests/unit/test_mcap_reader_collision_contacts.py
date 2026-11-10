from __future__ import annotations

import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_robots_msgs.msg")
pytest.importorskip("mcap_ros2.writer")

import polars as pl
import pyarrow.parquet as pq
from arena_robots_msgs.msg import CollisionEvent, CollisionEvents
from mcap_ros2.writer import Writer
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import MCAPReader

_TOPIC = "/arena/env_0/task_generator_node/jackal/collision_events"


def test_every_collision_event_becomes_a_contact_row(tmp_path: pathlib.Path) -> None:
    msg = CollisionEvents()
    msg.header.stamp.sec = 3
    ped = CollisionEvent(kind=CollisionEvent.KIND_PEDESTRIAN, obstacle_id="env_0/Pedestrian_6", distance=0.0)
    ped.obstacle_position.x, ped.obstacle_position.y = 22.5, 48.25
    msg.events = [ped, CollisionEvent(kind=CollisionEvent.KIND_WALL, obstacle_id="wall_2", distance=0.0)]
    bag = tmp_path / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, _TOPIC, "arena_robots_msgs/msg/CollisionEvents", "cdr"))
    writer.write(_TOPIC, serialize_message(msg), 3_000_000_000)
    writer.close()
    bundles = MCAPReader(bag).read(tmp_path / "out")

    rows = pq.read_table(tmp_path / "out" / "env_0_jackal" / "collision_contacts.parquet").to_pylist()
    assert rows == [
        {"time_ns": 3_000_000_000, "stamp_ns": 3_000_000_000, "kind": "pedestrian", "obstacle_id": "env_0/Pedestrian_6", "obstacle_x": 22.5, "obstacle_y": 48.25, "obstacle_z": 0.0, "distance": 0.0},
        {"time_ns": 3_000_000_000, "stamp_ns": 3_000_000_000, "kind": "wall", "obstacle_id": "wall_2", "obstacle_x": 0.0, "obstacle_y": 0.0, "obstacle_z": 0.0, "distance": 0.0},
    ]
    contacts = bundles["env_0_jackal"].collision_contacts
    assert isinstance(contacts, pl.LazyFrame)
    assert contacts.collect().height == 2


SEC = 1_000_000_000
SEP = "\n" + "=" * 80 + "\n"
TIME = "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec"
HEADER = "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id"
OLD_COLLISION_EVENTS = ("arena_robots_msgs/msg/CollisionEvents", SEP.join(["std_msgs/Header header\narena_robots_msgs/CollisionEvent[] events", HEADER, TIME, "MSG: arena_robots_msgs/CollisionEvent\nstring obstacle_id"]))
NO_HEADER_COLLISION_EVENTS = ("arena_robots_msgs/msg/CollisionEvents", SEP.join(["arena_robots_msgs/CollisionEvent[] events", "MSG: arena_robots_msgs/CollisionEvent\nstring obstacle_id"]))


def _write_rosbag(bag: pathlib.Path, msg: CollisionEvents, log_time: int) -> pathlib.Path:
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, _TOPIC, "arena_robots_msgs/msg/CollisionEvents", "cdr"))
    writer.write(_TOPIC, serialize_message(msg), log_time)
    writer.close()
    return bag


def _write_trimmed_bag(path: pathlib.Path, definition: tuple[str, str], message: dict[str, object], log_time: int) -> pathlib.Path:
    with path.open("wb") as fh:
        writer = Writer(fh)
        schema = writer.register_msgdef(*definition)
        writer.write_message(_TOPIC, schema, message, log_time=log_time)
        writer.finish()
    return path


def test_empty_collision_events_keep_an_empty_typed_contacts_table(tmp_path: pathlib.Path) -> None:
    msg = CollisionEvents()
    msg.header.stamp.sec = 4
    bag = _write_rosbag(tmp_path / "bag", msg, 4 * SEC)
    bundles = MCAPReader(bag).read(tmp_path / "out")

    path = tmp_path / "out" / "env_0_jackal" / "collision_contacts.parquet"
    assert pq.read_table(path).num_rows == 0
    assert list(pl.read_parquet(path).schema.items()) == [
        ("time_ns", pl.Int64),
        ("stamp_ns", pl.Int64),
        ("kind", pl.String),
        ("obstacle_id", pl.String),
        ("obstacle_x", pl.Float64),
        ("obstacle_y", pl.Float64),
        ("obstacle_z", pl.Float64),
        ("distance", pl.Float64),
    ]
    contacts = bundles["env_0_jackal"].collision_contacts
    assert isinstance(contacts, pl.LazyFrame)
    assert contacts.collect().is_empty()


def test_old_collision_events_read_missing_fields_as_null(tmp_path: pathlib.Path) -> None:
    message = {"header": {"stamp": {"sec": 3, "nanosec": 5}, "frame_id": "map"}, "events": [{"obstacle_id": "w1"}]}
    bag = _write_trimmed_bag(tmp_path / "old.mcap", OLD_COLLISION_EVENTS, message, 4 * SEC)
    MCAPReader(bag).read(tmp_path / "out")

    rows = pq.read_table(tmp_path / "out" / "env_0_jackal" / "collision_contacts.parquet").to_pylist()
    assert rows == [{"time_ns": 4 * SEC, "stamp_ns": 3 * SEC + 5, "kind": None, "obstacle_id": "w1", "obstacle_x": None, "obstacle_y": None, "obstacle_z": None, "distance": None}]


def test_collision_events_without_a_header_read_a_null_stamp_ns(tmp_path: pathlib.Path) -> None:
    bag = _write_trimmed_bag(tmp_path / "bare.mcap", NO_HEADER_COLLISION_EVENTS, {"events": [{"obstacle_id": "w1"}]}, 4 * SEC)
    MCAPReader(bag).read(tmp_path / "out")

    rows = pq.read_table(tmp_path / "out" / "env_0_jackal" / "collision_contacts.parquet").to_pylist()
    assert rows == [{"time_ns": 4 * SEC, "stamp_ns": None, "kind": None, "obstacle_id": "w1", "obstacle_x": None, "obstacle_y": None, "obstacle_z": None, "distance": None}]
