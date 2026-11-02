from __future__ import annotations

import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_robots_msgs.msg")

import polars as pl
import pyarrow.parquet as pq
from arena_robots_msgs.msg import CollisionEvent, CollisionEvents
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
