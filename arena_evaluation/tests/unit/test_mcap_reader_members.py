from collections import defaultdict

import pytest

pytest.importorskip("polars")

from arena_evaluation.processing.mcap_reader import MCAPReader


class _FakeEntity:
    def __init__(self, entity, kind, members):
        self.entity = entity
        self.kind = kind
        self.discrete_names = ["state"]
        self.discrete_values = ["open"]
        self.continuous_names = []
        self.continuous_values = []
        self.predicate_names = ["over_cap"]
        self.predicate_values = [True]
        self.members = members


def test_append_semantic_entity_includes_members_column():
    target = defaultdict(list)
    ent = _FakeEntity("env_0/lobby", "occupancy_cap", ["env_0/robot_0", "env_0/human_1"])

    MCAPReader._append_semantic_entity(target, 1_000_000_000, 0, "hospital_small", ent)

    assert target["time_ns"] == [1_000_000_000]
    assert target["env_id"] == [0]
    assert target["world"] == ["hospital_small"]
    assert target["entity"] == ["env_0/lobby"]
    assert target["kind"] == ["occupancy_cap"]
    assert target["discrete_names"] == [["state"]]
    assert target["predicate_values"] == [[True]]
    assert target["members"] == [["env_0/robot_0", "env_0/human_1"]]


def test_append_semantic_entity_defaults_to_empty_members():
    target = defaultdict(list)
    ent = _FakeEntity("env_0/door_1", "door", [])

    MCAPReader._append_semantic_entity(target, 0, 0, "world", ent)

    assert target["members"] == [[]]


def test_append_semantic_entity_accumulates_across_calls():
    target = defaultdict(list)
    ent_a = _FakeEntity("env_0/lobby", "occupancy_cap", ["env_0/robot_0"])
    ent_b = _FakeEntity("env_0/elevator_1", "elevator", ["env_0/robot_0", "env_0/robot_1"])

    MCAPReader._append_semantic_entity(target, 0, 0, "world", ent_a)
    MCAPReader._append_semantic_entity(target, 1_000_000_000, 0, "world", ent_b)

    assert target["entity"] == ["env_0/lobby", "env_0/elevator_1"]
    assert target["members"] == [["env_0/robot_0"], ["env_0/robot_0", "env_0/robot_1"]]
