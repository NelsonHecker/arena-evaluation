import json
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


def test_append_semantic_entity_flattens_fields_into_long_rows():
    target = defaultdict(list)
    ent = _FakeEntity("env_0/lobby", "occupancy_cap", ["env_0/robot_0", "env_0/human_1"])

    MCAPReader._append_semantic_entity(target, 1_000_000_000, 0, "hospital_small", ent)

    assert target["time_ns"] == [1_000_000_000, 1_000_000_000, 1_000_000_000]
    assert target["env_id"] == [0, 0, 0]
    assert target["world"] == ["hospital_small"] * 3
    assert target["entity"] == ["env_0/lobby"] * 3
    assert target["kind"] == ["occupancy_cap"] * 3
    assert target["field"] == ["state", "over_cap", "members"]
    assert target["field_kind"] == ["discrete", "predicate", "members"]
    assert target["value_str"] == ["open", None, None]
    assert target["value_num"] == [None, None, None]
    assert target["value_bool"] == [None, True, None]
    assert target["value_list"] == [None, None, ["env_0/robot_0", "env_0/human_1"]]


def test_append_semantic_entity_defaults_to_empty_members():
    target = defaultdict(list)
    ent = _FakeEntity("env_0/door_1", "door", [])

    MCAPReader._append_semantic_entity(target, 0, 0, "world", ent)

    assert target["value_list"][-1] == []
    assert target["field"][-1] == "members"


def test_append_semantic_entity_accumulates_across_calls():
    target = defaultdict(list)
    ent_a = _FakeEntity("env_0/lobby", "occupancy_cap", ["env_0/robot_0"])
    ent_b = _FakeEntity("env_0/elevator_1", "elevator", ["env_0/robot_0", "env_0/robot_1"])

    MCAPReader._append_semantic_entity(target, 0, 0, "world", ent_a)
    MCAPReader._append_semantic_entity(target, 1_000_000_000, 0, "world", ent_b)

    assert target["entity"] == ["env_0/lobby"] * 3 + ["env_0/elevator_1"] * 3
    assert target["value_list"][2] == ["env_0/robot_0"]
    assert target["value_list"][5] == ["env_0/robot_0", "env_0/robot_1"]


def test_append_semantic_field_appends_one_row():
    target = defaultdict(list)

    MCAPReader._append_semantic_field(
        target,
        2_000_000_000,
        1,
        "world",
        "env_1/door_1",
        "door",
        "open",
        "predicate",
        value_bool=True,
    )

    assert target["time_ns"] == [2_000_000_000]
    assert target["env_id"] == [1]
    assert target["entity"] == ["env_1/door_1"]
    assert target["kind"] == ["door"]
    assert target["field"] == ["open"]
    assert target["field_kind"] == ["predicate"]
    assert target["value_bool"] == [True]
    assert target["value_str"] == [None]
    assert target["value_num"] == [None]
    assert target["value_list"] == [None]


def _phase(name: str = "ramp_up_vx_1.00_h_1.00", kind: str = "ramp_up", **overrides: object) -> dict:
    phase = {
        "name": name,
        "kind": kind,
        "vx_target": 1.0,
        "vy_target": 0.0,
        "wz_target": 0.0,
        "duration_s": 1.0,
        "ramp_s": 1.0,
        "radius_m": 0.0,
    }
    phase.update(overrides)
    return phase


def test_schedule_rows_parses_a_valid_payload():
    payload = json.dumps(
        {
            "robot": "env_0_jackal",
            "model": "jackal",
            "envelope": {"vx_max": 2.0},
            "phases": [
                _phase(),
                _phase(name="arc_vx_0.50_r_1.00_left", kind="arc", vx_target=0.5, wz_target=0.5, radius_m=1.0, ramp_s=0.0),
            ],
        }
    )
    rows = MCAPReader._schedule_rows(payload)
    assert rows is not None
    assert rows["phase_label"] == ["ramp_up_vx_1.00_h_1.00", "arc_vx_0.50_r_1.00_left"]
    assert rows["phase_kind"] == ["ramp_up", "arc"]
    assert rows["vx_target"] == [1.0, 0.5]
    assert rows["wz_target"] == [0.0, 0.5]
    # turn_radius_m is taken from the phase's radius_m field
    assert rows["turn_radius_m"] == [0.0, 1.0]
    assert set(rows) == {
        "phase_label",
        "phase_kind",
        "vx_target",
        "vy_target",
        "wz_target",
        "duration_s",
        "ramp_s",
        "turn_radius_m",
    }


def test_schedule_rows_rejects_a_payload_missing_phases():
    payload = json.dumps({"robot": "env_0_jackal", "model": "jackal", "envelope": {}})
    assert MCAPReader._schedule_rows(payload) is None


def test_schedule_rows_rejects_a_phase_missing_a_key():
    incomplete = _phase()
    del incomplete["radius_m"]
    payload = json.dumps({"phases": [incomplete]})
    assert MCAPReader._schedule_rows(payload) is None


def test_schedule_rows_rejects_non_json():
    assert MCAPReader._schedule_rows("not json") is None


def test_frame_env_comes_from_the_frame_id_not_the_topic():
    # /tf carries every env, so a robot in env_1 must not be filed under the topic's default env_0.
    assert MCAPReader._frame_env("env_1/jackal/base_link", "env_0") == "env_1"
    assert MCAPReader._frame_env("jackal/base_link", "env_0") == "env_0"
