from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
import pytest
from flat_peds import flat_peds

from arena_evaluation.processing.metrics.safety.contact_attribution import (
    ContactAttributionCalculator,
    classify_contact_fault,
)
from arena_evaluation.processing.telemetry.writer import TABLE_SCHEMAS
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SECOND_NS = 1_000_000_000
SAMPLE_S = 0.05
ROBOT_RADIUS_M = 0.3
PED_RADIUS_M = 0.3


def _schema(topic: str) -> pl.Schema:
    """Recorded flat schema of a telemetry table."""
    return pl.from_arrow(TABLE_SCHEMAS[topic].empty_table()).schema


def _template(topic: str) -> pl.DataFrame:
    """Empty flat table carrying the recorded schema of `topic`."""
    return pl.DataFrame(schema=_schema(topic))


def _table(topic: str, rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    """Flat table of `rows` with the recorded schema of `topic`."""
    defaults = dict.fromkeys(_schema(topic))
    return pl.DataFrame([{**defaults, **row} for row in rows], schema=_schema(topic))


def _robot_tracks(t_s: np.ndarray, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> pl.DataFrame:
    """Flat robot_tracks rows sampled at `t_s`."""
    n = len(t_s)
    zeros = np.zeros(n, dtype=np.float64)
    return pl.DataFrame(
        {
            "t_ns": (t_s * SECOND_NS).astype(np.int64),
            "env_id": ["env_0"] * n,
            "robot_id": ["env_0_jackal"] * n,
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "z": zeros,
            "yaw": zeros.copy(),
            "vx": np.asarray(vx, dtype=np.float64),
            "vy": np.asarray(vy, dtype=np.float64),
            "vz": zeros.copy(),
            "omega_z": zeros.copy(),
            "speed_mps": np.hypot(np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64)),
            "pose_source": ["tf_gt"] * n,
        },
        schema=_schema("robot_tracks"),
    )


def _pedestrian_tracks(frames: Sequence[tuple[int, Sequence[Mapping[str, object]]]]) -> pl.DataFrame:
    """Flat pedestrian_tracks rows with a recorded pedestrian radius."""
    return flat_peds(frames).with_columns(pl.lit(PED_RADIUS_M, dtype=pl.Float64).alias("radius_m"))


def _episode(topics: Mapping[str, pl.DataFrame] | None) -> AlignedEpisodeBundle:
    """Episode bundle carrying the given flat topic tables."""
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[3.0, 0.0, 0.0],
        topics=None if topics is None else dict(topics),
    )


def _calculator(config: Mapping[str, object] | None = None) -> ContactAttributionCalculator:
    """Contact attribution calculator with the given metrics config."""
    calc = ContactAttributionCalculator(RobotParams(robot_radius=ROBOT_RADIUS_M))
    if config is not None:
        calc.metrics_config = dict(config)
    return calc


def _stationary_robot_episode() -> AlignedEpisodeBundle:
    """Stationary robot struck by Pedestrian_10 walking at 1 m/s along -x."""
    t_s = np.arange(0.0, 3.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, zeros, zeros, zeros, zeros)
    frames = [(int(t_ns), [{"ped_id": "Pedestrian_10", "ped_name": "Pedestrian_10", "x": 2.0 - t, "y": 0.0, "vx": -1.0, "vy": 0.0}]) for t, t_ns in zip(t_s, (t_s * SECOND_NS).astype(np.int64), strict=True)]
    return _episode({"robot_tracks": robot, "pedestrian_tracks": _pedestrian_tracks(frames)})


def _moving_robot_episode() -> AlignedEpisodeBundle:
    """Robot driving at 1.2 m/s along +x toward Pedestrian_5 standing at x=2 m."""
    t_s = np.arange(0.0, 3.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, 1.2 * t_s, zeros, np.full_like(t_s, 1.2), zeros)
    frames = [(int(t_ns), [{"ped_id": "Pedestrian_5", "ped_name": "Pedestrian_5", "x": 2.0, "y": 0.0, "vx": 0.0, "vy": 0.0}]) for t_ns in (t_s * SECOND_NS).astype(np.int64)]
    return _episode({"robot_tracks": robot, "pedestrian_tracks": _pedestrian_tracks(frames)})


def _multi_contact_episode() -> AlignedEpisodeBundle:
    """Robot passing two standing pedestrians, with a wall contact in the collision_contacts table."""
    t_s = np.arange(0.0, 6.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, t_s, zeros, np.ones_like(t_s), zeros)
    frames = [
        (
            int(t_ns),
            [
                {"ped_id": "Pedestrian_1", "ped_name": "Pedestrian_1", "x": 1.5, "y": 0.0, "vx": 0.0, "vy": 0.0},
                {"ped_id": "Pedestrian_2", "ped_name": "Pedestrian_2", "x": 3.5, "y": 0.0, "vx": 0.0, "vy": 0.0},
            ],
        )
        for t_ns in (t_s * SECOND_NS).astype(np.int64)
    ]
    contacts = _table(
        "collision_contacts",
        [
            {"t_ns": SECOND_NS, "obstacle_id": "env_0/Pedestrian_1", "kind": "pedestrian", "obstacle_x": 1.5, "obstacle_y": 0.0},
            {"t_ns": 3 * SECOND_NS, "obstacle_id": "env_0/Pedestrian_2", "kind": "pedestrian", "obstacle_x": 3.5, "obstacle_y": 0.0},
            {"t_ns": 5 * SECOND_NS, "obstacle_id": "wall_north", "kind": "wall", "obstacle_x": 5.5, "obstacle_y": 0.0},
        ],
    )
    return _episode({"robot_tracks": robot, "pedestrian_tracks": _pedestrian_tracks(frames), "collision_contacts": contacts})


def test_classify_closer_label_thresholds() -> None:
    """Fault matrix: wall, dominant robot, dominant pedestrian, mutual and two bodies at rest."""
    static = classify_contact_fault(collider_kind="wall", v_close_robot_mps=0.8, v_close_target_mps=0.0, counterfactual_freeze_persists=None, counterfactual_regime="unavailable")
    assert (static.closer_label, static.fault_attributed_to, static.is_robot_fault) == ("static", "robot", True)

    ped = classify_contact_fault(collider_kind="pedestrian", v_close_robot_mps=0.02, v_close_target_mps=1.0, counterfactual_freeze_persists=True, counterfactual_regime="blind_exact")
    assert (ped.closer_label, ped.fault_attributed_to, ped.is_robot_fault) == ("pedestrian", "pedestrian", False)
    assert ped.robot_closing_share < 0.05

    robot = classify_contact_fault(collider_kind="pedestrian", v_close_robot_mps=1.2, v_close_target_mps=0.01, counterfactual_freeze_persists=False, counterfactual_regime="blind_exact")
    assert (robot.closer_label, robot.fault_attributed_to, robot.is_robot_fault) == ("robot", "robot", True)
    assert robot.robot_closing_share > 0.95

    mutual = classify_contact_fault(collider_kind="pedestrian", v_close_robot_mps=0.6, v_close_target_mps=0.6, counterfactual_freeze_persists=True, counterfactual_regime="blind_exact")
    assert (mutual.closer_label, mutual.fault_attributed_to, mutual.is_robot_fault) == ("mutual", "unavoidable", None)
    assert mutual.robot_closing_share == pytest.approx(0.5)

    dominant_robot = classify_contact_fault(collider_kind="pedestrian", v_close_robot_mps=0.75, v_close_target_mps=0.25, counterfactual_freeze_persists=None, counterfactual_regime="unavailable")
    assert dominant_robot.closer_label == "robot"

    dominant_ped = classify_contact_fault(collider_kind="pedestrian", v_close_robot_mps=0.25, v_close_target_mps=0.75, counterfactual_freeze_persists=None, counterfactual_regime="unavailable")
    assert dominant_ped.closer_label == "pedestrian"


@pytest.mark.parametrize(
    ("v_robot", "v_target", "persists", "expected_fault", "expected_strict"),
    [
        (0.0, 0.0, True, "unavoidable", None),
        (0.0, 0.0, None, "unavoidable", None),
        (0.5, 0.5, False, "mutual", True),
    ],
    ids=["two-bodies-at-rest", "no-counterfactual", "avoidable-mutual"],
)
def test_mutual_contacts_are_never_silently_robot_faults(v_robot: float, v_target: float, persists: bool | None, expected_fault: str, expected_strict: bool | None) -> None:
    policy = classify_contact_fault(
        collider_kind="pedestrian",
        v_close_robot_mps=v_robot,
        v_close_target_mps=v_target,
        counterfactual_freeze_persists=persists,
        counterfactual_regime="blind_exact",
    )

    assert policy.closer_label == "mutual"
    assert policy.fault_attributed_to == expected_fault
    assert policy.is_robot_fault is expected_strict


def test_a_robot_collider_is_never_labelled_a_pedestrian() -> None:
    other_closes = classify_contact_fault(collider_kind="robot", v_close_robot_mps=0.1, v_close_target_mps=0.9, counterfactual_freeze_persists=None, counterfactual_regime="blind_exact")

    assert other_closes.closer_label == "other_robot"
    assert other_closes.fault_attributed_to == "other_robot"
    assert other_closes.is_robot_fault is False


def test_an_unknown_target_speed_leaves_every_verdict_null() -> None:
    policy = classify_contact_fault(
        collider_kind="pedestrian",
        v_close_robot_mps=1.0,
        v_close_target_mps=0.0,
        counterfactual_freeze_persists=None,
        counterfactual_regime="unavailable",
        target_speed_known=False,
    )

    assert policy.closer_label is None
    assert policy.fault_attributed_to is None
    assert policy.robot_closing_share is None
    assert policy.is_robot_fault is None


@pytest.mark.parametrize(("regime", "strict", "lower"), [("blind_exact", True, None), ("reactive_approx", None, True)])
def test_the_evidence_standard_gates_the_fault_verdict(regime: str, strict: bool | None, lower: bool | None) -> None:
    policy = classify_contact_fault(
        collider_kind="pedestrian",
        v_close_robot_mps=1.2,
        v_close_target_mps=0.01,
        counterfactual_freeze_persists=False,
        counterfactual_regime=regime,  # type: ignore[arg-type]
    )

    assert policy.is_robot_fault is strict
    assert policy.is_robot_fault_lower_bound is lower


def test_stationary_robot_struck_by_pedestrian_in_blind_regime_is_not_robot_fault() -> None:
    out = _calculator({"crowd_condition": "blind"}).calculate(_stationary_robot_episode(), {})

    assert out["contact_count"] == 1
    assert out["closer_label"] == "pedestrian"
    assert out["counterfactual_regime"] == "blind_exact"
    assert out["counterfactual_freeze_persists"] is True
    assert out["is_robot_fault"] is False
    assert out["is_robot_fault_lower_bound"] is None
    assert out["contact_robot_fault_count"] == 0
    assert out["contact_passive_ped_impact_count"] == 1
    assert out["contact_impact_speed_mps"] == pytest.approx(1.0, abs=1e-2)


def test_moving_robot_avoidable_by_freezing_is_robot_fault_and_respects_reactive_regime() -> None:
    out = _calculator({"crowd_condition": "visible"}).calculate(_moving_robot_episode(), {})

    assert out["contact_count"] == 1
    assert out["closer_label"] == "robot"
    assert out["counterfactual_regime"] == "reactive_approx"
    assert out["counterfactual_freeze_persists"] is False
    assert out["is_robot_fault"] is None
    assert out["is_robot_fault_lower_bound"] is True
    assert out["contact_impact_speed_mps"] == pytest.approx(1.2, abs=1e-2)


def test_multi_contact_episode_attributes_every_contact_including_static() -> None:
    out = _calculator({"crowd_condition": "blind"}).calculate(_multi_contact_episode(), {})

    assert out["contact_count"] == 3
    assert out["contact_closer_robot_count"] == 2
    assert out["contact_closer_static_count"] == 1
    # The wall contact is the robot's fault too, so every contact counts.
    assert out["contact_robot_fault_count"] == 3
    assert out["contact_passive_ped_impact_count"] == 0
    records = out["contact_records"]
    assert [record["t0_s"] for record in records] == pytest.approx([1.0, 3.0, 5.0])
    assert records[0]["target_ped_name"] == "Pedestrian_1"
    assert records[1]["target_ped_name"] == "Pedestrian_2"
    assert records[2]["closer_label"] == "static"
    assert records[2]["obstacle_kind"] == "wall"
    assert records[2]["counterfactual_regime"] == "unavailable"
    assert records[2]["counterfactual_freeze_min_dist_m"] is None
    assert records[2]["fault_attributed_to"] == "robot"
    assert records[2]["is_robot_fault"] is True


@pytest.mark.parametrize("topics", [None, {"robot_tracks": _template("robot_tracks")}], ids=["absent", "empty"])
def test_absent_or_empty_robot_tracks_yields_zeroed_outputs(topics: Mapping[str, pl.DataFrame] | None) -> None:
    out = _calculator().calculate(_episode(topics), {})

    assert set(out) == set(ContactAttributionCalculator.output_keys())
    assert out["contact_count"] == 0
    assert out["closer_label"] is None
    assert out["counterfactual_regime"] is None
    assert out["counterfactual_freeze_persists"] is None
    assert out["is_robot_fault"] is None
    assert out["is_robot_fault_lower_bound"] is None
    assert out["contact_impact_speed_mps"] is None
    assert out["contact_robot_fault_count"] is None
    assert out["contact_passive_ped_impact_count"] is None
    assert out["contact_records"] == []


def test_contact_free_episode_reports_zero_contacts() -> None:
    out = _calculator().calculate(_episode({"robot_tracks": _template("robot_tracks"), "pedestrian_tracks": _template("pedestrian_tracks")}), {})

    assert out["contact_count"] == 0
    assert out["contact_records"] == []


def test_empty_collision_contacts_falls_back_to_clearance() -> None:
    episode = _stationary_robot_episode()
    episode.topics["collision_contacts"] = _template("collision_contacts")

    out = _calculator({"crowd_condition": "blind"}).calculate(episode, {})

    assert out["contact_count"] == 1
    assert out["closer_label"] == "pedestrian"


@pytest.mark.parametrize(
    ("kind", "obstacle_id", "expected_kind", "expected_closer"),
    [
        ("pedestrian", "env_0/Pedestrian_1", "pedestrian", "robot"),
        ("ped_human", "env_0/Pedestrian_1", "pedestrian", "robot"),
        ("human", "env_0/Pedestrian_1", "pedestrian", "robot"),
        ("wall", "wall_north", "wall", "static"),
        ("static", "pillar_south", "static", "static"),
    ],
)
def test_collision_contact_kind_is_normalized(kind: str, obstacle_id: str, expected_kind: str, expected_closer: str) -> None:
    t_s = np.arange(0.0, 3.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, t_s, zeros, np.ones_like(t_s), zeros)
    frames = [(int(t_ns), [{"ped_id": "Pedestrian_1", "ped_name": "Pedestrian_1", "x": 2.0, "y": 0.0, "vx": 0.0, "vy": 0.0}]) for t_ns in (t_s * SECOND_NS).astype(np.int64)]
    contacts = _table(
        "collision_contacts",
        [{"t_ns": SECOND_NS, "obstacle_id": obstacle_id, "kind": kind, "obstacle_x": 2.0, "obstacle_y": 0.0}],
    )
    episode = _episode({"robot_tracks": robot, "pedestrian_tracks": _pedestrian_tracks(frames), "collision_contacts": contacts})

    out = _calculator().calculate(episode, {})

    assert out["contact_count"] == 1
    record = out["contact_records"][0]
    assert record["obstacle_kind"] == expected_kind
    assert record["closer_label"] == expected_closer


def test_pedestrian_contact_without_obstacle_id_uses_nearest_track() -> None:
    t_s = np.arange(0.0, 3.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, zeros, zeros, zeros, zeros)
    frames = [
        (
            int(t_ns),
            [
                {"ped_id": "Pedestrian_1", "ped_name": "Pedestrian_1", "x": 0.4, "y": 0.0, "vx": 0.0, "vy": 0.0},
                {"ped_id": "Pedestrian_2", "ped_name": "Pedestrian_2", "x": 5.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
            ],
        )
        for t_ns in (t_s * SECOND_NS).astype(np.int64)
    ]
    contacts = _table("collision_contacts", [{"t_ns": SECOND_NS, "obstacle_id": "", "kind": "pedestrian", "obstacle_x": 0.4, "obstacle_y": 0.0}])
    episode = _episode({"robot_tracks": robot, "pedestrian_tracks": _pedestrian_tracks(frames), "collision_contacts": contacts})

    out = _calculator().calculate(episode, {})

    record = out["contact_records"][0]
    assert out["contact_count"] == 1
    assert record["target_ped_id"] == "Pedestrian_1"
    assert record["target_ped_name"] == "Pedestrian_1"
    assert record["obstacle_id"] == "Pedestrian_1"


def test_numeric_suffix_resolves_pedestrian_track() -> None:
    t_s = np.arange(0.0, 3.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, zeros, zeros, zeros, zeros)
    frames = [(int(t_ns), [{"ped_id": "Pedestrian_7", "ped_name": "Pedestrian_7", "x": 1.0, "y": 0.0, "vx": 0.0, "vy": 0.0}]) for t_ns in (t_s * SECOND_NS).astype(np.int64)]
    contacts = _table("collision_contacts", [{"t_ns": SECOND_NS, "obstacle_id": "env_0/7", "kind": "pedestrian", "obstacle_x": 1.0, "obstacle_y": 0.0}])
    episode = _episode({"robot_tracks": robot, "pedestrian_tracks": _pedestrian_tracks(frames), "collision_contacts": contacts})

    out = _calculator().calculate(episode, {})

    assert out["contact_count"] == 1
    assert out["contact_records"][0]["target_ped_id"] == "Pedestrian_7"


def test_bumper_events_read_wall_and_static_counts() -> None:
    t_s = np.arange(0.0, 4.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, zeros, zeros, zeros, zeros)
    events = _table(
        "collision_events",
        [
            {"t_ns": SECOND_NS, "event_count": 1, "wall_count": 1, "static_count": 0, "ped_count": 0},
            {"t_ns": 2 * SECOND_NS, "event_count": 1, "wall_count": 0, "static_count": 1, "ped_count": 0},
            {"t_ns": 3 * SECOND_NS, "event_count": 1, "wall_count": 0, "static_count": 0, "ped_count": 1},
        ],
    )
    episode = _episode({"robot_tracks": robot, "collision_events": events})

    out = _calculator().calculate(episode, {})

    assert out["contact_count"] == 2
    assert out["contact_closer_static_count"] == 2
    assert [record["obstacle_kind"] for record in out["contact_records"]] == ["wall", "static"]


@pytest.mark.parametrize(("gap_s", "expected_count", "expected_duration_s"), [(0.2, 1, 0.2), (0.3, 2, 0.0)])
def test_collision_contacts_merge_within_the_gap_and_split_beyond_it(gap_s: float, expected_count: int, expected_duration_s: float) -> None:
    t_s = np.arange(0.0, 3.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, zeros, zeros, zeros, zeros)
    first_ns = SECOND_NS
    second_ns = int(round((1.0 + gap_s) * SECOND_NS))
    contacts = _table(
        "collision_contacts",
        [
            {"t_ns": first_ns, "obstacle_id": "env_0/Pedestrian_1", "kind": "pedestrian", "obstacle_x": 0.5, "obstacle_y": 0.0},
            {"t_ns": second_ns, "obstacle_id": "env_0/Pedestrian_1", "kind": "pedestrian", "obstacle_x": 0.5, "obstacle_y": 0.0},
        ],
    )
    episode = _episode({"robot_tracks": robot, "collision_contacts": contacts})

    out = _calculator().calculate(episode, {})

    assert out["contact_count"] == expected_count
    assert out["contact_records"][0]["duration_s"] == pytest.approx(expected_duration_s, abs=1e-6)


def test_configured_contact_honors_target_agent_id_and_zero_windows() -> None:
    config = {
        "crowd_condition": "blind",
        "contact_t0_ns": int(round(1.4 * SECOND_NS)),
        "contact_t_end_ns": int(round(1.4 * SECOND_NS)),
        "obstacle_kind": "pedestrian",
        "target_agent_id": 10,
        "pre_contact_window_s": 0.0,
        "post_contact_window_s": 0.0,
    }

    out = _calculator(config).calculate(_stationary_robot_episode(), {})

    assert out["contact_count"] == 1
    assert out["closer_label"] == "pedestrian"
    assert out["counterfactual_regime"] == "blind_exact"
    assert out["contact_records"][0]["target_ped_id"] == "Pedestrian_10"


def test_configured_wall_contact_uses_static_point() -> None:
    t_s = np.arange(0.0, 3.0, SAMPLE_S)
    zeros = np.zeros_like(t_s)
    robot = _robot_tracks(t_s, t_s, zeros, np.ones_like(t_s), zeros)
    config = {
        "contact_t0_ns": SECOND_NS,
        "obstacle_kind": "wall",
        "static_point": [2.0, 0.0],
    }

    out = _calculator(config).calculate(_episode({"robot_tracks": robot}), {})

    record = out["contact_records"][0]
    assert out["contact_count"] == 1
    assert record["obstacle_kind"] == "wall"
    assert record["obstacle_id"] == "wall"
    assert record["closer_label"] == "static"
    assert record["v_close_robot_mps"] == pytest.approx(1.0, abs=1e-2)
    assert record["counterfactual_regime"] == "unavailable"
