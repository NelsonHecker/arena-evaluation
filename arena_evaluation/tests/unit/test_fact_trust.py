"""Trust semantics of the published facts and dimensions (07 section 2, 08 H1.3 to H1.5).

The fixture is one 3 s episode at 10 Hz: a robot driving straight along +x, optionally with a
pedestrian crossing its path and with a recorded collision stream. Every expectation is derived
from that geometry or from the policy, never read back from the evaluator.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.scoped_evaluator import ScopedEvaluator, ScopedFactTables
from arena_evaluation.processing.validity import episode_validity, teleport_reason
from arena_evaluation.storage.declarations import (
    NullReason,
    canonical_null_reason,
    is_known_null_reason,
)
from arena_evaluation.storage.schemas import (
    AlignedEpisodeBundle,
    EpisodeDescriptor,
    RobotParams,
    RunMetadata,
)
from arena_evaluation.storage.star_schema import StarSchemaBuilder

NS_PER_S: int = 1_000_000_000
SAMPLE_RATE_HZ: int = 10
SAMPLE_COUNT: int = 30
ROBOT_RADIUS_M: float = 0.25
PED_RADIUS_M: float = 0.3
RUN_ID: str = "run-trust"
EPISODE_ID: int = 1
ROBOT_ID: str = "env_0_jackal"
PED_ID: str = "Pedestrian_1"

_ROBOT_TRACKS_SCHEMA: dict[str, pl.DataType] = {
    "robot_id": pl.String,
    "env_id": pl.String,
    "t_ns": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "yaw": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "pose_source": pl.String,
}

_PEDESTRIAN_TRACKS_SCHEMA: dict[str, pl.DataType] = {
    "t_ns": pl.Int64,
    "env_id": pl.String,
    "ped_id": pl.String,
    "ped_name": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "yaw": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "speed_mps": pl.Float64,
    "radius_m": pl.Float64,
    "desired_speed": pl.Float64,
    "policy": pl.String,
    "agent_type": pl.String,
}

_COLLISION_CONTACTS_SCHEMA: dict[str, pl.DataType] = {
    "t_ns": pl.Int64,
    "obstacle_id": pl.String,
    "kind": pl.String,
    "obstacle_x": pl.Float64,
    "obstacle_y": pl.Float64,
    "obstacle_z": pl.Float64,
    "distance": pl.Float64,
}

_TRANSFORM_SCHEMA: dict[str, pl.DataType] = {
    "t_ns": pl.Int64,
    "parent_frame": pl.String,
    "child_frame": pl.String,
    "tx": pl.Float64,
    "ty": pl.Float64,
    "yaw": pl.Float64,
}

_METRICS: dict[str, object] = {"result": "GOAL_REACHED", "success": True, "pose_frame": "map"}
_WALL_CONTACT_NS: int = 2 * NS_PER_S
_PED_CONTACT_NS: int = 2 * NS_PER_S


def _time_axis() -> np.ndarray:
    """Nanosecond stamps of the 3 s, 10 Hz sample grid."""
    return np.arange(SAMPLE_COUNT, dtype=np.int64) * (NS_PER_S // SAMPLE_RATE_HZ)


def _robot_tracks(*, x_m: np.ndarray | None = None) -> pl.DataFrame:
    """Robot track rows; ``x_m`` overrides the straight 1 m/s path for a planted pose jump."""
    t_ns = _time_axis()
    x = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE_HZ if x_m is None else np.asarray(x_m, dtype=np.float64)
    return pl.DataFrame(
        {
            "robot_id": [ROBOT_ID] * SAMPLE_COUNT,
            "env_id": ["env_0"] * SAMPLE_COUNT,
            "t_ns": t_ns,
            "x": x,
            "y": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "yaw": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "vx": np.ones(SAMPLE_COUNT, dtype=np.float64),
            "vy": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "pose_source": ["tf_gt"] * SAMPLE_COUNT,
        },
        schema=_ROBOT_TRACKS_SCHEMA,
    )


def _pedestrian_tracks() -> pl.DataFrame:
    """One pedestrian crossing the robot path at x = 1.5 m on +y."""
    t_ns = _time_axis()
    times_s = t_ns.astype(np.float64) / NS_PER_S
    speed_mps = 1.2
    return pl.DataFrame(
        {
            "t_ns": t_ns,
            "env_id": ["env_0"] * SAMPLE_COUNT,
            "ped_id": [PED_ID] * SAMPLE_COUNT,
            "ped_name": [PED_ID] * SAMPLE_COUNT,
            "x": np.full(SAMPLE_COUNT, 1.5, dtype=np.float64),
            "y": speed_mps * (times_s - 1.5),
            "yaw": np.full(SAMPLE_COUNT, np.pi / 2.0, dtype=np.float64),
            "vx": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "vy": np.full(SAMPLE_COUNT, speed_mps, dtype=np.float64),
            "speed_mps": np.full(SAMPLE_COUNT, speed_mps, dtype=np.float64),
            "radius_m": np.full(SAMPLE_COUNT, PED_RADIUS_M, dtype=np.float64),
            "desired_speed": np.full(SAMPLE_COUNT, speed_mps, dtype=np.float64),
            "policy": ["social_force"] * SAMPLE_COUNT,
            "agent_type": ["adult"] * SAMPLE_COUNT,
        },
        schema=_PEDESTRIAN_TRACKS_SCHEMA,
    )


def _collision_contacts(kind: str, obstacle_id: str, t_ns: int) -> pl.DataFrame:
    """One recorded contact of the given kind."""
    return pl.DataFrame(
        [{"t_ns": t_ns, "obstacle_id": obstacle_id, "kind": kind, "obstacle_x": 1.5, "obstacle_y": 0.0, "obstacle_z": 0.0, "distance": 0.0}],
        schema=_COLLISION_CONTACTS_SCHEMA,
    )


def _empty_collision_contacts() -> pl.DataFrame:
    """A recorded but empty collision stream: no collision happened, and that is known."""
    return pl.DataFrame(schema=_COLLISION_CONTACTS_SCHEMA)


def _episode(
    *,
    topics: dict[str, pl.DataFrame] | None = None,
    human_params: dict[str, object] | None = None,
    outcome_state: int | None = 2,
    run: RunMetadata | None = None,
) -> AlignedEpisodeBundle:
    """Aligned bundle with the robot track and the given extra topic tables."""
    base: dict[str, pl.DataFrame] = {"robot_tracks": _robot_tracks()}
    base.update(topics or {})
    return AlignedEpisodeBundle(
        episode_id=EPISODE_ID,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[3.0, 0.0, 0.0],
        robot_name=ROBOT_ID,
        human_params=human_params,
        outcome_state=outcome_state,
        run=run,
        map="map_empty",
        topics=base,
    )


def _descriptor() -> EpisodeDescriptor:
    """Descriptor of the fixture episode."""
    return EpisodeDescriptor(episode_dir="", benchmark_id=RUN_ID, episode_id=EPISODE_ID, planner="dwb-standard", stage="s0", map="map_empty")


def _metadata(*, seed: int | None = None) -> RunMetadata:
    """Run metadata of the fixture, optionally carrying the migrated seed parameter."""
    return RunMetadata(
        benchmark_id=RUN_ID,
        planner="dwb-standard",
        map="map_empty",
        stage="s0",
        recording_started_at="2026-01-01T00:00:00",
        python_version="3.12",
        ros_distro="jazzy",
        params={} if seed is None else {"seed": seed},
    )


def _evaluate(episode: AlignedEpisodeBundle, *, metrics: dict[str, object] | None = None) -> ScopedFactTables:
    """Evaluate the fixture episode with the given episode metrics (the registry is not re-run)."""
    evaluator = ScopedEvaluator(RobotParams(robot_radius=ROBOT_RADIUS_M))
    return evaluator.evaluate(
        episode,
        run_id=RUN_ID,
        episode_id=EPISODE_ID,
        episode_metrics=dict(_METRICS if metrics is None else metrics),
    )


def _episode_row(episode: AlignedEpisodeBundle, *, metrics: dict[str, object] | None = None) -> dict[str, object]:
    """The single published episode fact row."""
    return _evaluate(episode, metrics=metrics).episodes.row(0, named=True)


def _dim_row(episode: AlignedEpisodeBundle, *, metadata: RunMetadata | None = None) -> dict[str, object]:
    """The episode dimension row the fixture episode builds."""
    facts = _evaluate(episode)
    tables = StarSchemaBuilder(robot_params=RobotParams(robot_radius=ROBOT_RADIUS_M)).build_episode_tables(
        episode,
        facts,
        run_id=RUN_ID,
        episode_id=EPISODE_ID,
        descriptor=_descriptor(),
        metadata=metadata if metadata is not None else _metadata(),
    )
    return tables.dim_episodes.row(0, named=True)


# --- success flags -------------------------------------------------------------------------


def test_an_empty_recorded_collision_stream_proves_no_contact() -> None:
    row = _episode_row(_episode(topics={"collision_contacts": _empty_collision_contacts()}))

    assert row["success"] is True
    assert row["success_no_ped_contact"] is True
    assert row["success_no_robot_fault"] is True
    assert "success_no_ped_contact:" not in str(row["null_reason"])
    assert "success_no_robot_fault:" not in str(row["null_reason"])


def test_a_missing_collision_stream_leaves_the_success_flags_null() -> None:
    row = _episode_row(_episode())

    assert row["success"] is True
    assert row["success_no_ped_contact"] is None
    assert row["success_no_robot_fault"] is None
    reason = str(row["null_reason"])
    assert f"success_no_ped_contact:{NullReason.NOT_RECORDED.value}" in reason
    assert f"success_no_robot_fault:{NullReason.NOT_RECORDED.value}" in reason


def test_a_recorded_pedestrian_contact_breaks_success_without_ped_contact() -> None:
    episode = _episode(topics={"pedestrian_tracks": _pedestrian_tracks(), "collision_contacts": _collision_contacts("pedestrian", f"env_0/{PED_ID}", _PED_CONTACT_NS)})
    row = _episode_row(episode)

    assert row["success_no_ped_contact"] is False


def test_a_robot_fault_contact_breaks_success_without_robot_fault_but_not_the_pedestrian_flag() -> None:
    episode = _episode(topics={"collision_contacts": _collision_contacts("wall", "wall_north", _WALL_CONTACT_NS)})
    row = _episode_row(episode)

    assert row["success_no_robot_fault"] is False
    assert row["success_no_ped_contact"] is True


def test_a_failed_episode_is_never_a_success_without_contact() -> None:
    row = _episode_row(_episode(topics={"collision_contacts": _empty_collision_contacts()}, outcome_state=3), metrics={"success": False})

    assert row["success"] is False
    assert row["success_no_ped_contact"] is False
    assert row["success_no_robot_fault"] is False


# --- ranking validity ----------------------------------------------------------------------


def test_a_clean_recorded_episode_is_valid_and_has_no_invalid_reason() -> None:
    row = _episode_row(_episode(topics={"collision_contacts": _empty_collision_contacts()}))

    assert row["is_valid_for_ranking"] is True
    assert row["invalid_reason"] is None


def test_a_pose_teleport_makes_the_episode_unrankable() -> None:
    jumped = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE_HZ
    jumped[SAMPLE_COUNT // 2 :] += 50.0
    row = _episode_row(_episode(topics={"robot_tracks": _robot_tracks(x_m=jumped), "collision_contacts": _empty_collision_contacts()}))

    assert row["is_valid_for_ranking"] is False
    assert str(row["invalid_reason"]) == "invalid_episode:pose_teleport"


@pytest.mark.parametrize(
    ("outcome_state", "integrity", "expected"),
    [
        (None, (), "invalid_episode:incomplete_recording"),
        (1, (), "invalid_episode:incomplete_recording"),
        (5, (), "invalid_episode:simulator_error"),
        (2, ("dropped_messages",), "invalid_episode:incomplete_recording"),
    ],
)
def test_the_validity_policy_names_each_defect(outcome_state: int | None, integrity: tuple[str, ...], expected: str) -> None:
    valid, reason = episode_validity(robot_tracks=_robot_tracks(), outcome_state=outcome_state, integrity=integrity)

    assert valid is False
    assert reason is not None
    assert expected in reason.split(";")


def test_a_missing_robot_track_names_the_missing_stream() -> None:
    valid, reason = episode_validity(robot_tracks=None, outcome_state=2, integrity=())

    assert valid is False
    assert reason == "invalid_episode:missing_stream:robot_tracks"


def test_teleport_reason_is_none_for_a_physical_trajectory() -> None:
    assert teleport_reason(_robot_tracks()) is None
    assert teleport_reason(None) is None


# --- pose frame ----------------------------------------------------------------------------


def test_pose_frame_comes_from_the_published_episode_metrics() -> None:
    assert _episode_row(_episode(topics={"collision_contacts": _empty_collision_contacts()}))["pose_frame"] == "map"


def test_pose_frame_falls_back_to_the_bundles_map_frame_transforms() -> None:
    t_ns = _time_axis()
    transforms = pl.DataFrame(
        {
            "t_ns": t_ns,
            "parent_frame": ["map"] * SAMPLE_COUNT,
            "child_frame": [f"{ROBOT_ID}/base_link"] * SAMPLE_COUNT,
            "tx": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "ty": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "yaw": np.zeros(SAMPLE_COUNT, dtype=np.float64),
        },
        schema=_TRANSFORM_SCHEMA,
    )
    row = _episode_row(_episode(topics={"tf": transforms, "collision_contacts": _empty_collision_contacts()}), metrics={"success": True})

    assert row["pose_frame"] == "map"
    assert "pose_frame:" not in str(row["null_reason"])


def test_pose_frame_is_null_without_any_recorded_transform() -> None:
    row = _episode_row(_episode(topics={"collision_contacts": _empty_collision_contacts()}), metrics={"success": True})

    assert row["pose_frame"] is None
    assert f"pose_frame:{NullReason.NO_SAMPLES.value}" in str(row["null_reason"])


# --- dimension: seed and crowd condition ----------------------------------------------------


def test_dim_seed_reads_the_migrated_metadata_parameter() -> None:
    assert _dim_row(_episode(), metadata=_metadata(seed=101))["seed"] == 101


def test_dim_seed_falls_back_to_the_recorded_episode_lifecycle() -> None:
    lifecycle = pl.DataFrame({"t_ns": [0], "seed": [101]}, schema={"t_ns": pl.Int64, "seed": pl.Int64})
    row = _dim_row(_episode(topics={"episode_lifecycle": lifecycle}), metadata=_metadata())

    assert row["seed"] == 101


def test_dim_seed_is_null_rather_than_the_episode_id_when_the_run_recorded_none() -> None:
    seed = _dim_row(_episode(), metadata=_metadata())["seed"]

    assert seed is None
    assert seed != EPISODE_ID


def test_dim_crowd_condition_is_null_with_a_reason_when_the_visibility_is_unknown() -> None:
    row = _dim_row(_episode())

    assert row["crowd_condition"] is None
    assert row["null_reason"] == "crowd_condition:not_recorded"


def test_dim_crowd_condition_is_recorded_from_the_accepted_parameter() -> None:
    row = _dim_row(_episode(human_params={"humansim.robots_visible": False}))

    assert row["crowd_condition"] == "blind"
    assert row["null_reason"] is None


def test_a_status_episode_publishes_an_unknown_crowd_condition_with_a_reason() -> None:
    tables = StarSchemaBuilder(robot_params=RobotParams(robot_radius=ROBOT_RADIUS_M)).build_status_episode_tables(
        run_id=RUN_ID,
        episode_id=EPISODE_ID,
        descriptor=_descriptor(),
        metadata=_metadata(),
    )
    row = tables.dim_episodes.row(0, named=True)

    assert row["crowd_condition"] is None
    assert row["null_reason"] == "crowd_condition:not_recorded"
    assert row["seed"] is None


def test_a_pedestrian_is_blind_only_when_the_crowd_condition_is_known() -> None:
    episode = _episode(topics={"pedestrian_tracks": _pedestrian_tracks(), "collision_contacts": _empty_collision_contacts()})
    unknown = _evaluate(episode).pedestrians.row(0, named=True)
    blind_episode = dataclasses.replace(episode, human_params={"humansim.robots_visible": False})
    blind = _evaluate(blind_episode).pedestrians.row(0, named=True)

    assert unknown["is_blind"] is None
    assert f"is_blind:{NullReason.NOT_RECORDED.value}" in str(unknown["null_reason"])
    assert blind["is_blind"] is True


# --- null reason vocabulary ----------------------------------------------------------------


def test_every_published_null_reason_is_in_the_closed_vocabulary() -> None:
    episode = _episode(topics={"pedestrian_tracks": _pedestrian_tracks(), "collision_contacts": _collision_contacts("pedestrian", f"env_0/{PED_ID}", _PED_CONTACT_NS)})
    tables = _evaluate(episode, metrics={"success": True})

    for name, frame in tables.by_name().items():
        for reason in frame.get_column("null_reason").drop_nulls().to_list():
            assert is_known_null_reason(str(reason)), f"{name}: {reason}"


@pytest.mark.parametrize(
    ("produced", "canonical"),
    [
        ("path_efficiency:series_missing:path_series", "path_efficiency:not_recorded:path_series"),
        ("reference_exceeds_path", "invalid_episode:reference_exceeds_path"),
        ("path_length_m:no_samples", "path_length_m:no_samples"),
    ],
)
def test_legacy_producer_codes_are_canonicalised(produced: str, canonical: str) -> None:
    assert canonical_null_reason(produced) == canonical


def test_an_unknown_null_reason_code_is_rejected() -> None:
    assert not is_known_null_reason("path_length_m:made_up")
    assert not is_known_null_reason("topic_missing")
