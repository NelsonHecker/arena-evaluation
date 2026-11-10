"""End-to-end fact-table tests for the scoped evaluator (WP2).

The fixture is one synthetic 6 s episode at 10 Hz: a robot driving straight at 1 m/s, a
crossing pedestrian (alice) that produces exactly one encounter ending in a recorded
contact, a mid-distance stationary pedestrian (cara) used as the encounter-distance
boundary case, and a far pedestrian (bob) that never enters the 3 m detection radius.
All expected values below are derived analytically from that geometry; nothing is read
back from the evaluator to build an expectation.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
import typing
from collections.abc import Mapping

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics import MetricRegistry, MetricScope, ScopeExecutionOutcome, ScopeSlice
from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological import AcousticDoseCalculator, SurgeEnergyCalculator
from arena_evaluation.processing.metrics.safety import ContactAttributionCalculator
from arena_evaluation.processing.metrics.social import (
    EncounterCostsCalculator,
    PedestrianActivityCalculator,
    PedestrianIntrusionCalculator,
    ProbstResponsibilityCalculator,
)
from arena_evaluation.processing.scoped_evaluator import (
    EVALUATOR_VERSION,
    FACT_SCHEMAS,
    FACT_TABLE_NAMES,
    EpisodeScopeHierarchy,
    ScopedEvaluator,
    ScopedFactTables,
    write_fact_tables,
)
from arena_evaluation.processing.windows import DEFAULT_DETECTOR_PARAMS, DetectorParams
from arena_evaluation.storage.declarations import (
    METRIC_DECLARATIONS,
    FactTable,
    MetricDeclaration,
    declared_columns,
    declarations_for,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

NS_PER_S: typing.Final[int] = 1_000_000_000
SAMPLE_RATE_HZ: typing.Final[int] = 10
SAMPLE_COUNT: typing.Final[int] = 61  # 6 s at 10 Hz
ALICE_SPEED_MPS: typing.Final[float] = math.sqrt(5.0) / 2.0
ALICE_CROSSING_X_M: typing.Final[float] = 3.0
CARA_X_M: typing.Final[float] = 2.5
CARA_Y_M: typing.Final[float] = 2.0
BOB_X_M: typing.Final[float] = 20.0
BOB_Y_M: typing.Final[float] = 20.0
ROBOT_ID: typing.Final[str] = "env_0_jackal_0"
ROBOT_TOKEN: typing.Final[str] = "jackal_0"
ROBOT_NAME: typing.Final[str] = "jackal_0"
ENV_ID: typing.Final[str] = "env_0"
RUN_ID: typing.Final[str] = "run-x"
EPISODE_ID: typing.Final[int] = 7
ROBOT_POWER_W: typing.Final[float] = 100.0
ACOUSTIC_LEVEL_DBA: typing.Final[float] = 60.0
REFERENCE_PRESSURE_PA: typing.Final[float] = 20e-6
WALKING_SPEED_MPS: typing.Final[float] = 1.2

ALICE_START_NS: typing.Final[int] = 1 * NS_PER_S  # centre distance crosses 3 m exactly here
ALICE_END_NS: typing.Final[int] = 5 * NS_PER_S  # and returns to 3 m exactly here
ALICE_DURATION_S: typing.Final[float] = 4.0
CARA_START_NS: typing.Final[int] = 300_000_000
CARA_END_NS: typing.Final[int] = 4_700_000_000
CARA_DURATION_S: typing.Final[float] = 4.4
PED_CONTACT_NS: typing.Final[int] = 3 * NS_PER_S
WALL_CONTACT_NS: typing.Final[int] = 1 * NS_PER_S
ALICE_ENCOUNTER_ID: typing.Final[str] = f"enc-{EPISODE_ID:03d}-{ROBOT_TOKEN}-001"
CARA_ENCOUNTER_ID: typing.Final[str] = f"enc-{EPISODE_ID:03d}-{ROBOT_TOKEN}-000"
PED_CONTACT_ID: typing.Final[str] = f"con-{EPISODE_ID:03d}-{ROBOT_TOKEN}-001"
WALL_CONTACT_ID: typing.Final[str] = f"con-{EPISODE_ID:03d}-{ROBOT_TOKEN}-000"
ACOUSTIC_DOSE_PA2S: typing.Final[float] = REFERENCE_PRESSURE_PA**2 * 10.0 ** (ACOUSTIC_LEVEL_DBA / 10.0) * 6.0
ALICE_TOTAL_DISTANCE_M: typing.Final[float] = 6.0 * ALICE_SPEED_MPS
ALICE_SPAWN_Y_M: typing.Final[float] = -3.0 * ALICE_SPEED_MPS
ALICE_MIN_CLEARANCE_M: typing.Final[float] = -0.55  # robot radius 0.25 + pedestrian radius 0.3
ENCOUNTER_ENERGY_J: typing.Final[float] = ROBOT_POWER_W * ALICE_DURATION_S

_EPISODE_METRICS: typing.Final[dict[str, object]] = {
    "result": "success",
    "success": True,
    "time_to_goal": 6.0,
    "path_length": 6.0,
    "path_efficiency": 0.9,
    "energy_total_wh": 0.5,
    "energy_mechanical_wh": 0.2,
    "energy_thermal_wh": 0.1,
    "energy_per_meter": 0.05,
    "velocity_mean": 1.0,
    "velocity_max": 1.0,
    "acceleration_max": 2.0,
    "jerk_max": 3.0,
    "min_obstacle_clearance": 0.4,
    "min_pedestrian_clearance": 0.1,
    "collision_amount_pedestrian": 2,
    "collision_amount_wall": 1,
    "collision_amount_static": 3,
    "mar": 0.5,
    "pfi": 0.25,
    "robot_path_deviation_m": 0.75,
    "power_peak_w": 150.0,
    "friction_dissipation_j": 5.0,
}

_SAMPLE_BY_DTYPE: typing.Final[Mapping[pl.DataType, object]] = {
    pl.Float64: 0.5,
    pl.Int64: 2,
    pl.Boolean: True,
    pl.String: "declared_sample",
}


_CHILD_EPISODE_LISTS: typing.Final[dict[str, object]] = {
    "edge_case_steps": ["hold", "approach"],
    "edge_case_agents": ["alice"],
    "edge_case_missing": ["bob"],
    "edge_case_unplayed": ["cara"],
    "edge_case_formation_members": ["alice", "cara"],
    "edge_case_formation_absent": ["bob"],
    "phase_outcomes": ["met", "failed"],
    "judge_disagreements": ["phase 1 recorded as met but not met in replay"],
    "mechanism_states": [
        {
            "entity": "north_door",
            "level": "0",
            "kind": "door",
            "progress": [0.0, 0.0, 1.0],
            "locked": [False, False, True],
        }
    ],
    "collision_obstacles": ["wall_north"],
}


def _complete_episode_metrics() -> dict[str, object]:
    """Fixture metrics plus one typed sample per declared EPISODE and ROBOT source key."""
    metrics: dict[str, object] = {**_EPISODE_METRICS, **_CHILD_EPISODE_LISTS}
    for declaration in (
        *declarations_for(FactTable.EPISODES),
        *declarations_for(FactTable.ROBOTS),
    ):
        metrics.setdefault(declaration.name, _SAMPLE_BY_DTYPE[declaration.dtype])
    return metrics


_COMPLETE_EPISODE_METRICS: typing.Final[dict[str, object]] = _complete_episode_metrics()

_ROBOT_PARAMS: typing.Final[RobotParams] = RobotParams(model="test_robot", robot_radius=0.25)

_ROBOT_TRACKS_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
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

_PEDESTRIAN_TRACKS_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
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

_COLLISION_CONTACTS_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
    "t_ns": pl.Int64,
    "obstacle_id": pl.String,
    "kind": pl.String,
    "obstacle_x": pl.Float64,
    "obstacle_y": pl.Float64,
    "obstacle_z": pl.Float64,
    "distance": pl.Float64,
}

_CLOSER_LABELS: typing.Final[frozenset[str]] = frozenset({"robot", "pedestrian", "mutual", "static"})
_FAULT_ATTRIBUTIONS: typing.Final[frozenset[str]] = frozenset({"robot", "pedestrian", "unavoidable"})


@dataclasses.dataclass(frozen=True, slots=True)
class _PedestrianSpec:
    """One pedestrian's linear track: constant velocity, fixed spawn pose and policy."""

    ped_id: str
    ped_name: str
    x_m: np.ndarray
    y_m: np.ndarray
    vx_mps: float
    vy_mps: float
    yaw_rad: float
    desired_speed_mps: float
    policy: str = "social_force"
    radius_m: float = 0.3


def _metric(key: str) -> float:
    """Numeric value of the episode-metrics fixture, asserted to be a float."""
    value = _EPISODE_METRICS[key]
    assert isinstance(value, float)
    return value


def _metric_int(key: str) -> int:
    """Integer value of the episode-metrics fixture, asserted to be an int."""
    value = _EPISODE_METRICS[key]
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _time_axis() -> tuple[np.ndarray, np.ndarray]:
    """Nanosecond and second timestamps of the 6 s, 10 Hz sample grid."""
    t_ns = np.arange(SAMPLE_COUNT, dtype=np.int64) * (NS_PER_S // SAMPLE_RATE_HZ)
    return t_ns, t_ns.astype(np.float64) / NS_PER_S


def _robot_tracks(t_ns: np.ndarray, times_s: np.ndarray) -> pl.DataFrame:
    """Robot driving straight along +x at 1 m/s from the origin."""
    return pl.DataFrame(
        {
            "robot_id": [ROBOT_ID] * SAMPLE_COUNT,
            "env_id": [ENV_ID] * SAMPLE_COUNT,
            "t_ns": t_ns,
            "x": times_s,
            "y": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "yaw": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "vx": np.ones(SAMPLE_COUNT, dtype=np.float64),
            "vy": np.zeros(SAMPLE_COUNT, dtype=np.float64),
            "pose_source": ["ground_truth"] * SAMPLE_COUNT,
        },
        schema=_ROBOT_TRACKS_SCHEMA,
    )


def _pedestrian_track(t_ns: np.ndarray, spec: _PedestrianSpec) -> pl.DataFrame:
    """Rows of one pedestrian track on the shared sample grid."""
    speed_mps = math.hypot(spec.vx_mps, spec.vy_mps)
    return pl.DataFrame(
        {
            "t_ns": t_ns,
            "env_id": [ENV_ID] * SAMPLE_COUNT,
            "ped_id": [spec.ped_id] * SAMPLE_COUNT,
            "ped_name": [spec.ped_name] * SAMPLE_COUNT,
            "x": spec.x_m,
            "y": spec.y_m,
            "yaw": np.full(SAMPLE_COUNT, spec.yaw_rad, dtype=np.float64),
            "vx": np.full(SAMPLE_COUNT, spec.vx_mps, dtype=np.float64),
            "vy": np.full(SAMPLE_COUNT, spec.vy_mps, dtype=np.float64),
            "speed_mps": np.full(SAMPLE_COUNT, speed_mps, dtype=np.float64),
            "radius_m": np.full(SAMPLE_COUNT, spec.radius_m, dtype=np.float64),
            "desired_speed": np.full(SAMPLE_COUNT, spec.desired_speed_mps, dtype=np.float64),
            "policy": [spec.policy] * SAMPLE_COUNT,
            "agent_type": ["pedestrian"] * SAMPLE_COUNT,
        },
        schema=_PEDESTRIAN_TRACKS_SCHEMA,
    )


def _alice_spec() -> _PedestrianSpec:
    """Alice crosses the robot's path on x = 3 at sqrt(5)/2 m/s, hitting the robot at t = 3 s."""
    _, times_s = _time_axis()
    y_m = ALICE_SPEED_MPS * (times_s - 3.0)
    return _PedestrianSpec(
        ped_id="Pedestrian_1",
        ped_name="alice",
        x_m=np.full(SAMPLE_COUNT, ALICE_CROSSING_X_M, dtype=np.float64),
        y_m=y_m,
        vx_mps=0.0,
        vy_mps=ALICE_SPEED_MPS,
        yaw_rad=math.pi / 2.0,
        desired_speed_mps=WALKING_SPEED_MPS,
    )


def _static_spec(ped_id: str, ped_name: str, x_m: float, y_m: float) -> _PedestrianSpec:
    """Stationary pedestrian at a fixed map position."""
    return _PedestrianSpec(
        ped_id=ped_id,
        ped_name=ped_name,
        x_m=np.full(SAMPLE_COUNT, x_m, dtype=np.float64),
        y_m=np.full(SAMPLE_COUNT, y_m, dtype=np.float64),
        vx_mps=0.0,
        vy_mps=0.0,
        yaw_rad=0.0,
        desired_speed_mps=0.0,
    )


def _pedestrian_tracks(t_ns: np.ndarray) -> pl.DataFrame:
    """Alice (crossing contact), cara (2 m stationary mid-distance) and bob (far away)."""
    _, times_s = _time_axis()
    tracks = (
        _pedestrian_track(t_ns, _alice_spec()),
        _pedestrian_track(t_ns, _static_spec("Pedestrian_3", "cara", CARA_X_M, CARA_Y_M)),
        _pedestrian_track(t_ns, _static_spec("Pedestrian_2", "bob", BOB_X_M, BOB_Y_M)),
    )
    return pl.concat(tracks, how="vertical")


def _alice_track(t_ns: np.ndarray) -> pl.DataFrame:
    """The robot-free reference twin of alice: the same straight, unhindered path."""
    return _pedestrian_track(t_ns, _alice_spec())


def _robot_power(t_ns: np.ndarray) -> pl.DataFrame:
    """Constant 100 W total power so window energies are analytic."""
    return pl.DataFrame(
        {"t_ns": t_ns, "total_power_w": np.full(SAMPLE_COUNT, ROBOT_POWER_W, dtype=np.float64)},
        schema={"t_ns": pl.Int64, "total_power_w": pl.Float64},
    )


def _robot_acoustics(t_ns: np.ndarray) -> pl.DataFrame:
    """Constant 60 dBA emission so the dose integral is analytic."""
    return pl.DataFrame(
        {"t_ns": t_ns, "total_level_af_dba": np.full(SAMPLE_COUNT, ACOUSTIC_LEVEL_DBA, dtype=np.float64)},
        schema={"t_ns": pl.Int64, "total_level_af_dba": pl.Float64},
    )


def _collision_contacts() -> pl.DataFrame:
    """One wall contact and one pedestrian contact, each a single recorded sample."""
    return pl.DataFrame(
        [
            {
                "t_ns": WALL_CONTACT_NS,
                "obstacle_id": "wall_north",
                "kind": "wall",
                "obstacle_x": 1.0,
                "obstacle_y": 0.3,
                "obstacle_z": 0.0,
                "distance": 0.0,
            },
            {
                "t_ns": PED_CONTACT_NS,
                "obstacle_id": f"{ENV_ID}/Pedestrian_1",
                "kind": "pedestrian",
                "obstacle_x": ALICE_CROSSING_X_M,
                "obstacle_y": 0.0,
                "obstacle_z": 0.0,
                "distance": 0.0,
            },
        ],
        schema=_COLLISION_CONTACTS_SCHEMA,
    )


def _episode() -> AlignedEpisodeBundle:
    """Synthetic aligned bundle with the flat WP1 topic tables of the fixture."""
    t_ns, times_s = _time_axis()
    return AlignedEpisodeBundle(
        episode_id=EPISODE_ID,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[6.0, 0.0, 0.0],
        num_pedestrians=3,
        robot_name=ROBOT_NAME,
        human_params={"humansim.robots_visible": False},
        topics={
            "robot_tracks": _robot_tracks(t_ns, times_s),
            "pedestrian_tracks": _pedestrian_tracks(t_ns),
            "robot_power": _robot_power(t_ns),
            "robot_acoustics": _robot_acoustics(t_ns),
            "collision_contacts": _collision_contacts(),
        },
    )


def _reference_episode() -> AlignedEpisodeBundle:
    """Robot-free twin episode injected through metrics_config for detour/deflection facts."""
    t_ns, _ = _time_axis()
    return AlignedEpisodeBundle(
        episode_id=EPISODE_ID,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[6.0, 0.0, 0.0],
        robot_name=ROBOT_NAME,
        topics={"pedestrian_tracks": _alice_track(t_ns)},
    )


def _referenced_config() -> dict[str, object]:
    """Metrics config carrying the robot-free twin of the fixture."""
    return {"reference_episode": _reference_episode()}


def _without_topics(episode: AlignedEpisodeBundle, *names: str) -> AlignedEpisodeBundle:
    """Copy of the episode with the named topic tables removed."""
    topics = {name: table for name, table in (episode.topics or {}).items() if name not in names}
    return dataclasses.replace(episode, topics=topics)


def _evaluate(
    episode: AlignedEpisodeBundle,
    *,
    episode_metrics: Mapping[str, object] | None = None,
    metrics_config: Mapping[str, object] | None = None,
    detector_params: DetectorParams | None = None,
) -> ScopedFactTables:
    """Evaluate one episode with the given metrics, config and detector parameters."""
    evaluator = ScopedEvaluator(
        _ROBOT_PARAMS,
        metrics_config=metrics_config,
        detector_params=DEFAULT_DETECTOR_PARAMS if detector_params is None else detector_params,
    )
    return evaluator.evaluate(
        episode,
        run_id=RUN_ID,
        episode_id=EPISODE_ID,
        episode_metrics=_COMPLETE_EPISODE_METRICS if episode_metrics is None else episode_metrics,
    )


def _evaluate_referenced(
    episode: AlignedEpisodeBundle,
    *,
    episode_metrics: Mapping[str, object] | None = None,
    detector_params: DetectorParams | None = None,
) -> ScopedFactTables:
    """Evaluate one episode with the robot-free reference twin configured."""
    return _evaluate(
        episode,
        episode_metrics=episode_metrics,
        metrics_config=_referenced_config(),
        detector_params=detector_params,
    )


def _single_row(table: pl.DataFrame, column: str, value: object) -> dict[str, object]:
    """The single row matching one column equality, failing loudly otherwise."""
    selected = table.filter(pl.col(column) == value)
    assert selected.height == 1, f"expected exactly one row with {column}={value!r}, got {selected.height}"
    return selected.row(0, named=True)


def _pedestrian_row(tables: ScopedFactTables, ped_id: str) -> dict[str, object]:
    """Fact row of one pedestrian."""
    return _single_row(tables.pedestrians, "ped_id", ped_id)


def _encounter_row(tables: ScopedFactTables, actor_2: str) -> dict[str, object]:
    """Fact row of the encounter with one actor."""
    return _single_row(tables.encounters, "actor_2", actor_2)


def _assert_closer_label_is_consistent(closer_label: object, robot_share: object) -> None:
    """closer_label and robot_closing_share must tell the same story."""
    assert isinstance(closer_label, str)
    assert closer_label in _CLOSER_LABELS
    assert isinstance(robot_share, float)
    assert 0.0 <= robot_share <= 1.0
    if closer_label == "robot":
        assert robot_share > 0.5
    elif closer_label == "pedestrian":
        assert robot_share < 0.5
    elif closer_label == "mutual":
        assert 0.25 <= robot_share <= 0.75
    else:
        assert robot_share == pytest.approx(1.0)


@pytest.fixture(scope="module")
def episode() -> AlignedEpisodeBundle:
    """The synthetic episode of the suite."""
    return _episode()


@pytest.fixture(scope="module")
def tables(episode: AlignedEpisodeBundle) -> ScopedFactTables:
    """Fact tables of the full synthetic episode, reference twin configured."""
    return _evaluate_referenced(episode)


def test_evaluator_contract_constants() -> None:
    assert EVALUATOR_VERSION == "wp2-scoped-evaluator-v1"
    assert FACT_TABLE_NAMES == (
        "episodes",
        "encounters",
        "contacts",
        "pedestrians",
        "robots",
        "edge_cases",
        "formations",
        "phase_outcomes",
        "judge_disagreements",
        "mechanism_states",
        "collisions",
    )
    assert tuple(FACT_SCHEMAS) == FACT_TABLE_NAMES


@pytest.mark.parametrize("table_name", FACT_TABLE_NAMES)
def test_each_fact_table_matches_frozen_schema(table_name: str, tables: ScopedFactTables) -> None:
    frame = tables.by_name()[table_name]
    assert frame.columns == FACT_SCHEMAS[table_name].names()
    assert frame.schema == FACT_SCHEMAS[table_name]


def test_by_name_returns_the_five_tables(tables: ScopedFactTables) -> None:
    named = tables.by_name()
    assert tuple(named) == FACT_TABLE_NAMES
    for table in named.values():
        assert isinstance(table, pl.DataFrame)


def test_episode_fact_columns_match_frozen_schema(tables: ScopedFactTables) -> None:
    assert tables.episodes.columns == FACT_SCHEMAS["episodes"].names()


def test_episode_fact_maps_every_episode_metric(tables: ScopedFactTables) -> None:
    row = tables.episodes.row(0, named=True)
    assert row["run_id"] == RUN_ID
    assert row["episode_id"] == EPISODE_ID
    assert row["robot_id"] == ROBOT_ID
    assert row["result"] == _EPISODE_METRICS["result"]
    assert row["success"] is True
    assert row["time_to_goal_s"] == pytest.approx(_metric("time_to_goal"))
    assert row["path_length_m"] == pytest.approx(_metric("path_length"))
    assert row["path_efficiency"] == pytest.approx(_metric("path_efficiency"))
    assert row["total_energy_wh"] == pytest.approx(_metric("energy_total_wh"))
    assert row["mean_velocity_mps"] == pytest.approx(_metric("velocity_mean"))
    assert row["max_acceleration_mps2"] == pytest.approx(_metric("acceleration_max"))
    assert row["jerk_max_mps3"] == pytest.approx(_metric("jerk_max"))
    assert row["min_obstacle_clearance_m"] == pytest.approx(_metric("min_obstacle_clearance"))
    assert row["min_ped_clearance_m"] == pytest.approx(_metric("min_pedestrian_clearance"))
    assert row["mar"] == pytest.approx(_metric("mar"))
    assert row["pfi"] == pytest.approx(_metric("pfi"))
    assert row["robot_path_dev_m"] == pytest.approx(_metric("robot_path_deviation_m"))


def test_episode_energy_per_meter_is_converted_from_wh_per_m(tables: ScopedFactTables) -> None:
    row = tables.episodes.row(0, named=True)
    assert row["energy_per_meter_j_m"] == pytest.approx(_metric("energy_per_meter") * 3600.0)


def test_episode_collision_counts_and_robot_fault_flag(tables: ScopedFactTables) -> None:
    row = tables.episodes.row(0, named=True)
    assert row["total_collisions_ped"] == _EPISODE_METRICS["collision_amount_pedestrian"]
    assert row["total_collisions_static"] == _metric_int("collision_amount_wall") + _metric_int("collision_amount_static")
    assert row["success"] is True
    assert row["success_no_robot_fault"] is False


def test_episode_total_acoustic_dose_is_dba_integral(tables: ScopedFactTables) -> None:
    row = tables.episodes.row(0, named=True)
    assert row["total_acoustic_dose_pa2s"] == pytest.approx(ACOUSTIC_DOSE_PA2S, rel=1e-12)


def test_episode_null_reason_empty_for_complete_metrics(tables: ScopedFactTables) -> None:
    assert tables.episodes.row(0, named=True)["null_reason"] == ""


def test_episode_null_reason_lists_missing_metric_fields(episode: AlignedEpisodeBundle) -> None:
    tables = _evaluate_referenced(episode, episode_metrics={"success": True})
    row = tables.episodes.row(0, named=True)
    reason = row["null_reason"]
    assert isinstance(reason, str)
    assert reason != ""
    codes = set(reason.split(";"))
    assert {"result:no_samples", "time_to_goal_s:no_samples", "mar:no_samples", "pfi:no_samples"} <= codes
    assert "success:no_samples" not in codes
    assert row["total_collisions_ped"] == 1
    assert row["total_collisions_static"] == 1


def test_encounter_fact_columns_match_frozen_schema(tables: ScopedFactTables) -> None:
    assert tables.encounters.columns == FACT_SCHEMAS["encounters"].names()


def test_alice_encounter_reproduces_analytic_window(tables: ScopedFactTables) -> None:
    row = _encounter_row(tables, "Pedestrian_1")
    assert row["robot_id"] == ROBOT_ID
    assert row["actor_1"] == ROBOT_ID
    assert row["actor_2_kind"] == "pedestrian"
    assert row["t_start_ns"] == ALICE_START_NS
    assert row["t_end_ns"] == ALICE_END_NS
    assert row["duration_s"] == pytest.approx(ALICE_DURATION_S)
    assert row["min_clearance_m"] == pytest.approx(ALICE_MIN_CLEARANCE_M, abs=1e-9)
    assert row["approach_type"] == "crossing"


def test_encounter_ids_are_detector_independent(tables: ScopedFactTables) -> None:
    alice = _encounter_row(tables, "Pedestrian_1")
    cara = _encounter_row(tables, "Pedestrian_3")
    assert alice["encounter_id"] == ALICE_ENCOUNTER_ID
    assert cara["encounter_id"] == CARA_ENCOUNTER_ID
    pattern = re.compile(rf"^enc-{EPISODE_ID:03d}-{ROBOT_TOKEN}-\d{{3}}$")
    for encounter_id in tables.encounters["encounter_id"].to_list():
        assert pattern.fullmatch(encounter_id) is not None
    assert tables.encounters["encounter_id"].n_unique() == tables.encounters.height


def test_encounter_links_its_recorded_contact(tables: ScopedFactTables) -> None:
    alice = _encounter_row(tables, "Pedestrian_1")
    contact = _single_row(tables.contacts, "obstacle_id", "Pedestrian_1")
    assert alice["had_contact"] is True
    assert alice["contact_id"] == contact["contact_id"]
    assert alice["encounter_id"] == contact["encounter_id"]
    cara = _encounter_row(tables, "Pedestrian_3")
    assert cara["had_contact"] is False
    assert cara["contact_id"] is None


def test_encounter_energy_is_the_constant_power_integral(tables: ScopedFactTables) -> None:
    row = _encounter_row(tables, "Pedestrian_1")
    assert row["encounter_energy_j"] == pytest.approx(ENCOUNTER_ENERGY_J)
    assert row["encounter_energy_j"] > 0.0
    assert row["energy_surge_ratio"] == pytest.approx(1.0)


def test_encounter_sound_exposure_dose_is_positive(tables: ScopedFactTables) -> None:
    row = _encounter_row(tables, "Pedestrian_1")
    dose = row["sound_exposure_dose_pa2s"]
    assert isinstance(dose, float)
    assert dose > 0.0


def test_encounter_probst_shares_are_present_and_sum_to_one(tables: ScopedFactTables) -> None:
    row = _encounter_row(tables, "Pedestrian_1")
    shares = [row["probst_R_robot"], row["probst_R_actor2"], row["probst_R_time"]]
    assert all(isinstance(share, float) for share in shares)
    assert sum(typing.cast(float, share) for share in shares) == pytest.approx(1.0, abs=1e-9)
    assert isinstance(row["probst_E_robot"], float)
    assert isinstance(row["probst_E_actor2"], float)


def test_encounter_null_reason_names_exactly_the_null_window_columns(tables: ScopedFactTables) -> None:
    """Every null declared window column names its star column; no non-null column is reported as missing."""
    row = _encounter_row(tables, "Pedestrian_1")
    codes = set(typing.cast(str, row["null_reason"]).split(";"))
    for name in declared_columns(FactTable.ENCOUNTERS):
        assert (f"{name}:no_samples" in codes) == (row[name] is None)
    # A 3 m encounter window never reaches the public zone, so its max speed is structurally null.
    assert row["max_speed_public_zone"] is None
    assert "max_speed_public_zone:no_samples" in codes
    assert row["detour_time_s"] == pytest.approx(0.0)
    assert row["path_deviation_m"] == pytest.approx(0.0)


def test_encounter_fact_publishes_the_windowed_social_columns(tables: ScopedFactTables) -> None:
    """The windowed social metrics land in fact_encounters from the encounter's own telemetry."""
    row = _encounter_row(tables, "Pedestrian_1")
    assert row["num_pedestrians"] == 1  # the encounter window's population is its own actor
    peak_force = row["sfm_peak_force"]
    assert isinstance(peak_force, float)
    assert peak_force > 0.0
    assert row["max_speed_intimate_zone"] == pytest.approx(1.0)
    assert row["time_in_public_zone"] == pytest.approx(0.0)
    intrusion = row["personal_space_intrusion_integral"]
    assert isinstance(intrusion, float)
    assert intrusion > 0.0
    assert row["total_time_looking_at_pedestrians"] is not None


def test_contact_fact_publishes_the_windowed_hhi_columns(tables: ScopedFactTables) -> None:
    """The hhi metrics are the slice's own pre/post-contact window: one second around the 10 Hz contact sample."""
    pedestrian_contact = _single_row(tables.contacts, "obstacle_id", "Pedestrian_1")
    assert pedestrian_contact["hhi_episode_s"] == pytest.approx(1.0)
    assert pedestrian_contact["hhi_share_approach"] == pytest.approx(1.0)
    assert pedestrian_contact["hhi_share_hold"] == pytest.approx(0.0)
    assert pedestrian_contact["hhi_stall_count"] == 0
    assert pedestrian_contact["hhi_interaction"] is None
    wall_contact = _single_row(tables.contacts, "obstacle_id", "wall_north")
    assert wall_contact["hhi_episode_s"] == pytest.approx(1.0)
    assert wall_contact["hhi_share_approach"] == pytest.approx(1.0)


def test_no_reference_episode_nulls_detour_facts(episode: AlignedEpisodeBundle) -> None:
    tables = _evaluate(episode)
    row = _encounter_row(tables, "Pedestrian_1")
    assert row["detour_time_s"] is None
    assert row["path_deviation_m"] is None
    reason = typing.cast(str, row["null_reason"])
    assert "detour_time_s:reference_missing" in reason
    assert "path_deviation_m:reference_missing" in reason


def test_contact_fact_columns_match_frozen_schema(tables: ScopedFactTables) -> None:
    assert tables.contacts.columns == FACT_SCHEMAS["contacts"].names()


def test_pedestrian_contact_fact(tables: ScopedFactTables) -> None:
    row = _single_row(tables.contacts, "obstacle_id", "Pedestrian_1")
    assert row["collider_kind"] == "pedestrian"
    assert row["encounter_id"] == ALICE_ENCOUNTER_ID
    assert row["t_onset_ns"] == PED_CONTACT_NS
    assert row["duration_s"] == pytest.approx(0.0)
    assert row["impact_speed_mps"] == pytest.approx(1.5)  # |robot vel - pedestrian vel|
    assert row["robot_speed_at_impact_mps"] == pytest.approx(1.0)
    assert row["obstacle_speed_at_impact_mps"] == pytest.approx(ALICE_SPEED_MPS)
    assert row["counterfactual_freeze_persists"] is True
    assert row["fault_attributed_to"] in _FAULT_ATTRIBUTIONS


def test_pedestrian_contact_closer_label_matches_closing_share(tables: ScopedFactTables) -> None:
    row = _single_row(tables.contacts, "obstacle_id", "Pedestrian_1")
    assert row["closer_label"] == "mutual"
    _assert_closer_label_is_consistent(row["closer_label"], row["robot_closing_share"])


def test_static_contact_fact(tables: ScopedFactTables) -> None:
    row = _single_row(tables.contacts, "obstacle_id", "wall_north")
    assert row["collider_kind"] == "wall"
    assert row["encounter_id"] is None
    assert row["closer_label"] == "static"
    assert row["robot_closing_share"] == pytest.approx(1.0)
    assert row["fault_attributed_to"] == "robot"
    assert row["counterfactual_freeze_persists"] is None
    assert row["t_onset_ns"] == WALL_CONTACT_NS
    assert row["impact_speed_mps"] == pytest.approx(1.0)
    assert row["obstacle_speed_at_impact_mps"] == pytest.approx(0.0)


def test_contact_ids_are_detector_independent(tables: ScopedFactTables) -> None:
    assert tables.contacts["contact_id"].to_list() == [WALL_CONTACT_ID, PED_CONTACT_ID]
    pattern = re.compile(rf"^con-{EPISODE_ID:03d}-{ROBOT_TOKEN}-\d{{3}}$")
    for contact_id in tables.contacts["contact_id"].to_list():
        assert pattern.fullmatch(contact_id) is not None


def test_pedestrian_fact_columns_match_frozen_schema(tables: ScopedFactTables) -> None:
    assert tables.pedestrians.columns == FACT_SCHEMAS["pedestrians"].names()


def test_pedestrian_facts_are_one_row_per_pedestrian(tables: ScopedFactTables) -> None:
    assert tables.pedestrians.height == 3
    assert set(tables.pedestrians["ped_id"].to_list()) == {"Pedestrian_1", "Pedestrian_2", "Pedestrian_3"}
    assert tables.pedestrians["run_id"].unique().to_list() == [RUN_ID]
    assert tables.pedestrians["episode_id"].unique().to_list() == [EPISODE_ID]
    assert tables.pedestrians["env_id"].unique().to_list() == [ENV_ID]


def test_alice_pedestrian_kinematics_are_analytic(tables: ScopedFactTables) -> None:
    row = _pedestrian_row(tables, "Pedestrian_1")
    assert row["ped_name"] == "alice"
    assert row["ped_policy"] == "social_force"
    assert row["is_blind"] is True
    assert row["desired_speed"] == pytest.approx(WALKING_SPEED_MPS)
    assert row["radius_m"] == pytest.approx(0.3)
    assert row["spawn_x"] == pytest.approx(ALICE_CROSSING_X_M)
    assert row["spawn_y"] == pytest.approx(ALICE_SPAWN_Y_M, abs=1e-12)
    assert row["spawn_yaw"] == pytest.approx(math.pi / 2.0)
    assert row["total_distance_m"] == pytest.approx(ALICE_TOTAL_DISTANCE_M, rel=1e-12)
    assert row["mean_speed_mps"] == pytest.approx(ALICE_SPEED_MPS, rel=1e-12)


def test_far_pedestrian_has_no_encounters(tables: ScopedFactTables) -> None:
    row = _pedestrian_row(tables, "Pedestrian_2")
    assert row["total_encounters"] == 0
    assert row["total_distance_m"] == pytest.approx(0.0)
    assert row["mean_speed_mps"] == pytest.approx(0.0)
    assert row["cumulative_sound_dose_pa2s"] is None
    reason = typing.cast(str, row["null_reason"])
    assert "cumulative_sound_dose_pa2s:no_samples" in reason
    assert "ped_path_deflection_m:no_samples" in reason


def test_pedestrian_encounter_counts_per_actor(tables: ScopedFactTables) -> None:
    assert _pedestrian_row(tables, "Pedestrian_1")["total_encounters"] == 1
    assert _pedestrian_row(tables, "Pedestrian_3")["total_encounters"] == 1
    assert _pedestrian_row(tables, "Pedestrian_2")["total_encounters"] == 0


def test_pedestrian_cumulative_sound_dose_equals_encounter_dose(tables: ScopedFactTables) -> None:
    alice = _pedestrian_row(tables, "Pedestrian_1")
    alice_encounter = _encounter_row(tables, "Pedestrian_1")
    assert alice["cumulative_sound_dose_pa2s"] == pytest.approx(alice_encounter["sound_exposure_dose_pa2s"])
    cara = _pedestrian_row(tables, "Pedestrian_3")
    cara_encounter = _encounter_row(tables, "Pedestrian_3")
    assert cara["cumulative_sound_dose_pa2s"] == pytest.approx(cara_encounter["sound_exposure_dose_pa2s"])
    alice_codes = set(typing.cast(str, alice["null_reason"]).split(";"))
    for name in declared_columns(FactTable.PEDESTRIANS):
        assert (f"{name}:no_samples" in alice_codes) == (alice[name] is None)


def test_pedestrian_intrusion_counts(tables: ScopedFactTables) -> None:
    alice = _pedestrian_row(tables, "Pedestrian_1")
    assert alice["total_intrusions_intimate"] == 1
    assert alice["total_intrusions_personal"] == 1
    bob = _pedestrian_row(tables, "Pedestrian_2")
    assert bob["total_intrusions_intimate"] == 0
    assert bob["total_intrusions_personal"] == 0
    cara = _pedestrian_row(tables, "Pedestrian_3")
    assert cara["total_intrusions_intimate"] == 0
    assert cara["total_intrusions_personal"] == 0


@pytest.mark.parametrize(("robots_visible", "expected_blind"), [(False, True), (True, False)])
def test_pedestrian_is_blind_follows_human_params(episode: AlignedEpisodeBundle, robots_visible: bool, expected_blind: bool) -> None:
    visible_episode = dataclasses.replace(episode, human_params={"humansim.robots_visible": robots_visible})
    tables = _evaluate_referenced(visible_episode)
    assert tables.pedestrians["is_blind"].unique().to_list() == [expected_blind]


def test_robot_fact_columns_match_frozen_schema(tables: ScopedFactTables) -> None:
    assert tables.robots.columns == FACT_SCHEMAS["robots"].names()


def test_robot_fact_maps_energy_and_motion_metrics(tables: ScopedFactTables) -> None:
    row = tables.robots.row(0, named=True)
    assert row["run_id"] == RUN_ID
    assert row["episode_id"] == EPISODE_ID
    assert row["env_id"] == ENV_ID
    assert row["robot_id"] == ROBOT_ID
    assert row["robot_name"] == ROBOT_NAME
    assert row["path_length_m"] == pytest.approx(_metric("path_length"))
    assert row["mean_speed_mps"] == pytest.approx(_metric("velocity_mean"))
    assert row["max_speed_mps"] == pytest.approx(_metric("velocity_max"))
    assert row["mean_power_w"] == pytest.approx(_metric("energy_total_wh") * 3600.0 / _metric("time_to_goal"))
    assert row["peak_power_w"] == pytest.approx(_metric("power_peak_w"))
    assert row["mechanical_work_j"] == pytest.approx(_metric("energy_mechanical_wh") * 3600.0)
    assert row["thermal_energy_j"] == pytest.approx(_metric("energy_thermal_wh") * 3600.0)
    assert row["electrical_energy_wh"] == pytest.approx(_metric("energy_total_wh"))
    assert row["drivetrain_scrub_j"] == pytest.approx(_metric("friction_dissipation_j"))


def test_robot_acoustic_emission_equals_episode_dose(tables: ScopedFactTables) -> None:
    robot_dose = tables.robots.row(0, named=True)["acoustic_emission_pa2s"]
    episode_dose = tables.episodes.row(0, named=True)["total_acoustic_dose_pa2s"]
    assert robot_dose == pytest.approx(ACOUSTIC_DOSE_PA2S, rel=1e-12)
    assert robot_dose == pytest.approx(typing.cast(float, episode_dose))


def test_robot_null_reason_empty(tables: ScopedFactTables) -> None:
    assert tables.robots.row(0, named=True)["null_reason"] == ""


def test_write_fact_tables_writes_every_parquet_file(tables: ScopedFactTables, tmp_path: pathlib.Path) -> None:
    write_fact_tables(tables, tmp_path)
    written = {path.name for path in tmp_path.glob("fact_*.parquet")}
    assert written == {f"fact_{name}.parquet" for name in FACT_TABLE_NAMES}


@pytest.mark.parametrize("table_name", FACT_TABLE_NAMES)
def test_write_fact_tables_round_trips_schema_and_rows(table_name: str, tables: ScopedFactTables, tmp_path: pathlib.Path) -> None:
    write_fact_tables(tables, tmp_path)
    path = tmp_path / f"fact_{table_name}.parquet"
    assert path.is_file()
    original = tables.by_name()[table_name]
    reloaded = pl.read_parquet(path)
    assert reloaded.columns == original.columns
    assert reloaded.schema == original.schema
    assert reloaded.height == original.height
    assert_frame_equal(reloaded, original, check_dtypes=True, check_exact=False)


def test_absent_pedestrian_table_returns_empty_children(episode: AlignedEpisodeBundle) -> None:
    tables = _evaluate_referenced(_without_topics(episode, "pedestrian_tracks"))
    assert tables.encounters.height == 0
    assert tables.encounters.schema == FACT_SCHEMAS["encounters"]
    assert tables.pedestrians.height == 0
    assert tables.pedestrians.schema == FACT_SCHEMAS["pedestrians"]
    row = tables.episodes.row(0, named=True)
    assert "pedsim_unavailable" in typing.cast(str, row["null_reason"])


def test_absent_robot_tracks_yields_null_metrics(episode: AlignedEpisodeBundle) -> None:
    tables = _evaluate(_without_topics(episode, "robot_tracks"), episode_metrics={})
    row = tables.episodes.row(0, named=True)
    assert row["robot_id"] == ROBOT_NAME
    assert row["result"] is None
    assert row["success"] is None
    assert row["time_to_goal_s"] is None
    # No recorded collision stream: the count is unknown, not zero.
    assert row["total_collisions_ped"] is None
    assert row["total_collisions_static"] is None
    reason = typing.cast(str, row["null_reason"])
    assert "path_length_m:no_samples" in reason
    assert "topic_missing:robot_tracks" in reason
    assert "total_collisions_ped:" in reason
    assert "total_collisions_static:" in reason
    assert tables.encounters.height == 0
    assert tables.encounters.schema == FACT_SCHEMAS["encounters"]
    assert tables.contacts.height == 0
    assert tables.contacts.schema == FACT_SCHEMAS["contacts"]
    robot = tables.robots.row(0, named=True)
    assert robot["env_id"] is None
    assert robot["robot_name"] == ROBOT_NAME
    assert robot["path_length_m"] is None
    assert robot["mechanical_work_j"] is None
    assert robot["null_reason"] != ""


def test_absent_collision_contacts_yields_empty_contacts(episode: AlignedEpisodeBundle) -> None:
    tables = _evaluate_referenced(_without_topics(episode, "collision_contacts"))
    assert tables.contacts.height == 0
    assert tables.contacts.schema == FACT_SCHEMAS["contacts"]
    assert tables.encounters.height == 2
    row = tables.episodes.row(0, named=True)
    assert row["total_collisions_ped"] == _EPISODE_METRICS["collision_amount_pedestrian"]
    assert row["total_collisions_static"] == 4
    assert row["success_no_robot_fault"] is True


def test_detector_distance_parameter_changes_encounter_count(episode: AlignedEpisodeBundle, tables: ScopedFactTables) -> None:
    assert tables.encounters.height == 2
    tight = _evaluate_referenced(episode, detector_params=DetectorParams(encounter_distance_m=1.0))
    assert tight.encounters.height == 1
    row = _encounter_row(tight, "Pedestrian_1")
    assert row["t_start_ns"] == 2_400_000_000
    assert row["t_end_ns"] == 3_600_000_000
    assert row["duration_s"] == pytest.approx(1.2)
    assert row["encounter_id"] == CARA_ENCOUNTER_ID
    contact = _single_row(tight.contacts, "obstacle_id", "Pedestrian_1")
    assert contact["encounter_id"] == CARA_ENCOUNTER_ID


def test_lazy_topic_frames_are_accepted(episode: AlignedEpisodeBundle, tables: ScopedFactTables) -> None:
    lazy_episode = dataclasses.replace(episode, topics={name: table.lazy() for name, table in (episode.topics or {}).items()})
    lazy_tables = _evaluate_referenced(lazy_episode)
    for name, table in lazy_tables.by_name().items():
        assert isinstance(table, pl.DataFrame)
        assert table.height == tables.by_name()[name].height


def test_missing_power_and_acoustics_mark_nulls(episode: AlignedEpisodeBundle) -> None:
    tables = _evaluate(_without_topics(episode, "robot_power", "robot_acoustics"))
    episode_row = tables.episodes.row(0, named=True)
    assert episode_row["total_acoustic_dose_pa2s"] is None
    assert "total_acoustic_dose_pa2s:no_samples" in typing.cast(str, episode_row["null_reason"])
    assert "total_acoustic_dose_pa2s:topic_missing:robot_acoustics" in typing.cast(str, episode_row["null_reason"])
    encounter = _encounter_row(tables, "Pedestrian_1")
    assert encounter["energy_surge_ratio"] is None
    assert "energy_surge_ratio:no_samples" in typing.cast(str, encounter["null_reason"])
    assert "energy_surge_ratio:topic_missing:robot_power" in typing.cast(str, encounter["null_reason"])
    robot = tables.robots.row(0, named=True)
    assert robot["acoustic_emission_pa2s"] is None
    assert "acoustic_emission_pa2s:no_samples" in typing.cast(str, robot["null_reason"])
    assert "acoustic_emission_pa2s:topic_missing:robot_acoustics" in typing.cast(str, robot["null_reason"])


def test_decompose_hierarchy_materializes_topics_once_and_builds_five_scopes(episode: AlignedEpisodeBundle) -> None:
    lazy_episode = dataclasses.replace(episode, topics={name: table.lazy() for name, table in (episode.topics or {}).items()})
    evaluator = ScopedEvaluator(_ROBOT_PARAMS, metrics_config=_referenced_config())
    hierarchy = evaluator.decompose_hierarchy(lazy_episode, run_id=RUN_ID, episode_id=EPISODE_ID)

    assert isinstance(hierarchy, EpisodeScopeHierarchy)
    assert len(hierarchy.episodes) == 1
    assert len(hierarchy.robots) == 1
    assert len(hierarchy.pedestrians) == 3
    assert len(hierarchy.encounters) == 2
    assert len(hierarchy.contacts) == 2

    ep_slice = hierarchy.episodes[0]
    assert ep_slice.scope == MetricScope.EPISODE
    assert ep_slice.slice_id == f"{RUN_ID}:{EPISODE_ID}:{ROBOT_ID}"
    assert ep_slice.parent_id is None
    for table in (ep_slice.full_episode.topics or {}).values():
        assert isinstance(table, pl.DataFrame)

    ped_contact_slice = next(s for s in hierarchy.contacts if s.identity["obstacle_id"] == "Pedestrian_1")
    assert ped_contact_slice.scope == MetricScope.CONTACT
    assert ped_contact_slice.parent_id == ALICE_ENCOUNTER_ID
    assert ped_contact_slice.t_start_ns == PED_CONTACT_NS


def test_multi_scope_calculators_register_and_execute_across_all_scopes(episode: AlignedEpisodeBundle) -> None:
    assert ProbstResponsibilityCalculator.effective_scopes() == frozenset({MetricScope.ENCOUNTER, MetricScope.CONTACT, MetricScope.EPISODE})
    assert EncounterCostsCalculator.effective_scopes() == frozenset({MetricScope.ENCOUNTER, MetricScope.EPISODE})
    assert ContactAttributionCalculator.effective_scopes() == frozenset({MetricScope.CONTACT, MetricScope.EPISODE})
    assert AcousticDoseCalculator.effective_scopes() == frozenset({MetricScope.EPISODE, MetricScope.ROBOT, MetricScope.ENCOUNTER, MetricScope.PEDESTRIAN})
    assert SurgeEnergyCalculator.effective_scopes() == frozenset({MetricScope.EPISODE, MetricScope.ROBOT, MetricScope.ENCOUNTER})
    assert PedestrianActivityCalculator.effective_scopes() == frozenset({MetricScope.PEDESTRIAN})
    assert PedestrianIntrusionCalculator.effective_scopes() == frozenset({MetricScope.PEDESTRIAN})

    evaluator = ScopedEvaluator(_ROBOT_PARAMS, metrics_config=_referenced_config())
    hierarchy = evaluator.decompose_hierarchy(episode, run_id=RUN_ID, episode_id=EPISODE_ID)
    registry = MetricRegistry(evaluator.robot_params, evaluator.metrics_config)

    for scope in (MetricScope.CONTACT, MetricScope.ENCOUNTER, MetricScope.PEDESTRIAN, MetricScope.ROBOT, MetricScope.EPISODE):
        order = registry.get_execution_order_for_scope(scope)
        assert len(order) > 0

    ped_contact_slice = next(s for s in hierarchy.contacts if s.identity["obstacle_id"] == "Pedestrian_1")
    contact_outcome = registry.run_scope(
        MetricScope.CONTACT,
        ped_contact_slice.window_bundle,
        available_topics=set(ped_contact_slice.window_bundle.topics or {}),
        pedsim_available=True,
        scope_slice=ped_contact_slice,
    )
    assert isinstance(contact_outcome, ScopeExecutionOutcome)
    assert contact_outcome["probst_R_robot"] is not None
    assert contact_outcome["probst_R_actor2"] is not None
    assert contact_outcome["probst_R_time"] is not None
    assert (float(contact_outcome["probst_R_robot"]) + float(contact_outcome["probst_R_actor2"]) + float(contact_outcome["probst_R_time"])) == pytest.approx(1.0, abs=1e-9)


def test_numeric_gazebo_obstacle_id_and_clearance_fallback_link_encounter(episode: AlignedEpisodeBundle) -> None:
    numeric_contacts = pl.DataFrame(
        [
            {
                "t_ns": WALL_CONTACT_NS,
                "obstacle_id": "wall_north",
                "kind": "wall",
                "obstacle_x": 1.0,
                "obstacle_y": 0.3,
                "obstacle_z": 0.0,
                "distance": 0.0,
            },
            {
                "t_ns": PED_CONTACT_NS,
                "obstacle_id": f"{ENV_ID}/1",
                "kind": "pedestrian",
                "obstacle_x": ALICE_CROSSING_X_M,
                "obstacle_y": 0.0,
                "obstacle_z": 0.0,
                "distance": 0.0,
            },
        ],
        schema=_COLLISION_CONTACTS_SCHEMA,
    )
    numeric_topics = dict(episode.topics or {})
    numeric_topics["collision_contacts"] = numeric_contacts
    numeric_tables = _evaluate_referenced(dataclasses.replace(episode, topics=numeric_topics))
    contact = _single_row(numeric_tables.contacts, "obstacle_id", "Pedestrian_1")
    alice = _encounter_row(numeric_tables, "Pedestrian_1")
    assert contact["encounter_id"] == ALICE_ENCOUNTER_ID
    assert alice["had_contact"] is True
    assert alice["contact_id"] == contact["contact_id"]

    empty_topics = dict(episode.topics or {})
    empty_topics["collision_contacts"] = pl.DataFrame(schema=_COLLISION_CONTACTS_SCHEMA)
    fallback_tables = _evaluate_referenced(dataclasses.replace(episode, topics=empty_topics))
    fallback_contact = _single_row(fallback_tables.contacts, "obstacle_id", "Pedestrian_1")
    fallback_alice = _encounter_row(fallback_tables, "Pedestrian_1")
    assert fallback_contact["encounter_id"] == ALICE_ENCOUNTER_ID
    assert fallback_alice["had_contact"] is True
    assert fallback_alice["contact_id"] == fallback_contact["contact_id"]


def test_encounter_acoustics_attenuates_recorded_dba_and_fleet_encounter_has_chord_detour(episode: AlignedEpisodeBundle) -> None:
    louder_topics = dict(episode.topics or {})
    t_ns, _ = _time_axis()
    louder_topics["robot_acoustics"] = pl.DataFrame(
        {"t_ns": t_ns, "total_level_af_dba": np.full(SAMPLE_COUNT, 70.0, dtype=np.float64)},
        schema={"t_ns": pl.Int64, "total_level_af_dba": pl.Float64},
    )
    base_tables = _evaluate_referenced(episode)
    louder_tables = _evaluate_referenced(dataclasses.replace(episode, topics=louder_topics))
    base_dose = typing.cast(float, _encounter_row(base_tables, "Pedestrian_1")["sound_exposure_dose_pa2s"])
    louder_dose = typing.cast(float, _encounter_row(louder_tables, "Pedestrian_1")["sound_exposure_dose_pa2s"])
    assert louder_dose == pytest.approx(base_dose * 10.0, rel=1e-9)

    fleet_poses = pl.DataFrame(
        {"t_ns": t_ns, "pos_x": np.full(SAMPLE_COUNT, 3.0, dtype=np.float64), "pos_y": np.zeros(SAMPLE_COUNT, dtype=np.float64)},
        schema={"t_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64},
    )
    fleet_episode = dataclasses.replace(_without_topics(episode, "pedestrian_tracks", "collision_contacts"), fleet={"robot_peer": fleet_poses})
    fleet_tables = _evaluate(fleet_episode)
    peer_row = _encounter_row(fleet_tables, "robot_peer")
    assert peer_row["actor_2_kind"] == "robot"
    assert peer_row["path_deviation_m"] == pytest.approx(0.0, abs=1e-9)
    assert peer_row["detour_time_s"] is not None
    assert float(typing.cast(float, peer_row["detour_time_s"])) >= 0.0


def test_registry_reraises_metric_input_errors(episode: AlignedEpisodeBundle) -> None:
    class _MissingInputCalculator(BaseMetricCalculator):
        NAME = "missing_input_calc"
        SCOPE = MetricScope.ENCOUNTER
        SCOPES = frozenset({MetricScope.ENCOUNTER})

        @classmethod
        def output_keys(cls) -> list[str]:
            return ["missing_input_metric"]

        def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
            raise MetricInputError("required map is missing")

    registry = MetricRegistry(_ROBOT_PARAMS, {})
    registry.register(_MissingInputCalculator)
    evaluator = ScopedEvaluator(_ROBOT_PARAMS)
    hierarchy = evaluator.decompose_hierarchy(episode, run_id=RUN_ID, episode_id=EPISODE_ID)
    enc_slice = hierarchy.encounters[0]

    with pytest.raises(MetricInputError, match="required map is missing"):
        registry.run_scope(
            MetricScope.ENCOUNTER,
            enc_slice.window_bundle,
            available_topics=set(enc_slice.window_bundle.topics or {}),
            pedsim_available=True,
            scope_slice=enc_slice,
        )

def test_registry_provenance_records_topic_missing_no_samples_and_exception(episode: AlignedEpisodeBundle) -> None:
    class _BrokenCalculator(BaseMetricCalculator):
        NAME = "broken_calc"
        SCOPE = MetricScope.ENCOUNTER
        SCOPES = frozenset({MetricScope.ENCOUNTER})

        @classmethod
        def output_keys(cls) -> list[str]:
            return ["broken_metric"]

        def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
            raise RuntimeError("synthetic failure")

    registry = MetricRegistry(_ROBOT_PARAMS, {})
    registry.register(_BrokenCalculator)
    evaluator = ScopedEvaluator(_ROBOT_PARAMS)
    hierarchy = evaluator.decompose_hierarchy(episode, run_id=RUN_ID, episode_id=EPISODE_ID)
    enc_slice = hierarchy.encounters[0]

    outcome = registry.run_scope(
        MetricScope.ENCOUNTER,
        enc_slice.window_bundle,
        available_topics=set(enc_slice.window_bundle.topics or {}),
        pedsim_available=True,
        scope_slice=enc_slice,
    )
    assert outcome["broken_metric"] is None
    assert outcome.provenance.get("broken_metric") == "broken_metric:exception:RuntimeError"

    empty_power_topics = dict(enc_slice.window_bundle.topics or {})
    empty_power_topics["robot_power"] = pl.DataFrame(schema={"t_ns": pl.Int64, "total_power_w": pl.Float64})
    empty_window_slice = dataclasses.replace(
        enc_slice,
        window_bundle=dataclasses.replace(enc_slice.window_bundle, topics=empty_power_topics),
    )
    empty_outcome = registry.run_scope(
        MetricScope.ENCOUNTER,
        empty_window_slice.window_bundle,
        available_topics=set(empty_window_slice.window_bundle.topics or {}),
        pedsim_available=True,
        scope_slice=empty_window_slice,
    )
    assert empty_outcome.provenance.get("energy_surge_ratio") == "energy_surge_ratio:no_samples"

    missing_power_bundle = _without_topics(enc_slice.window_bundle, "robot_power")
    missing_power_slice = dataclasses.replace(
        enc_slice,
        window_bundle=missing_power_bundle,
        full_episode=_without_topics(enc_slice.full_episode, "robot_power"),
    )
    missing_outcome = registry.run_scope(
        MetricScope.ENCOUNTER,
        missing_power_slice.window_bundle,
        available_topics=set(missing_power_slice.window_bundle.topics or {}),
        pedsim_available=True,
        scope_slice=missing_power_slice,
    )
    assert missing_outcome.provenance.get("energy_surge_ratio") == "energy_surge_ratio:topic_missing:robot_power"


# ---------------------------------------------------------------------------
# Ported declarations: schema derivation, publication and null provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_name", FACT_TABLE_NAMES)
def test_declared_columns_are_typed_schema_columns(table_name: str) -> None:
    table = FactTable(table_name)
    schema = FACT_SCHEMAS[table_name]
    for declaration in declarations_for(table):
        assert schema[declaration.name] == declaration.dtype


def test_every_declaration_lands_in_its_table_schema() -> None:
    for declaration in METRIC_DECLARATIONS:
        assert FACT_SCHEMAS[declaration.fact_table.value][declaration.name] == declaration.dtype


def test_declared_column_names_are_unique_within_each_table() -> None:
    for table in FactTable:
        names = [declaration.name for declaration in declarations_for(table)]
        assert len(names) == len(set(names))


@pytest.mark.parametrize("table_name", FACT_TABLE_NAMES)
def test_declared_columns_keep_the_declaration_order(table_name: str) -> None:
    names = FACT_SCHEMAS[table_name].names()
    indexes = [names.index(declaration.name) for declaration in declarations_for(FactTable(table_name))]
    assert indexes == sorted(indexes)


@pytest.mark.parametrize(
    "declaration",
    declarations_for(FactTable.EPISODES),
    ids=lambda declaration: declaration.name,
)
def test_episode_fact_publishes_each_declared_metric(declaration: MetricDeclaration, tables: ScopedFactTables) -> None:
    row = tables.episodes.row(0, named=True)
    assert row[declaration.name] == _COMPLETE_EPISODE_METRICS[declaration.name]


@pytest.mark.parametrize(
    "declaration",
    declarations_for(FactTable.ROBOTS),
    ids=lambda declaration: declaration.name,
)
def test_robot_fact_falls_back_to_the_declared_episode_metrics(declaration: MetricDeclaration, tables: ScopedFactTables) -> None:
    assert tables.robots.item(0, declaration.name) == _COMPLETE_EPISODE_METRICS[declaration.name]


def test_pedestrian_fact_publishes_the_declared_disturbance_metrics(
    tables: ScopedFactTables,
) -> None:
    alice = _pedestrian_row(tables, "Pedestrian_1")
    assert alice["ped_round_trips_completed"] >= 0.0
    assert 0.0 <= alice["ped_velocity_delay_ratio"] <= 1.0


def test_declared_columns_stay_null_and_name_their_star_column(
    episode: AlignedEpisodeBundle,
) -> None:
    tables = _evaluate(episode, episode_metrics={"success": True})
    row = tables.episodes.row(0, named=True)
    reason = typing.cast(str, row["null_reason"])
    assert row["barn_score"] is None
    assert "barn_score:no_samples" in reason
    assert "barn_time:no_samples" in reason


# ---------------------------------------------------------------------------
# Child fact tables: the identity-grain list outputs reshaped into 1NF rows
# ---------------------------------------------------------------------------


def test_edge_case_fact_rows_reshape_each_case_list(tables: ScopedFactTables) -> None:
    frame = tables.edge_cases
    assert frame.height == 5
    assert frame.schema == FACT_SCHEMAS["edge_cases"]
    steps = frame.filter(pl.col("kind") == "step").sort("ordinal")
    assert steps["edge_case_steps"].to_list() == ["hold", "approach"]
    assert steps["ordinal"].to_list() == [0, 1]
    assert frame.filter(pl.col("kind") == "agent")["edge_case_agents"].to_list() == ["alice"]
    assert frame.filter(pl.col("kind") == "missing")["edge_case_missing"].to_list() == ["bob"]
    assert frame.filter(pl.col("kind") == "unplayed")["edge_case_unplayed"].to_list() == ["cara"]
    assert frame["case_idx"].unique().to_list() == [0]
    assert frame["null_reason"].unique().to_list() == [""]


def test_formation_fact_rows_carry_role_and_member(tables: ScopedFactTables) -> None:
    frame = tables.formations
    assert frame.height == 3
    members = frame.filter(pl.col("role") == "member").sort("edge_case_formation_members")
    assert members["edge_case_formation_members"].to_list() == ["alice", "cara"]
    absent = frame.filter(pl.col("role") == "absent")
    assert absent["edge_case_formation_absent"].to_list() == ["bob"]
    assert frame["formation_idx"].unique().to_list() == [0]


def test_phase_outcome_fact_rows_follow_the_outcome_list(tables: ScopedFactTables) -> None:
    frame = tables.phase_outcomes
    assert frame.height == 2
    assert frame["phase_idx"].to_list() == [0, 1]
    assert frame["phase_outcomes"].to_list() == ["met", "failed"]
    assert frame["met"].to_list() == [True, False]


def test_judge_disagreement_fact_rows_carry_the_text(tables: ScopedFactTables) -> None:
    frame = tables.judge_disagreements
    assert frame.height == 1
    assert frame["judge_disagreements"].to_list() == ["phase 1 recorded as met but not met in replay"]
    assert frame["disagreement_idx"].to_list() == [0]


def test_mechanism_state_fact_rows_are_change_points_with_one_flag(tables: ScopedFactTables) -> None:
    frame = tables.mechanism_states
    # progress [0, 0, 1] and locked [False, False, True] change at samples 0 and 2, two fields each.
    assert frame.height == 4
    assert frame["entity"].unique().to_list() == ["north_door"]
    assert frame["kind"].unique().to_list() == ["door"]
    assert frame["level"].unique().to_list() == ["0"]
    assert sorted(frame["field"].unique().to_list()) == ["locked", "progress"]
    assert int(frame["mechanism_states"].sum()) == 1
    assert sorted(frame.filter(pl.col("field") == "progress")["value"].to_list()) == [0.0, 1.0]


def test_mechanism_state_fact_rows_map_the_episode_clock(
    episode: AlignedEpisodeBundle,
) -> None:
    t_ns, _ = _time_axis()
    timed = dataclasses.replace(episode, data=pl.DataFrame({"time_ns": t_ns}))
    tables = _evaluate(timed)
    frame = tables.mechanism_states
    assert sorted(set(frame["t_ns"].drop_nulls().to_list())) == [0, 2 * (NS_PER_S // SAMPLE_RATE_HZ)]
    assert frame["null_reason"].unique().to_list() == [""]


def test_collision_fact_rows_carry_the_first_contact_of_each_obstacle(
    tables: ScopedFactTables,
) -> None:
    frame = tables.collisions
    assert frame.height == 1
    row = _single_row(frame, "collision_obstacles", "wall_north")
    assert row["collision_idx"] == 0
    assert row["obstacle_kind"] == "wall"
    assert row["t_ns"] == WALL_CONTACT_NS
    assert row["x_m"] == pytest.approx(1.0)
    assert row["y_m"] == pytest.approx(0.3)
    assert row["collisions"] == 1
    assert row["null_reason"] == ""


def test_child_fact_tables_are_empty_without_their_episode_lists(
    episode: AlignedEpisodeBundle,
) -> None:
    tables = _evaluate(episode, episode_metrics={"success": True})
    for name in ("edge_cases", "formations", "phase_outcomes", "judge_disagreements", "mechanism_states", "collisions"):
        frame = tables.by_name()[name]
        assert frame.height == 0
        assert frame.schema == FACT_SCHEMAS[name]
