from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from flat_peds import flat_peds

from arena_evaluation.processing.metrics.social.encounter_costs import (
    EncounterCostsCalculator,
)
from arena_evaluation.processing.telemetry.writer import TABLE_SCHEMAS
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SECOND_NS = 1_000_000_000
SAMPLE_DT_S = 0.1
ROBOT_RADIUS_M = 0.3


def _time_ns(duration_s: float) -> np.ndarray:
    """Robot clock of an episode sampled every 0.1 s."""
    seconds = np.arange(0.0, duration_s + SAMPLE_DT_S / 2.0, SAMPLE_DT_S)
    return np.round(seconds * SECOND_NS).astype(np.int64)


def _schema(topic: str) -> pl.Schema:
    """Recorded flat schema of a telemetry table."""
    return pl.from_arrow(TABLE_SCHEMAS[topic].empty_table()).schema


def _table(topic: str, rows: list[dict[str, object]]) -> pl.DataFrame:
    """Flat table of rows carrying the recorded schema of topic."""
    schema = _schema(topic)
    defaults = dict.fromkeys(schema)
    return pl.DataFrame([{**defaults, **row} for row in rows], schema=schema)


def _robot_tracks(
    time_ns: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    vx: np.ndarray | None = None,
    vy: np.ndarray | None = None,
    yaw: np.ndarray | None = None,
) -> pl.DataFrame:
    """Flat robot_tracks rows of a robot whose pose is sampled on time_ns, in the recorded schema."""
    count = len(time_ns)
    vx_arr = np.zeros(count) if vx is None else np.asarray(vx, dtype=np.float64)
    vy_arr = np.zeros(count) if vy is None else np.asarray(vy, dtype=np.float64)
    yaw_arr = np.zeros(count) if yaw is None else np.asarray(yaw, dtype=np.float64)
    return _table(
        "robot_tracks",
        [
            {
                "t_ns": int(time_ns[index]),
                "env_id": "env_0",
                "robot_id": "env_0_jackal",
                "x": float(np.asarray(x, dtype=np.float64)[index]),
                "y": float(np.asarray(y, dtype=np.float64)[index]),
                "z": 0.0,
                "yaw": float(yaw_arr[index]),
                "vx": float(vx_arr[index]),
                "vy": float(vy_arr[index]),
                "vz": 0.0,
                "omega_z": 0.0,
                "speed_mps": float(np.hypot(vx_arr[index], vy_arr[index])),
                "pose_source": "tf_gt",
            }
            for index in range(count)
        ],
    )


def _robot_power(time_ns: np.ndarray, total_power_w: np.ndarray) -> pl.DataFrame:
    """Flat robot_power rows sampled on time_ns, in the recorded schema."""
    power_w = np.asarray(total_power_w, dtype=np.float64)
    return _table(
        "robot_power",
        [
            {
                "t_ns": int(time_ns[index]),
                "env_id": "env_0",
                "robot_id": "env_0_jackal",
                "total_power_w": float(power_w[index]),
                "static_power_w": 15.0,
                "total_mech_power_w": float(power_w[index] - 15.0),
                "total_thermal_power_w": 0.0,
            }
            for index in range(len(time_ns))
        ],
    )


def _bundle(topics: dict[str, pl.DataFrame], *, episode_id: int = 1) -> AlignedEpisodeBundle:
    """Episode bundle whose telemetry lives only in the flat topic tables."""
    return AlignedEpisodeBundle(
        episode_id=episode_id,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[0.0, 0.0, 0.0],
        topics=topics,
    )


def _calculator(metrics_config: dict[str, object] | None = None) -> EncounterCostsCalculator:
    """Encounter costs calculator with an optional metrics config."""
    calculator = EncounterCostsCalculator(RobotParams(model="jackal", robot_radius=ROBOT_RADIUS_M))
    if metrics_config is not None:
        calculator.metrics_config = dict(metrics_config)
    return calculator


def _following_episode(duration_s: float, *, robot_power: pl.DataFrame | None = None) -> AlignedEpisodeBundle:
    """Robot driving +x at 1 m/s with pedestrian alice following one metre to its left."""
    time_ns = _time_ns(duration_s)
    seconds = time_ns.astype(np.float64) / SECOND_NS
    pedestrians = flat_peds(
        [
            (
                int(t_ns),
                [{"ped_id": "Pedestrian_7", "ped_name": "alice", "x": float(time_s), "y": 1.0, "vx": 1.0}],
            )
            for t_ns, time_s in zip(time_ns, seconds, strict=True)
        ]
    )
    topics = {
        "robot_tracks": _robot_tracks(time_ns, seconds.copy(), np.zeros_like(seconds), vx=np.ones_like(seconds)),
        "pedestrian_tracks": pedestrians,
    }
    if robot_power is not None:
        topics["robot_power"] = robot_power
    return _bundle(topics)


def test_reference_twin_matches_pedestrian_by_name() -> None:
    time_ns = _time_ns(5.0)
    seconds = time_ns.astype(np.float64) / SECOND_NS
    robot_x = 1.0 * seconds
    dynamic_peds = flat_peds(
        [
            (
                int(t_ns),
                [
                    {
                        "ped_id": "Pedestrian_7",
                        "ped_name": "alice",
                        "x": float(5.0 - time_s),
                        "y": float(np.sin(np.pi * time_s / 5.0)),
                        "vx": -1.0,
                    }
                ],
            )
            for t_ns, time_s in zip(time_ns, seconds, strict=True)
        ]
    )
    reference_peds = flat_peds(
        [
            (
                int(t_ns),
                [
                    {"ped_id": "Pedestrian_1", "ped_name": "bob", "x": 10.0, "y": 10.0},
                    {"ped_id": "Pedestrian_7", "ped_name": "alice", "x": float(5.0 - time_s), "y": 0.0, "vx": -1.0},
                ],
            )
            for t_ns, time_s in zip(time_ns, seconds, strict=True)
        ]
    )
    episode = _bundle(
        {
            "robot_tracks": _robot_tracks(time_ns, robot_x, np.zeros_like(robot_x), vx=np.ones_like(robot_x)),
            "pedestrian_tracks": dynamic_peds,
        },
        episode_id=1,
    )
    reference_episode = _bundle({"pedestrian_tracks": reference_peds}, episode_id=2)
    calculator = _calculator({"reference_episode": reference_episode, "encounter_distance_m": 10.0})

    outputs = calculator.calculate(episode, {})

    assert outputs["encounter_count"] == 1
    assert outputs["cost_time_s"] == pytest.approx(5.0, abs=1e-6)
    assert outputs["cost_path_m"] == pytest.approx(5.0, abs=1e-6)
    assert outputs["cost_energy_j"] > 0.0
    assert outputs["cost_sound_pa2s"] > 0.0
    assert outputs["cost_ped_detour_m"] is not None and outputs["cost_ped_detour_m"] > 0.3
    assert outputs["cost_ped_deflection_m"] is not None and outputs["cost_ped_deflection_m"] > 0.4


def test_overlapping_encounters_union_robot_costs_and_sum_sound() -> None:
    time_ns = _time_ns(4.0)
    seconds = time_ns.astype(np.float64) / SECOND_NS
    robot_x = 1.0 * seconds
    pedestrians = flat_peds(
        [
            (
                int(t_ns),
                [
                    {"ped_id": "Pedestrian_1", "ped_name": "ped_a", "x": float(time_s), "y": 1.0, "vx": 1.0},
                    {"ped_id": "Pedestrian_2", "ped_name": "ped_b", "x": float(time_s), "y": -1.0, "vx": 1.0},
                ],
            )
            for t_ns, time_s in zip(time_ns, seconds, strict=True)
        ]
    )
    episode = _bundle(
        {
            "robot_tracks": _robot_tracks(time_ns, robot_x, np.zeros_like(robot_x), vx=np.ones_like(robot_x)),
            "pedestrian_tracks": pedestrians,
        }
    )
    calculator = _calculator()

    records = calculator.calculate_encounters(episode)
    rolled = calculator.calculate(episode, {})

    assert len(records) == 2
    assert rolled["encounter_count"] == 2
    assert rolled["cost_time_s"] == pytest.approx(records[0].cost_time_s, abs=1e-6)
    assert rolled["cost_path_m"] == pytest.approx(records[0].cost_path_m, abs=1e-6)
    assert rolled["cost_energy_j"] == pytest.approx(records[0].cost_energy_j, abs=1e-6)
    assert rolled["cost_sound_pa2s"] == pytest.approx(records[0].cost_sound_pa2s + records[1].cost_sound_pa2s, abs=1e-9)


def test_no_pedestrians_yields_none_for_every_output() -> None:
    time_ns = _time_ns(2.0)
    seconds = time_ns.astype(np.float64) / SECOND_NS
    episode = _bundle({"robot_tracks": _robot_tracks(time_ns, seconds.copy(), np.zeros_like(seconds), vx=np.ones_like(seconds))})
    calculator = _calculator()

    outputs = calculator.calculate(episode, {})

    assert set(outputs) == set(calculator.output_keys())
    assert all(value is None for value in outputs.values())


def test_missing_robot_tracks_yields_none_for_every_output() -> None:
    time_ns = _time_ns(2.0)
    pedestrians = flat_peds([(int(t_ns), [{"ped_id": "Pedestrian_1", "ped_name": "alice", "x": 1.0, "y": 0.0}]) for t_ns in time_ns])
    episode = _bundle({"pedestrian_tracks": pedestrians})
    calculator = _calculator()

    outputs = calculator.calculate(episode, {})

    assert set(outputs) == set(calculator.output_keys())
    assert all(value is None for value in outputs.values())


def test_single_sample_robot_track_reports_missing_metrics() -> None:
    time_ns = np.array([0], dtype=np.int64)
    pedestrians = flat_peds([(0, [{"ped_id": "Pedestrian_1", "ped_name": "alice", "x": 1.0, "y": 0.0}])])
    episode = _bundle(
        {
            "robot_tracks": _robot_tracks(time_ns, np.array([0.0]), np.array([0.0])),
            "pedestrian_tracks": pedestrians,
        }
    )

    outputs = _calculator().calculate(episode, {})

    assert set(outputs) == set(EncounterCostsCalculator.output_keys())
    assert all(value is None for value in outputs.values())


def test_recorded_robot_power_is_interpolated_onto_robot_clock() -> None:
    time_ns = _time_ns(4.0)
    recorded_time_ns = time_ns + SECOND_NS // 10
    recorded_power = _robot_power(recorded_time_ns, np.full(len(recorded_time_ns), 200.0))
    episode = _following_episode(4.0, robot_power=recorded_power)

    outputs = _calculator().calculate(episode, {})

    assert outputs["cost_time_s"] == pytest.approx(4.0, abs=1e-6)
    assert outputs["cost_energy_j"] == pytest.approx(200.0 * 4.0, abs=1e-6)


def test_missing_robot_power_uses_kinematic_power_fallback() -> None:
    episode = _following_episode(4.0)

    outputs = _calculator().calculate(episode, {})

    kinematic_idle_power_w = 15.0
    kinematic_linear_power_w = 15.0 * 9.81 * 0.02
    kinematic_thermal_power_w = 0.1 * ((kinematic_linear_power_w / 24.0) ** 2)
    expected_energy_j = (kinematic_idle_power_w + kinematic_linear_power_w + kinematic_thermal_power_w) * 4.0
    assert outputs["cost_energy_j"] == pytest.approx(expected_energy_j, abs=1e-3)
    assert outputs["cost_energy_j"] > 0.0


def test_reference_episode_without_matching_name_reports_no_pedestrian_costs() -> None:
    time_ns = _time_ns(5.0)
    seconds = time_ns.astype(np.float64) / SECOND_NS
    dynamic_peds = flat_peds(
        [
            (
                int(t_ns),
                [{"ped_id": "Pedestrian_7", "ped_name": "alice", "x": float(5.0 - time_s), "y": 0.0, "vx": -1.0}],
            )
            for t_ns, time_s in zip(time_ns, seconds, strict=True)
        ]
    )
    reference_peds = flat_peds([(int(t_ns), [{"ped_id": "Pedestrian_1", "ped_name": "bob", "x": 10.0, "y": 10.0}]) for t_ns in time_ns])
    episode = _bundle(
        {
            "robot_tracks": _robot_tracks(time_ns, 1.0 * seconds, np.zeros_like(seconds), vx=np.ones_like(seconds)),
            "pedestrian_tracks": dynamic_peds,
        }
    )
    reference_episode = _bundle({"pedestrian_tracks": reference_peds}, episode_id=2)
    calculator = _calculator({"reference_episode": reference_episode, "encounter_distance_m": 10.0})

    outputs = calculator.calculate(episode, {})

    assert outputs["encounter_count"] == 1
    assert outputs["cost_ped_detour_m"] is None
    assert outputs["cost_ped_deflection_m"] is None
    assert outputs["cost_ped_delay_s"] is None


@pytest.mark.parametrize(("distance_threshold_m", "expected_count", "expected_time_s"), [(2.0, 0, 0.0), (4.0, 1, 1.0)])
def test_encounter_distance_threshold_controls_detection(distance_threshold_m: float, expected_count: int, expected_time_s: float) -> None:
    time_ns = _time_ns(1.0)
    stationary_x = np.zeros(len(time_ns))
    pedestrians = flat_peds([(int(t_ns), [{"ped_id": "Pedestrian_1", "ped_name": "alice", "x": 3.0, "y": 0.0}]) for t_ns in time_ns])
    episode = _bundle(
        {
            "robot_tracks": _robot_tracks(time_ns, stationary_x, np.zeros_like(stationary_x)),
            "pedestrian_tracks": pedestrians,
        }
    )
    calculator = _calculator({"encounter_distance_m": distance_threshold_m})

    outputs = calculator.calculate(episode, {})

    assert outputs["encounter_count"] == expected_count
    assert outputs["cost_time_s"] == pytest.approx(expected_time_s, abs=1e-6)
    if expected_count == 0:
        assert outputs["cost_energy_j"] == 0.0
        assert outputs["cost_ped_detour_m"] is None


def test_rollup_keeps_disjoint_encounter_windows_separate() -> None:
    time_ns = _time_ns(4.0)
    seconds = time_ns.astype(np.float64) / SECOND_NS
    first_ped_x = np.where(seconds <= 2.0, 1.0, 100.0)
    second_ped_x = np.where(seconds >= 2.1, 1.0, 100.0)
    pedestrians = flat_peds(
        [
            (
                int(t_ns),
                [
                    {"ped_id": "Pedestrian_1", "ped_name": "ped_a", "x": float(first_x), "y": 0.0},
                    {"ped_id": "Pedestrian_2", "ped_name": "ped_b", "x": float(second_x), "y": 0.0},
                ],
            )
            for t_ns, first_x, second_x in zip(time_ns, first_ped_x, second_ped_x, strict=True)
        ]
    )
    episode = _bundle(
        {
            "robot_tracks": _robot_tracks(time_ns, np.zeros_like(seconds), np.zeros_like(seconds)),
            "pedestrian_tracks": pedestrians,
        }
    )
    calculator = _calculator()

    records = calculator.calculate_encounters(episode)
    rolled = calculator.calculate(episode, {})

    assert len(records) == 2
    assert records[0].t_end_ns < records[1].t_start_ns
    assert rolled["encounter_count"] == 2
    assert rolled["cost_time_s"] == pytest.approx(records[0].cost_time_s + records[1].cost_time_s, abs=1e-6)
    assert rolled["cost_time_s"] < 4.0


def _scoping_episode() -> AlignedEpisodeBundle:
    """Robot driving +x at 1 m/s with alice following and bob far away."""
    time_ns = _time_ns(3.0)
    seconds = time_ns.astype(np.float64) / SECOND_NS
    pedestrians = flat_peds(
        [
            (
                int(t_ns),
                [
                    {"ped_id": "Pedestrian_7", "ped_name": "alice", "x": float(time_s), "y": 1.0, "vx": 1.0},
                    {"ped_id": "Pedestrian_9", "ped_name": "bob", "x": 100.0, "y": 100.0},
                ],
            )
            for t_ns, time_s in zip(time_ns, seconds, strict=True)
        ]
    )
    return _bundle(
        {
            "robot_tracks": _robot_tracks(time_ns, seconds.copy(), np.zeros_like(seconds), vx=np.ones_like(seconds)),
            "pedestrian_tracks": pedestrians,
        }
    )


@pytest.mark.parametrize("target_agent_id", [7, "7", "ped_7", "Pedestrian_7", "alice"])
def test_target_agent_id_selects_the_scoped_pedestrian(target_agent_id: int | str) -> None:
    calculator = _calculator({"target_agent_id": target_agent_id})

    records = calculator.calculate_encounters(_scoping_episode())

    assert len(records) == 1
    assert records[0].ped_id == "Pedestrian_7"
    assert records[0].ped_name == "alice"
    assert records[0].cost_time_s == pytest.approx(3.0, abs=1e-6)


def test_target_agent_id_takes_precedence_and_name_is_the_fallback() -> None:
    by_id = _calculator({"target_agent_id": "Pedestrian_7", "target_agent_name": "Pedestrian_9"})
    by_name = _calculator({"target_agent_id": "Pedestrian_99", "target_agent_name": "alice"})

    assert by_id.calculate_encounters(_scoping_episode())[0].ped_id == "Pedestrian_7"
    assert by_name.calculate_encounters(_scoping_episode())[0].ped_id == "Pedestrian_7"
