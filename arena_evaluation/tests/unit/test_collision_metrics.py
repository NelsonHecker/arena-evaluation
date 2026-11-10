import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.path import theta_star
from arena_evaluation.processing.path.theta_star import GeometricThetaStar

from arena_evaluation.processing.metrics.base import MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def test_collision_amount():
    df = pl.DataFrame(
        {
            "event_count": [
                0,  # no collision
                2,  # collision 1
                2,  # still collision 1
                0,  # no collision
                1,  # collision 2
            ]
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    prior = {"time_to_goal": 10.0}
    results = calc.calculate(episode, prior)

    assert results["collision_amount"] == 2
    assert results["result"] == "GOAL_REACHED"
    assert results["success"] == True


def test_collision_amount_split_by_kind():
    df = pl.DataFrame(
        {
            "event_count": [0, 2, 2, 0, 1, 1],
            "wall_count": [0, 1, 1, 0, 0, 0],
            "static_count": [0, 1, 1, 0, 0, 0],
            "ped_count": [0, 0, 0, 0, 1, 1],
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_amount"] == 2
    assert results["collision_amount_wall"] == 1
    assert results["collision_amount_static"] == 1
    assert results["collision_amount_pedestrian"] == 1


def test_collision_amount_split_by_kind_unknown():
    df = pl.DataFrame(
        {
            "event_count": [0, 2, 0],
            "wall_count": [None, None, None],
            "static_count": [None, None, None],
            "ped_count": [None, None, None],
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_amount"] == 1
    assert results["collision_amount_wall"] is None
    assert results["collision_amount_static"] is None
    assert results["collision_amount_pedestrian"] is None


def test_collision_timeout_and_fail():
    df = pl.DataFrame(
        {
            "event_count": [0, 1, 0, 1, 0, 1, 0, 1]  # 4 collisions, > MAX_COLLISIONS
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results_timeout = calc.calculate(episode, {"time_to_goal": 200.0})
    assert results_timeout["result"] == "TIMEOUT"
    assert results_timeout["success"] == False

    results_fail = calc.calculate(episode, {"time_to_goal": 50.0})
    assert results_fail["result"] == "COLLISION"
    assert results_fail["success"] == False


def test_collision_obstacles_recorded():
    df = pl.DataFrame({"event_count": [0, 2, 1]})
    contacts = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 1_000_000_000, 2_000_000_000],
            "obstacle_id": ["<wall>", "chair_1", "ped_3"],
        }
    )
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=df,
        start_pos=[],
        goal_pos=[],
        topics={"collision_contacts": contacts},
    )
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_obstacles"] == ["<wall>", "chair_1", "ped_3"]


def test_collision_obstacles_absent():
    df = pl.DataFrame({"event_count": [0, 1]})

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_obstacles"] == []


def test_recorded_outcome_overrides_derived_success():
    # A cancelled episode can still show a short time_to_goal, the runtime's verdict wins.
    df = pl.DataFrame({"event_count": [0, 0, 0]})
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[], outcome_state=4)
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))

    results = calc.calculate(episode, {"time_to_goal": 169.0})

    assert results["success"] is False
    assert results["result"] == "CANCELLED"


def test_recorded_success_keeps_the_derived_result():
    df = pl.DataFrame({"event_count": [0, 0, 0]})
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[], outcome_state=2)
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["success"] is True
    assert results["result"] == "GOAL_REACHED"


def test_action_type_stop_zones_no_longer_count_as_collisions():
    df = pl.DataFrame(
        {
            "event_count": [0, 0, 0, 0, 0],
            "action_type": [0, 1, 1, 0, 1],
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_amount"] == 0
    assert results["result"] == "GOAL_REACHED"
    assert results["success"] is True


def test_outcome_info_collision_overrides_result():
    df = pl.DataFrame({"event_count": [0, 0, 0]})
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=df,
        start_pos=[],
        goal_pos=[],
        outcome_state=3,
        outcome_info="collision",
    )
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["result"] == "COLLISION"
    assert results["success"] is False


def _episode_with_outcome(outcome_state: int | None) -> AlignedEpisodeBundle:
    df = pl.DataFrame({"event_count": [0, 0, 0]})
    return AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[], outcome_state=outcome_state)


def test_unresolved_record_overrides_trace_verdict() -> None:
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))
    for state in (0, 1):
        results = calc.calculate(_episode_with_outcome(state), {"time_to_goal": 200.0})
        assert results["result"] == "UNRESOLVED"
        assert results["success"] is False


def test_missing_record_keeps_trace_verdict() -> None:
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))
    results = calc.calculate(_episode_with_outcome(None), {"time_to_goal": 200.0})
    assert results["result"] == "TIMEOUT"
    assert results["success"] is False


def test_max_collisions_override() -> None:
    df = pl.DataFrame({"event_count": [0, 2, 0]})
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))
    calc.metrics_config = {"max_collisions": 1}

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_amount"] == 1
    assert results["result"] == "COLLISION"
    assert results["success"] is False


def test_max_collisions_default_unchanged() -> None:
    df = pl.DataFrame({"event_count": [0, 2, 0, 1]})
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))

    results = calc.calculate(episode, {"time_to_goal": 10.0})

    assert results["collision_amount"] == 2
    assert results["result"] == "GOAL_REACHED"
    assert results["success"] is True


# --- CONTACT scope -----------------------------------------------------------

SEC_NS = 1_000_000_000
_EPISODE_TOPIC_SCHEMA = {
    "t_ns": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "yaw": pl.Float64,
}
_EVENTS_SCHEMA = {
    "t_ns": pl.Int64,
    "event_count": pl.Int64,
    "wall_count": pl.Int64,
    "static_count": pl.Int64,
    "ped_count": pl.Int64,
}
_ALIGNED_SCHEMA = {
    "time_ns": pl.Int64,
    "pos_x": pl.Float64,
    "pos_y": pl.Float64,
    "pos_x_gt": pl.Float64,
    "pos_y_gt": pl.Float64,
    "yaw_gt": pl.Float64,
}


def _spl_episode(outcome_state: int = 2, span_s: float = 3.0) -> AlignedEpisodeBundle:
    """A straight 3 m ground-truth track to a goal 2 m away: native theta* is the 2 m straight line."""
    time_ns = [int(index * span_s * SEC_NS / 3.0) for index in range(4)]
    data = pl.DataFrame(
        {
            "time_ns": time_ns,
            "pos_x": [0.0, 1.0, 2.0, 3.0],
            "pos_y": [0.0, 0.0, 0.0, 0.0],
            "pos_x_gt": [0.0, 1.0, 2.0, 3.0],
            "pos_y_gt": [0.0, 0.0, 0.0, 0.0],
            "yaw_gt": [0.0, 0.0, 0.0, 0.0],
        },
        schema=_ALIGNED_SCHEMA,
    )
    tracks = pl.DataFrame(
        {"t_ns": time_ns, "x": [0.0, 1.0, 2.0, 3.0], "y": [0.0] * 4, "yaw": [0.0] * 4},
        schema=_EPISODE_TOPIC_SCHEMA,
    )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[2.0, 0.0],
        outcome_state=outcome_state,
        topics={"robot_tracks": tracks},
    )


def _events_episode(event_count, wall_count) -> AlignedEpisodeBundle:
    count = len(event_count)
    time_ns = [index * SEC_NS for index in range(count)]
    events = pl.DataFrame(
        {
            "t_ns": time_ns,
            "event_count": event_count,
            "wall_count": wall_count,
            "static_count": [0] * count,
            "ped_count": [0] * count,
        },
        schema=_EVENTS_SCHEMA,
    )
    episode = _spl_episode()
    return AlignedEpisodeBundle(
        episode_id=episode.episode_id,
        data=episode.data,
        start_pos=episode.start_pos,
        goal_pos=episode.goal_pos,
        outcome_state=episode.outcome_state,
        topics={**episode.topics, "collision_events": events},
    )


def _contact_slice(
    full: AlignedEpisodeBundle,
    t_start_ns: int,
    t_end_ns: int,
    obstacle_id: str = "ped_1",
) -> ScopeSlice:
    window = AlignedEpisodeBundle(
        episode_id=full.episode_id,
        data=pl.DataFrame(),
        start_pos=full.start_pos,
        goal_pos=full.goal_pos,
        topics=dict(full.topics),
    )
    return ScopeSlice(
        scope=MetricScope.CONTACT,
        slice_id="con-001",
        parent_id=None,
        t_start_ns=t_start_ns,
        t_end_ns=t_end_ns,
        identity={"obstacle_id": obstacle_id},
        window_bundle=window,
        full_episode=full,
    )


@pytest.mark.parametrize(
    ("t_start_ns", "t_end_ns", "expected_event", "expected_wall"),
    [
        (SEC_NS, 3 * SEC_NS, 1, 1),  # both onsets fall inside the contact interval
        (3 * SEC_NS, 3 * SEC_NS, 0, 0),  # a later contact window contains no onset of its own
        (0, 0, 0, 0),  # the interval before the first onset
    ],
)
def test_contact_counts_only_the_windows_own_onsets(t_start_ns, t_end_ns, expected_event, expected_wall):
    full = _events_episode(event_count=[0, 1, 1, 1, 0], wall_count=[0, 0, 1, 1, 0])
    contact = _contact_slice(full, t_start_ns, t_end_ns)
    result = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0)).calculate_for_scope(contact.window_bundle, {}, contact)
    assert result["collision_amount"] == expected_event
    assert result["collision_amount_wall"] == expected_wall


def test_contact_counts_are_null_without_the_collision_stream():
    full = _spl_episode()
    contact = _contact_slice(full, SEC_NS, 2 * SEC_NS)
    result = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0)).calculate_for_scope(contact.window_bundle, {}, contact)
    assert result["collision_amount"] is None
    assert result["collision_amount_wall"] is None


def test_contact_carries_the_subject_actor_and_the_legacy_dead_list():
    full = _events_episode(event_count=[0, 1, 1, 1, 0], wall_count=[0, 0, 0, 0, 0])
    contact = _contact_slice(full, SEC_NS, 3 * SEC_NS, obstacle_id="env_0/Pedestrian_4")
    result = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0)).calculate_for_scope(contact.window_bundle, {}, contact)
    assert result["collision_obstacles"] == ["env_0/Pedestrian_4"]
    assert result["collisions"] == []


def test_contact_spl_reproduces_the_episode_score():
    full = _spl_episode()
    calc = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0))
    contact = _contact_slice(full, SEC_NS, 2 * SEC_NS)
    window_result = calc.calculate_for_scope(contact.window_bundle, {}, contact)
    episode_result = calc.calculate(full, {"time_to_goal": 3.0, "path_length": 3.0, "theta_star_length": 2.0})
    assert_frame_equal(
        pl.DataFrame({"spl": [window_result["spl"]]}, schema={"spl": pl.Float64}),
        pl.DataFrame({"spl": [episode_result["spl"]]}, schema={"spl": pl.Float64}),
    )
    assert window_result["spl"] == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    ("outcome_state", "span_s"),
    [(3, 3.0), (2, 200.0)],
    ids=["failed_episode", "timed_out_episode"],
)
def test_contact_spl_is_zero_when_the_episode_does_not_succeed(outcome_state, span_s):
    full = _spl_episode(outcome_state=outcome_state, span_s=span_s)
    contact = _contact_slice(full, 0, SEC_NS)
    result = CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0)).calculate_for_scope(contact.window_bundle, {}, contact)
    assert result["spl"] == 0.0


def test_contact_scope_publishes_the_five_scalar_candidates():
    assert CollisionMetricsCalculator.output_keys_for_scope(MetricScope.CONTACT) == (
        "collision_amount",
        "collision_amount_wall",
        "collision_obstacles",
        "collisions",
        "spl",
    )


def test_other_scopes_are_rejected():
    full = _spl_episode()
    slice_ = _contact_slice(full, 0, SEC_NS)
    pedestrian_slice = ScopeSlice(
        scope=MetricScope.PEDESTRIAN,
        slice_id="ped_1",
        parent_id=None,
        t_start_ns=0,
        t_end_ns=SEC_NS,
        identity={},
        window_bundle=slice_.window_bundle,
        full_episode=full,
    )
    with pytest.raises(ValueError, match="does not serve the pedestrians scope"):
        CollisionMetricsCalculator(RobotParams(0.2, 0.0, 10.0)).calculate_for_scope(pedestrian_slice.window_bundle, {}, pedestrian_slice)


COLLISION_ROUTE_MAP = "collision_test_open_room"


@pytest.fixture(autouse=True, scope="module")
def collision_open_room_solver():
    theta_star._solver_instances[f"{COLLISION_ROUTE_MAP}_0.25"] = GeometricThetaStar(np.zeros((200, 200), dtype=bool), resolution=0.1, robot_radius=0.25)


def test_contact_scope_theta_star_uses_the_whole_mission_route():
    time_ns = [index * SEC_NS for index in range(7)]
    xs = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]
    data = pl.DataFrame(
        {
            "time_ns": time_ns,
            "pos_x": xs,
            "pos_y": [0.0] * 7,
            "pos_x_gt": xs,
            "pos_y_gt": [0.0] * 7,
            "yaw_gt": [0.0] * 7,
        },
        schema=_ALIGNED_SCHEMA,
    )
    tracks = pl.DataFrame({"t_ns": time_ns, "x": xs, "y": [0.0] * 7, "yaw": [0.0] * 7}, schema=_EPISODE_TOPIC_SCHEMA)
    plans = pl.DataFrame(
        {
            "t_ns": [SEC_NS, 2 * SEC_NS],
            "plan_id": ["plan_0", "plan_1"],
            "waypoint_idx": [0, 0],
            "x": [3.0, 0.0],
            "y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    )
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[0.0, 0.0],
        outcome_state=2,
        map=COLLISION_ROUTE_MAP,
        topics={"robot_tracks": tracks, "robot_plans": plans},
    )

    assert CollisionMetricsCalculator(RobotParams(robot_radius=0.25))._theta_star_length_m(episode) == pytest.approx(6.0)
