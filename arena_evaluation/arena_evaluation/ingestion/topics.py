from __future__ import annotations

import dataclasses

@dataclasses.dataclass
class TopicDefinition:
    """Definition of a topic to record."""
    name_template: str
    msg_type: type
    throttled: bool = True
    throttle_rate_hz: float = 10.0
    qos_transient_local: bool = False


def get_topics(namespace: str, parent_namespace: str = "") -> dict[str, TopicDefinition]:
    """
    Returns the dictionary of topics to subscribe to.
    """
    from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
    from sensor_msgs.msg import JointState
    from nav_msgs.msg import Path
    from tf2_msgs.msg import TFMessage

    # Optional dependencies
    try:
        from arena_people_msgs.msg import Pedestrians
        HAS_PEDSIM = True
    except ImportError:
        Pedestrians = type("Pedestrians", (), {})
        HAS_PEDSIM = False

    try:
        from arena_humansim_msgs.msg import AgentStates
        HAS_HUMANSIM = True
    except ImportError:
        AgentStates = type("AgentStates", (), {})
        HAS_HUMANSIM = False

    try:
        from task_generator_msgs.msg import EpisodeRecord, RobotFleet
        HAS_TASK_GEN = True
    except ImportError:
        EpisodeRecord = type("EpisodeRecord", (), {})
        RobotFleet = type("RobotFleet", (), {})
        HAS_TASK_GEN = False

    try:
        from arena_robots_msgs.msg import CollisionEvents
        HAS_COLLISION = True
    except ImportError:
        CollisionEvents = type("CollisionEvents", (), {})
        HAS_COLLISION = False

    try:
        from nav2_msgs.msg import CollisionMonitorState
        HAS_NAV2_COLLISION = True
    except ImportError:
        CollisionMonitorState = type("CollisionMonitorState", (), {})
        HAS_NAV2_COLLISION = False

    # Use default namespaces if empty
    ns = f"/{namespace}" if namespace else ""
    p_ns = f"/{parent_namespace}" if parent_namespace else ""
    if ns == "/": ns = ""
    if p_ns == "/": p_ns = ""

    topics = {
        "cmd_vel": TopicDefinition(f"{ns}/cmd_vel", Twist, throttled=True),
        "joint_states": TopicDefinition(f"{ns}/joint_states", JointState, throttled=True),
        "plan": TopicDefinition(f"{ns}/plan", Path, throttled=False),
        "goal_pose": TopicDefinition(f"{p_ns}/goal_pose", PoseStamped, throttled=False),
        "initialpose": TopicDefinition(f"{p_ns}/initialpose", PoseWithCovarianceStamped, throttled=False),
        "tf": TopicDefinition("/tf", TFMessage, throttled=True),
        "tf_static": TopicDefinition("/tf_static", TFMessage, throttled=False, qos_transient_local=True),
    }

    if HAS_PEDSIM:
        topics["peds"] = TopicDefinition(f"{p_ns}/arena_peds", Pedestrians, throttled=True)

    if HAS_HUMANSIM:
        topics["agent_states"] = TopicDefinition(f"{p_ns}/agent_states", AgentStates, throttled=True)

    if HAS_TASK_GEN:
        topics["episode_record"] = TopicDefinition(
            f"{p_ns}/state/episode", EpisodeRecord, throttled=False,
            qos_transient_local=False,
        )
        topics["robots_fleet"] = TopicDefinition(
            f"{p_ns}/state/robots", RobotFleet, throttled=False,
            qos_transient_local=True,
        )

    if HAS_COLLISION:
        topics["collision_events"] = TopicDefinition(
            f"{ns}/collision_events", CollisionEvents, throttled=False,
        )

    if HAS_NAV2_COLLISION:
        topics["collision_monitor_state"] = TopicDefinition(
            f"{ns}/collision_monitor_state", CollisionMonitorState, throttled=False,
        )

    return topics
