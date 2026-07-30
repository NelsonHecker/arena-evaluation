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
    from arena_people_msgs.msg import Pedestrians
    from arena_humansim_msgs.msg import AgentStates
    from task_generator_msgs.msg import EpisodeRecord, RobotFleet
    from arena_robots_msgs.msg import CollisionEvents, Power, Energy, Acoustics
    from nav2_msgs.msg import CollisionMonitorState

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
        "peds": TopicDefinition(f"{p_ns}/arena_peds", Pedestrians, throttled=True),
        "agent_states": TopicDefinition(f"{p_ns}/agent_states", AgentStates, throttled=True),
        "episode_record": TopicDefinition(f"{p_ns}/state/episode", EpisodeRecord, throttled=False, qos_transient_local=False),
        "robots_fleet": TopicDefinition(f"{p_ns}/state/robots", RobotFleet, throttled=False, qos_transient_local=True),
        "collision_events": TopicDefinition(f"{ns}/collision_events", CollisionEvents, throttled=False),
        "power": TopicDefinition(f"{ns}/power_publisher/power", Power, throttled=True),
        "energy": TopicDefinition(f"{ns}/power_publisher/energy", Energy, throttled=True),
        "acoustics": TopicDefinition(f"{ns}/acoustics", Acoustics, throttled=True),
        "collision_monitor_state": TopicDefinition(f"{ns}/collision_monitor_state", CollisionMonitorState, throttled=False),
    }

    return topics
