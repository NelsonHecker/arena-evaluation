#!/usr/bin/env python3

import argparse
import os
import shutil
import threading
import traceback
from datetime import datetime
import yaml

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.serialization import serialize_message

from rosbag2_py import (
    SequentialWriter,
    ConverterOptions,
    StorageOptions,
    TopicMetadata
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Int16

try:
    from arena_people_msgs.msg import Pedestrians
    HAS_PEDESTRIANS = True
except ImportError:
    HAS_PEDESTRIANS = False

from arena_robots_msgs.msg import CollisionEvents
from task_generator_msgs.msg import EpisodeRecord
import arena_evaluation_msgs.srv as arena_evaluation_srvs


class DataRecorder(Node):
    def __init__(self, result_dir: str):
        super().__init__("data_recorder_node")
        
        self.declare_parameter("data_recorder_autoprefix", "")
        self.result_dir = self.get_directory(result_dir)

        self.declare_parameter("model", "")
        self.model = self.get_parameter("model").value

        self.base_dir = get_package_share_directory("arena_evaluation")
        self.result_dir = os.path.join(self.base_dir, "data", self.result_dir)
        os.makedirs(self.result_dir, exist_ok=True)

        self.write_params()
        self.config = self.read_config()
        self.freqs = self.config.get("record_frequencies", {"default": 20.0})

        self.current_time = None
        self.last_recorded_times = {}

        # Thread-safe writer access — locked during episode rotation
        self.writer_lock = threading.Lock()
        self.writer = None
        self.topics_metadata = {}

        # Track all known topic registrations so we can re-register on rotation
        self._topic_registry: dict[str, TopicMetadata] = {}

        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_callback, self.qos)
        
        # Service
        self.change_directory_service = self.create_service(
            arena_evaluation_srvs.ChangeDirectory,
            'change_directory',
            self.change_directory_callback
        )

        self._setup_subscriptions()

        # Start the first episode recording
        self._start_new_recording()

    # ──────────────────────────────────────────────────────────────
    # Recording lifecycle
    # ──────────────────────────────────────────────────────────────

    def _start_new_recording(self):
        """Create a fresh 'current_episode' directory and open a new bag writer."""
        self.current_episode_dir = os.path.join(self.result_dir, "current_episode")

        # Purge any leftover partial episode
        if os.path.exists(self.current_episode_dir):
            shutil.rmtree(self.current_episode_dir)
        os.makedirs(self.current_episode_dir, exist_ok=True)

        bag_uri = os.path.join(self.current_episode_dir, "recording")
        storage_options = StorageOptions(
            uri=bag_uri,
            storage_id='mcap',
            max_bagfile_size=0,
            max_cache_size=0,
            storage_preset_profile='zstd_small'
        )
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        with self.writer_lock:
            self.writer = SequentialWriter()
            self.writer.open(storage_options, converter_options)
            self.topics_metadata = {}
            self.last_recorded_times = {}
            # Topics are registered lazily on first real write — no eager pre-registration needed

        self.get_logger().info(f"Started new episode recording at: {self.current_episode_dir}")

    def _finalize_episode(self, msg: EpisodeRecord):
        """Seal the current bag, write metadata, and rotate directory to episode_<id>."""
        episode_id = msg.episode_id

        with self.writer_lock:
            # Write the EpisodeRecord into the bag before sealing
            if self.writer is not None and self.current_time is not None:
                try:
                    parent_ns = "/" + "/".join(self.get_namespace().strip('/').split('/')[:-1]) \
                        if "/" in self.get_namespace().strip('/') else ""
                    episode_topic = f"{parent_ns}/state/episode"
                    self._ensure_topic_in_bag(episode_topic)
                    self.writer.write(episode_topic.strip('/'), serialize_message(msg), self.current_time)
                except BaseException:
                    pass

            # Seal the bag
            self.writer = None

        # Write metadata.yaml alongside the recording
        obstacles_params = {p.name: self._param_value_to_py(p.value) for p in msg.obstacles_params}
        robots_params = {p.name: self._param_value_to_py(p.value) for p in msg.robots_params}

        metadata = {
            "episode_id": episode_id,
            "robot_model": msg.robots,
            "planner_type": self.model,
            "environment_map": msg.world,
            "episode_result": msg.outcome_state,
            "episode_info": msg.outcome_info,
            "tm_obstacles": msg.tm_obstacles,
            "tm_robots": msg.tm_robots,
            "obstacles_params": self._unflatten_dict(obstacles_params),
            "robots_params": self._unflatten_dict(robots_params),
            "recorded_topics": list(self._topic_registry.keys()),
        }

        metadata_path = os.path.join(self.current_episode_dir, "metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        # Rename current_episode → episode_<id>
        episode_dir = os.path.join(self.result_dir, f"episode_{episode_id}")
        if os.path.exists(episode_dir):
            shutil.rmtree(episode_dir)
        os.rename(self.current_episode_dir, episode_dir)

        self.get_logger().info(
            f"Episode {episode_id} finalized → {episode_dir}"
        )

        # Immediately begin recording the next episode
        self._start_new_recording()

    # ──────────────────────────────────────────────────────────────
    # Subscriptions
    # ──────────────────────────────────────────────────────────────

    def _setup_subscriptions(self):
        namespace = self.get_namespace().strip('/')
        ns_prefix = f"/{namespace}" if namespace else ""
        
        # Hardcoded topics under the robot sub-namespace
        throttled_topics = [
            (f"{ns_prefix}/cmd_vel", Twist),
            (f"{ns_prefix}/joint_states", JointState),
        ]

        # arena_peds is published at the task generator level (parent namespace), not per-robot
        parent_ns = "/" + "/".join(namespace.split('/')[:-1]) if "/" in namespace else ""
        if HAS_PEDESTRIANS:
            throttled_topics.append((f"{parent_ns}/arena_peds", Pedestrians))
        else:
            self.get_logger().warn("arena_people_msgs not found! arena_peds will NOT be recorded.")
            
        self.subs = []
        for topic_name, msg_type in throttled_topics:
            self._register_topic(topic_name, msg_type)
            callback = self._create_throttled_callback(topic_name)
            sub = self.create_subscription(msg_type, topic_name, callback, self.qos)
            self.subs.append(sub)

        # Dynamic topic discovery timer (runs every 2 seconds) to find flexible topics like scan and odom
        self.discovery_timer = self.create_timer(2.0, self.discover_topics)

        # Unthrottled / Event topics
        self._register_topic(f"{ns_prefix}/collision_events", CollisionEvents)
        self.subs.append(self.create_subscription(
            CollisionEvents, f"{ns_prefix}/collision_events", self.collision_events_callback, self.qos))

        # EpisodeRecord is published by the task generator, one level above the robot namespace
        parent_ns = "/" + "/".join(namespace.split('/')[:-1]) if "/" in namespace else ""
        episode_topic = f"{parent_ns}/state/episode"
        self._register_topic(episode_topic, EpisodeRecord)
        self.subs.append(self.create_subscription(
            EpisodeRecord, episode_topic, self.episode_record_callback, self.qos))
        self.get_logger().info(f"Subscribed to EpisodeRecord on {episode_topic}")

        self._register_topic(f"{ns_prefix}/plan", Path)
        self.subs.append(self.create_subscription(
            Path, f"{ns_prefix}/plan", self._create_unthrottled_callback(f"{ns_prefix}/plan"), self.qos))

    def discover_topics(self):
        namespace = self.get_namespace().strip('/')
        ns_prefix = f"/{namespace}" if namespace else ""
        
        for name, types in self.get_topic_names_and_types():
            if name in self._topic_registry:
                continue  # Already registered

            if not name.startswith(ns_prefix):
                continue  # Not in our robot's namespace
                
            is_scan = "scan" in name or "lidar" in name
            is_odom = "odom" in name
            
            if is_scan and "sensor_msgs/msg/LaserScan" in types:
                self._subscribe_discovered_topic(name, LaserScan)
            elif is_odom and "nav_msgs/msg/Odometry" in types:
                self._subscribe_discovered_topic(name, Odometry)

    def _subscribe_discovered_topic(self, topic_name, msg_type):
        self._register_topic(topic_name, msg_type)
        callback = self._create_throttled_callback(topic_name)
        sub = self.create_subscription(msg_type, topic_name, callback, self.qos)
        self.subs.append(sub)
        self.get_logger().info(f"Dynamically discovered and subscribed to: {topic_name}")

    # ──────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────

    def _resolve_throttle_ms(self, topic_name: str) -> float:
        """Find the throttle interval (ms) for a topic.
        
        Checks all keys in `record_frequencies` as substrings of the full topic path,
        so 'lidar' matches '.../jackal/lidar', 'odom' matches '.../jackal_velocity_controller/odom', etc.
        Falls back to 'default' if nothing matches.
        """
        topic_lower = topic_name.lower()
        for key, ms in self.freqs.items():
            if key == "default":
                continue
            if key.lower() in topic_lower:
                return float(ms)
        return float(self.freqs.get("default", 20.0))

    def _create_throttled_callback(self, topic_name: str):
        throttle_ms = self._resolve_throttle_ms(topic_name)
        
        def callback(msg):
            if self.current_time is None:
                return
            last_time = self.last_recorded_times.get(topic_name, 0)
            time_diff = (self.current_time - last_time) / 1e6  # ms
            
            if time_diff >= throttle_ms:
                with self.writer_lock:
                    if self.writer is None:
                        return
                    try:
                        self._ensure_topic_in_bag(topic_name)
                        self.writer.write(topic_name.strip('/'), serialize_message(msg), self.current_time)
                        self.last_recorded_times[topic_name] = self.current_time
                    except BaseException as e:
                        self.get_logger().error(f"Error writing to {topic_name}: {e}")
        return callback

    def _create_unthrottled_callback(self, topic_name: str):
        def callback(msg):
            if self.current_time is None:
                return
            with self.writer_lock:
                if self.writer is None:
                    return
                try:
                    self._ensure_topic_in_bag(topic_name)
                    self.writer.write(topic_name.strip('/'), serialize_message(msg), self.current_time)
                except BaseException as e:
                    self.get_logger().error(f"Error writing to {topic_name}: {e}")
        return callback

    def _register_topic(self, topic_name, msg_type):
        """Track topic metadata. Does NOT register in the bag until a real message arrives."""
        if topic_name in self._topic_registry:
            return
        type_str = f"{os.path.dirname(msg_type.__module__.replace('.', '/'))}/{msg_type.__name__}"
        metadata = TopicMetadata(
            id=0,
            name=topic_name.strip('/'),
            type=type_str,
            serialization_format='cdr'
        )
        self._topic_registry[topic_name] = metadata

    def _ensure_topic_in_bag(self, topic_name):
        """Lazily register topic in the open bag on first real message. Must be called under writer_lock."""
        if topic_name not in self.topics_metadata and topic_name in self._topic_registry:
            self.writer.create_topic(self._topic_registry[topic_name])
            self.topics_metadata[topic_name] = self._topic_registry[topic_name]

    def clock_callback(self, clock: Clock):
        self.current_time = clock.clock.sec * int(1e9) + clock.clock.nanosec

    def collision_events_callback(self, msg: CollisionEvents):
        if self.current_time is None:
            return
        topic_name = f"{self.get_namespace().strip('/')}/collision_events".strip('/')
        with self.writer_lock:
            if self.writer is None:
                return
            try:
                self._ensure_topic_in_bag(f"/{topic_name}")
                self.writer.write(topic_name, serialize_message(msg), self.current_time)
            except BaseException as e:
                self.get_logger().error(f"Error writing collision: {e}")

    def episode_record_callback(self, msg: EpisodeRecord):
        self._finalize_episode(msg)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _param_value_to_py(self, val):
        from rcl_interfaces.msg import ParameterType
        if val.type == ParameterType.PARAMETER_BOOL:
            return val.bool_value
        if val.type == ParameterType.PARAMETER_INTEGER:
            return val.integer_value
        if val.type == ParameterType.PARAMETER_DOUBLE:
            return val.double_value
        if val.type == ParameterType.PARAMETER_STRING:
            return val.string_value
        if val.type == ParameterType.PARAMETER_BYTE_ARRAY:
            return list(val.byte_array_value)
        if val.type == ParameterType.PARAMETER_BOOL_ARRAY:
            return list(val.bool_array_value)
        if val.type == ParameterType.PARAMETER_INTEGER_ARRAY:
            return list(val.integer_array_value)
        if val.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            return list(val.double_array_value)
        if val.type == ParameterType.PARAMETER_STRING_ARRAY:
            return list(val.string_array_value)
        return str(val)

    def _unflatten_dict(self, d):
        res = {}
        for k, v in d.items():
            parts = k.split('.')
            curr = res
            for part in parts[:-1]:
                if part not in curr:
                    curr[part] = {}
                curr = curr[part]
            curr[parts[-1]] = v
        return res

    def change_directory_callback(self, request, response):
        self.result_dir = self.get_directory(request.data)
        response.success = True
        response.message = "Directory changed successfully"
        return response

    def get_directory(self, directory: str) -> str:
        AUTO_PREFIX = "auto:/"
        PARAM_AUTO_PREFIX = "data_recorder_autoprefix"
        if directory.startswith(AUTO_PREFIX):
            set_prefix = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            self.get_logger().info(f"Generated timestamp: {set_prefix}")
            param_value = self.get_parameter(PARAM_AUTO_PREFIX).value
            if param_value == "":
                self.set_parameters([Parameter(PARAM_AUTO_PREFIX, Parameter.Type.STRING, set_prefix)])
            else:
                set_prefix = param_value
            directory = os.path.join(str(set_prefix), directory[len(AUTO_PREFIX):])
        return directory

    def write_params(self):
        params_path = os.path.join(self.result_dir, "params.yaml")
        with open(params_path, "w") as file:
            self.declare_parameter("map_file", "")
            self.declare_parameter("scenario_file", "")

            map_file = self.get_parameter("map_file").value
            scenario_file = self.get_parameter("scenario_file").value
            namespace = self.get_namespace().strip('/')
            yaml.dump({
                "model": self.model,
                "map_file": map_file,
                "scenario_file": scenario_file,
                "namespace": namespace
            }, file)

    def read_config(self):
        config_path = os.path.join(self.base_dir, "config", "data_recorder_config.yaml")
        try:
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        except Exception:
            return {"record_frequencies": {"default": 20.0}}
    def destroy_node(self):
        """Clean up the in-progress episode directory on shutdown."""
        try:
            if hasattr(self, 'current_episode_dir') and os.path.exists(self.current_episode_dir):
                self.get_logger().warn(
                    f"Node shutting down mid-episode — removing incomplete recording at {self.current_episode_dir}"
                )
                shutil.rmtree(self.current_episode_dir)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", default="auto:/")
    arguments, extra_args = parser.parse_known_args()
    
    try:
        recorder = DataRecorder(arguments.dir)
        executor = MultiThreadedExecutor()
        executor.add_node(recorder)
        executor.spin()
    except BaseException as e:
        print(f"Exception in main: {e}")
        traceback.print_exc()
        raise e
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
