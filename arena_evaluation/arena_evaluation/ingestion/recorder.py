import argparse
import os
import shutil
import threading
import traceback
from datetime import datetime, timezone
import yaml
import pathlib
import sys
import signal

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.serialization import serialize_message

import rosbag2_py
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, JointState
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from tf2_msgs.msg import TFMessage

try:
    from arena_people_msgs.msg import Pedestrians
    HAS_PEDESTRIANS = True
except ImportError:
    class Pedestrians:
        pass
    HAS_PEDESTRIANS = False

try:
    from arena_robots_msgs.msg import CollisionEvents
    HAS_COLLISION = True
except ImportError:
    class CollisionEvents:
        pass
    HAS_COLLISION = False

try:
    from arena_robots_msgs.msg import Power, Energy
    HAS_POWER = True
except ImportError:
    class Power:
        pass
    class Energy:
        pass
    HAS_POWER = False

try:
    from task_generator_msgs.msg import EpisodeRecord, RobotFleet
    HAS_TASK_GEN = True
except ImportError:
    class EpisodeRecord:
        pass
    class RobotFleet:
        pass
    HAS_TASK_GEN = False

import arena_evaluation_msgs.srv as arena_evaluation_srvs

from .metadata import IngestionMetadata
from ..storage.manifest import MetadataWriter


class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('arena_evaluation_data_recorder')

        self.base_dir = get_package_share_directory("arena_evaluation")

        if not self.has_parameter("record_data_dir"):
            try:
                self.declare_parameter("record_data_dir", "")
            except Exception:
                pass
        record_data_dir = self.get_parameter("record_data_dir").get_parameter_value().string_value

        if not record_data_dir:
            for idx, arg in enumerate(sys.argv):
                if arg in ("--dir", "-d") and idx + 1 < len(sys.argv):
                    record_data_dir = sys.argv[idx + 1]
                    break
                elif arg.startswith("--dir="):
                    record_data_dir = arg[6:].strip()
                    break

        if not record_data_dir:
            record_data_dir = "auto:/"

        if record_data_dir.startswith("auto:/"):
            if not self.has_parameter("data_recorder_autoprefix"):
                try:
                    self.declare_parameter("data_recorder_autoprefix", "")
                except Exception:
                    pass
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            param_value = self.get_parameter("data_recorder_autoprefix").value
            if not param_value:
                self.set_parameters([Parameter("data_recorder_autoprefix", Parameter.Type.STRING, timestamp)])
            else:
                timestamp = param_value
            record_data_dir = os.path.join("data", "recordings", timestamp)

        record_data_dir_path = pathlib.Path(record_data_dir)
        if not record_data_dir_path.is_absolute():
            workspace_root = pathlib.Path(self.base_dir).parents[3]
            record_data_dir_path = workspace_root / record_data_dir_path

        self.run_dir = record_data_dir_path.resolve()

        bare_names = {"data", "recordings"}
        if self.run_dir.name in bare_names or not record_data_dir_path.parts[1:]:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_dir = self.run_dir / "recordings" / timestamp

        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.run_dir.chmod(0o777)
        except Exception:
            pass

        parts = self.run_dir.parts
        try:
            if len(parts) >= 5 and parts[-3] == "recordings":
                self.benchmark_id = parts[-4]
                self.planner = parts[-2]
                self.stage = parts[-1]
            else:
                self.benchmark_id = "unknown"
                self.planner = "unknown"
                self.stage = "unknown"
        except Exception:
            self.benchmark_id = "unknown"
            self.planner = "unknown"
            self.stage = "unknown"

        for param_name, default_val in [
            ("world", "unknown"),
            ("suite_name", "unknown"),
            ("contest_name", "unknown"),
            ("benchmark_id", ""),
            ("planner", ""),
            ("stage", ""),
        ]:
            if not self.has_parameter(param_name):
                try:
                    self.declare_parameter(param_name, default_val)
                except Exception:
                    pass
                    
        param_benchmark = self.get_parameter("benchmark_id").value
        if param_benchmark: self.benchmark_id = param_benchmark
        
        param_planner = self.get_parameter("planner").value
        if param_planner: self.planner = param_planner
        
        param_stage = self.get_parameter("stage").value
        if param_stage: self.stage = param_stage

        self.world = self.get_parameter("world").value
        self.suite_name = self.get_parameter("suite_name").value
        self.contest_name = self.get_parameter("contest_name").value
        
        self.robot_model = "unknown"
        self.known_robots = set()

        self.mcap_path = self.run_dir / "recording.mcap"
        self.metadata_path = self.run_dir / "metadata.yaml"

        if self.mcap_path.exists():
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = self.run_dir / f"recording_backup_{ts}.mcap"
            try:
                shutil.move(str(self.mcap_path), str(backup))
                self.get_logger().warn(f"Backed up existing MCAP to {backup}")
            except Exception:
                shutil.rmtree(self.mcap_path, ignore_errors=True)

        self._open_log_file()

        self.write_params()
        self.config = self.read_config()
        self.freqs = self.config.get("record_frequencies", {"default": 20.0})

        self._log_info(f"Writing initial metadata to {self.metadata_path}")
        self._write_initial_metadata()
        self._log_info("Initial metadata written OK")

        self.current_time = None
        self._pre_clock_buffer = []
        self._clock_received_count = 0
        self.last_recorded_times: dict[str, int] = {}
        self.writer_lock = threading.Lock()
        self.writer: rosbag2_py.SequentialWriter | None = None
        self.topics_metadata: dict[str, rosbag2_py.TopicMetadata] = {}
        self._topic_registry: dict[str, rosbag2_py.TopicMetadata] = {}
        self._write_drop_count = 0
        self._write_success_count = 0


        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=100,
        )

        self.is_shutting_down = False
        self.episodes_recorded = 0
        self._seen_episodes = set()
        self.recorded_topics: set[str] = set()

        self._log_info(f"Subscribing to /clock for sim time")
        self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_callback, self.qos)

        env_namespace = self.get_namespace().strip('/')
        env_ns_root = f"/{env_namespace}" if env_namespace else ""


        self._log_info(f"Setting up topic subscriptions")
        self._setup_subscriptions()
        self._log_info(f"Topic subscriptions ready. Opening MCAP writer...")

        self._start_recording()

    def _open_log_file(self):
        if hasattr(self, 'log_file') and self.log_file is not None and not self.log_file.closed:
            try:
                self.log_file.close()
            except Exception:
                pass
        self.log_file_path = self.run_dir / "recorder.log"
        try:
            self.log_file = open(self.log_file_path, "a", buffering=1) # Line buffered
        except Exception as e:
            print(f"[DataRecorder] Failed to open log file at {self.log_file_path}: {e}", flush=True)
            self.log_file = None

    def _start_recording(self):
        """Open the rosbag2 SequentialWriter on the continuous MCAP file.

        rosbag2 treats the `uri` as a directory prefix: it creates a folder
        named `uri` and puts `<uri>_0.mcap` inside.  We therefore pass only
        the stem (e.g. "recording"), which produces run_dir/recording/recording_0.mcap.
        """
        self._log_info(f"Opening MCAP writer in {self.run_dir}")
        mcap_uri = str(self.mcap_path.with_suffix(""))
        self._log_info(f"Opening MCAP writer: uri={mcap_uri}")

        mcap_config_path = os.path.join(self.base_dir, "config", "mcap_writer_options.yaml")
        if not os.path.exists(mcap_config_path):
            self._log_warn(f"mcap_writer_options.yaml not found at {mcap_config_path} — recording WITHOUT compression")
            mcap_config_path = ""

        storage_options = rosbag2_py.StorageOptions(
            uri=mcap_uri,
            storage_id='mcap',
            max_bagfile_size=0,
            max_cache_size=0,
            storage_config_uri=mcap_config_path,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )

        try:
            with self.writer_lock:
                self.writer = rosbag2_py.SequentialWriter()
                self._log_info("Calling writer.open() now...")
                self.writer.open(storage_options, converter_options)
                self.topics_metadata = {}
                self.last_recorded_times = {}
            self._log_info(f"MCAP writer opened successfully at {mcap_uri} (config: {mcap_config_path or 'none'})")
        except Exception as e:
            self._log_error(f"FATAL: Failed to open MCAP writer: {e}")
            import traceback as tb
            self._log_error(tb.format_exc())
            self.writer = None

        self._log_info(f"Started continuous recording at: {self.mcap_path}")

    def _setup_subscriptions(self):
        env_namespace = self.get_namespace().strip('/')
        env_prefix = f"/{env_namespace}" if env_namespace else ""

        self.subs = []

        from .topics import get_topics
        topics_dict = get_topics(namespace="", parent_namespace=env_namespace)

        for key, t_def in topics_dict.items():
            if key not in ("episode_record", "robots_fleet", "peds", "agent_states"):
                continue
                
            topic_name = t_def.name_template
            msg_type = t_def.msg_type

            if isinstance(msg_type, type) and msg_type.__name__ in ("Pedestrians", "AgentStates", "EpisodeRecord", "RobotFleet") and not msg_type.__module__.startswith("arena_") and not msg_type.__module__.startswith("task_generator_"):
                continue

            self._register_topic(topic_name, msg_type)

            qos_profile = self.latched_qos if t_def.qos_transient_local else self.qos

            if key == "episode_record":
                callback = self.episode_record_callback
            elif key == "robots_fleet":
                callback = self.robots_fleet_callback
            elif t_def.throttled:
                callback = self._create_throttled_callback(topic_name)
            else:
                callback = self._create_unthrottled_callback(topic_name)

            sub = self.create_subscription(msg_type, topic_name, callback, qos_profile)
            self.subs.append(sub)
            if key == "episode_record":
                self.get_logger().info(f"Subscribed to EpisodeRecord on {topic_name}")
            elif key == "robots_fleet":
                self.get_logger().info(f"Subscribed to RobotFleet on {topic_name}")

        self.create_timer(1.0, self.discover_topics)

    def discover_topics(self):
        namespace = self.get_namespace().strip('/')
        ns_prefix = f"/{namespace}" if namespace else ""

        for name, types in self.get_topic_names_and_types():
            if name in self._topic_registry:
                continue
            if not name.startswith(ns_prefix):
                continue

            name_lower = name.lower()
            is_scan = "scan" in name_lower or "lidar" in name_lower
            is_odom = "odom" in name_lower

            # if is_scan and "sensor_msgs/msg/LaserScan" in types:
            #     self._subscribe_discovered(name, LaserScan)
            if is_odom and "nav_msgs/msg/Odometry" in types:
                self._subscribe_discovered(name, Odometry)
                if self.robot_model in ("unknown", ""):
                    for part in reversed(name.split("/")):
                        if part and part not in ("odom", "eval_sim", "task_generator_node", "arena") \
                                and "velocity_controller" not in part \
                                and not part.startswith("env_"):
                            self.robot_model = part
                            if hasattr(self, 'metadata') and self.metadata is not None:
                                self.metadata.robot_model = [part]
                                MetadataWriter.write(self.metadata, self.metadata_path)
                            break

    def _subscribe_discovered(self, topic_name: str, msg_type):
        self._register_topic(topic_name, msg_type)
        sub = self.create_subscription(msg_type, topic_name, self._create_throttled_callback(topic_name), self.qos)
        self.subs.append(sub)
        self.get_logger().info(f"Dynamically subscribed to: {topic_name}")


    def clock_callback(self, msg: Clock):
        new_time = msg.clock.sec * int(1e9) + msg.clock.nanosec
        self._clock_received_count += 1
        if self._clock_received_count <= 5:
            self._log_info(f"/clock tick #{self._clock_received_count}: sim_time_ns={new_time} ({new_time/1e9:.3f}s)")
        self.current_time = new_time
        
        if hasattr(self, '_pre_clock_buffer') and self._pre_clock_buffer:
            self._log_info(f"Flushing {len(self._pre_clock_buffer)} pre-clock buffered messages")
            for topic, buffered_msg in self._pre_clock_buffer:
                self._write_to_bag_at(topic, buffered_msg, self.current_time)
            self._pre_clock_buffer.clear()
            del self._pre_clock_buffer

    def episode_record_callback(self, msg: EpisodeRecord):
        if msg.episode_id not in self._seen_episodes:
            self._seen_episodes.add(msg.episode_id)
            self.episodes_recorded += 1

        now = self.current_time
        if now is None:
            now = self.get_clock().now().nanoseconds
        
        env_namespace = self.get_namespace().strip('/')
        episode_topic = f"/{env_namespace}/state/episode" if env_namespace else "/state/episode"
        self._write_to_bag_at(episode_topic, msg, now)

        self._update_metadata_from_episode(msg)
        
    def robots_fleet_callback(self, msg):
        env_namespace = self.get_namespace().strip('/')
        now = self.current_time or self.get_clock().now().nanoseconds
        topic = f"/{env_namespace}/state/robots" if env_namespace else "/state/robots"
        self._write_to_bag_at(topic, msg, now)

        from .topics import get_topics
        
        for state in getattr(msg, 'robots', []):
            robot = state.descriptor
            robot_ns = robot.ns.strip('/')
            
            if robot_ns not in self.known_robots:
                self.get_logger().info(f"Discovered new robot from RobotFleet: {robot_ns} (Model: {robot.model})")
                self.known_robots.add(robot_ns)
                
                topics_dict = get_topics(namespace=robot_ns, parent_namespace=env_namespace)
                
                for key, t_def in topics_dict.items():
                    if key in ("episode_record", "robots_fleet", "peds", "agent_states"):
                        continue
                        
                    topic_name = t_def.name_template
                    msg_type = t_def.msg_type

                    if isinstance(msg_type, type) and msg_type.__name__ in ("CollisionEvents") and not msg_type.__module__.startswith("arena_") and not msg_type.__module__.startswith("task_generator_"):
                        continue

                    self._register_topic(topic_name, msg_type)

                    qos_profile = self.latched_qos if t_def.qos_transient_local else self.qos

                    if t_def.throttled:
                        callback = self._create_throttled_callback(topic_name)
                    else:
                        callback = self._create_unthrottled_callback(topic_name)

                    sub = self.create_subscription(msg_type, topic_name, callback, qos_profile)
                    self.subs.append(sub)
                    self.get_logger().info(f"Subscribed to robot topic: {topic_name}")
                    
                robot_name = robot_ns.split('/')[-1] if robot_ns else ""
                ns_prefix = f"/{robot_ns}" if robot_ns else ""
                if robot_name:
                    odom_topic = f"{ns_prefix}/{robot_name}_velocity_controller/odom"
                    self._register_topic(odom_topic, Odometry)
                    sub = self.create_subscription(Odometry, odom_topic, self._create_throttled_callback(odom_topic), self.qos)
                    self.subs.append(sub)
                    
                if self.robot_model == "unknown":
                    self.robot_model = robot.model
                    
                if hasattr(self, 'metadata') and self.metadata is not None:
                    if "unknown" in self.metadata.robot_model:
                        self.metadata.robot_model = []
                        
                    if robot.model not in self.metadata.robot_model:
                        self.metadata.robot_model.append(robot.model)
                        
                    try:
                        from ..storage.manifest import MetadataWriter
                        MetadataWriter.write(self.metadata, self.metadata_path)
                    except Exception as e:
                        self.get_logger().warn(f"Failed to write metadata: {e}")

    def _resolve_throttle_ms(self, topic_name: str) -> float:
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
            # Never write until simulation clock has been received
            if self.current_time is None:
                if hasattr(self, '_pre_clock_buffer'):
                    self._pre_clock_buffer.append((topic_name, msg))
                return
            now = self.current_time
            last_time = self.last_recorded_times.get(topic_name, 0)
            if (now - last_time) / 1e6 >= throttle_ms:
                self._write_to_bag_at(topic_name, msg, now)
                self.last_recorded_times[topic_name] = now
        return callback

    def _create_unthrottled_callback(self, topic_name: str):
        def callback(msg):
            # Never write until simulation clock has been received
            if self.current_time is None:
                if hasattr(self, '_pre_clock_buffer'):
                    self._pre_clock_buffer.append((topic_name, msg))
                return
            now = self.current_time
            self._write_to_bag_at(topic_name, msg, now)
        return callback

    def _log_info(self, msg: str):
        full_msg = f"[DataRecorder] [INFO] {msg}"
        if hasattr(self, 'log_file') and self.log_file is not None and not self.log_file.closed:
            self.log_file.write(f"[{datetime.now().isoformat()}] {full_msg}\n")
        try:
            if rclpy.ok():
                self.get_logger().info(msg)
            else:
                print(full_msg, flush=True)
        except Exception:
            print(full_msg, flush=True)

    def _log_warn(self, msg: str):
        full_msg = f"[DataRecorder] [WARN] {msg}"
        if hasattr(self, 'log_file') and self.log_file is not None and not self.log_file.closed:
            self.log_file.write(f"[{datetime.now().isoformat()}] {full_msg}\n")
        try:
            if rclpy.ok():
                self.get_logger().warn(msg)
            else:
                print(full_msg, flush=True)
        except Exception:
            print(full_msg, flush=True)

    def _log_error(self, msg: str):
        full_msg = f"[DataRecorder] [ERROR] {msg}"
        if hasattr(self, 'log_file') and self.log_file is not None and not self.log_file.closed:
            self.log_file.write(f"[{datetime.now().isoformat()}] {full_msg}\n")
        try:
            if rclpy.ok():
                self.get_logger().error(msg)
            else:
                print(full_msg, flush=True)
        except Exception:
            print(full_msg, flush=True)

    def _write_to_bag_at(self, topic_name: str, msg, timestamp_ns: int):
        if self.is_shutting_down:
            return

        try:
            serialized_msg = serialize_message(msg)
        except Exception as e:
            if not self.is_shutting_down:
                self._log_error(f"Serialization failed for {topic_name}: {e}")
            return

        with self.writer_lock:
            if self.writer is None:
                self._write_drop_count += 1
                if self._write_drop_count <= 10 or self._write_drop_count % 100 == 0:
                    self._log_warn(f"DROP (writer=None) topic={topic_name} drop_count={self._write_drop_count}")
                return
            try:
                self._ensure_topic_in_bag(topic_name)
                self.writer.write(topic_name.strip('/'), serialized_msg, timestamp_ns)
                self.recorded_topics.add(topic_name.strip('/'))
                self._write_success_count += 1
                if self._write_success_count <= 5 or self._write_success_count % 500 == 0:
                    self._log_info(f"Write success count reached {self._write_success_count}")
            except Exception as e:
                self._log_error(f"WRITE ERROR topic={topic_name} ts={timestamp_ns} err={e}")


    def _register_topic(self, topic_name: str, msg_type):
        """Pre-register a topic so we know its type string before the first message."""
        if topic_name in self._topic_registry:
            return
        try:
            parts = msg_type.__module__.split('.')
            type_str = f"{parts[0]}/msg/{msg_type.__name__}"
        except AttributeError:
            type_str = str(msg_type)

        self._topic_registry[topic_name] = rosbag2_py.TopicMetadata(
            id=0,
            name=topic_name.strip('/'),
            type=type_str,
            serialization_format='cdr',
        )

    def _ensure_topic_in_bag(self, topic_name: str):
        """Lazily call create_topic on first write. Must be called under writer_lock."""
        strip = topic_name.strip('/')
        if strip not in self.topics_metadata and topic_name in self._topic_registry:
            self.writer.create_topic(self._topic_registry[topic_name])
            self.topics_metadata[strip] = self._topic_registry[topic_name]


    def _write_initial_metadata(self):
        metadata = IngestionMetadata.create_initial_metadata(
            benchmark_id=self.benchmark_id,
            planner=self.planner,
            stage=self.stage,
            map_name=self.world,
            episodes_requested=0,
            robot_model=self.robot_model,
            suite_name=self.suite_name,
            contest_name=self.contest_name,
        )
        MetadataWriter.write(metadata, self.metadata_path)
        self.metadata = metadata

    def _update_metadata_from_episode(self, msg: EpisodeRecord):
        try:
            if not self.metadata.robot_model or self.metadata.robot_model == ["unknown"]:
                self.metadata.robot_model = list(msg.robots)
            if not self.metadata.map or self.metadata.map == "unknown":
                self.metadata.map = msg.world

            self.metadata.tm_obstacles = msg.tm_obstacles
            self.metadata.tm_robots = msg.tm_robots
            self.metadata.tm_modules = list(msg.tm_modules)

            obstacles_params = {p.name: self._param_value_to_py(p.value) for p in msg.obstacles_params}
            robots_params = {p.name: self._param_value_to_py(p.value) for p in msg.robots_params}
            self.metadata.obstacles_params = self._unflatten_dict(obstacles_params)
            self.metadata.robots_params = self._unflatten_dict(robots_params)

            MetadataWriter.write(self.metadata, self.metadata_path)
        except Exception as e:
            self._log_error(f"Failed to update metadata from episode: {e}")

    def _param_value_to_py(self, val):
        from rcl_interfaces.msg import ParameterType
        mapping = {
            ParameterType.PARAMETER_BOOL: lambda v: v.bool_value,
            ParameterType.PARAMETER_INTEGER: lambda v: v.integer_value,
            ParameterType.PARAMETER_DOUBLE: lambda v: v.double_value,
            ParameterType.PARAMETER_STRING: lambda v: v.string_value,
            ParameterType.PARAMETER_BYTE_ARRAY: lambda v: list(v.byte_array_value),
            ParameterType.PARAMETER_BOOL_ARRAY: lambda v: list(v.bool_array_value),
            ParameterType.PARAMETER_INTEGER_ARRAY: lambda v: list(v.integer_array_value),
            ParameterType.PARAMETER_DOUBLE_ARRAY: lambda v: list(v.double_array_value),
            ParameterType.PARAMETER_STRING_ARRAY: lambda v: list(v.string_array_value),
        }
        return mapping.get(val.type, lambda v: str(v))(val)

    def _unflatten_dict(self, d: dict) -> dict:
        res: dict = {}
        for k, v in d.items():
            parts = k.split('.')
            curr = res
            for part in parts[:-1]:
                curr = curr.setdefault(part, {})
            curr[parts[-1]] = v
        return res

    def write_params(self):
        params_path = self.run_dir / "params.yaml"
        for param_name, default_val in [("map_file", ""), ("scenario_file", "")]:
            if not self.has_parameter(param_name):
                try:
                    self.declare_parameter(param_name, default_val)
                except Exception:
                    pass

        with open(params_path, "w") as f:
            yaml.dump({
                "model": self.model if hasattr(self, 'model') else "",
                "map_file": self.get_parameter("map_file").value,
                "scenario_file": self.get_parameter("scenario_file").value,
                "namespace": self.get_namespace().strip('/'),
            }, f)

    def read_config(self):
        config_path = os.path.join(self.base_dir, "config", "data_recorder_config.yaml")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {"record_frequencies": {"default": 20.0}}

    def finalize(self):
        """Flush and close the writer, write final metadata."""
        if self.is_shutting_down:
            return
        self.is_shutting_down = True

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except Exception:
            pass

        print(f"[DataRecorder] finalize() called. clock_ticks={self._clock_received_count} writes_ok={self._write_success_count} writes_dropped={self._write_drop_count} episodes={self.episodes_recorded}", flush=True)

        self._log_info("Finalizing recording...")

        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.recording_ended_at = datetime.now(timezone.utc).isoformat()
            self.metadata.episodes_recorded = self.episodes_recorded
            self.metadata.pedsim_available = any("arena_peds" in t or "agent_states" in t for t in self.recorded_topics)
            self.metadata.recorded_topics = sorted(self.recorded_topics)
            print(f"[DataRecorder] Writing final run-level metadata to {self.metadata_path}...", flush=True)
            try:
                MetadataWriter.write(self.metadata, self.metadata_path)
                print("[DataRecorder] Run-level metadata written successfully.", flush=True)
            except Exception as e:
                print(f"[DataRecorder] Failed to write final metadata: {e}", flush=True)

        with self.writer_lock:
            if self.writer is not None:
                print("[DataRecorder] Closing MCAP writer...", flush=True)
                try:
                    self.writer.close()
                    print("[DataRecorder] MCAP writer closed successfully.", flush=True)
                except Exception as e:
                    print(f"[DataRecorder] Failed to close MCAP writer: {e}", flush=True)
                finally:
                    del self.writer
                    self.writer = None
                    import gc
                    gc.collect()

        self._log_info(f"Recording finished. {self.episodes_recorded} episodes recorded to {self.mcap_path}")

    def destroy_node(self):
        self.finalize()
        super().destroy_node()


def _bind_to_parent():
    """SIGTERM the recorder when its parent dies instead of orphaning to init."""
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass

    if os.getppid() == 1:
        os._exit(0)


def main(args=None):
    import os

    _bind_to_parent()

    def _on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_term)

    rclpy.init(args=args)

    node = None
    try:
        node = DataRecorderNode()

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            print("[DataRecorder] Finalizing node...", flush=True)
            if node:
                node.finalize()
            print("[DataRecorder] Shutting down executor...", flush=True)
            executor.shutdown()
            print("[DataRecorder] Destroying node...", flush=True)
            if node:
                node.destroy_node()
    except Exception as e:
        print(f"[DataRecorder] Exception in main: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if rclpy.ok():
            rclpy.shutdown()
        os._exit(1)

    if rclpy.ok():
        rclpy.shutdown()
    os._exit(0)


if __name__ == '__main__':
    main()