from __future__ import annotations

import pathlib
import math
import typing
import polars as pl
from collections import defaultdict
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import pyarrow as pa
import pyarrow.parquet as pq

from ..storage.schemas import TopicBundle


class MCAPReader:
    """
    Reads an MCAP file (or legacy CSVs) and produces a TopicBundle of raw DataFrames.
    """
    def __init__(self, data_path: pathlib.Path):
        self.data_path = data_path

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """Convert quaternion to yaw angle. Matches legacy metrics implementation."""
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _param_value_to_py(val) -> typing.Any:
        p_type = getattr(val, "type", 0)
        if p_type == 1:
            return getattr(val, "bool_value", False)
        elif p_type == 2:
            return getattr(val, "integer_value", 0)
        elif p_type == 3:
            return getattr(val, "double_value", 0.0)
        elif p_type == 4:
            return getattr(val, "string_value", "")
        elif p_type == 5:
            return list(getattr(val, "byte_array_value", []))
        elif p_type == 6:
            return list(getattr(val, "bool_array_value", []))
        elif p_type == 7:
            return list(getattr(val, "integer_array_value", []))
        elif p_type == 8:
            return list(getattr(val, "double_array_value", []))
        elif p_type == 9:
            return list(getattr(val, "string_array_value", []))
        return str(val)

    @staticmethod
    def _unflatten_dict(d: dict) -> dict:
        res: dict = {}
        for k, v in d.items():
            parts = k.split('.')
            curr = res
            for part in parts[:-1]:
                curr = curr.setdefault(part, {})
            curr[parts[-1]] = v
        return res

    def read(self) -> dict[str, TopicBundle]:
        """
        Reads the data source and returns raw DataFrames/LazyFrames for each topic,
        organized by robot namespace.
        """
        if self.data_path.is_dir():
            mcap_files = list(self.data_path.glob("*.mcap"))
            if mcap_files:
                actual_path = mcap_files[0]
            else:
                raise FileNotFoundError(f"No MCAP file found in directory: {self.data_path}")
        else:
            actual_path = self.data_path
            
        if not actual_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {actual_path}")

        path = self.data_path.resolve()
        if path.is_dir():
            run_dir = path
        elif path.parent.name == "recording":
            run_dir = path.parent.parent
        else:
            run_dir = path.parent
        topics_dir = run_dir / "topics"

        def new_robot_data():
            return {
                "odom": defaultdict(list),
                "scan": defaultdict(list),
                "cmd_vel": defaultdict(list),
                "joint_states": defaultdict(list),
                "collision_events": defaultdict(list),
                "collision_monitor_state": defaultdict(list),
                "plan": defaultdict(list),
                "initialpose": defaultdict(list),
                "tf_gt": defaultdict(list),
            }
            
        global_data = {
            "peds": defaultdict(list),
            "episode_record": defaultdict(list),
            "tf": defaultdict(list),
            "tf_static": defaultdict(list),
        }
        
        robot_data = defaultdict(new_robot_data)

        env_prefix = None

        from mcap.reader import NonSeekingReader

        topics_dir.mkdir(parents=True, exist_ok=True)

        writers = {}
        accumulated_count = 0

        def flush_buffers():
            for topic_name, topic_data in global_data.items():
                if not topic_data or len(topic_data.get("time_ns", [])) == 0:
                    continue
                
                batch = pa.RecordBatch.from_pydict(dict(topic_data))
                writer_key = ("__global__", topic_name)
                
                if writer_key not in writers:
                    final_path = topics_dir / f"{topic_name}.parquet"
                    writers[writer_key] = pq.ParquetWriter(
                        final_path, schema=batch.schema, compression="zstd"
                    )
                writers[writer_key].write_batch(batch)
                topic_data.clear()
                
            for robot_name, r_data in robot_data.items():
                for topic_name, topic_data in r_data.items():
                    if not topic_data or len(topic_data.get("time_ns", [])) == 0:
                        continue
                        
                    batch = pa.RecordBatch.from_pydict(dict(topic_data))
                    writer_key = (robot_name, topic_name)
                    
                    if writer_key not in writers:
                        robot_dir = topics_dir / robot_name
                        robot_dir.mkdir(parents=True, exist_ok=True)
                        final_path = robot_dir / f"{topic_name}.parquet"
                        writers[writer_key] = pq.ParquetWriter(
                            final_path, schema=batch.schema, compression="zstd"
                        )
                    writers[writer_key].write_batch(batch)
                    topic_data.clear()

        try:
            with open(actual_path, "rb") as f:
                reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])
                
                msg_count = 0
                decoders = {}
                
                for schema, channel, message in reader.iter_messages(log_time_order=False):
                    try:
                        decoder = decoders.get(message.channel_id)
                        if decoder is None:
                            for factory in reader._decoder_factories:
                                decoder = factory.decoder_for(channel.message_encoding, schema)
                                if decoder is not None:
                                    decoders[message.channel_id] = decoder
                                    break
                        if decoder is None:
                            continue
                        ros_msg = decoder(message.data)
                        msg_count += 1
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"Error reading MCAP message on {channel.topic} at index {msg_count}: {e}")
                        continue

                    topic = channel.topic
                    
                    if env_prefix is None and ("env_" in topic or "env" in topic):
                        import re
                        match = re.search(r'env_(\d+)', topic)
                        if match:
                            env_prefix = f"env_{match.group(1)}"
                    
                    # Filter other namespaced topics to ensure we only read data for the active environment
                    if env_prefix and topic not in ("/tf", "tf", "/tf_static", "tf_static", "/state/robots", "state/robots", "/state/episode", "state/episode", "/arena_peds", "arena_peds"):
                        if env_prefix not in topic:
                            continue
                            
                    ts_ns = message.log_time
                    appended = False
                    
                    # Determine robot name for non-global topics
                    parts = [p for p in topic.strip('/').split('/') if p]
                    
                    # Odom
                    if topic.endswith("/odom") and "velocity_controller" not in topic:
                        # e.g. /arena/env_0/jackal1/odom -> parts = ['arena', 'env_0', 'jackal1', 'odom']
                        robot_name = parts[-2] if len(parts) >= 2 else "unknown"
                        if robot_name.endswith("_velocity_controller"):
                            robot_name = robot_name.replace("_velocity_controller", "")
                            
                        target = robot_data[robot_name]["odom"]
                        target["time_ns"].append(ts_ns)
                        target["pos_x"].append(ros_msg.pose.pose.position.x)
                        target["pos_y"].append(ros_msg.pose.pose.position.y)
                        
                        yaw = self._quaternion_to_yaw(
                            ros_msg.pose.pose.orientation.x,
                            ros_msg.pose.pose.orientation.y,
                            ros_msg.pose.pose.orientation.z,
                            ros_msg.pose.pose.orientation.w
                        )
                        target["yaw"].append(yaw)
                        target["vel_linear"].append(ros_msg.twist.twist.linear.x)
                        target["vel_angular"].append(ros_msg.twist.twist.angular.z)
                        appended = True
                    
                    # Scan
                    elif topic.endswith("/scan") or topic.endswith("/lidar"):
                        robot_name = parts[-2] if len(parts) >= 2 else "unknown"
                        target = robot_data[robot_name]["scan"]
                        target["time_ns"].append(ts_ns)
                        target["scan_ranges"].append(list(ros_msg.ranges))
                        target["scan_min"].append(ros_msg.range_min)
                        appended = True
                    
                    # Cmd_vel
                    elif topic.endswith("/cmd_vel"):
                        robot_name = parts[-2] if len(parts) >= 2 else "unknown"
                        target = robot_data[robot_name]["cmd_vel"]
                        target["time_ns"].append(ts_ns)
                        target["linear_x"].append(ros_msg.linear.x)
                        target["linear_y"].append(ros_msg.linear.y)
                        target["linear_z"].append(ros_msg.linear.z)
                        target["angular_x"].append(ros_msg.angular.x)
                        target["angular_y"].append(ros_msg.angular.y)
                        target["angular_z"].append(ros_msg.angular.z)
                        # Keep legacy columns
                        target["cmd_linear"].append(ros_msg.linear.x)
                        target["cmd_angular"].append(ros_msg.angular.z)
                        appended = True
                        
                    # Joint states
                    elif topic.endswith("/joint_states"):
                        robot_name = parts[-2] if len(parts) >= 2 else "unknown"
                        target = robot_data[robot_name]["joint_states"]
                        target["time_ns"].append(ts_ns)
                        target["name"].append(list(ros_msg.name))
                        target["position"].append(list(ros_msg.position))
                        target["velocity"].append(list(ros_msg.velocity))
                        target["effort"].append(list(ros_msg.effort))
                        appended = True

                    # Pedestrians
                    elif topic.endswith("/arena_peds") or topic.endswith("/peds") or topic.endswith("/agent_states"):
                        target = global_data["peds"]
                        target["time_ns"].append(ts_ns)
                        
                        if hasattr(ros_msg, "pedestrians"):
                            agents = ros_msg.pedestrians
                            is_pose2d = False
                        else:
                            agents = [a for a in ros_msg.agents if getattr(a, "kind", 0) == 0]
                            is_pose2d = True

                        target["num_pedestrians"].append(len(agents))
                        
                        positions = []
                        headings = []
                        twists = []
                        
                        for p in agents:
                            if is_pose2d:
                                positions.extend([p.pose.x, p.pose.y, 0.0])
                                headings.append(p.pose.theta)
                                twists.extend([p.velocity.x, p.velocity.y, p.velocity.z])
                            else:
                                # Positions: flattened list [x1, y1, z1, x2, y2, z2, ...]
                                positions.extend([p.pose.position.x, p.pose.position.y, p.pose.position.z])
                                
                                # Headings: calculate yaw from quaternion
                                yaw = self._quaternion_to_yaw(
                                    p.pose.orientation.x,
                                    p.pose.orientation.y,
                                    p.pose.orientation.z,
                                    p.pose.orientation.w
                                )
                                headings.append(yaw)
                                
                                # Twists: flattened list of linear velocities [vx1, vy1, vz1, vx2, vy2, vz2, ...]
                                twists.extend([p.twist.linear.x, p.twist.linear.y, p.twist.linear.z])
                                
                        target["peds_positions"].append(positions)
                        target["peds_headings"].append(headings)
                        target["peds_twists"].append(twists)
                        
                        appended = True

                    # Episode records
                    elif topic.endswith("/state/episode"):
                        target = global_data["episode_record"]
                        target["time_ns"].append(ts_ns)
                        target["episode_id"].append(ros_msg.episode_id)
                        target["outcome_state"].append(ros_msg.outcome_state)
                        target["outcome_info"].append(ros_msg.outcome_info)
                        target["goal_uuid"].append(ros_msg.goal_uuid)
                        
                        robots_yaml = ""
                        try:
                            import yaml
                            robots_dict = {}
                            for p in ros_msg.robots_params:
                                robots_dict[p.name] = self._param_value_to_py(p.value)
                            robots_dict = self._unflatten_dict(robots_dict)
                            robots_yaml = yaml.dump(robots_dict)
                        except Exception:
                            pass
                        target["robots_params"].append(robots_yaml)
                        appended = True

                    # Collision Events (Taskgen simulation)
                    elif topic.endswith("/collision_events"):
                        robot_name = parts[-2] if len(parts) >= 2 else "unknown"
                        target = robot_data[robot_name]["collision_events"]
                        target["time_ns"].append(ts_ns)
                        target["collision_event"].append(len(ros_msg.events))
                        appended = True
                        
                    # Collision Monitor State (Nav2)
                    elif topic.endswith("/collision_monitor_state"):
                        robot_name = parts[-2] if len(parts) >= 2 else "unknown"
                        target = robot_data[robot_name]["collision_monitor_state"]
                        target["time_ns"].append(ts_ns)
                        target["action_type"].append(ros_msg.action_type)
                        target["polygon_name"].append(ros_msg.polygon_name)
                        appended = True

                    # Global plan
                    elif topic.endswith("/plan") or topic.endswith("/global_plan"):
                        robot_name = parts[-2] if len(parts) >= 2 else "unknown"
                        target = robot_data[robot_name]["plan"]
                        target["time_ns"].append(ts_ns)
                        poses_x = []
                        poses_y = []
                        poses_yaw = []
                        for pose_stamped in ros_msg.poses:
                            poses_x.append(pose_stamped.pose.position.x)
                            poses_y.append(pose_stamped.pose.position.y)
                            poses_yaw.append(self._quaternion_to_yaw(
                                pose_stamped.pose.orientation.x,
                                pose_stamped.pose.orientation.y,
                                pose_stamped.pose.orientation.z,
                                pose_stamped.pose.orientation.w
                            ))
                        target["poses_x"].append(poses_x)
                        target["poses_y"].append(poses_y)
                        target["poses_yaw"].append(poses_yaw)
                        appended = True
                        
                    # Initialpose
                    elif topic.endswith("/initialpose"):
                        robot_name = parts[-2] if len(parts) >= 2 else "unknown"
                        target = robot_data[robot_name]["initialpose"]
                        target["time_ns"].append(ts_ns)
                        target["pos_x"].append(ros_msg.pose.pose.position.x)
                        target["pos_y"].append(ros_msg.pose.pose.position.y)
                        target["yaw"].append(self._quaternion_to_yaw(
                            ros_msg.pose.pose.orientation.x,
                            ros_msg.pose.pose.orientation.y,
                            ros_msg.pose.pose.orientation.z,
                            ros_msg.pose.pose.orientation.w
                        ))
                        appended = True
                        
                    # TF processing
                    elif topic in ("/tf", "tf", "/tf_static", "tf_static"):
                        target_dict = global_data["tf_static"] if "static" in topic else global_data["tf"]
                        for t in ros_msg.transforms:
                            target_dict["time_ns"].append(ts_ns)
                            target_dict["frame_id"].append(t.header.frame_id)
                            target_dict["child_frame_id"].append(t.child_frame_id)
                            target_dict["trans_x"].append(t.transform.translation.x)
                            target_dict["trans_y"].append(t.transform.translation.y)
                            target_dict["trans_z"].append(t.transform.translation.z)
                            target_dict["rot_x"].append(t.transform.rotation.x)
                            target_dict["rot_y"].append(t.transform.rotation.y)
                            target_dict["rot_z"].append(t.transform.rotation.z)
                            target_dict["rot_w"].append(t.transform.rotation.w)
                            appended = True

                            # Detect ground-truth map/world/odom -> base_link transform if available
                            parent = t.header.frame_id.strip('/')
                            child = t.child_frame_id.strip('/')
                            parent_lower = parent.lower()
                            child_lower = child.lower()
                            
                            is_world_frame = (
                                parent_lower in ("map", "world", "odom") or
                                parent_lower.endswith("/map") or
                                parent_lower.endswith("/world") or
                                parent_lower.endswith("/odom")
                            )
                            is_base_frame = (
                                child_lower.endswith("base_link") or
                                child_lower.endswith("base_footprint") or
                                "base_link" in child_lower or
                                "base_footprint" in child_lower
                            )
                            
                            # Filter by env_prefix if detected
                            if env_prefix and env_prefix not in child_lower:
                                continue
                                
                            if is_world_frame and is_base_frame:
                                child_parts = child.split('/')
                                robot_name = child_parts[-2] if len(child_parts) >= 2 else "unknown"
                                                               
                                yaw_val = self._quaternion_to_yaw(
                                    t.transform.rotation.x,
                                    t.transform.rotation.y,
                                    t.transform.rotation.z,
                                    t.transform.rotation.w
                                )
                                target = robot_data[robot_name]["tf_gt"]
                                target["time_ns"].append(ts_ns)
                                target["pos_x_gt"].append(t.transform.translation.x)
                                target["pos_y_gt"].append(t.transform.translation.y)
                                target["yaw_gt"].append(yaw_val)
                                appended = True

                    if appended:
                        accumulated_count += 1
                        if accumulated_count >= 10000:
                            flush_buffers()
                            accumulated_count = 0
        finally:
            flush_buffers()
            for writer in writers.values():
                writer.close()

        # Reconstruct dict[str, TopicBundle]
        bundles = {}
        
        def load_parquet(path):
            if path.exists():
                lf = pl.scan_parquet(path)
                if "time_ns" in lf.collect_schema().names():
                    lf = lf.sort("time_ns")
                return lf
            return None

        # Load global data
        global_bundle = TopicBundle()
        for t_name in global_data.keys():
            lf = load_parquet(topics_dir / f"{t_name}.parquet")
            if lf is not None:
                setattr(global_bundle, t_name, lf)

        # Build each robot's bundle
        robot_dirs = [d for d in topics_dir.iterdir() if d.is_dir()]
        
        # If no explicit robot dirs but we have data, maybe it was named "unknown"
        for robot_dir in robot_dirs:
            robot_name = robot_dir.name
            rb = TopicBundle()
            
            # Copy global references
            rb.peds = global_bundle.peds
            rb.episode_record = global_bundle.episode_record
            rb.tf = global_bundle.tf
            rb.tf_static = global_bundle.tf_static
            
            for t_name in new_robot_data().keys():
                lf = load_parquet(robot_dir / f"{t_name}.parquet")
                if lf is not None:
                    setattr(rb, t_name, lf)
                    
            bundles[robot_name] = rb
            
        return bundles
