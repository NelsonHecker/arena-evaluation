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

    def _read_legacy_csv(self) -> TopicBundle:
        """Fallback to reading legacy CSVs directly into Polars."""
        bundle = TopicBundle()
        
        # Odom
        odom_path = self.data_path / "odom.csv"
        if odom_path.exists():
            df = pl.read_csv(odom_path)
            # Legacy CSV has pos_x, pos_y, pos_z, orientation_x/y/z/w, linear_x/y/z, angular_x/y/z
            if "orientation_w" in df.columns:
                # Need to compute yaw
                yaws = [
                    self._quaternion_to_yaw(
                        row["orientation_x"], row["orientation_y"], row["orientation_z"], row["orientation_w"]
                    )
                    for row in df.iter_rows(named=True)
                ]
                df = df.with_columns(pl.Series("yaw", yaws))
            bundle.odom = df
            
        # Scan
        scan_path = self.data_path / "scan.csv"
        if scan_path.exists():
            bundle.scan = pl.read_csv(scan_path)
            
        # Episode record / goal (legacy didn't have episode_record as csv typically, we'll try to find it)
        # Note: True legacy support requires parsing custom formats, simplified here
        
        return bundle

    def read(self) -> TopicBundle:
        """
        Reads the data source and returns raw DataFrames/LazyFrames for each topic.
        """
        if self.data_path.is_dir():
            mcap_files = list(self.data_path.glob("*.mcap"))
            if mcap_files:
                actual_path = mcap_files[0]
            else:
                # Legacy CSV mode
                return self._read_legacy_csv()
        else:
            actual_path = self.data_path
            
        if not actual_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {actual_path}")

        # Resolve topics directory
        path = self.data_path.resolve()
        if path.is_dir():
            run_dir = path
        elif path.parent.name == "recording":
            run_dir = path.parent.parent
        else:
            run_dir = path.parent
        topics_dir = run_dir / "topics"

        # Data collection buffers
        data = {
            "odom": defaultdict(list),
            "scan": defaultdict(list),
            "cmd_vel": defaultdict(list),
            "joint_states": defaultdict(list),
            "peds": defaultdict(list),
            "episode_record": defaultdict(list),
            "collision_events": defaultdict(list),
            "collision_monitor_state": defaultdict(list),
            "plan": defaultdict(list),
            "initialpose": defaultdict(list),
            "tf": defaultdict(list),
            "tf_static": defaultdict(list),
            "tf_gt": defaultdict(list),
        }

        env_prefix = None

        from mcap.reader import NonSeekingReader

        # Ensure topics directory exists
        topics_dir.mkdir(parents=True, exist_ok=True)

        writers = {}
        accumulated_count = 0

        def flush_buffers():
            for topic_name, topic_data in data.items():
                if not topic_data or len(topic_data.get("time_ns", [])) == 0:
                    continue
                
                # Convert the topic buffer dict of lists to a pyarrow.RecordBatch
                batch = pa.RecordBatch.from_pydict(dict(topic_data))
                
                # Get or create the ParquetWriter
                if topic_name not in writers:
                    final_path = topics_dir / f"{topic_name}.parquet"
                    writers[topic_name] = pq.ParquetWriter(
                        final_path,
                        schema=batch.schema,
                        compression="zstd"
                    )
                
                # Write the RecordBatch to the Parquet file
                writers[topic_name].write_batch(batch)
                
                # Clear the topic buffer
                topic_data.clear()

        try:
            with open(actual_path, "rb") as f:
                reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])
                
                msg_count = 0
                def safe_iter(it):
                    nonlocal msg_count
                    try:
                        for item in it:
                            msg_count += 1
                            yield item
                    except Exception as e:
                        import logging
                        logging.getLogger("mcap_reader").warning(
                            f"MCAP reading hit a record error: {e}. Salvaged {msg_count} messages."
                        )

                for schema, channel, message, ros_msg in safe_iter(reader.iter_decoded_messages(log_time_order=False)):
                    topic = channel.topic
                    
                    # Dynamically detect the environment prefix (e.g. "env_0") to filter global /tf transforms
                    if env_prefix is None and ("env_" in topic or "env" in topic):
                        import re
                        match = re.search(r'env_(\d+)', topic)
                        if match:
                            env_prefix = f"env_{match.group(1)}"
                    
                    # Filter other namespaced topics to ensure we only read data for the active environment
                    if env_prefix and topic not in ("/tf", "tf", "/tf_static", "tf_static"):
                        if env_prefix not in topic:
                            continue
                            
                    ts_ns = message.log_time
                    appended = False
                    
                    # Odom
                    if topic.endswith("/odom") and "velocity_controller" not in topic:
                        data["odom"]["time_ns"].append(ts_ns)
                        data["odom"]["pos_x"].append(ros_msg.pose.pose.position.x)
                        data["odom"]["pos_y"].append(ros_msg.pose.pose.position.y)
                        
                        yaw = self._quaternion_to_yaw(
                            ros_msg.pose.pose.orientation.x,
                            ros_msg.pose.pose.orientation.y,
                            ros_msg.pose.pose.orientation.z,
                            ros_msg.pose.pose.orientation.w
                        )
                        data["odom"]["yaw"].append(yaw)
                        data["odom"]["vel_linear"].append(ros_msg.twist.twist.linear.x)
                        data["odom"]["vel_angular"].append(ros_msg.twist.twist.angular.z)
                        appended = True
                    
                    # Scan
                    elif topic.endswith("/scan") or topic.endswith("/lidar"):
                        data["scan"]["time_ns"].append(ts_ns)
                        # Convert ranges array to list for parquet storage
                        data["scan"]["scan_ranges"].append(list(ros_msg.ranges))
                        data["scan"]["scan_min"].append(ros_msg.range_min)
                        appended = True
                    
                    # Cmd_vel
                    elif topic.endswith("/cmd_vel"):
                        data["cmd_vel"]["time_ns"].append(ts_ns)
                        data["cmd_vel"]["linear_x"].append(ros_msg.linear.x)
                        data["cmd_vel"]["linear_y"].append(ros_msg.linear.y)
                        data["cmd_vel"]["linear_z"].append(ros_msg.linear.z)
                        data["cmd_vel"]["angular_x"].append(ros_msg.angular.x)
                        data["cmd_vel"]["angular_y"].append(ros_msg.angular.y)
                        data["cmd_vel"]["angular_z"].append(ros_msg.angular.z)
                        # Keep legacy columns
                        data["cmd_vel"]["cmd_linear"].append(ros_msg.linear.x)
                        data["cmd_vel"]["cmd_angular"].append(ros_msg.angular.z)
                        appended = True
                        
                    # Joint states
                    elif topic.endswith("/joint_states"):
                        data["joint_states"]["time_ns"].append(ts_ns)
                        data["joint_states"]["name"].append(list(ros_msg.name))
                        data["joint_states"]["position"].append(list(ros_msg.position))
                        data["joint_states"]["velocity"].append(list(ros_msg.velocity))
                        data["joint_states"]["effort"].append(list(ros_msg.effort))
                        
                        vels = list(ros_msg.velocity)
                        # Keep legacy columns
                        data["joint_states"]["joint_vel_left"].append(vels[0] if len(vels) > 0 else 0.0)
                        data["joint_states"]["joint_vel_right"].append(vels[1] if len(vels) > 1 else 0.0)
                        appended = True
                        
                    elif topic.endswith("/plan") or topic.endswith("/global_plan"):
                        data["plan"]["time_ns"].append(ts_ns)
                        poses_x = []
                        poses_y = []
                        poses_yaw = []
                        for pose_stamped in ros_msg.poses:
                            poses_x.append(pose_stamped.pose.position.x)
                            poses_y.append(pose_stamped.pose.position.y)
                            yaw = self._quaternion_to_yaw(
                                pose_stamped.pose.orientation.x,
                                pose_stamped.pose.orientation.y,
                                pose_stamped.pose.orientation.z,
                                pose_stamped.pose.orientation.w
                            )
                            poses_yaw.append(yaw)
                        data["plan"]["poses_x"].append(poses_x)
                        data["plan"]["poses_y"].append(poses_y)
                        data["plan"]["poses_yaw"].append(poses_yaw)
                        appended = True
                        
                    # Pedestrians (arena_people_msgs/Pedestrians)
                    elif topic.endswith("/arena_peds"):
                        data["peds"]["time_ns"].append(ts_ns)
                        
                        pos_list = []
                        head_list = []
                        for ped in ros_msg.pedestrians:
                            pos_list.extend([ped.pose.position.x, ped.pose.position.y])
                            # Get yaw from quaternion for heading
                            yaw = self._quaternion_to_yaw(
                                ped.pose.orientation.x,
                                ped.pose.orientation.y,
                                ped.pose.orientation.z,
                                ped.pose.orientation.w
                            )
                            head_list.append(yaw)
                            
                        data["peds"]["peds_positions"].append(pos_list)
                        data["peds"]["peds_headings"].append(head_list)
                        data["peds"]["num_pedestrians"].append(len(ros_msg.pedestrians))
                        appended = True
                        
                    # Episode Record
                    elif topic.endswith("/state/episode"):
                        data["episode_record"]["time_ns"].append(ts_ns)
                        data["episode_record"]["episode_id"].append(ros_msg.episode_id)
                        
                        # Convert robots_params array to dict to extract start/goal
                        robots_params_list = getattr(ros_msg, 'robots_params', [])
                        try:
                            import yaml
                            flat_params = {}
                            for p in robots_params_list:
                                flat_params[p.name] = MCAPReader._param_value_to_py(p.value)
                            unflattened = MCAPReader._unflatten_dict(flat_params)
                            robots_params_str = yaml.dump(unflattened)
                        except Exception as e:
                            robots_params_str = ""
                        data["episode_record"]["robots_params"].append(robots_params_str)
                        appended = True
                        
                    # Collision Events
                    elif topic.endswith("/collision_events"):
                        data["collision_events"]["time_ns"].append(ts_ns)
                        data["collision_events"]["collision_event"].append(len(ros_msg.events))
                        appended = True
                        
                    # Collision Monitor State
                    elif topic.endswith("/collision_monitor_state"):
                        data["collision_monitor_state"]["time_ns"].append(ts_ns)
                        data["collision_monitor_state"]["action_type"].append(ros_msg.action_type)
                        data["collision_monitor_state"]["polygon_name"].append(ros_msg.polygon_name)
                        appended = True
                        
                    # Initial Pose
                    elif topic.endswith("/initialpose"):
                        data["initialpose"]["time_ns"].append(ts_ns)
                        data["initialpose"]["pos_x"].append(ros_msg.pose.pose.position.x)
                        data["initialpose"]["pos_y"].append(ros_msg.pose.pose.position.y)
                        
                        yaw = self._quaternion_to_yaw(
                            ros_msg.pose.pose.orientation.x,
                            ros_msg.pose.pose.orientation.y,
                            ros_msg.pose.pose.orientation.z,
                            ros_msg.pose.pose.orientation.w
                        )
                        data["initialpose"]["yaw"].append(yaw)
                        appended = True

                    # TF and TF Static
                    elif topic in ("/tf", "tf", "/tf_static", "tf_static"):
                        target_dict = data["tf_static"] if "static" in topic else data["tf"]
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
                            
                            # Filter by env_prefix if detected to avoid mixing up TF frames from other environments in shared /tf topic
                            if env_prefix and env_prefix not in child_lower:
                                continue
                                
                            if is_world_frame and is_base_frame:
                                yaw_val = self._quaternion_to_yaw(
                                    t.transform.rotation.x,
                                    t.transform.rotation.y,
                                    t.transform.rotation.z,
                                    t.transform.rotation.w
                                )
                                data["tf_gt"]["time_ns"].append(ts_ns)
                                data["tf_gt"]["pos_x_gt"].append(t.transform.translation.x)
                                data["tf_gt"]["pos_y_gt"].append(t.transform.translation.y)
                                data["tf_gt"]["yaw_gt"].append(yaw_val)
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

        # Reconstruct TopicBundle using pl.scan_parquet for the written files!
        bundle = TopicBundle()
        for topic_name in data.keys():
            final_path = topics_dir / f"{topic_name}.parquet"
            if final_path.exists():
                lf = pl.scan_parquet(final_path)
                lf = lf.sort("time_ns")
                setattr(bundle, topic_name, lf)
                
        # Save TF ground-truth poses directly on the bundle
        tf_gt_path = topics_dir / "tf_gt.parquet"
        if tf_gt_path.exists():
            count = pl.scan_parquet(tf_gt_path).collect().height
            print(f"  [MCAPReader] Found {count} TF ground-truth transforms. Storing as separate tf_gt channel...")
                
        return bundle
