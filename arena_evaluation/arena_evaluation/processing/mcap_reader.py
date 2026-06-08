from __future__ import annotations

import pathlib
import math
import typing
import polars as pl
from collections import defaultdict
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

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
        Reads the data source and returns raw DataFrames for each topic.
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

        # Data collection buffers
        data = {
            "odom": defaultdict(list),
            "scan": defaultdict(list),
            "cmd_vel": defaultdict(list),
            "joint_states": defaultdict(list),
            "peds": defaultdict(list),
            "episode_record": defaultdict(list),
            "collision_events": defaultdict(list),
            "plan": defaultdict(list),
        }

        with open(actual_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                topic = channel.topic
                ts_ns = message.log_time
                
                # Odom
                if topic.endswith("/odom"):
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
                
                # Scan
                elif topic.endswith("/scan") or topic.endswith("/lidar"):
                    data["scan"]["time_ns"].append(ts_ns)
                    # Convert ranges array to list for parquet storage
                    data["scan"]["scan_ranges"].append(list(ros_msg.ranges))
                    data["scan"]["scan_min"].append(ros_msg.range_min)
                
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
                    
                # Plan
                elif topic.endswith("/plan") or topic.endswith("/global_plan"):
                    data["plan"]["time_ns"].append(ts_ns)
                    poses_x = []
                    poses_y = []
                    for pose_stamped in ros_msg.poses:
                        poses_x.append(pose_stamped.pose.position.x)
                        poses_y.append(pose_stamped.pose.position.y)
                    data["plan"]["poses_x"].append(poses_x)
                    data["plan"]["poses_y"].append(poses_y)
                    
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
                    
                # Collision Events
                elif topic.endswith("/collision_events"):
                    data["collision_events"]["time_ns"].append(ts_ns)
                    data["collision_events"]["collision_event"].append(str(ros_msg))

        # Convert dicts of lists to Polars DataFrames
        bundle = TopicBundle()
        
        for key, dict_data in data.items():
            if len(dict_data) > 0 and len(dict_data.get("time_ns", [])) > 0:
                df = pl.DataFrame(dict_data)
                # Sort by time to ensure join_asof works correctly later
                df = df.sort("time_ns")
                setattr(bundle, key, df)
                
        return bundle
