#!/usr/bin/env python3

import os
import glob
import math
import yaml
import argparse
import polars as pl
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.LATEST)
deserialize_cdr = typestore.deserialize_cdr

def quaternion_to_yaw(x, y, z, w):
    """Calculate yaw from quaternion to match old euler_from_quaternion output."""
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def process_episode_bag(episode_dir):
    """Reads an MCAP bag and outputs legacy CSVs using Polars."""
    bag_path = os.path.join(episode_dir, "recording")
    metadata_path = os.path.join(episode_dir, "metadata.yaml")

    # If the MCAP directory doesn't exist, skip it (might already be converted/deleted)
    if not os.path.exists(bag_path):
        return

    data_dict = {
        "scan": {"time": [], "data": []},
        "odom": {"time": [], "data": []},
        "cmd_vel": {"time": [], "data": []}
    }
    
    episode_id = 0
    start_pos = [0.0, 0.0, 0.0]
    goal_pos = [0.0, 0.0, 0.0]
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = yaml.safe_load(f)
            episode_id = meta.get("episode_id", 0)
            robots_params = meta.get("robots_params", {})
            for robot, params in robots_params.items():
                if isinstance(params, dict):
                    start_pos = [float(x) for x in params.get("start", start_pos)]
                    goal_pos = [float(x) for x in params.get("goal", goal_pos)]

    first_time = None

    # Deserialize MCAP offline
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if first_time is None:
                first_time = timestamp

            topic = connection.topic
            
            if "scan" in topic or "lidar" in topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                ranges = [
                    float(round(msg.range_max, 3)) if math.isnan(v) or math.isinf(v) else float(round(v, 3)) 
                    for v in msg.ranges
                ]
                data_dict["scan"]["time"].append(timestamp)
                data_dict["scan"]["data"].append(str(ranges)) 

            elif "odom" in topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                pose = msg.pose.pose
                twist = msg.twist.twist
                
                yaw = quaternion_to_yaw(
                    pose.orientation.x, pose.orientation.y, 
                    pose.orientation.z, pose.orientation.w
                )
                
                odom_data = {
                    "position": [float(round(pose.position.x, 3)), float(round(pose.position.y, 3)), float(round(yaw, 3))],
                    "velocity": [float(round(twist.linear.x, 3)), float(round(twist.linear.y, 3)), float(round(twist.angular.z, 3))]
                }
                data_dict["odom"]["time"].append(timestamp)
                data_dict["odom"]["data"].append(str(odom_data))

            elif "cmd_vel" in topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                action_data = [float(round(msg.linear.x, 3)), float(round(msg.linear.y, 3)), float(round(msg.angular.z, 3))]
                data_dict["cmd_vel"]["time"].append(timestamp)
                data_dict["cmd_vel"]["data"].append(str(action_data))

    # Write CSVs
    max_len = 0
    for topic_key, columns in data_dict.items():
        if not columns["time"]:
            continue
        max_len = max(max_len, len(columns["time"]))
        df = pl.DataFrame(columns)
        out_csv = os.path.join(episode_dir, f"{topic_key}.csv")
        df.write_csv(out_csv, include_header=True)

    if first_time is not None and max_len > 0:
        pl.DataFrame({
            "time": [first_time] * max_len,
            "data": [episode_id] * max_len
        }).write_csv(
            os.path.join(episode_dir, "episode.csv"), include_header=True
        )
        pl.DataFrame({
            "episode": [episode_id] * max_len,
            "start": [str(start_pos)] * max_len,
            "goal": [str(goal_pos)] * max_len
        }).write_csv(
            os.path.join(episode_dir, "start_goal.csv"), include_header=True
        )

def convert_directory(base_dir):
    """Finds all episode folders in the base directory and converts MCAP to legacy CSV."""
    episode_dirs = glob.glob(os.path.join(base_dir, "episode_*"))
    
    # Also support nested structures if data is inside contestant/stage/robot
    if not episode_dirs:
        # Recursively search for episode directories
        for root, dirs, files in os.walk(base_dir):
            for d in dirs:
                if d.startswith("episode_"):
                    episode_dirs.append(os.path.join(root, d))
                    
    if not episode_dirs:
        print(f"No MCAP episode directories found in {base_dir} to convert. Proceeding with existing CSVs if present.")
        return

    # Sort the episode directories to ensure they are merged in order
    episode_dirs = sorted(episode_dirs, key=lambda x: int(os.path.basename(x).split("_")[-1]))

    print(f"Found {len(episode_dirs)} episodes. Converting MCAP to CSV...")
    for ep_dir in episode_dirs:
        if os.path.isdir(ep_dir):
            process_episode_bag(ep_dir)
            
    print("MCAP to legacy CSV conversion complete. Merging episode CSVs...")

    # Merge all individual episode CSVs into the base directory
    csv_files = ["odom.csv", "scan.csv", "episode.csv", "start_goal.csv", "cmd_vel.csv"]
    for csv_file in csv_files:
        merged_lines = []
        is_first_file = True
        for ep_dir in episode_dirs:
            ep_csv_path = os.path.join(ep_dir, csv_file)
            if os.path.exists(ep_csv_path):
                with open(ep_csv_path, "r") as f:
                    lines = f.readlines()
                    if lines:
                        if is_first_file:
                            merged_lines.extend(lines)
                            is_first_file = False
                        else:
                            # Skip the header row (first line)
                            merged_lines.extend(lines[1:])
        
        if merged_lines:
            merged_csv_path = os.path.join(base_dir, csv_file)
            with open(merged_csv_path, "w") as f:
                f.writelines(merged_lines)
            print(f"Written merged {csv_file} to {merged_csv_path}")

    # Update params.yaml with model and environment_map from metadata
    map_file = "demo"
    model = "jackal"
    first_ep_dir = episode_dirs[0] if episode_dirs else None
    if first_ep_dir:
        metadata_path = os.path.join(first_ep_dir, "metadata.yaml")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    meta = yaml.safe_load(f)
                    map_file = meta.get("environment_map", map_file)
                    if meta.get("robot_model"):
                        model = meta.get("robot_model")[0]
            except Exception:
                pass

    params_path = os.path.join(base_dir, "params.yaml")
    params_content = {}
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                params_content = yaml.safe_load(f) or {}
        except Exception:
            pass
    
    params_content["map_file"] = params_content.get("map_file") or map_file
    params_content["model"] = params_content.get("model") or model
    if "namespace" not in params_content:
        params_content["namespace"] = f"arena/env_0/task_generator_node/{model}"
    
    try:
        with open(params_path, 'w') as f:
            yaml.dump(params_content, f)
        print(f"Updated {params_path} with map_file={map_file}, model={model}")
    except Exception as e:
        print(f"Failed to write params.yaml: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert new Arena MCAP evaluation bags to legacy CSV formats.")
    parser.add_argument("--dir", "-d", required=True, help="Base directory containing episode_X folders")
    args = parser.parse_args()
    convert_directory(args.dir)