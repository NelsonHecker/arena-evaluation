# ingestion — Layer 1: Live Data Recording

This package contains the ROS 2 node that records a live simulation run into a structured MCAP file.

It is the **only** component that requires a running ROS 2 environment. Everything else in the pipeline (`processing/`, `presentation/`) operates entirely offline.

---

## Responsibilities

- Subscribe to all relevant ROS 2 topics during a simulation run.
- Write a **single continuous MCAP file** for the entire step (one planner × one stage × N episodes).
- Embed `EpisodeRecord` messages directly into the MCAP stream so the processing layer can use them as episode boundary markers.
- Write and continuously update `metadata.yaml` with run-level context.
- Flush and close the MCAP cleanly on SIGTERM / SIGINT.

---

## Files

| File | Purpose |
|---|---|
| `recorder.py` | `DataRecorderNode` — the main ROS 2 recording node |
| `metadata.py` | `IngestionMetadata` helper — builds initial `RunMetadata` from environment |
| `topics.py` | Topic name constants and canonical type-string mapping |

---

## recorder.py — DataRecorderNode

### How It Is Launched

The node is registered as the `record` console script entry point in `setup.py` and is spawned automatically by the `task_generator` launch system when `record_data_dir` is passed.

```bash
# Manual launch (auto-timestamped output folder)
arena launch sim:=gazebo task_mode:=random record_data_dir:=data \
    world:=map_empty robot:=jackal

# The recorder is started internally with something equivalent to:
arena evaluation record \
    --ros-args -p record_data_dir:=/opt/arena_ws/data/recordings/20260528-210000
```

### Output Path Resolution

The `record_data_dir` parameter is resolved in priority order:

1. **ROS parameter** `record_data_dir` — set by the benchmark runner launch args.
2. **Command-line arg** `--dir` / `-d` — for manual invocation.
3. **Default** `auto:/` — generates a timestamped folder automatically.

If the resolved path ends in a bare root directory name (`data` or `recordings`), a timestamp subdirectory is automatically appended so recordings are never written directly into the root.

**Benchmark run structure:**
```
data/<benchmark_id>/recordings/<planner>/<stage>/
├── metadata.yaml
├── params.yaml
└── recording/
    └── recording_0.mcap
```

**Ad-hoc run structure:**
```
data/recordings/<YYYYMMDD-HHMMSS>/
├── metadata.yaml
├── params.yaml
└── recording/
    └── recording_0.mcap
```

### Recorded Topics

All topics use **simulation time** from `/clock` as timestamps. Messages are dropped until the first `/clock` tick to prevent timestamp monotonicity violations.

| Topic | Message Type | Throttle |
|---|---|---|
| `/{ns}/cmd_vel` | `geometry_msgs/Twist` | 20 ms |
| `/{ns}/lidar` | `sensor_msgs/LaserScan` | 100 ms |
| `/{ns}/{robot}_velocity_controller/odom` | `nav_msgs/Odometry` | 20 ms |
| `/{ns}/joint_states` | `sensor_msgs/JointState` | 20 ms |
| `/{ns}/plan` | `nav_msgs/Path` | unthrottled |
| `/{ns}/collision_events` | `arena_robots_msgs/CollisionEvents` | unthrottled |
| `/{parent_ns}/goal_pose` | `geometry_msgs/PoseStamped` | unthrottled |
| `/{parent_ns}/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | unthrottled |
| `/{parent_ns}/arena_peds` | `arena_people_msgs/Pedestrians` | 20 ms |
| `/{parent_ns}/state/episode` | `task_generator_msgs/EpisodeRecord` | unthrottled |
| `/{parent_ns}/state/robots` | `task_generator_msgs/RobotFleet` | latched |
| `/tf` | `tf2_msgs/TFMessage` | 20 ms |
| `/tf_static` | `tf2_msgs/TFMessage` | latched |
| `/clock` | `rosgraph_msgs/Clock` | (time tracking only) |

A dynamic discovery timer fires every 1 second to detect non-standard odom and scan topic names.

### Shutdown Behaviour

The node installs `SIGTERM` and `SIGINT` handlers that trigger `finalize()`:

1. Sets `is_shutting_down = True` so no new writes occur.
2. Writes the final `metadata.yaml` with `recording_ended_at`, `episodes_recorded`, `pedsim_available`, `recorded_topics`.
3. Calls `writer.close()` to flush rosbag2 buffers and write the MCAP index.

> If the process is killed with `SIGKILL`, `finalize()` is not called. The MCAP may be incomplete or unindexed. rosbag2 can still read it, but the final metadata fields will be missing.

---

## metadata.py — IngestionMetadata

Static helper that builds the initial `RunMetadata` object written to `metadata.yaml` at node startup.

```python
metadata = IngestionMetadata.create_initial_metadata(
    benchmark_id="my_benchmark",
    planner="dwa",
    stage="stage_1",
    episodes_requested=10,
    robot_model="jackal",
    suite_name="default",
    contest_name="dwa_vs_teb",
)
```

Fields populated at startup:
- `recording_started_at` — UTC ISO timestamp
- `arena_git_sha` / `arena_git_dirty` — from `git rev-parse HEAD` in the workspace
- `python_version` — from `sys.version`
- `ros_distro` — from `$ROS_DISTRO`
- `map` — from the world/map parameter
- `inter_planner` — parsed from the planner/contestant name
- `agent_name` — the agent/robot name

---

## topics.py — Topic Definitions

Provides the canonical list of topic names and their expected ROS message type strings (used when registering topics with rosbag2).

```python
from arena_evaluation.ingestion.topics import get_topics

topics = get_topics(namespace="arena/env_0/task_generator_node/jackal")
# Returns list of (topic_name, msg_type_string) tuples
```

---

## Throttle Configuration

Edit `config/data_recorder_config.yaml` to control sampling rates:

```yaml
record_frequencies:
  default: 20.0   # ms — fallback
  lidar:  100.0   # ms — 10 Hz
  odom:    20.0   # ms — 50 Hz
```

Keys are matched as **substrings** of the full topic name, so `"lidar"` matches both `…/lidar` and `…/gpu_lidar/scan`.
