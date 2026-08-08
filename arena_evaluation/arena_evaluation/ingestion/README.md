# ingestion — Layer 1: Live Data Recording

This package contains the ROS 2 node that records a live simulation run into structured MCAP files.

It is the **only** component that requires a running ROS 2 environment. Everything else in the pipeline (`processing/`, `presentation/`) operates entirely offline.

---

## Responsibilities

- Subscribe to all relevant ROS 2 topics during a simulation run.
- Write **one MCAP file per episode** into `episodes/episode_XXX/` (the flat per-episode structure).
- Open/close episode writers through the **`start_episode` service** — driven authoritatively by the
  benchmark runner, so a missed or late `EpisodeRecord` topic message can never corrupt the recording
  lifecycle.
- Enrich each episode's `episode_XXX.yaml` metadata (planner, local/inter planner, suite, contest,
  stage, map, robot, task params, outcome) from recorder parameters and the `EpisodeRecord` topic.
- Flush and close the MCAP cleanly on SIGTERM / SIGINT.

---

## Files

| File | Purpose |
|---|---|
| `recorder.py` | `DataRecorderNode` — the main ROS 2 recording node |
| `metadata.py` | `IngestionMetadata` — builds the full per-episode `RunMetadata` |
| `topics.py` | Topic name constants and canonical type-string mapping |

---

## recorder.py — DataRecorderNode

### Episode lifecycle (service-driven)

The recorder does **not** start/stop on `EpisodeRecord` topic messages. The benchmark runner controls
the lifecycle through the `RecordEpisode` service (`arena_evaluation_msgs/srv/RecordEpisode.srv`):

```
COMMAND_START  episode_id → opens episode_XXX.mcap + writes episode_XXX.yaml
COMMAND_STOP   episode_id → closes the writer, records outcome_state/outcome_info in the yaml
```

The runner sends START when it observes an episode beginning (QUEUED/RUNNING record) and STOP when the
episode terminates — and it **awaits** the STOP ack before killing the recorder, so the MCAP is always
flushed per episode.

The `EpisodeRecord` topic subscription remains, but only for:
- **metadata enrichment** (`map`, `robot_model`, `tm_*` params, `obstacles_params`/`robots_params`), and
- a **start fallback for manual runs** without a runner (a first-seen QUEUED/RUNNING record opens a writer).

Terminal records are logged but never drive lifecycle.

### Output structure

Each episode is recorded into its own directory (per-episode MCAPs make the processing layer trivial —
one MCAP == one episode, no in-bag splitting needed):

```
data/benchmarks/<run_id>/episodes/
├── .episode_counter
├── episode_000/
│   ├── episode_000.mcap      ← flattened from rosbag2's subdirectory
│   └── episode_000.yaml      ← full context: identity + task params + outcome
├── episode_001/ ...
└── recorder.log
```

### Parameters

The runner passes the full episode context as ROS parameters, which are written into every
`episode_XXX.yaml`:

| Parameter | Meaning |
|---|---|
| `record_data_dir` | The `episodes/` root |
| `benchmark_id` | Run id (also the benchmark directory name) |
| `contestant` | Contestant/planner name (written as `planner`) |
| `stage`, `map`, `world` | Stage identity |
| `suite_name`, `contest_name` | Benchmark context |
| `local_planner`, `inter_planner` | Explicit planner identity from the contest config |
| `robot` | Robot model (seeds `robot_model` before topic discovery) |
| `episodes_requested` | Episode count for the step |
| `is_reference`, `reference_type` | Reference-step flags |
| `episode_id_offset` | Continued episode numbering across steps (`.episode_counter`) |

### Recorded Topics

All topics use **simulation time** from `/clock` as timestamps. Messages are buffered until the first
`/clock` tick to prevent timestamp monotonicity violations.

| Topic | Message Type | Throttle |
|---|---|---|
| `/{ns}/cmd_vel` | `geometry_msgs/Twist` | rate-limited |
| `/{ns}/joint_states` | `sensor_msgs/JointState` | rate-limited |
| `/{ns}/plan` | `nav_msgs/Path` | unthrottled |
| `/{ns}/collision_events` | `arena_robots_msgs/CollisionEvents` | unthrottled |
| `/{ns}/collision_monitor_state` | `nav2_msgs/CollisionMonitorState` | unthrottled |
| `/{ns}/power_publisher/power` | `arena_robots_msgs/Power` | rate-limited |
| `/{ns}/power_publisher/energy` | `arena_robots_msgs/Energy` | rate-limited |
| `/{ns}/acoustics` | `arena_robots_msgs/Acoustics` | rate-limited |
| `/{ns}/characterization_phase` | `std_msgs/String` | unthrottled (open-loop sweep markers) |
| `/{robot_ns}/odom` | `nav_msgs/Odometry` | rate-limited |
| `/{parent_ns}/arena_peds` | `arena_people_msgs/Pedestrians` | rate-limited |
| `/{parent_ns}/agent_states` | `arena_humansim_msgs/AgentStates` | rate-limited |
| `/{parent_ns}/state/episode` | `task_generator_msgs/EpisodeRecord` | unthrottled |
| `/{parent_ns}/state/robots` | `task_generator_msgs/RobotFleet` | latched |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | rate-limited / latched |
| `/clock` | `rosgraph_msgs/Clock` | (time tracking only) |

A dynamic discovery timer detects non-standard odom topic names. Robot topics are subscribed when the
`RobotFleet` message announces a robot (namespace from the fleet).

### Shutdown Behaviour

The node installs `SIGTERM`/`SIGINT` handlers that trigger `finalize()`:

1. Sets `is_shutting_down = True` so no new writes occur.
2. Closes the current writer (flushing rosbag2 buffers, flattening the MCAP into `episode_XXX.mcap`).
3. Finalizes `episode_XXX.yaml` with `recording_ended_at`, `recorded_topics`, `pedsim_available`.

> If the process is killed with `SIGKILL`, `finalize()` is not called. The MCAP may be incomplete or
> unindexed; the final metadata fields will be missing.

---

## metadata.py — IngestionMetadata

Builds the `RunMetadata` object written to each `episode_XXX.yaml` — the **single self-contained
source of context** for later processing/reporting:

```python
metadata = IngestionMetadata.create_episode_metadata(
    benchmark_id="my_benchmark",
    planner="dwb",
    stage="stage_1",
    map_name="map_empty",
    episode_id=0,
    robot_model="jackal",
    suite_name="basic",
    contest_name="basic",
    local_planner="dwb",          # explicit, from the contestant's mobile config
    inter_planner="navigate_w_replanning_time",
    episodes_requested=5,
    task_generator_episode_id=1,  # sim-side id for correlating with progress.csv
)
```

When `local_planner`/`inter_planner` are not provided (manual runs), they fall back to the
`<prefix>-<local>-<inter>` name convention via `split_planner_name` (`storage/planner_names.py`).

---

## topics.py — Topic Definitions

Provides the canonical list of topic names and their expected ROS message type strings (used when
registering topics with rosbag2).

```python
from arena_evaluation.ingestion.topics import get_topics

topics = get_topics(namespace="arena/env_0/task_generator_node/jackal")
```

---

## Throttle Configuration

Edit `config/data_recorder_config.yaml` to control sampling rates:

```yaml
record_frequencies:
  default: 20.0   # ms — fallback
  lidar:   100.0  # ms — 10 Hz
  odom:     20.0  # ms — 50 Hz
```

Keys are matched as **substrings** of the full topic name.
