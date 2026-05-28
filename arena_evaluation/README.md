# arena_evaluation

> **ROS 2 package** — Record, process, and visualise navigational evaluation metrics for Arena-Rosnav planners.

The package provides a complete, end-to-end evaluation pipeline that turns live simulation data into structured metric reports. It is built around a layered architecture that separates recording, processing, and presentation concerns so that each layer can be used, replaced, or extended independently.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Arena Simulation                           │
│   Gazebo  ──  task_generator  ──  nav2  ──  planner               │
└────────────────────────┬────────────────────────────────────────────┘
                         │ ROS 2 topics
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Ingestion  (ingestion/)                                  │
│  DataRecorderNode subscribes to all relevant topics and writes a    │
│  single continuous MCAP file per benchmark step (planner × stage). │
│  EpisodeRecord messages are embedded in the MCAP as boundaries.    │
│  Output: run_dir/recording/recording_0.mcap + metadata.yaml        │
└────────────────────────┬────────────────────────────────────────────┘
                         │ MCAP file (one per step)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Processing  (processing/)                                │
│  MCAPReader → TopicAligner → EpisodeSplitter → MetricRegistry      │
│  Reads the MCAP offline (no live ROS required), aligns multi-rate  │
│  topics, splits into per-episode bundles, computes all metrics.    │
│  Output: metrics.parquet + combined_metrics.parquet                │
└────────────────────────┬────────────────────────────────────────────┘
                         │ Parquet files
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5 — Presentation  (presentation/)                            │
│  ReportBuilder reads the combined parquet + viz_manifest.yaml and  │
│  generates an interactive HTML report and static PNG plots.        │
│  Output: report.html + plots/*.png                                 │
└─────────────────────────────────────────────────────────────────────┘
```

> **Note:** Layers are numbered 1, 3, 5 to align with the SRD v2.0 which reserves Layer 2 (Storage/Folder Management) as a shared infrastructure module and Layer 4 (LLM Orchestration) for Phase 2.

---

## Directory Structure

```
arena_evaluation/
├── arena_evaluation/          ← Python package root
│   ├── ingestion/             ← Layer 1: Live recording (ROS node)
│   │   ├── recorder.py        ← DataRecorderNode
│   │   ├── metadata.py        ← Metadata helpers
│   │   └── topics.py          ← Topic definitions and type registry
│   ├── storage/               ← Layer 2: Shared schemas and path management
│   │   ├── schemas.py         ← Pydantic models (RunMetadata, TopicBundle, …)
│   │   ├── folder_manager.py  ← Path resolution and run discovery
│   │   ├── manifest.py        ← MetadataWriter (YAML read/write)
│   │   └── exceptions.py      ← Domain-specific exceptions
│   ├── processing/            ← Layer 3: Offline metric computation
│   │   ├── mcap_reader.py     ← Reads MCAP → TopicBundle
│   │   ├── topic_aligner.py   ← Aligns multi-rate topics via join_asof
│   │   ├── episode_splitter.py← Splits continuous stream into episodes
│   │   ├── parquet_store.py   ← Reads/writes metrics.parquet
│   │   ├── pipeline.py        ← ProcessingPipeline orchestrator
│   │   └── metrics/           ← Pluggable metric calculators
│   │       ├── base.py        ← BaseMetricCalculator ABC
│   │       ├── registry.py    ← Auto-discovery + topological execution
│   │       ├── performance/   ← Path, motion, time, collision, efficiency
│   │       ├── social/        ← Proxemics, gaze
│   │       ├── naturalness/   ← (Phase 2)
│   │       └── ecological/    ← (Phase 2)
│   ├── presentation/          ← Layer 5: Report and plot generation
│   │   ├── report_builder.py  ← Generates report.html
│   │   ├── viz_manifest.py    ← Loads/validates viz_manifest.yaml
│   │   ├── plotly_renderer.py ← Interactive HTML chart dispatcher
│   │   ├── seaborn_renderer.py← Static PNG chart dispatcher
│   │   └── plot_types/        ← violin, box, bar, trajectory, radar
│   ├── benchmark/             ← Benchmark runner and CLI
│   │   ├── runner.py          ← BenchmarkRunner (orchestrates simulation)
│   │   └── cli.py             ← Benchmark management CLI (argparse)
│   └── cli.py                 ← Evaluation pipeline CLI (argparse)
├── config/
│   └── data_recorder_config.yaml  ← Topic throttle frequencies
├── configs/benchmark/
│   ├── suites/                ← Benchmark suite YAML definitions
│   └── contests/              ← Contest (planner set) YAML definitions
└── tests/
    ├── unit/                  ← Pure Python unit tests (no ROS required)
    └── integration/           ← Full-pipeline tests with fixture data
```

---

## Data Flow

### Recording Phase (live simulation)

```
arena launch sim:=gazebo task_mode:=random record_data_dir:=data \
    world:=map_empty robot:=jackal
```

The benchmark runner (or manual launch) spawns the `DataRecorderNode` with the `record_data_dir` ROS parameter. The node:

1. Resolves the output directory and creates it.
2. Writes an initial `metadata.yaml` with known fields (robot, world, git SHA, …).
3. Subscribes to all configured ROS topics (cmd_vel, lidar, odom, joint_states, plan, goal_pose, collision_events, arena_peds, EpisodeRecord, RobotFleet).
4. Opens a single rosbag2 MCAP writer and records continuously for the entire step (all episodes).
5. On each `EpisodeRecord` message, updates `metadata.yaml` with episode metadata.
6. On clean shutdown (SIGTERM / SIGINT), writes the final `metadata.yaml` fields (episodes_recorded, recorded_topics, recording_ended_at) and flushes the MCAP.

**Output directory structure:**
```
data/recordings/<timestamp>/
├── metadata.yaml          ← Run-level metadata (start → end)
├── params.yaml            ← ROS parameter snapshot
└── recording/
    └── recording_0.mcap   ← All topics, all episodes, continuous
```

For benchmark runs the structure is:
```
data/<benchmark_id>/recordings/<planner>/<stage>/
├── metadata.yaml
├── params.yaml
└── recording/
    └── recording_0.mcap
```

### Processing Phase (offline)

```bash
arena evaluation process --benchmark-dir /opt/arena_ws/data/<benchmark_id>
```

Reads each `recording_0.mcap`, decodes all messages, temporally aligns them, splits by `EpisodeRecord` boundaries, and runs all metric calculators. Writes per-run and combined Parquet files.

### Presentation Phase (offline)

```bash
arena evaluation report --benchmark-dir /opt/arena_ws/data/<benchmark_id>
# Or both at once:
arena evaluation run --benchmark-dir /opt/arena_ws/data/<benchmark_id>
```

Reads `combined_metrics.parquet` and `viz_manifest.yaml`, generates `report.html` and `plots/*.png`.

---

## CLI Reference

The `arena evaluation` command is registered as part of the Arena feature system.

```
usage: arena evaluation <command> [--run-dir DIR | --benchmark-dir DIR]

Commands:
  process   Layer 3: Read MCAP(s) and compute metrics.parquet (no plots)
  run       Full pipeline: process → report + plots
  report    Layer 5: Generate report.html from existing metrics.parquet
  plot      Layer 5: Generate static PNG plots only (no HTML)
```

`process` and `run` accept **either** `--run-dir` (single recording) or `--benchmark-dir` (full benchmark). `report` and `plot` only accept `--benchmark-dir`.

### Processing a Single Run (ad-hoc recording)

```bash
# After running: arena launch ... record_data_dir:=data
arena evaluation process --run-dir /opt/arena_ws/data/recordings/20260528-215316
# Output: /opt/arena_ws/data/recordings/20260528-215316/metrics.parquet
```

### Processing a Full Benchmark

```bash
arena evaluation process --benchmark-dir /opt/arena_ws/data/my_benchmark
# Output: metrics.parquet per run + combined_metrics.parquet at root
```

### Full Pipeline (process + report)

```bash
# Single run (metrics only — no HTML report for single runs):
arena evaluation run --run-dir /opt/arena_ws/data/recordings/20260528-215316

# Full benchmark (metrics + HTML report + PNGs):
arena evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark
```

### Regenerate Report Only

```bash
arena evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark
# Reads combined_metrics.parquet + viz_manifest.yaml
# Writes: report.html + plots/*.png
```

The benchmark management CLI is accessed via:

```bash
arena evaluation list
arena evaluation status <run_id>
arena evaluation tail <run_id>
```

---

## Running Tests

All unit tests are pure Python — no running ROS environment required.

```bash
# Inside the Docker container
cd /opt/arena_ws/src/Arena/arena_evaluation/arena_evaluation
pytest tests/unit -v

# With integration tests (requires fixture data)
pytest tests/ -v
```

---

## Configuration

### Topic throttle frequencies (`config/data_recorder_config.yaml`)

Controls how often high-frequency topics are sampled into the MCAP. Keys are matched as substrings of the full topic name.

```yaml
record_frequencies:
  default: 20.0   # ms — fallback for any unmatched topic
  lidar:   100.0  # ms — LaserScan (10 Hz)
  odom:     20.0  # ms — Odometry (50 Hz)
  cmd_vel:  20.0  # ms — Twist (50 Hz)
```

### Visualization manifest (`viz_manifest.yaml`)

Placed at the benchmark directory root. Defines which plots to generate and what data to use.

```yaml
plots:
  - id: path_length_violin
    type: violin
    title: "Path Length Distribution"
    data_key: path_length
    group_by: planner
  - id: success_rate_bar
    type: bar
    title: "Success Rate"
    data_key: success
    group_by: planner
```

---

## Dependencies

All Python dependencies are managed in `src/Arena/pyproject.toml`:

| Package | Used For |
|---|---|
| `polars` | Fast DataFrame operations for metric computation |
| `pyarrow` | Parquet read/write with embedded metadata |
| `pydantic` | Schema validation for RunMetadata, PlotSpec |
| `mcap` + `mcap-ros2-support` | Decoding MCAP files offline (no live ROS) |
| `plotly` | Interactive HTML charts |
| `seaborn` + `matplotlib` | Static PNG fallback charts |
| `PyYAML` | YAML config and metadata file I/O |

ROS 2 dependencies (declared in `package.xml`): `rclpy`, `rosbag2_py`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `task_generator_msgs`, `arena_evaluation_msgs`.

---

## Rebuilding After Changes

```bash
# Inside the container — incremental rebuild with symlink install
# (source edits immediately reflected without rebuilding again)
colcon build --packages-select arena_evaluation --symlink-install
source /opt/arena_ws/install/setup.bash
```

---

## Phase Roadmap

| Phase | Status | Scope |
|---|---|---|
| **Phase 1** | ✅ In Progress | Ingestion, processing core, performance + social metrics, HTML report |
| **Phase 2** | 🔲 Planned | LLM Orchestration Layer (Layer 4), naturalness metrics, danger metrics |
| **Phase 3** | 🔲 Planned | Ecological metrics, ADE/FDE, topological complexity, noise contours |
