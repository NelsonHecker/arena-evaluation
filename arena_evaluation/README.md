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
│  MCAPReader → TopicParquetStore → TopicAligner → EpisodeSplitter   │
│  → MetricRegistry → ParquetStore                                   │
│  Reads the MCAP offline (no live ROS required), aligns multi-rate  │
│  topics, splits into per-episode bundles, computes all metrics.    │
│  Includes MapRegistry for map overlay caching.                     │
│  Output: metrics.parquet + combined_metrics.parquet                │
└────────────────────────┬────────────────────────────────────────────┘
                         │ Parquet files
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5 — Presentation  (presentation/)                            │
│  ReportBuilder reads the combined parquet + viz_manifest and        │
│  generates an interactive HTML report (with time-slider trajectory │
│  plots), static PNG plots, and optional animated GIFs.             │
│  Uses a global accessibility color palette for all charts.         │
│  Output: report.html + plots/*.png (+ plots/*.gif)                 │
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
│   │   ├── schemas.py         ← Pydantic models (RunMetadata, PlotSpec, TopicBundle, …)
│   │   ├── folder_manager.py  ← Path resolution and run discovery
│   │   ├── manifest.py        ← MetadataWriter (YAML read/write)
│   │   └── exceptions.py      ← Domain-specific exceptions
│   ├── processing/            ← Layer 3: Offline metric computation
│   │   ├── mcap_reader.py     ← Reads MCAP → TopicBundle
│   │   ├── topic_aligner.py   ← Aligns multi-rate topics via join_asof
│   │   ├── episode_splitter.py← Splits continuous stream into episodes
│   │   ├── parquet_store.py   ← Reads/writes metrics.parquet
│   │   ├── pipeline.py        ← ProcessingPipeline orchestrator
│   │   ├── map_registry.py    ← Discovers and caches map images (PGM→PNG)
│   │   └── metrics/           ← Pluggable metric calculators
│   │       ├── base.py        ← BaseMetricCalculator ABC (with UNITS support)
│   │       ├── registry.py    ← Auto-discovery + topological execution
│   │       ├── performance/   ← Path, motion, time, collision, efficiency, pedestrian_path
│   │       ├── social/        ← Proxemics, gaze
│   │       ├── naturalness/   ← (Phase 2)
│   │       └── ecological/    ← (Phase 2)
│   ├── presentation/          ← Layer 5: Report and plot generation
│   │   ├── report_builder.py  ← Generates report.html (with GIF support)
│   │   ├── viz_manifest.py    ← Default plot manifest + PlotSpec loader
│   │   ├── plotly_renderer.py ← Interactive HTML chart dispatcher
│   │   ├── seaborn_renderer.py← Static PNG chart dispatcher
│   │   ├── color_utils.py     ← Global accessibility color palette (YAML-driven)
│   │   ├── dimension_detector.py ← Auto-detects varying dimensions for compound labels
│   │   ├── report_template.html.j2 ← Jinja2 HTML template
│   │   └── plot_types/        ← violin, box, bar, histogram, scatter, trajectory, radar, heatmap
│   ├── benchmark/             ← Benchmark runner and CLI
│   │   ├── runner.py          ← BenchmarkRunner (orchestrates simulation)
│   │   └── cli.py             ← Benchmark management CLI (argparse)
│   └── cli.py                 ← Evaluation pipeline CLI (argparse)
├── config/
│   ├── data_recorder_config.yaml  ← Topic throttle frequencies
│   ├── mcap_writer_options.yaml   ← MCAP writer tuning
│   └── color_palette.yaml         ← Accessibility color palette
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
3. Subscribes to all configured ROS topics (cmd_vel, lidar, odom, joint_states, plan, goal_pose, collision_events, arena_peds, EpisodeRecord, RobotFleet, tf, tf_static).
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
ros2 run arena_evaluation evaluation extract --benchmark-dir /opt/arena_ws/data/<benchmark_id>
ros2 run arena_evaluation evaluation process --benchmark-dir /opt/arena_ws/data/<benchmark_id>
```

The `extract` command reads each `recording_0.mcap` and saves topics into compressed Parquet files under `run_dir/topics/` (the extraction cache).
The `process` command reads these cached parquet files (or extracts them on-the-fly if missing), temporally aligns them, splits by `EpisodeRecord` boundaries, and runs all metric calculators. Writes per-run and combined Parquet files.

### Presentation Phase (offline)

```bash
ros2 run arena_evaluation evaluation report --benchmark-dir /opt/arena_ws/data/<benchmark_id>
# Or both at once:
ros2 run arena_evaluation evaluation run --benchmark-dir /opt/arena_ws/data/<benchmark_id>
# With animated GIFs:
ros2 run arena_evaluation evaluation report --benchmark-dir /opt/arena_ws/data/<benchmark_id> --generate-gifs
```

Reads `combined_metrics.parquet` and the default visualization manifest (hardcoded in `viz_manifest.py`), generates `report.html` and `plots/*.png`. When `--generate-gifs` is passed, animated trajectory GIFs are also saved.

---

## CLI Reference

`arena_evaluation` commands are ROS 2 console scripts, runnable standalone via
`ros2 run arena_evaluation <entry>`: the `evaluation` entry for data ops
(`extract`, `process`, `run`, `report`, `plot`), `evaluation_cli` for run
inspection (`list`, `status`, `tail`), and `benchmark` for running a suite.
Inside the Arena meta-repo, `arena evaluation <verb>` wraps all three (for
example `arena evaluation extract` runs `ros2 run arena_evaluation evaluation
extract`). The examples below use the standalone form.

```
usage: ros2 run arena_evaluation evaluation <command> [--run-dir DIR | --benchmark-dir DIR] [--output-dir DIR] [--generate-gifs]

Commands:
  extract   Layer 3: Extract topics from MCAP into fast Parquet files (cache)
  process   Layer 3: Compute metrics and write metrics.parquet (uses cached extraction by default)
  run       Full pipeline: Extract (overwrite) → process → report + plots
  report    Layer 5: Generate report.html from existing metrics.parquet
  plot      Layer 5: Generate static PNG plots only (no HTML)
```

`extract`, `process`, and `run` accept **either** `--run-dir` (single recording) or `--benchmark-dir` (full benchmark). `report` and `plot` only accept `--benchmark-dir`.

### Flags

| Flag | Commands | Description |
|---|---|---|
| `--output-dir DIR` | all | Output directory for reports/plots (defaults to first input dir) |
| `--generate-gifs` | report, plot, run | Generate animated trajectory GIFs (computationally intensive) |
| `--force-extract` | process | Force re-extraction of MCAP files, overwriting cached topics |

### Extracting Topics (Cache)

```bash
# Extract MCAP data into fast, compressed Parquet files per topic:
ros2 run arena_evaluation evaluation extract --benchmark-dir /opt/arena_ws/data/my_benchmark
# Output: data/my_benchmark/recordings/<planner>/<stage>/topics/*.parquet
```

### Processing a Single Run (ad-hoc recording)

```bash
# After running: arena launch ... record_data_dir:=data
ros2 run arena_evaluation evaluation process --run-dir /opt/arena_ws/data/recordings/20260528-215316
# Output: /opt/arena_ws/data/recordings/20260528-215316/metrics.parquet
```

### Processing a Full Benchmark

```bash
ros2 run arena_evaluation evaluation process --benchmark-dir /opt/arena_ws/data/my_benchmark
# Output: metrics.parquet per run + combined_metrics.parquet at root
```

### Full Pipeline (process + report)

```bash
# Single run (metrics only — no HTML report for single runs):
ros2 run arena_evaluation evaluation run --run-dir /opt/arena_ws/data/recordings/20260528-215316

# Full benchmark (metrics + HTML report + PNGs):
ros2 run arena_evaluation evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark

# With GIF generation:
ros2 run arena_evaluation evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark --generate-gifs
```

### Regenerate Report Only

```bash
# Single benchmark report
ros2 run arena_evaluation evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark
# Reads combined_metrics.parquet + viz_manifest
# Writes: report.html + plots/*.png

# Multi-benchmark merged report
ros2 run arena_evaluation evaluation report --benchmark-dir /opt/arena_ws/data/bench1 /opt/arena_ws/data/bench2 --output-dir ./merged_report
```

The benchmark management CLI is accessed via:

```bash
ros2 run arena_evaluation evaluation_cli list
ros2 run arena_evaluation evaluation_cli status <run_id>
ros2 run arena_evaluation evaluation_cli tail <run_id>
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

### Accessibility color palette (`config/color_palette.yaml`)

Defines a qualitative color sequence loaded globally by `color_utils.py`. The palette is applied to both Plotly and Seaborn at startup via `set_global_color_palette()`. White and black are excluded from the drawing sequence.

### Visualization manifest (`viz_manifest.py`)

The default set of plots is defined in `viz_manifest.py` as Python code (not a YAML file). A `viz_manifest.yaml` placed at the benchmark root can override this. Plot types include: `violin`, `box`, `bar`, `histogram`, `scatter`, `trajectory`, `radar`, `heatmap`.

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
| `seaborn` + `matplotlib` | Static PNG fallback charts + GIF animation |
| `PyYAML` | YAML config and metadata file I/O |
| `jinja2` | HTML report templating |
| `Pillow` | Map image conversion (PGM→PNG) |

ROS 2 dependencies (declared in `package.xml`): `rclpy`, `rosbag2_py`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `task_generator_msgs`, `arena_evaluation_msgs`.

---

## Rebuilding After Changes

```bash
arena build arena_evaluation
```
