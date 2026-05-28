# storage — Layer 2: Shared Schemas and Path Management

This package contains the shared data types and infrastructure used by **all** other layers in the pipeline. It has no external dependencies beyond `pydantic` and `PyYAML`, and it does not import from `ingestion`, `processing`, or `presentation`.

---

## Files

| File | Purpose |
|---|---|
| `schemas.py` | Pydantic and dataclass type definitions (the data model) |
| `folder_manager.py` | `FolderManager` — resolves and validates all output paths |
| `manifest.py` | `MetadataWriter` — reads and writes `metadata.yaml` |
| `exceptions.py` | Domain-specific exceptions for the whole pipeline |

---

## schemas.py — Data Models

### `RunMetadata`

Pydantic model matching the `metadata.yaml` schema. Written by the recorder and read by the processing pipeline.

```python
from arena_evaluation.storage.schemas import RunMetadata

meta = RunMetadata(
    benchmark_id="my_benchmark",
    planner="dwa",
    robot_model=["jackal"],
    map="map_empty",
    stage="stage_1",
    episodes_requested=10,
    suite_name="default",
    contest_name="dwa_vs_teb",
    inter_planner="",
    agent_name="",
    recording_started_at="2026-05-28T21:03:48+00:00",
    python_version="3.12.3",
    ros_distro="jazzy",
)
```

**Write-on-startup fields** — known at recording start: `benchmark_id`, `planner`, `robot_model`, `map`, `stage`, `suite_name`, `contest_name`, `recording_started_at`, `arena_git_sha`, `arena_git_dirty`, `python_version`, `ros_distro`.

**Write-on-episode fields** — populated from `EpisodeRecord` messages: `tm_obstacles`, `tm_robots`, `tm_modules`, `obstacles_params`, `robots_params`.

**Write-on-shutdown fields** — finalized on clean exit: `recording_ended_at`, `episodes_recorded`, `pedsim_available`, `recorded_topics`.

**Write-on-processing fields** — added by the processing pipeline: `processing_completed_at`, `episodes_valid`, `pipeline_version`.

### `RunDescriptor`

Frozen dataclass identifying a discovered run. Returned by `FolderManager.discover_runs()`.

```python
@dataclass(frozen=True)
class RunDescriptor:
    run_dir: str
    benchmark_id: str
    planner: str
    stage: str
```

### `TopicBundle`

Dataclass holding one `polars.DataFrame | None` per topic. Produced by `MCAPReader` and consumed by `TopicAligner`.

```python
@dataclass
class TopicBundle:
    odom: pl.DataFrame | None
    scan: pl.DataFrame | None
    cmd_vel: pl.DataFrame | None
    joint_states: pl.DataFrame | None
    peds: pl.DataFrame | None
    episode_record: pl.DataFrame | None
    collision_events: pl.DataFrame | None
```

Each DataFrame has a `time_ns` column (nanoseconds since epoch) as its primary time axis.

### `AlignedEpisodeBundle`

Dataclass produced by `EpisodeSplitter` after aligning all topics onto the odom time axis for a single episode.

```python
@dataclass
class AlignedEpisodeBundle:
    episode_id: int
    data: pl.DataFrame      # all topics joined onto odom time_ns
    start_pos: list[float]  # [x, y] from EpisodeRecord
    goal_pos: list[float]   # [x, y] from EpisodeRecord
    num_pedestrians: int
```

### `RobotParams`

Frozen dataclass loaded from the `arena_robots` caps YAML at processing time.

```python
params = RobotParams.load("jackal")
# params.robot_radius, params.laser_min_range, params.laser_max_range
```

### `PlotSpec`

Pydantic model for entries in `viz_manifest.yaml`.

---

## folder_manager.py — FolderManager

Resolves all output paths relative to a `data_root`. Ensures no path escapes the root (path traversal protection).

```python
from arena_evaluation.storage.folder_manager import FolderManager
from pathlib import Path

fm = FolderManager(data_root=Path("/opt/arena_ws/data"))

# Resolve paths
run_dir   = fm.run_dir("my_benchmark", "dwa", "stage_1")
# → /opt/arena_ws/data/my_benchmark/recordings/dwa/stage_1

mcap_path = fm.mcap_path(run_dir)
# → /opt/arena_ws/data/my_benchmark/recordings/dwa/stage_1/recording.mcap

metrics   = fm.metrics_path(run_dir)
# → /opt/arena_ws/data/my_benchmark/recordings/dwa/stage_1/metrics.parquet

combined  = fm.combined_metrics_path("my_benchmark")
# → /opt/arena_ws/data/my_benchmark/combined_metrics.parquet

# Discover all valid runs in a benchmark
runs = fm.discover_runs("my_benchmark")
# → list[RunDescriptor]
```

**Path traversal protection:** every resolved path is checked with `resolved.relative_to(data_root)`. If the path escapes `data_root`, a `ValueError` is raised.

**Run discovery:** `discover_runs()` scans `data_root/<benchmark_id>/recordings/<planner>/<stage>/` for directories containing a `metadata.yaml` file.

---

## manifest.py — MetadataWriter

Static helper for reading and writing `metadata.yaml` files.

```python
from arena_evaluation.storage.manifest import MetadataWriter
from pathlib import Path

dest = Path("/opt/arena_ws/data/recordings/20260528-210000/metadata.yaml")

# Write
MetadataWriter.write(metadata, dest)

# Read
metadata = MetadataWriter.read(dest)

# Update specific fields in-place
MetadataWriter.update(dest, episodes_recorded=10, recording_ended_at="2026-05-28T22:00:00+00:00")
```

All YAML is loaded with `yaml.safe_load()` — no arbitrary code execution. Written files are `chmod 0o666` to ensure they are readable/writable by any user in the Docker environment.

---

## exceptions.py

| Exception | Raised When |
|---|---|
| `MetricCalculationError` | A metric calculator fails to produce a result |
| `CircularDependencyError` | Metric registry detects a circular `DEPENDS_ON` chain |
| `SchemaViolationError` | A Parquet file or YAML does not match the expected schema |
| `RobotNotFoundError` | `RobotParams.load()` cannot find the robot caps YAML |
| `ManifestGenerationError` | `MetadataWriter` fails to read or write a `metadata.yaml` |
