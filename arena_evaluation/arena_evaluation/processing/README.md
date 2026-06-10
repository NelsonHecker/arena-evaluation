# processing — Layer 3: Offline Metric Computation

This package reads recorded MCAP files and computes all navigational metrics. It operates entirely **offline** — no running ROS 2 environment is required.

The processing pipeline follows a strict left-to-right data flow:

```
MCAPReader → TopicParquetStore → TopicAligner → EpisodeSplitter → MetricRegistry → ParquetStore
```

---

## Files

| File | Purpose |
|---|---|
| `mcap_reader.py` | Decodes MCAP → `TopicBundle` (one DataFrame per topic) |
| `parquet_store.py` | Reads/writes cache (`TopicParquetStore`) and output (`ParquetStore`) |
| `topic_aligner.py` | Aligns multi-rate topics onto a common odom time axis |
| `episode_splitter.py` | Splits continuous stream into per-episode bundles |
| `pipeline.py` | `ProcessingPipeline` orchestrator — runs the full chain |
| `metrics/` | Pluggable metric calculators |

---

## mcap_reader.py — MCAPReader

Reads a rosbag2 MCAP file using the `mcap` + `mcap-ros2-support` Python libraries (no `ros2 bag play` required) and produces a `TopicBundle`.

```python
from arena_evaluation.processing.mcap_reader import MCAPReader

reader = MCAPReader(
    mcap_path=Path("/opt/arena_ws/data/my_benchmark/recordings/dwa/stage_1/recording/recording_0.mcap"),
    robot_params=RobotParams.load("jackal"),
)
bundle = reader.read()
# bundle.odom        → pl.DataFrame with columns: time_ns, pos_x, pos_y, yaw, vel_linear, vel_angular
# bundle.scan        → pl.DataFrame with columns: time_ns, scan_min, scan_ranges
# bundle.cmd_vel     → pl.DataFrame with columns: time_ns, cmd_linear, cmd_angular
# bundle.episode_record → pl.DataFrame with columns: time_ns, episode_id, outcome, ...
```

**Extracted fields per topic:**

| Topic | Polars Columns |
|---|---|
| odom | `time_ns`, `pos_x`, `pos_y`, `yaw` (from quaternion), `vel_linear`, `vel_angular` |
| scan | `time_ns`, `scan_min`, `scan_ranges` (filtered to valid range) |
| cmd_vel | `time_ns`, `cmd_linear`, `cmd_angular` |
| joint_states | `time_ns`, `joint_vel_left`, `joint_vel_right` |
| peds | `time_ns`, `peds_positions`, `peds_headings`, `num_pedestrians` |
| episode_record | `time_ns`, `episode_id`, `outcome_state`, `start_x`, `start_y`, `goal_x`, `goal_y`, … |
| collision_events | `time_ns`, `collision_count` |

---

## topic_aligner.py — TopicAligner

Aligns all topics onto the odom time axis using Polars `join_asof` with a configurable backward-looking tolerance.

```python
from arena_evaluation.processing.topic_aligner import TopicAligner
from arena_evaluation.storage.schemas import TopicBundle

aligner = TopicAligner(tolerance_ns=200_000_000)  # 200 ms default
aligned_df = aligner.align(bundle)
# Returns a single pl.DataFrame with all topic columns, indexed by odom time_ns
# Rows where no matching secondary data is within tolerance have null values
```

**Design rationale:**
- Odom is chosen as the primary time axis because it runs at the highest consistent frequency (~50 Hz).
- `join_asof` with `strategy="backward"` means each odom row is joined to the most recent scan/cmd_vel/peds sample within the tolerance window.
- Null values in the aligned frame indicate gaps. All metric calculators must handle nulls correctly.

**Tuning the tolerance:**
The default of 200 ms is conservative. After collecting real data, inspect the distribution of inter-message gaps (e.g. `bundle.scan["time_ns"].diff().describe()`) and tighten the tolerance to avoid spurious null rows.

---

## episode_splitter.py — EpisodeSplitter

Uses `EpisodeRecord` messages embedded in the MCAP to split the aligned DataFrame into per-episode bundles.

```python
from arena_evaluation.processing.episode_splitter import EpisodeSplitter

splitter = EpisodeSplitter(min_odom_frames=5)
episodes = splitter.split(bundle, robot_params)
# Returns list[AlignedEpisodeBundle]
```

Each `AlignedEpisodeBundle` contains:
- The episode's aligned DataFrame slice.
- `start_pos` and `goal_pos` extracted from `EpisodeRecord.robots_params`.
- `episode_id` and `num_pedestrians`.

Episodes with fewer than `min_odom_frames` rows (default: 5) are discarded — they indicate aborted or degenerate runs.

---

## parquet_store.py — ParquetStore

Reads and writes Parquet files with `RunMetadata` embedded in the file footer.

```python
from arena_evaluation.processing.parquet_store import ParquetStore

# Write
ParquetStore.write(df, metadata, dest_path)

# Read
df, metadata = ParquetStore.read(source_path)

# Combine multiple runs or multiple benchmarks
ParquetStore.combine([path1, path2, path3], combined_path)
```

The metadata is serialized as JSON and stored under the Parquet key `arena_evaluation_metadata`. This ensures the processing context (planner, robot, stage, git SHA) stays permanently attached to the metrics data.

Array columns (e.g., `path`, `scan_ranges`) are stored as Parquet `List` type rather than JSON strings.

---

## pipeline.py — ProcessingPipeline

Orchestrates the full processing chain for a benchmark.

```python
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.folder_manager import FolderManager

fm = FolderManager(data_root=Path("/opt/arena_ws/data"))
pipeline = ProcessingPipeline(fm)
pipeline.process_benchmark("my_benchmark")
```

For each discovered run:
1. Reads `metadata.yaml` → `RunMetadata`
2. Loads `RobotParams` from `arena_robots` caps
3. Uses `TopicParquetStore.read()` to load cached topics from `run_dir/topics/`. If missing (or forced), calls `MCAPReader.read()` and writes the cache.
4. `TopicAligner.align()` → aligned DataFrame
5. `EpisodeSplitter.split()` → `list[AlignedEpisodeBundle]`
6. `MetricRegistry.run()` → metrics DataFrame per episode
7. Adds identity columns: `planner`, `robot_model`, `stage`, `benchmark_id`
8. `ParquetStore.write()` → `metrics.parquet`
9. Updates `metadata.yaml` with `processing_completed_at`, `episodes_valid`

After all runs: `ParquetStore.combine()` → `combined_metrics.parquet`.

---

## metrics/ — Metric Calculator Framework

### Adding a New Metric

1. Create a new file under the appropriate category (`performance/`, `social/`, `naturalness/`, `ecological/`).
2. Subclass `BaseMetricCalculator` and set the required class attributes.
3. Implement `output_keys()` and `calculate()`.

```python
from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
import polars as pl

class MyMetricCalculator(BaseMetricCalculator):
    NAME = "my_metric"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["path_metrics"]  # must run after path_metrics

    def output_keys(self) -> list[str]:
        return ["my_value", "my_mean"]

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        robot_params: RobotParams,
        prior_results: dict,
    ) -> dict:
        path_length = prior_results.get("path_length", 0)
        my_value = ...  # compute using episode.data (pl.DataFrame)
        return {"my_value": my_value, "my_mean": ...}
```

The `MetricRegistry` auto-discovers all `BaseMetricCalculator` subclasses, resolves `DEPENDS_ON` via Kahn's topological sort, and runs them in dependency order. If a calculator raises an exception, its output keys are NaN-filled and processing continues.

### Phase 1 Calculators

| Calculator | NAME | Key Outputs |
|---|---|---|
| `PathMetricsCalculator` | `path_metrics` | `path_length`, `curvature`, `roughness` |
| `MotionMetricsCalculator` | `motion_metrics` | `velocity_mean`, `velocity_max`, `jerk_mean` |
| `TimeMetricsCalculator` | `time_metrics` | `time_to_goal`, `idling_time` |
| `CollisionMetricsCalculator` | `collision_metrics` | `collision_amount`, `result`, `success` |
| `PathEfficiencyCalculator` | `path_efficiency` | `path_efficiency` (straight-line / actual) |
| `ProxemicsCalculator` | `proxemics` | `time_in_personal_space`, `avg_velocity_in_personal_space` |
| `GazeMetricsCalculator` | `gaze_metrics` | `time_looking_at_peds`, `time_looked_at_by_peds` |

### Execution Order

Inspect the resolved topological order at runtime:

```python
from arena_evaluation.processing.metrics.registry import MetricRegistry
registry = MetricRegistry()
print(registry.execution_order())
```
