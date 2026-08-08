# processing — Layer 3: Offline Metric Computation

This package reads recorded episode MCAPs and computes all navigational metrics. It operates entirely **offline** — no running ROS 2 environment is required.

The processing pipeline follows a strict left-to-right data flow:

```
MCAPReader → TopicParquetStore → TopicAligner → MetricRegistry → ParquetStore
```

With the **per-episode recording structure** (one MCAP per episode) there is no in-bag splitting
step: each `episode_XXX.mcap` is aligned directly onto its odom axis.

---

## Files

| File | Purpose |
|---|---|
| `mcap_reader.py` | Decodes MCAP → `TopicBundle` (one DataFrame per topic) |
| `parquet_store.py` | Reads/writes the extraction cache (`TopicParquetStore`) and outputs (`ParquetStore`) |
| `topic_aligner.py` | Aligns multi-rate topics onto the odom time axis |
| `pipeline.py` | `ProcessingPipeline` orchestrator — extract + process for a benchmark |
| `map_registry.py` | Discovers, converts (PGM→PNG), and caches map images with resolved origins |
| `metrics/` | Pluggable metric calculators (performance/, social/, ecological/, naturalness/) |

The open-loop characterization analysis lives in `arena_evaluation/characterization/` and consumes
the same extraction cache (see below).

---

## mcap_reader.py — MCAPReader

Reads an episode MCAP using the `mcap` + `mcap-ros2-support` Python libraries (no `ros2 bag play`
required) and produces a `TopicBundle` per robot namespace.

```python
from arena_evaluation.processing.mcap_reader import MCAPReader

bundle = MCAPReader(path).read()
# bundle["env_0_jackal"].odom   → pl.DataFrame with time_ns, pos_x, pos_y, yaw, vel_linear, vel_angular
# bundle["env_0_jackal"].cmd_vel → time_ns, linear_x, linear_y, …, angular_z
# bundle["env_0_jackal"].power   → time_ns, total_power_w, static_power_w, total_mechanical_power_w, …
# bundle["env_0_jackal"].acoustics → time_ns, total_level_af_dba, baseline_level_dba, …
# bundle["env_0_jackal"].characterization_phase → time_ns, label
```

**Extracted fields per topic** (all columns are stored **unprefixed** — the aligner does not rename):

| Topic | Polars Columns |
|---|---|
| odom | `time_ns`, `pos_x`, `pos_y`, `yaw`, `vel_linear`, `vel_angular` |
| scan | `time_ns`, `scan_ranges`, `scan_min` |
| cmd_vel | `time_ns`, `linear_x`, `linear_y`, `linear_z`, `angular_x`, `angular_y`, `angular_z` |
| joint_states | `time_ns`, `name`, `position`, `velocity`, `effort` (lists) |
| power | `time_ns`, `total_power_w`, `static_power_w`, `total_mechanical_power_w`, `total_thermal_power_w`, `joint_*_power_w` |
| energy | `time_ns`, `total_energy_consumed_wh`, `battery_soc_percent` |
| acoustics | `time_ns`, `total_level_af_dba`, `total_level_zf_db`, `baseline_level_dba`, `drivetrain_level_dba`, `uncertainty_1sigma_dba`, `validity_flags` |
| characterization_phase | `time_ns`, `label` |
| peds | `time_ns`, `peds_positions`, `peds_headings`, `num_pedestrians` |
| episode_record | `time_ns`, `episode_id`, `outcome_state`, `outcome_info`, `goal_uuid`, `robots_params` |
| collision_events | `time_ns`, `collision_event` |
| collision_monitor_state | `time_ns`, `action_type`, `polygon_name` |
| tf / tf_static | `time_ns`, frame/child/trans/rot columns |
| tf_gt | `time_ns`, `pos_x_gt`, `pos_y_gt`, `yaw_gt`, `frame_id` |

---

## topic_aligner.py — TopicAligner

Aligns all topics onto the odom time axis using Polars `join_asof` with a configurable backward-looking tolerance.

```python
aligned_df = TopicAligner(tolerance_ns=100_000_000).align(bundle)
```

**Design rationale:**
- Odom is the primary time axis (highest consistent frequency).
- `join_asof` with `strategy="backward"`: each odom row is joined to the most recent sample of each
  secondary topic within the tolerance window.
- Joined columns are **not prefixed** — `power` arrives as `total_power_w`, `joint_states` as
  `velocity`/`effort`, `cmd_vel` as `linear_x`/`angular_z`. Consumers must use the unprefixed names.
- Null values in the aligned frame indicate gaps; metric calculators must handle nulls correctly.

---

## pipeline.py — ProcessingPipeline

Orchestrates extraction and metric computation for a benchmark.

```python
fm = FolderManager(data_root=Path("/opt/arena_ws/data"))
ProcessingPipeline(fm, workers=-1).process_benchmark("my_benchmark")
```

For each discovered episode (`FolderManager.discover_episodes` → `episodes/episode_XXX/`):

1. Reads `episode_XXX.yaml` → `RunMetadata` (planner, local/inter planner, stage, map, …)
2. Loads `RobotParams` from the `arena_robots` caps
3. Loads cached topics from `episode_XXX/topics/` (extracts via `MCAPReader` when missing)
4. `TopicAligner.align()` → the episode's aligned DataFrame
5. `MetricRegistry.run()` → the episode's metric row
6. Adds identity columns (`planner`, `local_planner`, `inter_planner` from the yaml when present,
   `robot`, `stage`, `map`, `benchmark_id`, …)
7. Writes `episode_XXX/metrics.parquet`; all episodes are combined into `combined_metrics.parquet`

`--workers -1` (the default) auto-detects the CPU count for parallel extraction/processing.

### Legacy recordings

Older recordings that embed `EpisodeRecord` boundary markers in a single continuous MCAP are still
readable: the reader extracts `episode_record` and the aligner output can be windowed by outcome
state (the former `EpisodeSplitter` logic). New per-episode recordings skip windowing entirely.

---

## Open-Loop Characterization (ecological metric)

Characterization is a regular metric calculator —
[`metrics/ecological/characterization.py`](metrics/ecological/characterization.py)
(`CharacterizationCalculator`, NAME `characterization`). It attaches the recorded
`characterization_phase` markers to every sample (the aligner carries them forward with a
no-tolerance asof join into the `label` column; the calculator maps labels to
kind/vx_target/wz_target via the task mode's schedule, with a cmd_vel fallback classifier) and
emits per-episode list columns into the metrics row:

- `timeseries_char_time_s`, `timeseries_char_power_total_w`, `timeseries_char_power_mech_w`
  (`Σ|τ·ω|`), `timeseries_char_dba` (recorded acoustics, with a steady-state joint-model
  fallback), `timeseries_char_vx_achieved`, `timeseries_char_energy_intensity` (J/m per sample),
  `timeseries_char_leq_power` (10^(dBA/10) — for exact L_Aeq), `timeseries_char_phase_kind`,
  `timeseries_char_vx_target`, `timeseries_char_wz_target`.

The report layer derives long frames and per-working-point aggregates from these columns (the
`line` renderer explodes the lists and aggregates mean ± std / L_Aeq / max — see the
`characterization` report manifest). No separate analysis module or artifacts are needed;
`arena evaluation run --report-manifest characterization` works on `combined_metrics.parquet`
directly.

---

## metrics/ — Metric Calculator Framework

### Adding a New Metric

1. Create a new file under the appropriate category (`performance/`, `social/`, `ecological/`, `naturalness/`).
2. Subclass `BaseMetricCalculator` and set the required class attributes.
3. Implement `output_keys()` and `calculate()`.
4. Optionally set `UNITS` to attach SI unit labels to output keys for display on plot axes.

```python
from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
import polars as pl

class MyMetricCalculator(BaseMetricCalculator):
    NAME = "my_metric"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["path_metrics"]  # must run after path_metrics
    REQUIRED_TOPICS = ["odom"]     # topics that must be present

    UNITS = {"my_value": "m/s"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["my_value", "my_mean"]

    def calculate(self, episode: AlignedEpisodeBundle, robot_params: RobotParams, prior_results: dict) -> dict:
        return {"my_value": ..., "my_mean": ...}
```

The `MetricRegistry` auto-discovers all `BaseMetricCalculator` subclasses, resolves `DEPENDS_ON`
via Kahn's topological sort, and runs them in dependency order. If a calculator raises, its output
keys are None-filled and processing continues.

### BaseMetricCalculator Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `NAME` | `str` | Unique calculator name (registry key, `DEPENDS_ON` target) |
| `CATEGORY` | `str` | `"performance"`, `"social"`, `"ecological"`, `"naturalness"` |
| `REQUIRES_PEDSIM` | `bool` | Requires pedestrian simulation data |
| `DEPENDS_ON` | `list[str]` | Calculators that must run first |
| `REQUIRED_TOPICS` | `list` | Topics that must be present |
| `UNITS` | `dict[str, str]` | Output-key → SI unit strings (also feeds report axis labels) |

The base class also provides `resolve_robot_pose()` which handles TF ground-truth resolution with
fallback to raw odom.

### Phase 1 Calculators

| Calculator | NAME | Key Outputs | Units |
|---|---|---|---|
| `PathMetricsCalculator` | `path_metrics` | `path_length`, `path`, `curvature`, `roughness_mean` | m, —, rad/m, — |
| `MotionMetricsCalculator` | `motion_metrics` | `velocity_mean`, `velocity_max`, `jerk_mean` | m/s, m/s, m/s³ |
| `TimeMetricsCalculator` | `time_metrics` | `time_to_goal`, `idling_time` | s, s |
| `CollisionMetricsCalculator` | `collision_metrics` | `collision_amount`, `result`, `success` | —, —, — |
| `PathEfficiencyCalculator` | `path_efficiency` | `path_efficiency` (straight-line / actual) | — |
| `PedestrianPathMetricsCalculator` | `pedestrian_path_metrics` | `pedestrian_path` | m |
| `ProxemicsCalculator` | `proxemics` | `time_in_personal_space`, `avg_velocity_in_personal_space` | s, m/s |
| `GazeMetricsCalculator` | `gaze_metrics` | `time_looking_at_peds`, `time_looked_at_by_peds` | s, s |
| `EnergyMetricCalculator` | `energy` | `energy_*_wh`, `battery_soc_final`, power timeseries | Wh, % |
