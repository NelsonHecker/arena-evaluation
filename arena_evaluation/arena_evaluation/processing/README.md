# processing (Layer 3: Offline Metric Computation)

Computes navigational and ecological metrics from recorded episode MCAP files offline without requiring active ROS 2 nodes.

Pipeline data flow:

```
MCAPReader -> TopicParquetStore -> TopicAligner -> MetricRegistry -> ParquetStore
```

Each `episode_XXX.mcap` is aligned directly onto its odom axis.

---

## Files

| File | Purpose |
|---|---|
| `mcap_reader.py` | Decodes MCAP to `TopicBundle` (one DataFrame per topic) |
| `parquet_store.py` | Reads and writes extraction cache (`TopicParquetStore`) and output metrics (`ParquetStore`) |
| `topic_aligner.py` | Aligns multi-rate topics onto the odom time axis |
| `pipeline.py` | `ProcessingPipeline` orchestrates benchmark processing |
| `recorded_episodes.py` | `RecordedEpisodes`: per-step frames of the episodes of one run directory, recorded or imported |
| `map_registry.py` | Discovers, converts (PGM to PNG), and caches map images |
| `metrics/` | Metric calculators (performance, social, ecological, naturalness, edge_case) |

---

## mcap_reader.py - MCAPReader

Reads episode MCAP files using Python `mcap` libraries and produces a `TopicBundle`.

```python
from arena_evaluation.processing.mcap_reader import MCAPReader

bundle = MCAPReader(path).read()
```

Positions in the shared `map` frame (`tf_gt`, `plan`, `initialpose`, `peds`, episode start and goal) are moved into the env's world frame by the recorded `map -> env_<n>/map` anchor. A world swap moves the anchor mid-recording, so each row takes the anchor in effect at its `time_ns`: the last one at or before it, a later message winning a tie, the first one for rows before any.

### Flat Relational Telemetry Tables

All ingested streams are normalized into canonical flat relational tables stored as Parquet and loaded as Polars `LazyFrame`/`DataFrame`. All tables use primitive scalar dtypes (zero `pl.Object`, zero nested lists) with canonical timestamps `t_ns` (`Int64` nanoseconds).

| Table | Polars Columns | Description |
|---|---|---|
| `robot_tracks` | `t_ns`, `env_id`, `robot_id`, `x`, `y`, `z`, `yaw`, `vx`, `vy`, `vz`, `omega_z`, `speed_mps`, `pose_source` | Ground truth robot track strictly derived from `/tf_gt` |
| `robot_odom` | `t_ns`, `env_id`, `robot_id`, `x`, `y`, `z`, `yaw`, `vx`, `vy`, `vz`, `omega_z`, `speed_mps` | Robot wheel odometry track |
| `pedestrian_tracks` | `t_ns`, `env_id`, `ped_id`, `ped_name`, `x`, `y`, `z`, `yaw`, `vx`, `vy`, `vz`, `speed_mps`, `radius_m`, `desired_speed`, `policy`, `agent_type`, `animation_state`, `interaction_id`, `interaction_type`, `gait_phase`, `gait_cadence` | Flat 1 row per pedestrian per timestamp |
| `robot_scan_summary` | `t_ns`, `env_id`, `robot_id`, `min_range_m`, `mean_range_m`, `valid_samples`, `total_samples`, `front_min_m`, `rear_min_m`, `left_min_m`, `right_min_m` | Vectorized LiDAR range summary |
| `collision_contacts` | `t_ns`, `env_id`, `robot_id`, `obstacle_id`, `contact_type`, `impact_force_n` | Flat contact events with exact obstacle identifiers |
| `collision_events` | `t_ns`, `env_id`, `robot_id`, `event_count` | Continuous collision rising edges |
| `world_semantics` | `t_ns`, `env_id`, `world`, `entity_name`, `kind`, `field_name`, `field_kind`, `value_str`, `value_num`, `value_bool` | Flat world mechanisms and zone semantics |
| `robot_acoustics` | `t_ns`, `env_id`, `robot_id`, `total_dba`, `drivetrain_dba`, `payload_dba` | Acoustic emissions and exposure levels |
| `robot_power` | `t_ns`, `env_id`, `robot_id`, `total_power_w`, `static_power_w`, `mechanical_power_w` | Electrical and mechanical power draw |
| `robot_energy` | `t_ns`, `env_id`, `robot_id`, `total_energy_consumed_wh`, `battery_soc_percent` | Cumulative energy and battery state of charge |
| `pedestrian_gestures` | `t_ns`, `ped_id`, `slot`, `clip`, `hand`, `at_x`, `at_y`, `at_z`, `render_pose_override` | One row per active pedestrian gesture |

---

## topic_aligner.py - TopicAligner

Aligns topics onto the odom time axis using `join_asof` with backward-looking tolerance.

```python
aligned_df = TopicAligner(tolerance_ns=100_000_000).align(bundle)
```

- Odom is the primary time axis.
- Columns are preserved without prefixes.
- Missing values in the aligned frame appear as nulls.

---

## pipeline.py - ProcessingPipeline

Orchestrates extraction and metric computation.

```python
fm = FolderManager(data_root=Path("/opt/arena_ws/data"))
ProcessingPipeline(fm, workers=-1).process_benchmark("my_benchmark")
```

For each episode:
1. Reads `episode_XXX.yaml` metadata.
2. Loads `RobotParams`.
3. Loads cached topics or extracts them via `MCAPReader`.
4. Resolves the pose stream (`pose_anchor.py`): dense `tf_gt` as is, sparse `tf_gt` anchors the odom track onto the map frame with one rigid fit, none leaves the track in the odom frame without pedestrian-relative metrics.
5. Aligns topics with `TopicAligner`.
6. Computes metrics with `MetricRegistry`.
7. Attaches run identifiers and writes `metrics.parquet`.

Every recorded episode yields a row. `status` is `evaluated`, `path_only` (no map-frame pose), `no_trajectory`, `no_recording` or `error`, with `status_reason`; rows without a trajectory carry the runtime verdict in `result` and `success` and null metrics. `pose_source` (`tf`, `anchored`, `tf_sparse`, `odom`), `pose_samples` and `pose_anchor_residual_m` say how the robot pose was obtained. `integrity` (`List(String)`) holds the reasons the episode is compromised from the last recorded `EpisodeRecord`, or from the episode yaml when none was recorded, and is empty when the episode is intact. The report lists every non-`evaluated` row above the summary.

---

## Open-Loop Characterization

Characterization is evaluated by `CharacterizationCalculator` (`metrics/ecological/characterization.py`). It maps recorded `characterization_phase` markers across working points and outputs per-episode timeseries columns. The presentation layer derives working point aggregates directly from `combined_metrics.parquet`.

---

## Metric Calculator Framework

### Adding a Metric

1. Create a file in `metrics/<category>/`.
2. Subclass `BaseMetricCalculator` and define class attributes.
3. Implement `output_keys()` and `calculate()`.
4. Set `UNITS` dictionary for display labels.

```python
from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

class MyMetricCalculator(BaseMetricCalculator):
    NAME = "my_metric"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["path_metrics"]
    REQUIRED_TOPICS = ["odom"]
    UNITS = {"my_value": "m/s"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["my_value"]

    def calculate(self, episode: AlignedEpisodeBundle, robot_params: RobotParams, prior_results: dict) -> dict:
        return {"my_value": 1.0}
```

`MetricRegistry` discovers all `BaseMetricCalculator` classes, orders them topologically by `DEPENDS_ON`, and handles errors gracefully with `None` fills.

