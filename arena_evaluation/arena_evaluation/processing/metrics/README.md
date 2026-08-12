# Metrics

Auto-discovered metric calculators for the Arena evaluation pipeline. Every
`BaseMetricCalculator` subclass in this package is registered by
`MetricRegistry` and executed in topological order (Kahn's algorithm on
`DEPENDS_ON`). Results are written to `combined_metrics.parquet`, one row per
episode.

## Categories

| Folder | Focus |
|---|---|
| [`social/`](social/README.md) | Robot–pedestrian interaction: social forces, proxemics, gaze, disturbance |
| [`ecological/`](ecological/README.md) | Energy, acoustics, world-condition compliance |
| [`performance/`](performance/README.md) | Path, motion, time, collision, efficiency, clearance |
| [`naturalness/`](naturalness/README.md) | Trajectory naturalness against the unobstructed baseline |
| [`holistic/`](holistic/README.md) | Composite scores (not implemented) |

## Calculator contract

Each calculator declares:

- `NAME`, `CATEGORY`, `REQUIRES_PEDSIM`, `DEPENDS_ON` (execution-order edges),
  `REQUIRED_TOPICS` (topic gates; list/tuple entries mean "any of"),
  `UNITS`, `PRIMARY_OUTPUTS` (headline keys for default comparisons),
  `OUTPUT_DIRECTIONS` ("lower" / "higher" per output)
- `output_keys()` — exhaustive list of produced keys
- `calculate(episode, prior_results)` — returns every key, `None`-filled on
  error or missing data; `prior_results` contains the accumulated outputs of
  all upstream calculators

## Conventions

- **Multi-rate:** `AlignedEpisodeBundle.topics` carries raw native-rate topic
  frames (odom, tf_gt, peds, scan, cmd_vel, power, energy, acoustics).
  Rate-sensitive calculators compute on their own time base; `episode.data`
  remains the odom-aligned frame for robot-trajectory metrics.
- **Ground truth first:** `resolve_native_pose` prefers the tf_gt world pose;
  odom is the fallback. Ground-truth velocity is derived by differentiating
  GT positions.
- **Proxemic distances:** zone metrics use the effective edge-to-edge distance
  `d_eff = d_center − (r_robot + r_ped)` against Hall's zones (0.45 / 1.2 /
  3.6 m) — comparable across robot footprints.
- **Energy:** reported in watt-hours; `specific_cost_of_transport` uses the
  total energy consumed.
- **Reference-based metrics** (`pfi`, `mar`, `ped_path_deflection_m`) are
  computed post-hoc in `pipeline.process_benchmark` against the reference
  runs (unobstructed_robot / unhindered_peds), not inside the calculators.

## Data sources

Per-topic parquets (`topics/<topic>.parquet`) are extracted by `MCAPReader`
and aligned by `TopicAligner` (backward asof, 100 ms tolerance, onto the odom
axis). See the per-category READMEs for the topics each metric consumes.

## Invocation

```bash
arena evaluation run --benchmark-dir <run_id>                          # extract + metrics
arena evaluation run --benchmark-dir <run_id> --report-manifest <name> # + report
```

List registered calculators:

```python
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.schemas import RobotParams

reg = MetricRegistry(RobotParams())
for m in reg.list_metrics():
    print(m["name"], m["outputs"])
```
