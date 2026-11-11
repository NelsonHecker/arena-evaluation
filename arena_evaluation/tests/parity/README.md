# Parity certificate (`tests/parity`)

Evidence that the canonical catalog views reproduce the frozen legacy per-episode
metrics before the legacy evaluation path is removed. The harness measures; it does
not port metrics.

## What it compares

The legacy side is the frozen baseline under
`/home/nelson/arena_ws/data/_parity_baseline/20261109/`: one `metrics.parquet` frame per
episode (24 episodes, two benchmarks). The new side is a processed benchmark directory
whose facts live once per episode:

```
<run>/
  dim_episodes.parquet, dim_robots.parquet, dim_pedestrians.parquet,
  dim_world_entities.parquet
  episodes/episode_NNN/metrics/fact_episodes.parquet
                             fact_encounters.parquet
                             fact_contacts.parquet
                             fact_pedestrians.parquet
                             fact_robots.parquet
                             fact_edge_cases.parquet ... fact_collisions.parquet
                             series_*.parquet
```

`certify.py` builds `BenchmarkCatalog.from_directory(run_dir)` and compares the
episode facts through `v_episode_summary`; every other scope is read through its
catalog fact view. There is no `combined_metrics.parquet` and no benchmark-level
`metrics/fact_*.parquet` in the new layout. A baseline parquet is never copied into
this repository.

## Running it

```bash
python tests/parity/certify.py \
  --baseline-dir /home/nelson/arena_ws/data/_parity_baseline/20261109 \
  --run-dir /path/to/benchmarks/<run-id> \
  [--benchmark <baseline-benchmark-id>] \
  [--output-dir /tmp/parity-out]

python tests/parity/certify.py --self-check
```

`--benchmark` defaults to the run's `dim_episodes.run_id`. Both sides are joined on
`(run_id, episode_id, robot)`; the runtime robot instance suffix is stripped, so
`env_0_jackal_0` matches the frozen `env_0_jackal` without treating the rename as a
metric difference.

## Statuses

| status | meaning | gate |
| --- | --- | --- |
| `exact` | every compared episode carries the same value | passes |
| `within_tolerance` | all differences are inside `--rel-tol` (default `1e-5`, relative) | passes |
| `expected_diff` | every difference is covered by `expected_diffs.yaml` | passes |
| `missing` | no usable comparison: an unexplained difference, no joined episode, all values null on both sides, or the legacy column is absent | fails |
| `no_producer` | the mapped catalog view or column does not exist in this run | fails |

Every row is printed to `certificate.csv` when `--output-dir` is given. A certificate
cannot go green on null==null, and an uncovered numeric difference is reported as
`missing` with an `UNEXPLAINED` detail rather than silently passing. Exit codes: `0`
green, `1` failing status, `2` invalid input or manifest.

## Reviewed data

| file | role |
| --- | --- |
| `metric_map.yaml` | legacy key -> catalog table/column, list reductions, legacy combine |
| `expected_diffs.yaml` | reviewed allowlist plus the overhaul agreements: 2.0 m surface-to-surface encounter detector, one fault/success policy, NULL-not-zero, odometry -> `/tf`, unknown seed/crowd as NULL, runtime robot-name suffix |
| `inventory_254.csv` | the historical 267-key inventory, kept for provenance; the certificate only compares the reviewed map |

`expected_diffs.yaml` kinds:

* `unit_scale` -- catalog = legacy * `factor`, verified inside `max_rel` (default `1e-9`).
* `derived` -- reviewed non-factor relationship, verified inside its mandatory `max_rel`.
* `ground_truth` -- the value may move because the source changed (odometry -> `/tf`,
  detector membership, the fault policy); accepted without a numeric bound.
* `null_fill` -- the legacy `fill_null(0.0)` masking became null; accepted only when the
  catalog value is null and the legacy value was not.
* `dtype` -- a reviewed dtype-only change; the comparison casts.

Entries without `star_table`/`star_column` apply to every mapping of that metric key.
`notes:` records the cross-cutting agreements that have no single metric key.

## Tests

`tests/parity/test_certify.py` builds synthetic baseline and run directories in
`tmp_path` and covers exact, tolerance, covered null-fill, uncovered difference,
all-null evidence, absent legacy column, absent catalog column, the robot-name change
and the self-check. It never touches a live run or the frozen baseline.

`--self-check` exercises the same comparison engine on in-memory frames.

## Current real run

Use the command above with a benchmark reprocessed by the current pipeline. The old
`results/` certificates were removed: they were produced against the retired
`combined_metrics.parquet` layout and would pin a path the pipeline no longer writes.
