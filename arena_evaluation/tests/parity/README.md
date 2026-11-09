# G1 parity certificate (`tests/parity`)

Evidence that the star-schema lakehouse reproduces the legacy `combined_metrics.parquet`
columns before the legacy evaluation path is removed. This directory holds the
harness, its reviewed data, and the certificate artefacts produced from a real
dual-path run.

The harness does **not** port metrics. It measures.

## What it does

`certify.py` reads one benchmark directory that holds both outputs of the current
dual-path pipeline:

```
<benchmark>/
  combined_metrics.parquet          # legacy one-row-per-episode frame
  dim_episodes.parquet              # star schema dimensions
  metrics/fact_{episodes,encounters,contacts,pedestrians,robots}.parquet
```

For every key in `inventory_254.csv` it emits one certificate row per reviewed
mapping, joining the legacy frame to the star table on
`(run_id, episode_id[, robot_id])` (legacy `benchmark_id`/`episode`/`robot` map to
the star keys). The certificate columns are

`metric_key | legacy_column | star_table | star_column | status | episodes_compared |
max_abs_diff | max_rel_diff | detail`

with statuses

| status | meaning |
| --- | --- |
| `EXACT` | every compared episode carries the same value in both tables |
| `WITHIN_TOL` | numeric values agree within `--rel-tol` (default `1e-5`, relative per episode) |
| `EXPECTED_DIFF` | a reviewed `expected_diff.yaml` entry explains the difference |
| `MISSING_STAR` | the inventory key has no scoped producer in this pipeline revision |
| `MISSING_LEGACY` | a mapped legacy column is absent from this run's frame |
| `NULL_REASON` | legacy value is `0.0`, star value is null and `null_reason` explains it |
| `MISMATCH` | unexplained difference; the certificate exits non-zero |

`MISMATCH` is deliberately outside the accepted set: the gate fails even though the
row is still reported, so a first certificate can be produced without hiding a
regression. Exit code `2` means the input is not a dual-path run or a manifest is
invalid.

## Files

| file | role |
| --- | --- |
| `certify.py` | generator (not a unittest); `--self-check` exercises the comparison engine on synthetic frames |
| `inventory_254.csv` | the 267 legacy metric keys (267 rows; the historical `254` name is kept for continuity) |
| `metric_map.yaml` | reviewed correspondence: inventory key -> star table/column, list reduction, legacy combine |
| `expected_diff.yaml` | reviewed allowlist of legitimate value changes, each with a reason |
| `results/certificate.csv` | the certificate for the committed run |
| `results/gaps.csv` | inventory keys without a scoped producer, with the triaged owner |
| `results/gap_summary.md` | the gap list grouped by owner, calculator and category |
| `results/star_coverage.csv` | numeric star metric columns with no reviewed legacy counterpart (sanity check) |
| `results/summary.yaml` | machine-readable counts and provenance |

Only identity-grain targets are claimed in this revision: `fact_episodes` and
`fact_robots` hold one row per episode and robot, so a legacy episode value maps
directly. `fact_encounters`, `fact_contacts` and `fact_pedestrians` need a reviewed
reduction per legacy episode aggregate (count/min/max/sum over rows); they are not
claimed yet and their columns show up in `star_coverage.csv` until a reviewer adds
the reduction entries.

## Running it

```bash
python arena_evaluation/tests/parity/certify.py --benchmark-dir /path/to/benchmark
python arena_evaluation/tests/parity/certify.py --self-check
```

Artifacts are written to `tests/parity/results/` unless `--output-dir` overrides it.

## Finding a real input

Historic runs predate the star schema. `bench-live-social-dwb`, the `20260914-*`
runs and the frozen `data/_parity_baseline/20261109` parquets carry only the legacy
side, so the harness refuses them (`ParityInputError`, exit 2). The frozen baseline
is the legacy reference for a later replay, not a certificate input.

The verified reprocessing recipe is:

1. Copy at most ten episode directories from a historic run into a scratch
   benchmark directory. An episode directory is self-contained
   (`episode_NNN.mcap` + `episode_NNN.yaml`); copy the run's `manifest.yaml` too so
   suite-level metric overrides resolve.
2. Run the current pipeline over the scratch benchmark:

   ```bash
   source /opt/arena_ws/source
   PYTHONPATH=<branch>/arena_evaluation arena evaluation process \
     --benchmark-dir /opt/arena_ws/data/benchmarks/<scratch> --workers 1 --force-extract
   ```

   (`ProcessingPipeline.process_benchmark` is what the CLI calls: it extracts topics
   into `episodes/episode_NNN/topics/`, evaluates metrics, writes the per-episode
   `metrics.parquet` + `metrics/` star tables, then concatenates both benchmark
   artefacts. `folder_manager.describe_episode(benchmark_id)` overrides the episode
   yaml's original `benchmark_id` with the scratch directory name, so the legacy
   `benchmark_id` and the star `run_id` agree.)
3. `--force-extract` is only needed for the first pass. Episodes whose
   `metrics.parquet` **and** `metrics/` already exist are skipped by
   `process_episode`; delete both to recompute metrics while keeping the cached
   `topics/`.

## Reviewed manifest policy

`metric_map.yaml` claims a correspondence only where the dual-path producer itself
documents it, e.g. `ScopedEvaluator._episode_fact_row` writes
`energy_total_wh -> total_energy_wh` and
`collision_amount_wall + collision_amount_static -> total_collisions_static`.
Multiple entries per inventory key are allowed (one legacy value can feed two fact
tables). The producer function is named in each `note`.

`expected_diff.yaml` entries are consulted only when values differ, and each must
carry a reason. `unit_scale` entries are verified arithmetically before they are
accepted; `derived` entries are verified inside a reviewed relative bound;
`ground_truth` is the reviewed allowlist for odometry-to-`/tf` changes; `null_fill`
documents the removal of `fill_null(0.0)` masking; `dtype` documents a dtype-only
change. Cross-cutting notes (nullability of `dim_episodes.crowd_condition`, the
`fill_null(0.0)` -> null + `null_reason` policy, the odometry-to-`/tf` source move)
are recorded under `notes:`.

## Gap ownership

Keys without a reviewed mapping are the port gaps. They are triaged as

* `port-social` -- category `social`, `safety`, `edge_case`, `holistic`
* `port-eco` -- category `ecological`, `performance`, `naturalness`
* `parity-g1` -- keys whose legacy column is a list/struct (timeseries,
  trajectory, edge-case provenance) and therefore needs a G1 decision about its
  1NF/series star representation before a scalar mapping can exist

The inventory's original owner (`Agent 4 (parity)` / `Agent 7 (perf)`) is kept in
`gaps.csv` as `inventory_owner` for traceability.

## Current result

See `results/summary.yaml`, `results/certificate.csv` and `results/gap_summary.md`.
The certificate is produced from the committed branch pipeline on the scratch
benchmark described in `results/summary.yaml`; the number of `MISSING_STAR` rows is
the size of the remaining port, not a harness failure.
