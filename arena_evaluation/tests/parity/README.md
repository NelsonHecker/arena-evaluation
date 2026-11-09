# G1 parity certificate (`tests/parity`)

Evidence that the star-schema lakehouse reproduces the legacy `combined_metrics.parquet`
columns before the legacy evaluation path is removed. This directory holds the
harness, its reviewed data, and the certificate artefacts produced from real
dual-path runs.

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
| `UNMAPPED` | no reviewed mapping, but a star column with that name exists in the run: manifest review needed |
| `NO_PRODUCER` | no reviewed mapping and no produced star column: a real porting gap |
| `MISSING_LEGACY` | a mapped legacy column is absent from this run's frame |
| `NULL_REASON` | legacy value is `0.0`, star value is null and `null_reason` explains it |
| `RELOCATED_SERIES` | legacy per-sample list moved to a 1NF series table; no scalar fact column is intended |
| `CERT_NULL_ONLY` | every compared episode was null on both sides; no real comparison |
| `INSUFFICIENT_EVIDENCE` | fewer than `--min-nonnull-episodes` non-null comparisons |
| `MISMATCH` | unexplained difference |

The certificate row also carries `null_compared` and `nonnull_compared`; `EXACT` and
`WITHIN_TOL` are only issued when at least `--min-nonnull-episodes` (default 3) pairs
were non-null on both sides. `CERT_NULL_ONLY`, `INSUFFICIENT_EVIDENCE` and `MISMATCH`
are gate-failing statuses (exit code 1): a certificate cannot go green on null==null.

`MISMATCH` is deliberately outside the accepted set: the gate fails even though the
row is still reported, so a first certificate can be produced without hiding a
regression. Exit code `2` means the input is not a dual-path run or a manifest is
invalid.

No inventory metric maps to `dim_episodes`: the 267 keys are all measurement columns.
The dimension's contribution to the certificate is the join key itself; its
nullability changes are recorded as notes in `expected_diff.yaml`.

## Files

| file | role |
| --- | --- |
| `certify.py` | generator (not a unittest); `--self-check` exercises the comparison engine on synthetic frames |
| `inventory_254.csv` | the 267 legacy metric keys (267 rows; the historical `254` name is kept for continuity) |
| `metric_map.yaml` | reviewed correspondence: inventory key -> star table/column, list reduction, legacy combine |
| `expected_diff.yaml` | reviewed allowlist of legitimate value changes, each with a reason |
| `results/<run>/certificate.csv` | the certificate for one run |
| `results/<run>/gaps.csv` | NO_PRODUCER porting gaps, with the triaged owner |
| `results/<run>/unmapped.csv` | keys whose star column exists but whose correspondence is unreviewed, with the candidate table/column |
| `results/<run>/gap_summary.md` | the gap list grouped by owner, calculator and category |
| `results/<run>/star_coverage.csv` | numeric star metric columns not claimed by any reviewed inventory mapping |
| `results/<run>/summary.yaml` | machine-readable counts and provenance |
| `results/<run>/relocated_series.csv` | reviewed list of keys whose per-sample data moved to a 1NF series table |
| `results/<run>/structural_replacements.csv` | new star keys that are not parity keys |
| `results/<run>/list_violations.csv` | legacy list columns knowingly still not in 1NF |

Only identity-grain targets are claimed in this revision: `fact_episodes` and
`fact_robots` hold one row per episode and robot, so a legacy episode value maps
directly. `fact_encounters`, `fact_contacts` and `fact_pedestrians` need a reviewed
reduction per legacy episode aggregate (count/min/max/sum over rows); they are not
claimed yet and their columns show up in `star_coverage.csv` until a reviewer adds
the reduction entries.

## Running it

```bash
python arena_evaluation/tests/parity/certify.py --benchmark-dir /path/to/benchmark \
  --output-dir arena_evaluation/tests/parity/results/<run>
python arena_evaluation/tests/parity/certify.py --self-check
```

`--self-check` builds synthetic legacy and star frames and asserts `EXACT`,
`EXPECTED_DIFF` (unit scale), `NULL_REASON`, `MISSING_STAR` and the gap list, so the
comparison engine can be verified without a benchmark.

## Finding a real input

Historic runs predate the star schema. `bench-live-social-dwb`, the `20260914-*`
runs and the frozen `data/_parity_baseline/20261109` parquets carry only the legacy
side, so the harness refuses them (`ParityInputError`, exit 2) with the reprocess
hint. The frozen baseline is the legacy reference for a later replay, not a
certificate input.

Two real dual-path inputs were used:

1. **`/opt/arena_ws/data/benchmarks/test-eval3-agent-mcp`** — an existing run whose
   dated artefacts are `combined_metrics.parquet` + `dim_episodes.parquet` +
   `metrics/fact_*.parquet`, all five episodes. This proves the harness works on a
   run it did not produce.
2. **A scratch benchmark reprocessed from a historic run** —
   `parity-g1-roster-ped6`, six `ped6` episode directories copied from
   `20260914-101024-roster_crowd_density-full_suite_planners` (episode 029/032/034/
   038/042/045, six distinct route stages) plus the run's `manifest.yaml`.

The verified reprocessing recipe is:

1. Copy at most ten episode directories into a scratch benchmark directory. An
   episode directory is self-contained (`episode_NNN.mcap` + `episode_NNN.yaml`);
   copy the run's `manifest.yaml` too so suite-level metric overrides resolve.
2. Run the branch pipeline over the scratch benchmark:

   ```bash
   source /opt/arena_ws/source
   PYTHONPATH=<branch>/arena_evaluation arena evaluation process \
     --benchmark-dir /opt/arena_ws/data/benchmarks/<scratch> --workers 1 --force-extract
   ```

   `ProcessingPipeline.process_benchmark` is what the CLI calls: it extracts topics
   into `episodes/episode_NNN/topics/`, evaluates metrics, writes the per-episode
   `metrics.parquet` + `metrics/` star tables, then concatenates the benchmark
   `combined_metrics.parquet` and the star tables. `FolderManager.describe_episode`
   overrides the episode yaml's original `benchmark_id` with the scratch directory
   name, so the legacy `benchmark_id` and the star `run_id` agree.
3. `--force-extract` is only needed for the first pass. An episode whose
   `metrics.parquet` **and** `metrics/` already exist is skipped by
   `process_episode`; delete both to recompute metrics while keeping the cached
   `topics/`.
4. `--workers 1` avoids the spawn-manager status dict; the CLI/API default parallel
   path still writes the same artefacts.

Observed on the GPU container: six episodes, one metrics pass; the run-level
`combined_metrics.parquet` and all five fact tables are written only after every
episode has been evaluated (the pipeline concatenates per-episode star tables at the
end). A run interrupted halfway has per-episode `metrics/` star dirs but no benchmark
star tables, and the harness still refuses it (correctly).

## Reviewed manifest policy

The `UNMAPPED`/`NO_PRODUCER` split is derived from the produced artefacts: a key is
`UNMAPPED` when a column with its name exists in the run's `dim_episodes.parquet` or
`metrics/fact_*.parquet`, `NO_PRODUCER` otherwise. Gap ownership is triaged from
`NO_PRODUCER` only.

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
are recorded under `notes:`. An entry without `star_table`/`star_column` applies to
every mapping of its key and is reported as pending until that producer lands.

The manifest also routes the non-scalar work: `relocated_series` (keys whose
per-sample data moved to 1NF series tables), `structural_replacements` (new star
keys that are not parity keys) and `list_violations` (legacy list columns knowingly
still not in 1NF, kept visible in the gap list instead of silently dropped).

## Gap ownership

Keys without a reviewed mapping are the port gaps. They are triaged as

* `port-social` -- category `social`, `safety`, `edge_case`, `holistic`
* `port-eco` -- category `ecological`, `performance`, `naturalness`
* `parity-g1` -- keys whose legacy column is a list/struct (timeseries,
  trajectory, edge-case provenance) and therefore needs a G1 decision about its
  1NF/series star representation before a scalar mapping can exist

Keys in `relocated_series` are reported with the `RELOCATED_SERIES` status and are
not port gaps. Keys in `list_violations` stay gaps but their note names the known
1NF violation. The inventory's original owner (`Agent 4 (parity)` /
`Agent 7 (perf)`) is kept in `gaps.csv` as `inventory_owner` for traceability.

## Current results

| run | input | EXACT | CERT_NULL_ONLY | RELOCATED_SERIES | UNMAPPED | NO_PRODUCER | MISMATCH |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `parity-g1-merged` | 6 ped6 episodes reprocessed by the merged integration branch | 20 | 4 | 30 | 1 | 215 | 0 |
| `parity-g1-roster-ped6` | 6 ped6 episodes reprocessed by this branch | 20 | 4 | 30 | 1 | 215 | 0 |
| `test-eval3-agent-mcp` | existing dual-path run, 5 episodes | 20 | 4 | 30 | 1 | 215 | 0 |

The merged-run certificate **does not pass the gate**: four mapped keys
(`mar`, `pfi`, `min_obstacle_clearance`, `robot_path_deviation_m` on this run) are
null on both sides in all six evaluable episodes, so they report `CERT_NULL_ONLY`
instead of `EXACT`. The other 20 mapped comparisons are `EXACT` with 6 non-null
episodes each. No `MISMATCH`, no `INSUFFICIENT_EVIDENCE`.

Finding behind the small `UNMAPPED` count: the merged run's star fact tables still
carry exactly the pre-port column set (`fact_episodes` 23, `fact_encounters` 25,
`fact_contacts` 18, `fact_pedestrians` 21, `fact_robots` 16). The port commits in the
integration branch (`dcbae0e`, `00572e2`) changed calculators and the legacy
`combined_metrics` frame (325 columns, new `char_*` keys, moved series) but did not
change `storage/declarations.py` or the star row builders, so no ported key reaches a
fact column yet. The one `UNMAPPED` key is `ped_path_deflection_m`, which does exist
in `fact_pedestrians`. `NO_PRODUCER` (215) is therefore the honest porting gap at the
star-schema boundary, not a mapping missing from `metric_map.yaml`.

Per-run provenance is in `results/<run>/PROVENANCE.md`.