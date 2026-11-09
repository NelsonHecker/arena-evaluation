# Certificate provenance

- run_id: `parity-g1-roster-ped6`
- benchmark directory in the GPU container: `/opt/arena_ws/data/benchmarks/parity-g1-roster-ped6`
- source episodes: six `stage_1_ped6_routeXX` episode directories copied from
  `data/benchmarks/20260914-101024-roster_crowd_density-full_suite_planners`
  (`episode_029`, `episode_032`, `episode_034`, `episode_038`, `episode_042`,
  `episode_045`) plus that run's `manifest.yaml`
- pipeline and harness: commit `7b44a8c` on `enhancement/eval-parity-g1`
  (base `1f744fd` of `enhancement/eval-integration` merged in)
- command (equivalent to `arena evaluation process --benchmark-dir ... --workers 1`):

  ```bash
  PYTHONPATH=<branch>/arena_evaluation python3 -c "
  import pathlib
  from arena_evaluation.storage.folder_manager import FolderManager
  from arena_evaluation.processing.pipeline import ProcessingPipeline
  ProcessingPipeline(FolderManager(data_root=pathlib.Path('/opt/arena_ws/data')), workers=1).process_benchmark('parity-g1-roster-ped6')
  "
  ```

- both artefacts were written by the same metric pass: `combined_metrics.parquet`
  (24 rows: 6 evaluated robots + 18 no-trajectory robot rows) and the star tables
  (6 fact_episodes/fact_robots rows, 5 fact_contacts rows, 36 fact_pedestrians rows,
  0 fact_encounters rows).
- the `topics/` cache was produced by the first reprocessing pass (pipeline commit
  `72c22e5`); the metric and star pass above was rerun on the merged base so the
  certificate reflects the current pipeline behaviour.
- certificate: 20 `EXACT`, 4 `CERT_NULL_ONLY`, 30 `RELOCATED_SERIES`, 1 `UNMAPPED`,
  215 `NO_PRODUCER`, 0 `MISMATCH`. Gate: failed on the four null-only keys.
