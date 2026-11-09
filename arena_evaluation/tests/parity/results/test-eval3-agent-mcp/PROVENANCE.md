# Certificate provenance

- run_id: `test-eval3-agent-mcp`
- benchmark directory in the GPU container: `/opt/arena_ws/data/benchmarks/test-eval3-agent-mcp`
- input: an existing dual-path run (5 episodes, `map_empty`, planner `dwb`) whose
  `combined_metrics.parquet`, `dim_episodes.parquet` and `metrics/fact_*.parquet`
  were written by the pipeline current on 2026-11-08, before this branch existed
- harness: commit `46d79eb` on `enhancement/eval-parity-g1`
  (base `1f744fd` of `enhancement/eval-integration` merged in)
- command:

  ```bash
  python3 arena_evaluation/tests/parity/certify.py \
    --benchmark-dir /opt/arena_ws/data/benchmarks/test-eval3-agent-mcp \
    --output-dir arena_evaluation/tests/parity/results/test-eval3-agent-mcp
  ```

- this run proves the harness measures a dual-path run it did not produce; the
  pipeline commit that wrote the artefacts is not the branch HEAD, so use
  `results/parity-g1-roster-ped6/` as the current-pipeline certificate.
- certificate status: 24 `EXACT`, 30 `RELOCATED_SERIES`, 216 `MISSING_STAR`,
  0 `MISMATCH`.
