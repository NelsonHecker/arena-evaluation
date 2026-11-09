# Certificate provenance

- run_id: `test-eval3-agent-mcp`
- benchmark directory in the GPU container: `/opt/arena_ws/data/benchmarks/test-eval3-agent-mcp`
- input: an existing dual-path run (5 episodes, `map_empty`, planner `dwb`) whose
  `combined_metrics.parquet`, `dim_episodes.parquet` and `metrics/fact_*.parquet`
  were written by the pipeline current on 2026-11-08, before this branch existed
- harness: commit `7b44a8c` on `enhancement/eval-parity-g1`
- command:

  ```bash
  python3 arena_evaluation/tests/parity/certify.py \
    --benchmark-dir /opt/arena_ws/data/benchmarks/test-eval3-agent-mcp \
    --output-dir arena_evaluation/tests/parity/results/test-eval3-agent-mcp
  ```

- this run proves the harness measures a dual-path run it did not produce; the
  pipeline commit that wrote the artefacts is not the branch HEAD, so use
  `results/parity-g1-merged/` as the current merged-branch certificate.
- certificate: 20 `EXACT`, 4 `CERT_NULL_ONLY`, 30 `RELOCATED_SERIES`, 1 `UNMAPPED`,
  215 `NO_PRODUCER`, 0 `MISMATCH`. Gate: failed on the four null-only keys.
