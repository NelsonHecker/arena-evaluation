# Certificate provenance

- run_id: `parity-g1-merged`
- benchmark directory in the GPU container: `/opt/arena_ws/data/benchmarks/parity-g1-merged`
- input: the six `stage_1_ped6_routeXX` episodes reprocessed by the merged
  integration branch (`enhancement/eval-integration` tip `1480551`, including
  `dcbae0e` ecological/performance/naturalness and `00572e2` social/safety/edge-case/
  holistic calculator ports, `c26f2f3` this branch). 24 legacy rows: 6 `evaluated`
  robots (`env_0_jackal`) plus 18 `no_trajectory` rows for the other three envs,
  whose odometry window is empty (the legacy odometry gate).
- harness: commit `7b44a8c` on `enhancement/eval-parity-g1`
- command:

  ```bash
  python3 arena_evaluation/tests/parity/certify.py \
    --benchmark-dir /opt/arena_ws/data/benchmarks/parity-g1-merged \
    --output-dir arena_evaluation/tests/parity/results/parity-g1-merged
  ```

- certificate: 20 `EXACT` (6 non-null episodes each), 4 `CERT_NULL_ONLY`
  (`mar`, `pfi`, `min_obstacle_clearance`, `robot_path_deviation_m`: both sides null
  in all 6 compared episodes), 30 `RELOCATED_SERIES`, 1 `UNMAPPED`
  (`ped_path_deflection_m`, which exists in `fact_pedestrians`), 215 `NO_PRODUCER`,
  0 `MISMATCH`, 0 `INSUFFICIENT_EVIDENCE`. Gate: **failed** on the four null-only
  keys.
- star-schema boundary: the merged run's fact tables still carry the pre-port column
  set (`fact_episodes` 23, `fact_encounters` 25, `fact_contacts` 18,
  `fact_pedestrians` 21, `fact_robots` 16). The port commits changed calculators and
  the legacy frame (325 columns, new `char_*` keys, moved series) but not
  `storage/declarations.py` or the fact row builders, so ported keys do not reach a
  fact column yet. `NO_PRODUCER` (215) is therefore the porting gap at the
  star-schema boundary, not a missing `metric_map.yaml` entry.
