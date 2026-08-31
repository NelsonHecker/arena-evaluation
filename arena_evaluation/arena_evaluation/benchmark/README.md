# benchmark

Runs multi-planner benchmark campaigns: spawns simulation environments, drives episodes, records results, and manages run state for pause/resume.

## Files

| File | Purpose |
|---|---|
| `runner.py` | `BenchmarkRunner`: spawns environments, executes episodes via `RunEpisode`, writes `progress.csv` and `.benchmark_state.json` |
| `config.py` | `Suite` and `Contest` parsers: YAML schema, sweep expansion, inline contest resolution |
| `state.py` | `Manifest`, `RunDir`, `StateFile`: run manifest, config hash, resume discovery, git SHA capture |
| `step.py` | `Step`, `StepResult`, `StepErrorKind`: grid model for contestant x stage combinations |
| `debug.py` | Process introspection: `running_processes()`, `tail_console()`, `console_log_path()` |
| `profiler.py` | `PipelineProfiler`: CPU, GPU, RAM, duration metrics |
## CLI (`arena evaluation`)

```bash
arena evaluation list                          # List all benchmark runs
arena evaluation status [<run_id>]             # Show run progress (--watch for live)
arena evaluation tail [<run_id>]               # tail -F progress.csv
arena evaluation ps                            # List running arena OS processes
arena evaluation kill [<pid>...] [-9]          # Terminate running arena/sim processes (-9 for SIGKILL)
arena evaluation console [<run_id>]            # Tail runner.log (--follow for streaming)
arena evaluation benchmark                     # Launch a benchmark campaign
```

## Execution Flow

The runner takes a suite (ordered stages) and a contest (planner lineup). It creates a step grid as the Cartesian product of contestants and stages, groups consecutive steps by `(contestant, robot, simulator)`, and spawns one environment per group.

For each group:
1. Calls `/arena/spawn_env` with launch arguments derived from the first step.
2. Waits for environment registration on `/arena/state/envs`.
3. For each step: calls `QueueEpisode` to set stage config, then executes `RunEpisode` action.
4. Despawns the environment and proceeds to the next group.

Run output structure in `$ARENA_DATA_DIR/benchmarks/<run_id>/`:
```
<run_id>/
|-- manifest.yaml              # Config snapshot
|-- progress.csv               # Append-only, one row per episode
|-- runner.log                 # Python logging and launch output
|-- .benchmark_state.json      # Per-step status
|-- episodes/                  # One MCAP per episode
|   |-- episode_000/
|   |   |-- episode_000.mcap
|   |   `-- episode_000.yaml
|   `-- ...
|-- combined_metrics.parquet   # Processed metrics
`-- report_manifest.yaml       # Used report manifest
```

## Programmatic Usage

```python
from arena_evaluation.benchmark.runner import BenchmarkRunner
from arena_evaluation.benchmark.config import Suite, Contest

suite = Suite.parse("basic", yaml.safe_load(suite_yaml))
contest = Contest.parse("basic", yaml.safe_load(contest_yaml))
runner = BenchmarkRunner(suite, contest, headless=True, env_n=4)
runner.run()
```

## Run State

`StateFile` manages `.benchmark_state.json` atomically. Each step has status: `ok | partial | failed | skipped | in_progress`. Interrupted runs resume with `--resume <run_id>`.

Run ID format: `{YYYYMMDD-HHMMSS}-{suite}-{contest}`.

