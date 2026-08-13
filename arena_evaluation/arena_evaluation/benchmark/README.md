# benchmark

Orchestrates multi-planner benchmark campaigns: spawns simulator environments, drives episodes, records results, and manages run state for pause/resume.

## Files

| File | Purpose |
|---|---|
| `runner.py` | `BenchmarkRunner`: spawns envs, drives episodes via `RunEpisode` action, writes `progress.csv` and `.benchmark_state.json` |
| `config.py` | `Suite` and `Contest` parsers: YAML schema, sweep expansion, inline contest resolution |
| `state.py` | `Manifest`, `RunDir`, `StateFile`: run manifest, config hash, resume discovery, git SHA capture |
| `step.py` | `Step` / `StepResult` / `StepErrorKind`: grid model for contestant x stage combinations |
| `debug.py` | Process introspection: `running_processes()`, `tail_console()`, `console_log_path()` |
| `profiler.py` | `PipelineProfiler`: per-phase CPU/GPU/RAM/duration metrics (NVML-accelerated) |
| `cli.py` | `evaluation_cli` entry point: `list`, `status`, `tail`, `ps`, `console` subcommands |

## CLI (evaluation_cli)

```bash
evaluation_cli list                          # List all benchmark runs
evaluation_cli status [<run_id>]             # Show run progress (--watch for live)
evaluation_cli tail [<run_id>]               # tail -F progress.csv
evaluation_cli ps                            # List running arena OS processes
evaluation_cli console [<run_id>]            # Tail runner.log (--follow for streaming)
```

## How It Works

The runner takes a suite (ordered stages) and a contest (planner lineup). It generates a step grid as the cartesian product of contestants x stages, groups consecutive steps by `(contestant, robot, simulator)`, and spawns one env per group.

For each group:
1. Calls `/arena/spawn_env` with launch args derived from the first step
2. Waits for env registration on `/arena/state/envs`
3. For each step: calls `QueueEpisode` to set stage config, then drives `RunEpisode` action
4. Despawns the env and advances to the next group

Run output lands in `$ARENA_DATA_DIR/benchmarks/<run_id>/`:
```
<run_id>/
|-- manifest.yaml              # config snapshot (never overwritten)
|-- progress.csv               # append-only, one row per episode
|-- runner.log                 # python logging + launch output
|-- .benchmark_state.json      # per-step status (atomic write)
|-- episodes/                  # one MCAP per episode
|   |-- episode_000/
|   |   |-- episode_000.mcap
|   |   `-- episode_000.yaml
|   `-- ...
|-- combined_metrics.parquet   # after processing
`-- report_manifest.yaml       # note: which manifest was used
```

## Programmatic Use

```python
from arena_evaluation.benchmark.runner import BenchmarkRunner
from arena_evaluation.benchmark.config import Suite, Contest

suite = Suite.parse("basic", yaml.safe_load(suite_yaml))
contest = Contest.parse("basic", yaml.safe_load(contest_yaml))
runner = BenchmarkRunner(suite, contest, headless=True, env_n=4)
runner.run()
```

## Run State

`StateFile` manages `.benchmark_state.json` with atomic writes. Each step has status: `ok | partial | failed | skipped | in_progress`. Interrupted runs resume with `--resume <run_id>`.

Run ID format: `{YYYYMMDD-HHMMSS}-{suite}-{contest}` (lex sort = chronological). Override with `--run-id`.
