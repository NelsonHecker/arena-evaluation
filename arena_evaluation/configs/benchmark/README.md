# Benchmark Configurations

The `arena evaluation benchmark` runner reads benchmark configurations from `arena_evaluation/configs/benchmark/` at startup.

Invocation:
```bash
arena evaluation benchmark --suite <name> --contest <name> [--scale-episodes N]
```

## Directory Layout

```
configs/benchmark/
|-- suites/           - Stage sequences (maps, episodes, task modes)
|   |-- basic.yaml
|   |-- meta_suite.yaml
|   |-- all_maps_random.yaml
|   |-- arena_corridor.yaml
|   |-- arena_hospital_small.yaml
|   |-- map_empty.yaml
|   `-- characterization.yaml
`-- contests/         - Planner lineups
    |-- basic.yaml
    |-- allplanners.yaml
    |-- inter.yaml
    |-- planners.yaml
    `-- characterization.yaml
```

## Suite Files

A suite defines an ordered list of stages executed sequentially across all contestants.

```yaml
stages:
  - name: scenario
    map: arena_hospital_small
    robot: jackal
    tm_robots: scenario
    tm_obstacles: random
    episodes: 1
    config:
      scenario:
        file: 4.json
      random:
        dynamic:  {min: 3, max: 5, models: [arenian]}
        static:   {min: 5, max: 10, models: [shelf]}
```

### Stage Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Stage label |
| `map` | string | World name |
| `robot` | string | Robot model |
| `tm_robots` | string | Task mode for robots (`scenario`, `random`, `characterization`) |
| `tm_obstacles` | string | Task mode for obstacles (`scenario`, `random`) |
| `episodes` | int | Episode count (scaled by `--scale-episodes`) |
| `config` | dict | Mode parameters forwarded via `QueueEpisode` |
| `seed` | int | Random seed (auto-derived if omitted) |
| `timeout` | string | Per-episode timeout (e.g. `60s`) |

## Contest Files

A contest defines the set of planner configurations (contestants) to evaluate.

### List Form

```yaml
- name: teb-polite
  mobile:
    driver: nav2
    local_planner: teb
    inter_planner: polite
- name: dwa-rl
  mobile:
    driver: rosnav_rl
    agent: my_agent
```

### Sweep Form

Cartesian product generated across list-valued parameters:

```yaml
name: basic
mobile:
  driver: nav2
  inter_planner: bypass
  local_planner: [teb, dwa, rosnav]
```

Produces: `basic-teb`, `basic-dwa`, `basic-rosnav`.

### Inline Contest (CLI)

```bash
arena evaluation benchmark --suite basic --contest '[{name: teb, mobile: {driver: nav2, local_planner: teb}}]'
arena evaluation benchmark --suite basic --contest '{mobile: {driver: nav2, local_planner: [teb, dwa]}}'
```

## Runner Outputs

Output directory: `$ARENA_DATA_DIR/benchmarks/<run_id>/`

```
<run_id>/
|-- manifest.yaml              # Config snapshot
|-- progress.csv               # Episode results
|-- runner.log                 # Execution logs
|-- .benchmark_state.json      # Step state (atomic)
|-- episodes/                  # Per-episode MCAP files
|-- combined_metrics.parquet   # Computed metrics
`-- report_manifest.yaml       # Used report manifest
```

## Inspection & Management Commands
```bash
arena evaluation list                          # List runs
arena evaluation status [run_id] [--watch]     # Show progress
arena evaluation tail [run_id]                 # Tail progress.csv
arena evaluation ps                            # List running arena processes
arena evaluation kill [pid...] [-9]            # Terminate running processes (-9 for SIGKILL)
arena evaluation console [run_id] [--follow]   # Tail runner.log
```

