# Benchmark configs

The `arena evaluation benchmark` runner reads all benchmark configuration from
`arena_evaluation/configs/benchmark/` at startup.

Invocation: `arena evaluation benchmark --suite <name> --contest <name> [--scale-episodes N] [sim:=...] [headless:=...] [env_n:=...]` (Arena meta-repo shortcut: `arena evaluation benchmark ...`).

## Directory layout

```
configs/benchmark/
├── suites/           — stage sequences (maps, episodes, task modes)
│   ├── basic.yaml
│   ├── meta_suite.yaml
│   ├── all_maps_random.yaml
│   ├── arena_corridor.yaml
│   ├── arena_hospital_small.yaml
<<<<<<< HEAD
│   ├── compliance.yaml
│   └── map_empty.yaml
=======
│   ├── map_empty.yaml
│   └── characterization.yaml   — open-loop energy/acoustic sweep
>>>>>>> origin-fork/jazzy
└── contests/         — planner lineups
    ├── basic.yaml
    ├── allplanners.yaml
    ├── inter.yaml
    ├── planners.yaml
    └── characterization.yaml  — dummy contestant (the task mode drives, not a planner)
```

## Suite files

A suite is an ordered list of stages. The runner steps through the
stages sequentially, cycling through all contestants at each stage.

```yaml
stages:
  - name: scenario            # human-readable label (used in log output)
    map: arena_hospital_small # world/map name
    robot: jackal             # robot model
    tm_robots: scenario       # TM_Robots kind (string, upper-cased to enum key)
    tm_obstacles: random      # TM_Obstacles kind
    episodes: 1               # number of episodes at this stage
    config:                   # per-mode params, leaf-keyed (see task_generator/tasks/obstacles/README.md)
      scenario:               # top-level key matches tm_robots / tm_obstacles
        file: 4.json          # → task.scenario.file
      random:
        dynamic:  {min: 3, max: 5, models: [arenian]}  # → task.random.dynamic.{min,max,models}
        static:   {min: 5, max: 10, models: [shelf]}
```

### Stage fields

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Stage label |
| `map` | string | World name (sets `Arena.WORLD` param on the task-generator node) |
| `robot` | string | Robot model |
| `tm_robots` | string | `Constants.TaskMode.TM_Robots` enum key (case-insensitive) |
| `tm_obstacles` | string | `Constants.TaskMode.TM_Obstacles` enum key (case-insensitive) |
| `episodes` | int | Episode count (scaled by `scale_episodes` launch arg) |
| `config` | dict | Per-mode params; top-level keys must match `tm_robots`/`tm_obstacles` (e.g. `scenario`, `random`). Inner leaves map to `task.<mode>.<leaf>` via QueueEpisode (see [task_generator/tasks/obstacles/README.md](../../../../task_generator/task_generator/tasks/obstacles/README.md)) |
| `seed` | int | Auto-derived from a SHA-1 hash of the stage fields (excluding `config`); can be set explicitly |
| `timeout` | string | Per-episode timeout; defaults to `Constants.Robot.TIMEOUT` if absent |

<<<<<<< HEAD
`suites/compliance.yaml` targets the worlds annotated with zone/door semantics
(`hospital_1`, `reception`, `three_storied_residential`), so the compliance
metric columns (`speed_zone_violations`, `speed_zone_violation_seconds`,
`quiet_zone_dwell_seconds`, `restricted_zone_entries`,
`doorway_blocking_time`) land in reports. Run it against any existing contest,
e.g. `ros2 run arena_evaluation benchmark --suite compliance --contest basic`.
=======
Suite-level `references: false` disables the automatically generated reference steps
(`unobstructed_robot` / `unhindered_peds`); characterization suites use it because the sweep
itself is the reference.

### Characterization suite

```yaml
references: false
stages:
  - name: characterization
    map: map_empty
    robot: jackal
    episodes: 3            # repetitions → cross-episode confidence bands
    tm_robots: characterization   # TM_Robots mode that drives the open-loop sweep
    tm_obstacles: random
    config:
      random:
        dynamic: {min: 0, max: 0}
        static: {min: 0, max: 0}
        interactive: {min: 0, max: 0}
    timeout: 300s
```

The `characterization` robot task mode (in `task_generator`) drives `cmd_vel` directly through the
robot's rated envelope — idle blocks, 0.25→vx_max linear steps with 5 s out-and-back dwells,
transient ramps, pivot rates — tagging every maneuver with `characterization_phase` markers. Run it
like any benchmark and analyse with the characterization report manifest:

```bash
arena evaluation benchmark --suite characterization --contest characterization
arena evaluation run --benchmark-dir <run_id> --report-manifest characterization
```
>>>>>>> origin-fork/jazzy

## Contest files

A contest defines the set of planner configurations (contestants) to evaluate.
The runner iterates over all contestants at each suite stage.

Contestant args use **cap-scoped dicts** — each capability (e.g. `mobile`,
`arm`) is a dict with a `driver` key identifying the driver and any extra
kwargs forwarded as launch args:

```yaml
mobile:
  driver: nav2
  local_planner: teb
  inter_planner: polite
```

This produces launch args: `mobile:=nav2 mobile.local_planner:=teb mobile.inter_planner:=polite`

There are two forms: **list** and **sweep**.

### List form

Top-level YAML is a sequence. Each entry must have `name`; all other keys
become `args` forwarded to `Robot.parse` via the `SpawnRobot` service.

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

### Sweep form

Top-level YAML is a mapping. Inside a cap dict, list values are sweep axes;
non-list values are constants shared by all contestants. The runner takes the
cartesian product of all list-valued keys.

```yaml
mobile:
  driver: nav2
  local_planner: [teb, dwa, rosnav]
  inter_planner: bypass
```

produces three contestants. `name` is auto-derived from the keys that vary
across the product. If only one axis varies, names are the values of that axis.
If two or more vary, names are the varying values joined by `-`, in yaml-dict
order.

```yaml
# 4 contestants: dwa-navfn, dwa-smac, teb-navfn, teb-smac
mobile:
  driver: nav2
  local_planner: [dwa, teb]
  global_planner: [navfn, smac]
```

A constant `name: <prefix>` prepends the prefix to all auto-derived names:

```yaml
name: basic
mobile:
  driver: nav2
  inter_planner: bypass
  local_planner: [teb, dwa, rosnav]
# produces: basic-teb, basic-dwa, basic-rosnav
```

`description` (optional) is stored in the contest manifest and is not forwarded
to `Robot.parse`.

### Backward compatibility

The old flat dot-notation format is still accepted in both list and sweep forms:

```yaml
# Old flat list form — still works
- name: teb
  mobile.local_planner: teb
  mobile.inter_planner: bypass

# Old flat sweep form — still works
mobile.local_planner: [teb, dwa]
mobile.inter_planner: bypass
```

### Inline contest (CLI)

Pass the YAML inline as the `--contest` value when the string starts with `[`
or `{`:

```
arena evaluation benchmark --suite basic --contest '[{name: teb, mobile: {driver: nav2, local_planner: teb}}]'
arena evaluation benchmark --suite basic --contest '{mobile: {driver: nav2, local_planner: [teb, dwa]}}'
```

### Contestant args

Contestant `args` keys are forwarded verbatim as launch args to the env on
spawn (so nav2, the controller, the agent, etc. come up correctly from the
start). See
[BRINGUP.md → Cap-scoped overrides](../../../../arena_bringup/BRINGUP.md#cap-scoped-overrides)
for the recommended key shapes.

The runner drops keys that collide with stage-owned launch args (`sim`,
`robot`, `world`, `tm_robots`, `tm_obstacles`, `run_seed`, `auto_reset`,
`tm_modules`, `record_data_dir`) and logs a warning, since those are
controlled by the suite stage. Anything else is passed through to the launch
layer, which binds it if declared or raises an error if not.

A useful passthrough example: `fail_on_collision: true` makes the env abort
an episode as FAILED (`outcome_info='collision'`) the moment the robot
footprint contacts a wall, static obstacle, or pedestrian, instead of the
default run-to-goal-or-timeout. See
[BRINGUP.md → Common options](../../../../arena_bringup/BRINGUP.md#common-options).

## How the runner consumes these files

The runner is the `benchmark` console script: `arena evaluation benchmark`,
no launch file. In the Arena meta-repo, install the feature first with
`arena feature evaluation install`.

Steps are grouped by `(contestant.name, stage.robot, simulator)`. Consecutive
steps with the same key share one env; suite order is preserved within and
across groups. Robot is fixed within a group. If a contestant's stages mix
robots, the runner splits into multiple groups per contestant. Authoring
suggestion: keep one robot per contestant for fastest runs.

`env_n` caps how many groups (parallel contestants) run at once, with the rest
queued. Run time scales as `bringup_time × num_groups + episode_time ×
total_episodes` spread across the `env_n` workers, not `bringup_time ×
num_steps`, since steps within a group reuse one env.

For each group the runner:

1. Calls `/arena/spawn_env` once with the first step's launch args: `sim`,
   `robot`, `world`, `tm_robots`, `tm_obstacles`, `run_seed`,
   `auto_reset:=false`, `tm_modules:=` (empty), and any contestant args of
   shape `mobile`, `arm`, `mobile.<key>`, or `arm.<key>`. `record_data_dir`
   is added when recording is enabled.
   Per-mode params (`task.scenario.file`, `task.random.*`, ...) are not passed
   as launch args; the runner sets them via QueueEpisode in step 3.
2. Waits for the env to publish on `/arena/state/envs` and resolves the env
   namespace from `EnvRecord.fqn`. Sets up a `RunEpisode` action client and a
   `QueueEpisode` service client at `<env_ns>/config/queue_episode`.
3. For each step in the group:
   - Calls `<env_ns>/config/queue_episode` with the stage's `world`,
     `tm_robots`, `tm_obstacles`, and the per-mode `config` blocks routed to
     `obstacles_params` / `robots_params` as leaf-keyed `Parameter[]` (e.g.
     `file`, `dynamic.min`). MERGE semantics: empty fields leave the prior
     queued value untouched. Called for every step including the first; the
     env owns no stage-specific config until the runner pushes it.
   - Drives `step.episodes` goals via the `RunEpisode` action at
     `<env_ns>/lifecycle/run_episode`, with `goal.world = stage.map` and
     `goal.seed = stage.seed` overriding the queued world/seed if needed.
     Per-episode `EpisodeRecord` rows are pulled from `<env_ns>/state/episode`
     and appended to `progress.csv`.
   - Per-step failures (timeout, missing EpisodeRecord, mid-run robot stuck)
     advance to the next step; only systemic setup failures
     (`error_kind in {env_setup, robot_setup}`) **before any episode in the
     run has completed** trigger a run-level abort.
4. Despawns the env via `/arena/despawn_env` at the end of the group, then
   advances to the next group.
5. Writes `progress.csv` and `.benchmark_state.json` to
   `$ARENA_DATA_DIR/benchmarks/<run_id>/` so an interrupted run can be
   resumed with `--resume <run_id>`.

`progress.csv` schema is unchanged (one row per episode; `env_id` is shared
within a group). `.benchmark_state.json` schema is unchanged.

Run-id default format: `{YYYYMMDD-HHMMSS}-{suite}-{contest}` (lex sort = chronological).
For inline contests the contest segment is `inline`. Override with `--run-id`.

Output dir: `$ARENA_DATA_DIR/benchmarks/<run_id>/` (default `$ARENA_WS_DIR/data/benchmarks/<run_id>/`).
Override with `--data-root`. Inside Docker: `/opt/arena_ws/data/benchmarks/<run_id>/`.

```
$ARENA_DATA_DIR/benchmarks/<run_id>/
├── manifest.yaml              # requested config snapshot (never overwritten)
├── progress.csv               # append-only, one row per episode
├── runner.log
├── .benchmark_state.json      # per-step status, atomic write
├── episodes/                  # recorder output — one MCAP per episode
│   ├── episode_000/
│   │   ├── episode_000.mcap
│   │   └── episode_000.yaml
│   └── ...
├── combined_metrics.parquet   # after arena evaluation run
└── report_manifest.yaml       # note: which manifest produced the last report
```

The recorder is spawned per step and writes one MCAP per episode into `episodes/`; its episode
lifecycle is driven by the `start_episode` service (see the [ingestion README](../../arena_evaluation/ingestion/README.md)).
`arena evaluation run --report-manifest <name>` performs extraction + metrics, then renders
`report.html`. Characterization is a regular metric calculator — the report derives the
per-working-point curves from its `timeseries_char_*` columns, with no separate analysis step.

Inspection helpers:

- `arena evaluation evaluation_cli list` — table of all runs under `$ARENA_DATA_DIR/benchmarks/`.
- `arena evaluation evaluation_cli status [run_id] [--watch]` — snapshot or live view (subscribes to `/arena/benchmark/state` TRANSIENT_LOCAL).
- `arena evaluation evaluation_cli tail [run_id]` — tail -F on `progress.csv` of the most recent (or named) run.

`progress.csv` schema: `ts_iso, run_id, step_key, contestant, stage, env_id, episode_id, world, seed, tm_robots, tm_obstacles, tm_modules, robots, outcome_state, outcome_info, started_at, ended_at, runtime_s, robots_params_json, obstacles_params_json, error_kind, error_detail`

Step status values in `.benchmark_state.json`: `ok | partial | failed | skipped | in_progress`.

## Known limitations

- Recorder topic-namespace mismatch (`/scenario_reset` vs `<ns>/task_reset`)
  means per-episode segmentation in CSV output is currently unreliable;
  aggregate-per-run metrics are still usable. Tracked separately.
- Heartbeat-eviction may rarely mark a stalled step `ok` instead of `failed`.
  If you see suspiciously fast steps in `progress.csv`, use
  `arena evaluation benchmark --resume <run_id> --retry-failed`.
