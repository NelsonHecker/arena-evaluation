# arena_evaluation_msgs

ROS 2 message and service definitions for the Arena evaluation system.

## Messages

### BenchmarkState

Published on `/arena/benchmark/state` (TRANSIENT_LOCAL, RELIABLE). Carries live benchmark progress.

| Field | Type | Description |
|---|---|---|
| `stamp` | `builtin_interfaces/Time` | Publication timestamp |
| `run_id` | `string` | Benchmark run identifier |
| `suite` | `string` | Suite name |
| `contest` | `string` | Contest name |
| `simulator` | `string` | Simulator name (empty when `--sim` not given) |
| `env_n` | `uint32` | Parallel env cap |
| `headless` | `bool` | Headless flag |
| `steps_total` | `uint32` | Total steps in the grid |
| `steps_done` | `uint32` | Steps with status `ok` |
| `steps_partial` | `uint32` | Steps with status `partial` |
| `steps_failed` | `uint32` | Steps with status `failed` |
| `steps_skipped` | `uint32` | Steps with status `skipped` |
| `steps_in_flight` | `uint32` | Steps with status `in_progress` |
| `active_keys` | `string[]` | Currently-running step keys |

## Services

### RecordEpisode

Authoritative episode lifecycle control for the `DataRecorderNode`. Driven by the benchmark runner.

**Request:**
| Field | Type | Description |
|---|---|---|
| `command` | `uint8` | `COMMAND_START=1` or `COMMAND_STOP=2` |
| `episode_id` | `uint32` | Episode number |
| `outcome_state` | `uint8` | Terminal outcome (only for STOP): `SUCCESS=2`, `FAILED=3`, `SKIPPED=4`, `FATAL=5` |
| `outcome_info` | `string` | Outcome detail string |

**Response:**
| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the command was processed |
| `message` | `string` | Status message |

### ChangeDirectory

Service to update the recorder's output directory at runtime.

**Request:** `data` (`string`), the new directory path
**Response:** `result` (`bool`), success flag

## Build

```bash
# Built as part of arena_evaluation via CMake
arena build arena_evaluation
```

## Dependencies

- `builtin_interfaces` for the `Time` stamp
- `rosidl_default_generators` for message code generation
