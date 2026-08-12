# arena_evaluation

> **ROS 2 package** — Record, process, and visualise navigation and ecological evaluation metrics for Arena robots.

The package provides a complete, end-to-end evaluation pipeline that turns live simulation data into structured metric reports. It is built around a layered architecture that separates recording, processing, and presentation concerns so that each layer can be used, replaced, or extended independently.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Arena Simulation                           │
│   Gazebo  ──  task_generator  ──  nav2  ──  planner / task mode    │
└────────────────────────┬────────────────────────────────────────────┘
                         │ ROS 2 topics
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Ingestion  (ingestion/)                                  │
│  DataRecorderNode writes ONE MCAP file per episode into            │
│  benchmark/episodes/episode_XXX/. The episode lifecycle (start and │
│  stop) is driven authoritatively by the benchmark runner through   │
│  the `start_episode` service — the EpisodeRecord topic is used     │
│  for metadata enrichment only.                                     │
│  Output: episodes/episode_XXX/episode_XXX.mcap + .yaml            │
└────────────────────────┬────────────────────────────────────────────┘
                         │ MCAP file (one per episode)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Processing  (processing/)                                │
│  MCAPReader → TopicParquetStore → TopicAligner → MetricRegistry    │
│  → ParquetStore                                                     │
│  Reads each episode MCAP offline (no live ROS required), aligns    │
│  multi-rate topics onto the odom axis (no splitting needed — one   │
│  MCAP == one episode), and computes all metrics.                   │
│  characterization/ adds open-loop energy/acoustic analysis.       │
│  Output: metrics.parquet + combined_metrics.parquet               │
└────────────────────────┬────────────────────────────────────────────┘
                         │ Parquet files
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5 — Presentation  (presentation/)                            │
│  ReportBuilder reads the parquet selected by a DECLARATIVE         │
│  manifest (configs/benchmark/manifests/*.yaml) and generates an    │
│  interactive HTML report (trajectory time-sliders, line charts     │
│  with confidence bands), static PNG plots, and optional GIFs.      │
│  Output: report.html + plots/*.png (+ plots/*.gif)                 │
└─────────────────────────────────────────────────────────────────────┘
```

> **Note:** Layers are numbered 1, 3, 5 to align with the SRD which reserves Layer 2 (Storage/Folder Management) as a shared infrastructure module and Layer 4 (LLM Orchestration) for a later phase.

---

## Directory Structure

```
arena_evaluation/
├── arena_evaluation/          ← Python package root
│   ├── ingestion/             ← Layer 1: Live recording (ROS node)
│   │   ├── recorder.py        ← DataRecorderNode (service-driven episode lifecycle)
│   │   ├── metadata.py        ← IngestionMetadata (full per-episode yaml context)
│   │   └── topics.py          ← Topic definitions and type registry
│   ├── storage/               ← Layer 2: Shared schemas and path management
│   │   ├── schemas.py         ← Pydantic models (RunMetadata, PlotSpec, VizManifest, TopicBundle, …)
│   │   ├── folder_manager.py  ← Path resolution and run discovery
│   │   ├── manifest.py        ← MetadataWriter (YAML read/write)
│   │   ├── planner_names.py   ← split_planner_name (shared by ingestion + presentation)
│   │   ├── data_root.py        ← benchmarks_root() and latest_benchmark() helpers
│   │   └── exceptions.py      ← Domain-specific exceptions
│   ├── processing/            ← Layer 3: Offline metric computation
│   │   ├── mcap_reader.py     ← Reads MCAP → TopicBundle (incl. acoustics, characterization_phase)
│   │   ├── topic_aligner.py   ← Aligns multi-rate topics via join_asof (unprefixed columns)
│   │   ├── parquet_store.py   ← Reads/writes the topic cache and metrics
│   │   ├── pipeline.py        ← ProcessingPipeline orchestrator
│   │   ├── map_registry.py    ← Discovers and caches map images (PGM→PNG)
│   │   └── metrics/           ← Pluggable metric calculators (performance/, social/, ecological/, naturalness/)
│   ├── presentation/          ← Layer 5: Report and plot generation
│   │   ├── report_builder.py  ← Generates report.html (data-source aware, declarative groups/summary)
│   │   ├── manifest_registry.py ← Resolves named/inline/path manifests (like suites/contests)
│   │   ├── viz_manifest.py    ← VizManifest model + default manifest loader
│   │   ├── plotly_renderer.py ← Interactive HTML chart dispatcher
│   │   ├── seaborn_renderer.py← Static PNG chart dispatcher
│   │   ├── color_utils.py     ← Global accessibility color palette (YAML-driven)
│   │   ├── dimension_detector.py ← Auto-detects varying dimensions for compound labels
│   │   ├── report_template.html.j2 ← Jinja2 HTML template
│   │   └── plot_types/        ← violin, box, bar, histogram, scatter, trajectory, radar, heatmap, timeseries, line
│   ├── benchmark/             ← Benchmark runner and CLI
│   │   ├── runner.py          ← BenchmarkRunner (orchestrates simulation)
│   │   ├── config.py          ← Suite/Contest parsing
│   │   ├── state.py           ← Run manifest, progress, resume
│   │   ├── step.py            ← Step grid model
│   │   ├── debug.py           ← Process introspection (ps, console)
│   │   ├── profiler.py        ← PipelineProfiler (CPU/GPU/RAM per phase)
│   │   └── cli.py             ← Benchmark management CLI (argparse)
│   ├── cli.py                 ← Evaluation pipeline CLI (argparse)
│   └── cli_acoustic.py        ← Acoustic analysis subcommands (list, animate, snapshot)
├── config/
│   ├── data_recorder_config.yaml  ← Topic throttle frequencies
│   ├── mcap_writer_options.yaml   ← MCAP writer tuning
│   └── color_palette.yaml         ← Accessibility color palette
├── configs/benchmark/
│   ├── suites/                ← Benchmark suite YAML definitions
│   ├── contests/              ← Contest (planner set) YAML definitions
│   └── manifests/             ← Declarative report manifests (standard, ecological, social, safety, characterization)
├── arena_evaluation_msgs/     ← ROS 2 message and service definitions
│   ├── msg/BenchmarkState.msg ← Live benchmark progress (TRANSIENT_LOCAL on /arena/benchmark/state)
│   └── srv/                   ← RecordEpisode, ChangeDirectory services
└── tests/
    ├── unit/                  ← Pure Python unit tests (no ROS required)
    ├── integration/           ← Full-pipeline tests with fixture data
    └── test_benchmark_*.py    ← Benchmark runner tests (top-level, no ROS required)
```

---

## The Full Flow

### 1. Run a benchmark (live simulation)

```bash
arena evaluation benchmark --suite basic --contest basic
```

The benchmark runner spawns envs, drives episodes, and the recorder writes one MCAP per episode into
`$ARENA_DATA_DIR/benchmarks/<run_id>/episodes/episode_XXX/`.

**Open-loop characterization** is a benchmark like any other — the `characterization` robot task mode
(in `task_generator`) drives `cmd_vel` directly through the robot's operating envelope while
`tm_robots: characterization`:

```bash
arena evaluation benchmark --suite characterization --contest characterization
```

### 2. Process + report (offline)

```bash
# Standard metrics report:
arena evaluation run --benchmark-dir <run_id>

# Characterization report (energy/acoustic profiles per working point):
arena evaluation run --benchmark-dir <run_id> --report-manifest characterization
```

Characterization is a plain metric calculator (per-episode `timeseries_char_*` columns in
`combined_metrics.parquet`); the report manifest simply selects the `metrics` data source and
derives the curves/table from those columns.

### Report manifests

Report layouts are **declarative named YAMLs** in `configs/benchmark/manifests/`, resolved like
suites/contests (`manifest_registry.py`): by name, by path, or inline `{...}` YAML. Precompiled
manifests: `standard`, `ecological`, `social`, `safety`, `characterization`.

```bash
arena evaluation run --list-manifests            # list available layouts
arena evaluation run --benchmark-dir X --report-manifest ecological
arena evaluation run --benchmark-dir X --report-manifest '{name: inline, plots: [...]}'
```

Each manifest declares its `data_source` (metrics / characterization_samples / characterization_summary),
`groups`, `summary` table, `units`, and the `plots` list. A `report_manifest.yaml` note is written into
the benchmark dir recording which manifest produced the report.

---

## CLI Reference

```
usage: arena evaluation <command> [--run-dir DIR | --benchmark-dir DIR] [--output-dir DIR]
                            [--workers N] [--report-manifest NAME|PATH|{...}] [--list-manifests]

Commands:
  extract   Layer 3: Extract topics from MCAP into fast Parquet files (cache)
  process   Layer 3: Compute metrics and write metrics.parquet (uses cached extraction by default)
  run       Full pipeline: extract → process → (characterization analysis) → report + plots
  report    Layer 5: Generate report.html from existing metrics.parquet
  plot      Layer 5: Generate static PNG plots only (no HTML)
  acoustic  Acoustic analysis: list episodes, animate field snapshots, export single-frame PNG
```

| Flag | Description |
|---|---|
| `--output-dir DIR` | Output directory for reports/plots (defaults to first input dir) |
| `--workers N` | Worker processes for parallel extraction/processing (`-1` = auto-detect CPU count) |
| `--force-extract` | Force re-extraction of MCAP files, overwriting the cached topic cache |
| `--report-manifest NAME\|PATH\|{...}` | Report layout: named manifest, YAML path, or inline YAML |
| `--list-manifests` | List the available named report manifests and exit |

---

## Running Tests

Unit tests are pure Python — no running ROS environment required.

```bash
cd /opt/arena_ws/src/Arena/arena_evaluation/arena_evaluation
pytest tests/unit -v
pytest tests/ -v                # Includes benchmark runner and integration tests
```

---

## Rebuilding After Changes

```bash
arena build arena_evaluation
```

See the sub-package READMEs for details:
- [ingestion/README.md](arena_evaluation/ingestion/README.md) — recording, topics, episode lifecycle
- [processing/README.md](arena_evaluation/processing/README.md) — extraction, alignment, metrics
- [presentation/README.md](arena_evaluation/presentation/README.md) — declarative manifests, plot types
- [storage/README.md](arena_evaluation/storage/README.md) — schemas, paths, metadata
- [benchmark/README.md](arena_evaluation/benchmark/README.md) — runner, state, CLI, profiler
- [processing/metrics/README.md](arena_evaluation/processing/metrics/README.md) — metric calculator framework
- [configs/benchmark/README.md](../configs/benchmark/README.md) — suites, contests, characterization
- [arena_evaluation_msgs/README.md](../../arena_evaluation_msgs/README.md) — ROS 2 message and service definitions
