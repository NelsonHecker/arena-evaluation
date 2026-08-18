# presentation - Layer 5: Report and Plot Generation

This package reads processed metric data (Parquet files) and produces human-readable output: an interactive HTML report, static PNG plots, and optional animated GIFs. It operates entirely **offline** - no running ROS 2 environment is required.

---

## Files

| File | Purpose |
|---|---|
| `report_builder.py` | `ReportBuilder` - resolves the manifest, loads the right data source, renders `report.html` + `plots/` |
| `manifest_registry.py` | Resolves report manifests by name/path/inline YAML (mirrors suite/contest resolution) |
| `viz_manifest.py` | `VizManifest` / `PlotSpec` models + default (`standard`) manifest loader |
| `plotly_renderer.py` | Dispatches `PlotSpec` to the correct interactive chart renderer |
| `seaborn_renderer.py` | Dispatches `PlotSpec` to the correct static PNG renderer |
| `color_utils.py` | Accessibility color palette loader + global Plotly/Seaborn application |
| `dimension_detector.py` | Auto-detects varying dimensions and builds compound labels |
| `report_template.html.j2` | Jinja2 HTML template for the final report |
| `plot_types/` | Individual chart type implementations (incl. the long-format `line`) |

---

## Usage

```bash
# Standard metrics report:
arena evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark

# A named declarative manifest:
arena evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark --report-manifest characterization

# List the available named manifests:
arena evaluation run --list-manifests

# Merge multiple benchmarks into one report:
arena evaluation report --benchmark-dir /opt/arena_ws/data/bench1 /opt/arena_ws/data/bench2 --output-dir ./merged_report
```

Programmatically:

```python
from arena_evaluation.presentation.report_builder import ReportBuilder
from pathlib import Path

builder = ReportBuilder.from_dirs(
    source_dirs=[Path("/opt/arena_ws/data/my_benchmark")],
    output_dir=Path("."),
    manifest="characterization",   # name, path, inline YAML, or a VizManifest object
)
builder.build()
```

---

## Declarative Report Manifests

Report layouts are **declarative named YAML files** in
`configs/benchmark/manifests/*.yaml`, mirroring the suite/contest pattern. Precompiled manifests:

| Manifest | Focus |
|---|---|
| `standard` | The default benchmark report (safety, efficiency, motion, smoothness, social, ecological) |
| `ecological` | Energy consumption, power splits, battery behaviour |
| `social` | Proxemics, gaze, pedestrian interaction |
| `safety` | Success rate, collisions, trajectories |
| `characterization` | Open-loop energy/acoustic profiles (line charts, confidence bands, per-working-point table) |

### Resolution precedence (`manifest_registry.py`)

1. Inline YAML (reference starts with `{` or `[`)
2. Explicit path to an existing YAML file
3. Name -> `configs/benchmark/manifests/<name>.yaml` (share dir, then source tree)
4. Legacy `benchmark_dir/viz_manifest.yaml` (only when no reference is given)
5. The `report_manifest.yaml` note file in the benchmark dir (records which manifest produced the last report)
6. Default `standard`

### Manifest schema

```yaml
manifest_version: "1.0"
name: characterization
title: Open-Loop Characterization Report
data_source: characterization_samples   # metrics | characterization_samples | characterization_summary
groups:                                  # report sections (id -> heading)
  - {id: power_curves, title: Power vs. Velocity Curves}
summary:                                 # declarative summary table columns
  - {metric: mean_power_total_w, label: Mean Power, format: "{:.1f}"}
summary_group_by: [phase_kind, robot]
units:                                   # per-column units, merged OVER metric UNITS
  p_total_w: W
  leq_af_dba: dBA
plots: [ ... ]                           # list of PlotSpec
```

- **`data_source`** selects the Parquet the report reads: `metrics`
  (`combined_metrics.parquet` -> `metrics.parquet`) or any `<file>.parquet` name. Per-plot
  `data_source` overrides are supported. The `characterization` manifest uses `metrics` - the
  per-sample data lives in the metric row's `timeseries_char_*` list columns (from the
  `CharacterizationCalculator`), which the `line` renderer explodes into a long frame and
  aggregates per working point.
- Empty `groups`/`summary` fall back to the legacy hardcoded behavior, keeping old manifests valid.
- A `report_manifest.yaml` note is written into the output dir recording which manifest was used.

### PlotSpec fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier, used as PNG filename |
| `type` | `str` | `violin`, `box`, `bar`, `histogram`, `scatter`, `trajectory`, `radar`, `heatmap`, `timeseries`, `line`, `table` |
| `title` | `str` | Human-readable plot title |
| `data_key` | `str` | Column name in the data source (or `"*"` for multi-metric types) |
| `group_by` | `str \| list[str]` | Column(s) for grouping (trajectory faceting, line traces) |
| `differentiate` | `str` | Color differentiation column (default: `"planner"`) |
| `auto_differentiate` | `bool` | Auto-detect varying dimensions and build compound labels |
| `filter` | `dict` | Row filter before plotting (e.g. `{"is_reference": false}`) |
| `options` | `dict` | Renderer-specific options (see below) |
| `layout_group` | `str` | Report section (`"overview"` renders in its own top section) |
| `data_source` | `str \| None` | Per-plot data source override (defaults to the manifest's) |

### Renderer options

| Option | Plot Types | Description |
|---|---|---|
| `overlay_markers`, `show_map` | `trajectory` | Markers / map background |
| `metrics` | `radar`, `bar` (stacked) | Column list |
| `x`, `y` | `heatmap`, `scatter`, `line` | Axis columns |
| `error_y` | `line` | Std column -> confidence band (default) or error bars (`error_style: bars`) |
| `mode` | `line` | `lines` \| `lines+markers` \| `markers` |
| `time_to_s` | `line` | x is `time_ns` -> divide by 1e9, label "Time [s]" |
| `time_relative` | `line` | subtract per-trace x minimum so episodes overlay at t=0 |
| `max_points_per_trace`, `max_traces` | `line` | Downsampling / trace caps for long frames |
| `group_by`, `columns`, `notes` | `table` | see *Tables & agent notes* below |

---

## Plot Types

- **`violin` / `box`** - distribution of a metric per group.
- **`bar`** - mean +/- std bar chart (optionally `stacked: true` with a `metrics` list for
  percentage splits, e.g. energy breakdown).
- **`histogram`** - smooth area distribution per group.
- **`scatter`** - X-Y scatter (auto-explodes List columns).
- **`timeseries`** - wide-format per-episode list columns against a time axis (e.g.
  `timeseries_power_total_w` vs `timeseries_time_s`).
- **`line`** - **long-format** line charts: one trace per `group_by` combination from per-sample or
  per-working-point frames (the characterization data shape), with optional mean+/-std confidence
  bands (`error_y`), `time_to_s`/`time_relative` transforms, and trace/point caps.
- **`table`** - a declarative HTML table combining **data-derived columns** and **agent-written
  notes** (see below). Renders as a styled `<table>` directly in the report.
- **`trajectory`** - (x, y) paths with map overlay, time slider, dynamic markers, spawn-jump
  detection, multi-agent support, optional GIF export.
- **`radar`** - normalized multi-metric profile per group.
- **`heatmap`** - correlation matrix (`data_key="*"`) or pivot grid (options `x`/`y`).

---

## Tables & agent notes

A `table` plot is fully manifest-defined:

```yaml
- id: overview_table
  type: table
  title: Benchmark Overview
  data_key: "*"
  layout_group: overview
  options:
    group_by: [local_planner]                    # one row per group
    columns:                                      # data-derived columns (means)
      - {metric: success, label: Success, format: "{:.0%}"}
      - {metric: time_to_goal, label: Avg Time, format: "{:.1f}"}
    notes: notes.yaml                             # agent-written content (optional)
```

The **notes source** is how an agent (e.g. through MCP tools) writes dynamic overviews, summaries,
or parsed values into the report: it places a `notes.yaml` in the benchmark dir and the table pulls
it in on the next `arena evaluation report` run. Accepted shapes:

```yaml
# structured rows
- {label: Conclusion, value: "DWB achieved the lowest energy intensity"}
- {label: Best planner, value: dwb}
# or a plain mapping
Conclusion: "DWB achieved the lowest energy intensity"
# or free text lines
Conclusion: DWB achieved the lowest energy intensity
# or inline in the manifest instead of a file:
notes:
  - {label: Run, value: 20260808-011247-characterization-characterization}
```

The manifest's `columns` also accept list-valued metrics (they are exploded before aggregating), so
tables can summarize the `timeseries_char_*` characterization columns too.

## dimension_detector.py - Auto-Differentiation

Scans identity columns (`local_planner`, `inter_planner`, `robot`, `stage`, `map`, `benchmark_id`)
for columns with >1 unique value; when multiple vary, builds a compound `__label__` column.
`split_planner_name()` (now in `storage/planner_names.py`) parses contestant names like
`trial-dwb-bypass` into `local_planner="dwb"` / `inter_planner="bypass"` and is shared with
ingestion so episode yamls and reports agree.

---

## report_builder.py - ReportBuilder

Produces a self-contained `report.html` via the Jinja2 template, plus `plots/<id>.png` static
exports at 300 DPI. Structure: header -> (declarative) summary table -> overview section ->
grouped plot sections. `--generate-gifs` also saves animated trajectory GIFs.

Units for axis labels come from `MetricRegistry.get_all_units()` merged with the manifest's
`units` map (manifest wins).

---

## Extending with a New Plot Type

1. Create `plot_types/<mytype>.py`, subclass `BasePlotRenderer`, set `PLOT_TYPE`.
2. Implement `render_plotly(df)` and `render_seaborn(df, out_path)`.
3. Register the class in `plot_types/__init__.py` **and** in both dispatchers
   (`plotly_renderer.py`, `seaborn_renderer.py`) - dispatch is by explicit registry, not auto-discovery.
4. All renderers must tolerate empty/missing-column frames (the adversarial test suite iterates
   every registered type).
