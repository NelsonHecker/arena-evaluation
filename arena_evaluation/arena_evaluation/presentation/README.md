# presentation — Layer 5: Report and Plot Generation

This package reads the processed metric data (Parquet files) and produces human-readable output: an interactive HTML report and static PNG plots.

It operates entirely **offline** and requires no running ROS 2 environment.

---

## Files

| File | Purpose |
|---|---|
| `report_builder.py` | `ReportBuilder` — generates `report.html` and `plots/` directory |
| `viz_manifest.py` | Loads and validates `viz_manifest.yaml` |
| `plotly_renderer.py` | Dispatches `PlotSpec` to the correct interactive chart renderer |
| `seaborn_renderer.py` | Dispatches `PlotSpec` to the correct static PNG renderer |
| `plot_types/` | Individual chart type implementations |

---

## Usage

```bash
# Via the evaluation CLI (single benchmark)
evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark

# Merge multiple benchmarks into one report
evaluation report --benchmark-dir /opt/arena_ws/data/bench1 /opt/arena_ws/data/bench2 --output-dir ./merged_report

# Merge specific ad-hoc runs
evaluation report --run-dir /opt/arena_ws/data/recordings/runA /opt/arena_ws/data/recordings/runB --output-dir ./merged_report

# Or as part of the full pipeline
evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark
```

Or programmatically:

```python
from arena_evaluation.presentation.report_builder import ReportBuilder
from pathlib import Path

# Single source
builder = ReportBuilder(benchmark_dir=Path("/opt/arena_ws/data/my_benchmark"))
builder.build()

# Multi-source merge
builder = ReportBuilder.from_dirs(
    source_dirs=[Path("/opt/arena_ws/data/bench1"), Path("/opt/arena_ws/data/bench2")],
    output_dir=Path("./merged_report")
)
builder.build()
```

---

## viz_manifest.yaml — Visualization Manifest

The `viz_manifest.yaml` file placed at the benchmark directory root controls which plots are generated and how they are rendered. If the file does not exist, a default set of plots is generated automatically.

```yaml
plots:
  - id: path_length_violin
    type: violin
    title: "Path Length Distribution"
    data_key: path_length      # column name in combined_metrics.parquet
    group_by: planner          # x-axis grouping

  - id: success_rate_bar
    type: bar
    title: "Success Rate by Planner"
    data_key: success
    group_by: planner

  - id: velocity_box
    type: box
    title: "Velocity Distribution"
    data_key: velocity_mean
    group_by: planner

  - id: trajectory_overlay
    type: trajectory
    title: "Episode Trajectories"
    data_key: path
    group_by: planner
    filter:
      result: "GOAL_REACHED"   # optional: filter rows before plotting

  - id: performance_radar
    type: radar
    title: "Multi-Metric Overview"
    data_key: radar_summary    # aggregated from multiple columns
    group_by: planner
```

### PlotSpec Fields

| Field | Required | Description |
|---|---|---|
| `id` | ✅ | Unique identifier, used as filename for PNG output |
| `type` | ✅ | One of: `violin`, `box`, `bar`, `trajectory`, `radar` |
| `title` | ✅ | Human-readable plot title |
| `data_key` | ✅ | Column name in the combined metrics Parquet file |
| `group_by` | — | Column for x-axis grouping (default: `"planner"`) |
| `differentiate` | — | Additional color differentiation column. If set to `null` (the default behaviour), the pipeline will automatically detect varying dimensions across merged data (e.g. planner, robot, stage) and build compound labels (e.g., `dwa / burger / stage_1`) to prevent data collapse. |
| `filter` | — | Dict of `{column: value}` to filter rows before plotting |
| `options` | — | Dict of renderer-specific options |

---

## Plot Types

### `violin`
Distribution of a continuous metric per group. Uses Plotly Violin with embedded box and individual data points. Best for showing the spread and shape of distributions (path_length, velocity_mean, time_to_goal).

### `box`
Box-and-whisker plot per group. More compact than violin; better for comparing medians and outliers across many planners.

### `bar`
Mean ± standard deviation bar chart. Best for scalar summary metrics (success rate, collision rate). Bars are coloured by planner.

### `trajectory`
Scatter/line plot of recorded (x, y) path coordinates on a blank canvas. Intended for visual inspection of robot paths across episodes. Map image overlay is deferred to Phase 2.

### `radar`
Spider / Scatterpolar chart normalising multiple metrics onto a 0–1 scale. Best for comparing overall planner profiles across performance, safety, and social dimensions.

---

## report_builder.py — ReportBuilder

Produces a single self-contained `report.html` file using Plotly (with `include_plotlyjs="cdn"` — no local JS bundle required).

**HTML structure:**
1. **Header** — benchmark ID, run date, robot, planner list
2. **Summary table** — aggregate statistics per planner (success rate, mean path length, mean time to goal)
3. **Plot blocks** — one section per `PlotSpec` in the manifest, with interactive Plotly chart
4. **Methodology notes** — metric definitions and data source reference

Also generates `plots/<id>.png` static exports at 300 DPI using Seaborn for each plot in the manifest.

---

## Extending with a New Plot Type

1. Create `plot_types/<mytype>.py`.
2. Subclass `BasePlotRenderer` and set `PLOT_TYPE = "mytype"`.
3. Implement `render_plotly(spec, df) -> go.Figure` and `render_seaborn(spec, df, ax)`.
4. The renderers are auto-discovered — no registration required.

```python
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer
import plotly.graph_objects as go
import polars as pl

class MyRenderer(BasePlotRenderer):
    PLOT_TYPE = "mytype"

    def render_plotly(self, spec, df: pl.DataFrame) -> go.Figure:
        fig = go.Figure()
        # ... build figure from df and spec ...
        return fig

    def render_seaborn(self, spec, df: pl.DataFrame, ax):
        # ... draw onto ax ...
        pass
```
