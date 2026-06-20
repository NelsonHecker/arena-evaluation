# presentation — Layer 5: Report and Plot Generation

This package reads the processed metric data (Parquet files) and produces human-readable output: an interactive HTML report, static PNG plots, and optional animated GIFs.

It operates entirely **offline** and requires no running ROS 2 environment.

---

## Files

| File | Purpose |
|---|---|
| `report_builder.py` | `ReportBuilder` — generates `report.html`, `plots/` directory, and optional GIFs |
| `viz_manifest.py` | Default plot manifest (hardcoded in Python) + YAML loader |
| `plotly_renderer.py` | Dispatches `PlotSpec` to the correct interactive chart renderer |
| `seaborn_renderer.py` | Dispatches `PlotSpec` to the correct static PNG renderer |
| `color_utils.py` | Accessibility color palette loader + global Plotly/Seaborn application |
| `dimension_detector.py` | Auto-detects varying dimensions and builds compound labels |
| `report_template.html.j2` | Jinja2 HTML template for the final report |
| `plot_types/` | Individual chart type implementations |

---

## Usage

```bash
# Via the evaluation CLI (single benchmark)
arena evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark

# Merge multiple benchmarks into one report
arena evaluation report --benchmark-dir /opt/arena_ws/data/bench1 /opt/arena_ws/data/bench2 --output-dir ./merged_report

# With animated GIF generation
arena evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark --generate-gifs

# Or as part of the full pipeline
arena evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark
```

Or programmatically:

```python
from arena_evaluation.presentation.report_builder import ReportBuilder
from pathlib import Path

# Single source
builder = ReportBuilder(benchmark_dir=Path("/opt/arena_ws/data/my_benchmark"))
builder.build()

# Multi-source merge with GIF generation
builder = ReportBuilder.from_dirs(
    source_dirs=[Path("/opt/arena_ws/data/bench1"), Path("/opt/arena_ws/data/bench2")],
    output_dir=Path("./merged_report"),
    generate_gifs=True,
)
builder.build()
```

---

## Visualization Manifest

The default set of plots is defined programmatically in `viz_manifest.py` via `VizManifest._default_manifest()`. If a `viz_manifest.yaml` file is placed at the benchmark directory root, it will be loaded instead.

The default manifest includes plots organized into layout groups:
- **`overview`** — Radar charts per local/inter planner, correlation heatmap, success pivot heatmap
- **`metrics`** — Violin, box, bar, histogram, and scatter plots for all key metrics
- **`details`** — Trajectory plots (robot paths with map overlay, pedestrian paths)

### PlotSpec Fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier, used as filename for PNG output |
| `type` | `str` | One of: `violin`, `box`, `bar`, `histogram`, `scatter`, `trajectory`, `radar`, `heatmap` |
| `title` | `str` | Human-readable plot title |
| `data_key` | `str` | Column name in the combined metrics Parquet file (or `"*"` for multi-metric types like radar/heatmap) |
| `group_by` | `str \| list[str]` | Column(s) for grouping (e.g. `["stage"]` for trajectory faceting) |
| `differentiate` | `str` | Color differentiation column (default: `"planner"`) |
| `auto_differentiate` | `bool` | If `true` (default), auto-detect varying dimensions across merged data and build compound labels to prevent data collapse |
| `filter` | `dict[str, str]` | Filter rows before plotting (e.g., `{"result": "GOAL_REACHED"}`) |
| `options` | `dict` | Renderer-specific options (see below) |
| `layout_group` | `str` | Groups plots into sections in the HTML report (`"overview"`, `"metrics"`, `"details"`) |

### Renderer-Specific Options

| Option | Plot Types | Description |
|---|---|---|
| `overlay_markers` | `trajectory` | `true` (default) to show Start/Goal/Collision markers, `false` to hide them (used for pedestrian plots) |
| `show_map` | `trajectory` | `true` (default) to overlay the map image as background |
| `metrics` | `radar` | List of metric column names to include in the radar chart |
| `x_key` / `y_key` | `scatter`, `heatmap` | Override the x/y axis data keys |

---

## Plot Types

### `violin`
Distribution of a continuous metric per group. Uses Plotly Violin with embedded box and individual data points. Best for showing the spread and shape of distributions (path_length, velocity_mean, time_to_goal).

### `box`
Box-and-whisker plot per group. More compact than violin; better for comparing medians and outliers across many planners.

### `bar`
Mean ± standard deviation bar chart. Best for scalar summary metrics (success rate, collision rate). Bars are coloured by the differentiation column.

### `histogram`
Rendered as smooth line charts with shaded area underneath (not traditional bar histograms). Shows the distribution of a metric with optional KDE-style smoothing. Supports differentiation by planner.

### `scatter`
X-Y scatter plot with optional differentiation. Useful for correlating two metrics (e.g., path_length vs time_to_goal).

### `trajectory`
Line plot of recorded (x, y) path coordinates with:
- **Map image overlay** — automatically loaded from `arena_simulation_setup` via `MapRegistry`, with TF-static origin correction.
- **Interactive time slider** — JavaScript-powered slider that progressively reveals the trajectory frame-by-frame.
- **Dynamic markers** — Start (circle), Goal (star), and Collision (cross) markers that appear dynamically as the time slider reaches the relevant frame. Uses Plotly `customdata` for zero-overhead index tracking instead of NaN-padded arrays.
- **Spawn jump detection** — Intelligently identifies teleportation jumps (spawn → actual start) by checking segment travel distances. Only the first segment with >0.1m of actual movement is treated as the true start.
- **Multi-agent support** — Handles concurrent robot fleet and pedestrian path rendering natively.
- **Per-plot marker toggle** — The `overlay_markers` option can be set to `false` for pedestrian trajectory plots to keep them uncluttered.
- **GIF export** — When `--generate-gifs` is enabled, animated GIFs are saved alongside PNGs showing the trajectory being drawn over time.

Trajectory plots are grouped by `stage` (or other `group_by` columns) and each group gets its own plot with the corresponding map overlay.

### `radar`
Spider / Scatterpolar chart normalising multiple metrics onto a 0–1 scale. Best for comparing overall planner profiles across performance, safety, and social dimensions. The `metrics` option in `options` specifies which columns to include.

### `heatmap`
Supports two modes:
1. **Correlation matrix** — When `data_key="*"`, computes and displays the Pearson correlation between all numeric metrics.
2. **Pivot heatmap** — When `x_key` and `y_key` are specified, pivots the data to show the mean of `data_key` across two categorical dimensions.

---

## color_utils.py — Global Color Palette

Loads a qualitative color palette from `config/color_palette.yaml` and applies it globally to both Plotly (via `pio.templates`) and Seaborn (via `sns.set_theme`). The palette is cached after the first load.

Colors are hand-picked for accessibility and maximum visual distinction. White and black entries in the YAML are excluded from the drawing palette.

---

## dimension_detector.py — Auto-Differentiation

When merging data from multiple benchmarks that vary across multiple dimensions (e.g., different planners AND different robots), a single `differentiate` column is insufficient. The dimension detector:

1. Scans the identity columns (`local_planner`, `inter_planner`, `robot`, `stage`, `map`, `benchmark_id`) for columns with >1 unique value.
2. If multiple dimensions vary, builds a compound label column (`__label__`) by concatenating values (e.g., `"dwb / burger / stage_1"`).
3. All plot renderers use `resolve_differentiate()` which handles this transparently.

The `split_planner_name()` function parses contestant names like `"trial-dwb-bypass"` into `local_planner="dwb"` and `inter_planner="bypass"` using a known planner vocabulary.

---

## report_builder.py — ReportBuilder

Produces a single self-contained `report.html` file using a Jinja2 template (`report_template.html.j2`) with Plotly charts (using `include_plotlyjs="cdn"` — no local JS bundle required).

**HTML structure:**
1. **Header** — benchmark ID, run date, robot, planner list
2. **Summary table** — aggregate statistics per planner (success rate, mean path length, mean time to goal)
3. **Plot sections** — organized by `layout_group` (overview → metrics → details), each with interactive Plotly charts
4. **Methodology notes** — metric definitions and data source reference

Also generates `plots/<id>.png` static exports at 300 DPI using Seaborn for each plot in the manifest.

When `generate_gifs=True`, trajectory plots also produce animated `.gif` files.

---

## Extending with a New Plot Type

1. Create `plot_types/<mytype>.py`.
2. Subclass `BasePlotRenderer` and set `PLOT_TYPE = "mytype"`.
3. Implement `render_plotly(df) -> str | list[str] | None` and `render_seaborn(df, out_path)`.
4. The renderers are auto-discovered via the `__init__.py` registry — no manual registration required.

```python
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer
import plotly.graph_objects as go
import polars as pl
import pathlib

class MyRenderer(BasePlotRenderer):
    PLOT_TYPE = "mytype"

    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        # Build and return HTML string(s)
        fig = go.Figure()
        # ... build figure from df and self.spec ...
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        # Draw and save to out_path
        import matplotlib.pyplot as plt
        # ... draw plot ...
        plt.savefig(out_path, dpi=300)
        plt.close()
```

### BasePlotRenderer API

| Method | Description |
|---|---|
| `render_plotly(df)` | Returns HTML string(s) for interactive charts |
| `render_seaborn(df, out_path)` | Saves static PNG to `out_path` |
| `resolve_diff_col(df)` | Returns `(col_name, df)` with auto-differentiation applied |
| `format_label(label, data_key)` | Appends unit suffix (e.g., `"Path Length [m]"`) if a unit is defined for `data_key` |
| `_apply_filters(df)` | Applies `self.spec.filter` to the DataFrame |

The `units` dict is passed to the renderer at construction time from the metric calculator's `UNITS` class attribute.
