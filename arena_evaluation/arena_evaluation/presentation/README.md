# presentation (Layer 5: Report and Plot Generation)

Reads processed metric data (Parquet) and produces interactive HTML reports, static PNG plots, and optional animated GIFs offline.

---

## Files

| File | Purpose |
|---|---|
| `report_builder.py` | `ReportBuilder`: resolves manifests, loads data sources, renders `report.html` and `plots/` |
| `manifest_registry.py` | Resolves report manifests by name, path, or inline YAML |
| `viz_manifest.py` | `VizManifest` and `PlotSpec` models and default manifest loader |
| `plotly_renderer.py` | Dispatches `PlotSpec` to interactive chart renderers |
| `seaborn_renderer.py` | Dispatches `PlotSpec` to static PNG renderers |
| `color_utils.py` | Accessibility color palette loader and global plotting styles |
| `dimension_detector.py` | Auto-detects varying dimensions and builds compound labels |
| `report_template.html.j2` | Jinja2 HTML template for reports |
| `plot_types/` | Individual chart type implementations (incl. `acoustic_field_texture` and static-PNG `timeseries`) |
| `../paper/` | Data preparation for publication figures (`hero_timeseries.py`; never renders graphs itself) |

---

## Usage

```bash
# Standard metrics report
arena evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark

# Named declarative manifest
arena evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark --report-manifest characterization

# List available named manifests
arena evaluation run --list-manifests

# Merge multiple benchmarks into one report
arena evaluation report --benchmark-dir /opt/arena_ws/data/bench1 /opt/arena_ws/data/bench2 --output-dir ./merged_report
```

Programmatically:

```python
from arena_evaluation.presentation.report_builder import ReportBuilder
from pathlib import Path

builder = ReportBuilder.from_dirs(
    source_dirs=[Path("/opt/arena_ws/data/my_benchmark")],
    output_dir=Path("."),
    manifest="characterization",
)
builder.build()
```

---

## Declarative Report Manifests

Report layouts are defined in `configs/benchmark/manifests/*.yaml`:

| Manifest | Focus |
|---|---|
| `standard` | Default benchmark report (safety, efficiency, motion, smoothness, social, ecological) |
| `ecological` | Energy consumption, power splits, battery behavior |
| `social` | Proxemics, gaze, pedestrian interaction |
| `safety` | Success rate, collisions, trajectories |
| `characterization` | Open-loop energy/acoustic profiles |
| `ecobench_hero` | Hero-figure panels: stacked decomposed power trace + twin-axis acoustic emission/exposure (data source: `hero_timeseries.parquet`) |

### Resolution Precedence (`manifest_registry.py`)

1. Inline YAML (starts with `{` or `[`).
2. Explicit YAML file path.
3. Named manifest in `configs/benchmark/manifests/<name>.yaml`.
4. Legacy `benchmark_dir/viz_manifest.yaml`.
5. `report_manifest.yaml` in benchmark directory.
6. Default `standard`.

### Manifest Schema

```yaml
manifest_version: "1.0"
name: characterization
title: Open-Loop Characterization Report
data_source: characterization_samples
groups:
  - {id: power_curves, title: Power vs. Velocity Curves}
summary:
  - {metric: mean_power_total_w, label: Mean Power, format: "{:.1f}"}
summary_group_by: [phase_kind, robot]
units:
  p_total_w: W
  leq_af_dba: dBA
plots: [ ... ]
```

### PlotSpec Fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique plot identifier and PNG filename |
| `type` | `str` | Plot type (`violin`, `box`, `bar`, `histogram`, `scatter`, `trajectory`, `radar`, `heatmap`, `timeseries`, `line`, `table`) |
| `title` | `str` | Plot title |
| `data_key` | `str` | Column name in data source (or `"*"` for multi-metric plots) |
| `group_by` | `str \| list[str]` | Grouping columns |
| `differentiate` | `str` | Color differentiation column (default: `"planner"`) |
| `auto_differentiate` | `bool` | Auto-detect varying dimensions and construct compound labels |
| `filter` | `dict` | Row filter before plotting (e.g. `{"is_reference": false}`) |
| `options` | `dict` | Renderer options |
| `layout_group` | `str` | Report section identifier |
| `data_source` | `str \| None` | Per-plot data source override |

---

## Plot Types

- `violin` / `box`: Distribution of a metric per group.
- `bar`: Mean +/- std bar chart (supports `stacked: true`).
- `histogram`: Smooth area distribution per group.
- `scatter`: X-Y scatter (expands list columns).
- `timeseries`: Wide-format per-episode list columns vs time axis.
- `line`: Long-format line charts with optional confidence bands (`error_y`).
- `table`: Declarative HTML summary table.
- `trajectory`: Spatial paths with map overlay, time slider, and optional GIF export.
- `radar`: Normalized multi-metric profile.
- `heatmap`: Correlation matrix or pivot grid.
- `acoustic_field_texture`: Borderless raw acoustic MP4 texture for Blender floor mapping (see below).
- `timeseries` (static PNG): Row-per-sample or list-column time series with stacked area, twin axes, reference lines, and annotated shade windows (see below).

---

## Adding a Plot Type

1. Create `plot_types/<mytype>.py`, subclass `BasePlotRenderer`, and define `PLOT_TYPE`.
2. Implement `render_plotly(df)` and `render_seaborn(df, out_path)`.
3. Register class in `plot_types/__init__.py`, `plotly_renderer.py`, and `seaborn_renderer.py`.
4. Ensure renderers handle empty or missing columns safely.

---

## Acoustic Texture MP4 (Blender Floor Mapping)

`acoustic_field_texture` (`AcousticFieldRenderer.render_raw_texture_video`) generates the borderless, full-duration raw MP4 that `arena-blender-viz build --acoustic` maps onto the floor:

```bash
python -m arena_evaluation.cli acoustic texture \
  --benchmark-dir <run> --episode worst --fps 30 \
  --vmin 20 --vmax 60 --downsample 2 [--max-frames 0] [--output ...]
```

Behavior and invariants:

- Output: `plots/<episode>_acoustic_raw.mp4` (auto-discovered by `arena-blender-viz` `bundle_builder`), 30 fps, one frame per simulation frame.
- **Spatial-temporal cache**: the C++ solver re-solves only when the robot moved more than 0.04 m or the door set changed; output frames interpolate linearly in time with door-state changes stepped at the interval midpoint.
- **Pinned colormap limits** (`--vmin/--vmax`, defaults 20/60 dBA): constant across all frames (GEMINI §12) so the floor stays comparable with the HUD colorbar; the texture is written with cv2 `mp4v` in BGR order.
- A pinned sidecar manifest `<episode>_acoustic_texture.yaml` is written with `texture: {vmin, vmax, cmap, fps, n_frames, duration_s, map, origin, resolution, downsample, width, height, episode, mp4}` — the single source of truth for any colorbar composited with the texture.
- Verified bit-faithful to the previous hardcoded episode-040 generator (byte-identical output in the same environment).

## Static PNG `timeseries` (hero-style panels)

`TimeseriesRenderer.render_seaborn` renders publication PNGs (300 dpi) from either a row-per-sample parquet (one row per time step) or the wide-format list columns of `combined_metrics` (`timeseries_time_s`, `timeseries_power_*_w`, ...).

Manifest options:

| Option | Description |
|---|---|
| `x` | Time column (default `timeseries_time_s`) |
| `metrics` | Primary (left) axis columns |
| `stacked: true` | Single stackplot accumulating all metrics (fixed per-metric order) |
| `colors` | `{column: hex}` fixed series colors |
| `twin` | Columns on the secondary (right) axis; `twin_fill: true` adds a soft area fill |
| `hline` | `{value, label, color, style, axis}` horizontal reference line (e.g. the 100 dBA cutoff) |
| `shade_window` | `{start, end, label, color, alpha}` annotated band (e.g. the evasion window with its marginal cost) |
| `ylabel` / `twin_ylabel` / `xlabel` | Axis labels (defaults derive from the units map) |
| `figsize` / `dpi` / `linewidth` | Layout (defaults `[7.2, 2.8]`, 300, 1.8) |

## Hero Figure Workflow (`ecobench_hero`)

One command produces both panels — the manifest declares `data_source_builder: hero_timeseries`, so the pipeline materializes `hero_timeseries.parquet` automatically (power layers + emitted dBA from the topic parquets; received pedestrian exposure via single-target transmission-loss solves, door-state aware) when it is missing:

```bash
arena evaluation run --benchmark-dir <run> --report-manifest ecobench_hero
#  (or `report`/`plot` on an already-processed run)
#  -> plots/hero_power_stacked.png      (Panel B: stacked mech/static/heat + shaded evasion window)
#  -> plots/hero_acoustic_exposure.png  (Panel C: emitted dBA + received exposure twin axis + 100 dBA cutoff)
```

- The hero episode is pinned via `builder_options.episode` in `configs/benchmark/manifests/ecobench_hero.yaml` (defaults to the first episode with extracted topics when omitted; a clear error suggests `evaluation process` when no topics exist).
- Tune the evasion window and the marginal-cost label by editing `shade_window` in the manifest — no code changes needed. The two panels share the same `t_s` column, so their x-axes are locked 1:1 by construction.
- The parquet is only rebuilt when missing; delete it to force a rebuild. The standalone prep CLI (`python -m arena_evaluation.paper.hero_timeseries ...`) remains available for manual use.

## Running the CLI in this workspace (WSL)

The `evaluation` console scripts carry stale shebangs; invoke via module with the explicit environment:

```bash
export PYTHONPATH=/opt/arena_ws/src/Arena/arena_evaluation/arena_evaluation:/opt/arena_ws/src/Arena/arena_simulation_setup/src:/opt/arena_ws/install/arena_evaluation/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages
export AMENT_PREFIX_PATH=$(find /opt/arena_ws/install -mindepth 1 -maxdepth 1 -type d | paste -sd:):/opt/ros/jazzy
python -m arena_evaluation.cli <command> ...
```

