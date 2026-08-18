# SESSION SNAPSHOT (2026-08-10) — current acoustic_field.py (modular renderer).
# Path: src/Arena/arena_evaluation/arena_evaluation/arena_evaluation/presentation/plot_types/acoustic_field.py
from __future__ import annotations

import pathlib
import logging
import numpy as np
import polars as pl
from PIL import Image

from .base import BasePlotRenderer
from ...processing.map_registry import MapRegistry
from ...processing.acoustics.impedance_grid import downsample_occupancy
from ...processing.acoustics.door_map import door_segments, build_pixel_tl

try:
    from ...processing.acoustics.impedance_grid import compute_attenuations
except ImportError:
    compute_attenuations = None

logger = logging.getLogger(__name__)

_CELL_FIGSIZE = (5, 4)
_CELL_DPI = 150

# Color-scale floor (dBA).  Set below hospital ambient so quiet frames still
# show field structure rather than collapsing to solid black.
_FIELD_VMIN_DBA = 30.0


class AcousticFieldRenderer(BasePlotRenderer):
    PLOT_TYPE = "acoustic_field"

    # ── Manifest-driven options ─────────────────────────────────────────────────
    #  filter:            {col: value} or {col: [v1, v2]} — select specific runs
    #  differentiate:     rows of the grid (e.g. local_planner)
    #  group_by:          columns of the grid (e.g. [stage])
    #  options.mode:      "grid" (default) | "single" — single = one worst-case image
    #  options.downsample: stride factor for the field solver (default 1)
    #  options.vmin:      color-scale floor (default = hospital ambient 42 dBA)
    #  options.include_reference: include reference/unhindered runs (default false)

    @staticmethod
    def _load_grid_and_meta(map_name, run_dir=None):
        meta = MapRegistry.get_map(map_name, run_dir=run_dir)
        if not meta or "png_path" not in meta:
            logger.warning("Map %r not found in registry (run_dir=%s)", map_name, run_dir)
            return None
        try:
            img = Image.open(meta["png_path"]).convert("L")
            grid = np.ascontiguousarray(np.flipud((np.array(img) < 200).astype(np.uint8)))
        except Exception as e:
            logger.warning("Could not load map image %r: %s", meta["png_path"], e)
            return None
        return grid, meta

    def _compute_full_field(self, grid, resolution, ox, oy,
                            rx_m, ry_m, source_dba, downsample=1, pixel_tl=None):
        if downsample > 1:
            # max-pool keeps 1-px walls instead of dropping them (strided slicing loses thin walls)
            grid = downsample_occupancy(grid, downsample)
            resolution = resolution * downsample

        h, w = grid.shape
        logger.info("AcousticFieldRenderer: full-field %dx%d (ds=%d)...", w, h, downsample)

        rx_px = (rx_m - ox) / resolution
        ry_px = (ry_m - oy) / resolution

        yy, xx = np.mgrid[0:h, 0:w]
        tx = np.ascontiguousarray(xx.flatten().astype(np.float32))
        ty = np.ascontiguousarray(yy.flatten().astype(np.float32))

        attenuations = compute_attenuations(
            grid, resolution, rx_px, ry_px, tx, ty,
            wall_tl=47.0, mic_distance=1.0,
            pixel_tl=pixel_tl,
        )

        att_grid = attenuations.reshape((h, w))
        field_dba = source_dba - att_grid
        field_dba = np.clip(field_dba, 0, None)
        logger.info("AcousticFieldRenderer: full-field done.")
        return field_dba, resolution, (h, w)

    def _render_cell_png(self, grid, resolution, ox, oy,
                         rx_m, ry_m, source_dba, peds, title, out_path,
                         downsample=1, vmin=None, vmax=None, pixel_tl=None, doors=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._compute_full_field(grid, resolution, ox, oy,
                                          rx_m, ry_m, source_dba,
                                          downsample=downsample, pixel_tl=pixel_tl)
        if result is None:
            return False
        field_dba, eff_res, (h, w) = result

        if downsample > 1:
            mask_grid = grid[::downsample, ::downsample]
        else:
            mask_grid = grid

        # Build door mask (all doors) and open-door mask (carved to 0 dB)
        door_mask = np.zeros_like(mask_grid, dtype=bool)
        open_door_mask = np.zeros_like(mask_grid, dtype=bool)
        if doors:
            for _name, (m, _tl) in doors.items():
                m_ds = m[::downsample, ::downsample] if downsample > 1 else m
                door_mask |= m_ds
                if pixel_tl is not None:
                    open_door_mask |= m_ds & (pixel_tl == 0.0)

        # NaN walls and closed doors, but NOT open doors — they show free-space colour
        wall_or_closed = (mask_grid == 1) & ~open_door_mask
        render_grid = np.where(wall_or_closed | np.isinf(field_dba), np.nan, field_dba)
        render_grid = np.flipud(render_grid)

        door_overlay = door_mask if door_mask.any() else None

        plt.figure(figsize=_CELL_FIGSIZE)
        ax = plt.gca()
        ax.set_facecolor("black")
        ax.grid(False)
        extent = [ox, ox + w * eff_res, oy, oy + h * eff_res]

        im = plt.imshow(render_grid, cmap="inferno", origin="upper", extent=extent,
                        vmin=vmin, vmax=vmax)
        # Subtle wall outline so room structure is always visible (even when field is dark)
        cy = np.linspace(extent[2], extent[3], mask_grid.shape[0])
        cx = np.linspace(extent[0], extent[1], mask_grid.shape[1])
        wall_outline = (mask_grid == 1).astype(np.uint8)
        ax.contour(cx, cy, wall_outline, levels=[0.5], colors=["#ffffff"],
                   linewidths=0.3, alpha=0.25)
        if door_overlay is not None:
            # Cyan contour outlines for ALL doors
            ax.contour(cx, cy, door_overlay.astype(np.uint8), levels=[0.5],
                       colors=["#00ffd5"], linewidths=1.2, alpha=0.85)
            # Brighter green contour for OPEN doors
            if open_door_mask.any():
                ax.contour(cx, cy, open_door_mask.astype(np.uint8), levels=[0.5],
                           colors=["#00ff00"], linewidths=2.0, alpha=0.9)
        plt.colorbar(im, label="dBA", fraction=0.046, pad=0.04)

        plt.plot(rx_m, ry_m, "g*", markersize=8, label="Robot")
        if peds:
            px = [p[0] for p in peds if len(p) >= 2]
            py = [p[1] for p in peds if len(p) >= 2]
            if px:
                plt.plot(px, py, "ro", markersize=4, label="Peds")

        plt.title(title, fontsize=9)
        plt.xlabel("X (m)", fontsize=8)
        plt.ylabel("Y (m)", fontsize=8)
        plt.tick_params(labelsize=7)
        plt.legend(fontsize=7)
        plt.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=_CELL_DPI, bbox_inches="tight")
        plt.close()
        logger.info("AcousticFieldRenderer: saved %s", out_path)
        return True

    @staticmethod
    def _pick_worst_row(pdf):
        if "ped_max_exposure_dba" not in pdf.columns:
            return None
        max_idx = pdf["ped_max_exposure_dba"].arg_max()
        if max_idx is None:
            return None
        row = pdf.row(max_idx, named=True)
        if not row.get("worst_case_acoustic_frame"):
            return None
        wf = row["worst_case_acoustic_frame"]
        if isinstance(wf, str):
            import json
            try:
                wf = json.loads(wf)
            except Exception:
                return None
        if not wf or wf.get("robot_x") is None or wf.get("robot_y") is None:
            return None
        return {
            "robot_x": float(wf["robot_x"]),
            "robot_y": float(wf["robot_y"]),
            "source_dba": float(wf.get("source_dba", 60.0)),
            "pedestrians": wf.get("pedestrians", []),
            "door_states": wf.get("door_states", {}),
            "ped_max_exposure_dba": float(row["ped_max_exposure_dba"]),
            "planner": row.get("planner", row.get("local_planner", "")),
            "stage": row.get("stage", ""),
        }

    def _prepared_df(self, df):
        """Apply manifest filters + reference-run exclusion."""
        work_df = self._apply_filters(df)
        include_ref = bool(self.spec.options.get("include_reference", False))
        if not include_ref and "is_reference" in work_df.columns:
            work_df = work_df.filter(
                pl.col("is_reference").is_null() | (pl.col("is_reference") == False)
            )
        return work_df

    def _group_values(self, work_df):
        """Return (row_values, col_values) from differentiate / group_by."""
        diff_col = self.spec.differentiate if self.spec.differentiate and self.spec.differentiate in work_df.columns else None
        group_cols_raw = self.spec.group_by
        if isinstance(group_cols_raw, str):
            group_cols_raw = [group_cols_raw]
        group_cols = [c for c in (group_cols_raw or []) if c in work_df.columns]

        if diff_col:
            row_values = sorted(work_df[diff_col].unique().to_list())
        else:
            row_values = [""]

        if group_cols:
            col_values = sorted(work_df.select(group_cols).unique().sort(group_cols).rows())
        else:
            col_values = [()]

        return diff_col, group_cols, row_values, col_values

    def _filter_group(self, work_df, diff_col, group_cols, rv, cv):
        pdf = work_df
        if diff_col:
            pdf = pdf.filter(pl.col(diff_col) == rv)
        for ci, cname in enumerate(group_cols or []):
            pdf = pdf.filter(pl.col(cname) == cv[ci])
        return pdf

    def render_plotly(self, df):
        """Return an HTML string embedding the acoustic-field image(s)."""
        if compute_attenuations is None:
            logger.warning("AcousticFieldRenderer: C++ solver not available.")
            return ""

        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            logger.info("AcousticFieldRenderer: no rows after filters.")
            return ""

        map_name = work_df["map"][0] if "map" in work_df.columns else None
        if not map_name:
            return ""

        result = self._load_grid_and_meta(map_name, run_dir=self.run_dir)
        if result is None:
            return ""
        grid, meta = result
        resolution = meta["resolution"]
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        downsample = int(self.spec.options.get("downsample", 1))
        vmin = float(self.spec.options.get("vmin", _FIELD_VMIN_DBA))
        mode = str(self.spec.options.get("mode", "grid")).lower()

        doors = door_segments(map_name, grid, resolution, (ox, oy, 0.0), run_dir=self.run_dir)

        plots_dir = pathlib.Path(self.run_dir) / "plots" if self.run_dir else pathlib.Path("plots")

        # ── single mode: one worst-case image ───────────────────────────────────
        if mode == "single":
            worst = self._pick_worst_row(work_df)
            if worst is None:
                logger.info("AcousticFieldRenderer: no worst-case frame in filtered set.")
                return ""

            row_label = str(worst["planner"]) if worst["planner"] else ""
            col_label = str(worst["stage"]) if worst["stage"] else ""
            safe = self.spec.id.replace("/", "_").replace(" ", "_")
            png_path = plots_dir / f"{safe}.png"

            src_dba = float(worst.get("source_dba", 60.0))
            # Boost source to 100 dBA if this frame has a collision impulse
            if float(worst.get("ped_max_exposure_dba", 0)) >= 100.0 and src_dba < 100.0:
                src_dba = 100.0
            vmax = float(np.ceil(src_dba / 10.0) * 10.0)
            if vmax <= vmin:
                vmax = vmin + 20.0

            pixel_tl = None
            door_states = worst.get("door_states") or {}
            if doors:
                open_set = {n for n, st in door_states.items() if st == "open"}
                pixel_tl = build_pixel_tl(grid, doors, open_doors=open_set)

            ok = self._render_cell_png(
                grid, resolution, ox, oy,
                worst["robot_x"], worst["robot_y"], src_dba,
                worst["pedestrians"],
                title=f"{row_label} / {col_label}  |  {worst['ped_max_exposure_dba']:.0f} dBA",
                out_path=png_path,
                downsample=downsample,
                vmin=vmin,
                vmax=vmax,
                pixel_tl=pixel_tl,
                doors=doors if pixel_tl is not None else None,
            )
            if not ok:
                return ""
            return (
                f'<div style="text-align:center;">'
                f'<img src="plots/{png_path.name}" style="max-width:100%;border-radius:4px;" '
                f'alt="{self.spec.title}">'
                f'<br><span style="font-size:0.78em;color:#475569;">'
                f'{row_label} / {col_label} &mdash; {worst["ped_max_exposure_dba"]:.0f} dBA'
                f'</span></div>'
            )

        # ── grid mode: one cell per (differentiate x group_by) ─────────────────
        diff_col, group_cols, row_values, col_values = self._group_values(work_df)

        # first pass: collect (worst, row_label, col_label) to compute global max
        entries = []
        for rv in row_values:
            for cv in col_values:
                pdf = self._filter_group(work_df, diff_col, group_cols, rv, cv)
                if len(pdf) == 0:
                    continue
                worst = self._pick_worst_row(pdf)
                if worst is None:
                    continue
                row_label = str(rv) if rv else ""
                col_label = "/".join(str(v) for v in cv) if cv else ""
                entries.append((worst, row_label, col_label))

        if not entries:
            return ""

        def _eff_src(w):
            s = float(w.get("source_dba", 60.0))
            return 100.0 if float(w.get("ped_max_exposure_dba", 0)) >= 100.0 and s < 100.0 else s
        global_max = max((_eff_src(w) for w, _, _ in entries), default=80.0)
        global_max = float(np.ceil(global_max / 10.0) * 10.0)
        if global_max <= vmin:
            global_max = vmin + 20.0
        logger.info("AcousticFieldRenderer: shared color scale %.0f..%.0f dBA across %d cells.",
                    vmin, global_max, len(entries))

        # second pass: render each cell with the shared vmin/vmax
        cells = []
        for worst, row_label, col_label in entries:
            safe = f"{self.spec.id}"
            if row_label:
                safe += f"_{row_label}"
            if col_label:
                safe += f"_{col_label}"
            safe = safe.replace("/", "_").replace(" ", "_")
            png_path = plots_dir / f"{safe}.png"

            pixel_tl = None
            door_states = worst.get("door_states") or {}
            if doors:
                open_set = {n for n, st in door_states.items() if st == "open"}
                pixel_tl = build_pixel_tl(grid, doors, open_doors=open_set)

            cell_src = _eff_src(worst)
            ok = self._render_cell_png(
                grid, resolution, ox, oy,
                worst["robot_x"], worst["robot_y"], cell_src,
                worst["pedestrians"],
                title=f"{row_label} / {col_label}  |  {worst['ped_max_exposure_dba']:.0f} dBA",
                out_path=png_path,
                downsample=downsample,
                vmin=vmin,
                vmax=global_max,
                pixel_tl=pixel_tl,
                doors=doors if pixel_tl is not None else None,
            )
            if ok:
                cells.append({
                    "row_val": row_label,
                    "col_label": col_label,
                    "img_rel_path": f"plots/{png_path.name}",
                    "exposure": worst["ped_max_exposure_dba"],
                })

        if not cells:
            return ""

        ncols = max(len(col_values), 1)
        html = (
            '<div style="display:grid;'
            f'grid-template-columns:repeat({ncols},1fr);'
            'gap:12px;margin-top:8px;">'
        )
        for c in cells:
            html += (
                f'<div style="text-align:center;font-size:0.78em;color:#475569;">'
                f'<img src="{c["img_rel_path"]}" style="width:100%;border-radius:4px;" '
                f'alt="{c["row_val"]}/{c["col_label"]}">'
                f'<br>{c["row_val"]} / {c["col_label"]} &mdash; {c["exposure"]:.0f} dBA'
                f'</div>'
            )
        html += '</div>'

        return html

    def render_seaborn(self, df, out_path):
        import matplotlib
        matplotlib.use("Agg")

        if compute_attenuations is None:
            return

        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            return

        map_name = work_df["map"][0] if "map" in work_df.columns else None
        if not map_name:
            return
        result = self._load_grid_and_meta(map_name, run_dir=self.run_dir)
        if result is None:
            return
        grid, meta = result
        resolution = meta["resolution"]
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        downsample = int(self.spec.options.get("downsample", 1))
        vmin = float(self.spec.options.get("vmin", _FIELD_VMIN_DBA))

        doors = door_segments(map_name, grid, resolution, (ox, oy, 0.0), run_dir=self.run_dir)

        worst = self._pick_worst_row(work_df)
        if worst is None:
            return

        src_dba = float(worst.get("source_dba", 60.0))
        if float(worst.get("ped_max_exposure_dba", 0)) >= 100.0 and src_dba < 100.0:
            src_dba = 100.0  # collision impulse
        vmax = float(np.ceil(src_dba / 10.0) * 10.0)
        if vmax <= vmin:
            vmax = vmin + 20.0

        pixel_tl = None
        door_states = worst.get("door_states") or {}
        if doors:
            open_set = {n for n, st in door_states.items() if st == "open"}
            pixel_tl = build_pixel_tl(grid, doors, open_doors=open_set)

        self._render_cell_png(
            grid, resolution, ox, oy,
            worst["robot_x"], worst["robot_y"], src_dba,
            worst["pedestrians"],
            title=f"{self.spec.title} (Robot: {src_dba:.0f} dBA)",
            out_path=out_path,
            downsample=downsample,
            vmin=vmin,
            vmax=vmax,
            pixel_tl=pixel_tl,
            doors=doors if pixel_tl is not None else None,
        )

    # ── Animation / timeseries ─────────────────────────────────────────────────

    @staticmethod
    def _parse_pedestrian_positions(row) -> list[tuple[float, float]]:
        """Parse one frame's pedestrian positions (flat or nested schema)."""
        pts: list[tuple[float, float]] = []
        if isinstance(row, str):
            import json
            try:
                row = json.loads(row)
            except Exception:
                row = []
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) == 0:
            return pts
        if isinstance(row[0], (list, tuple, np.ndarray)):
            for item in row:
                if len(item) >= 2 and not np.isnan(item[0]) and not np.isnan(item[1]):
                    pts.append((float(item[0]), float(item[1])))
        else:
            for j in range(0, len(row), 3):
                if j + 1 < len(row):
                    if not np.isnan(row[j]) and not np.isnan(row[j + 1]):
                        pts.append((float(row[j]), float(row[j + 1])))
        return pts

    @staticmethod
    def _load_episode_data(benchmark_dir: pathlib.Path, episode_id: str):
        """Load the time-aligned episode DataFrame from cached topic parquets.

        Returns a Polars DataFrame with columns ``time_ns``, ``pos_x_gt``,
        ``pos_y_gt``, ``source_dba``, ``peds_positions``, or ``None`` when
        the topic cache is missing.
        """
        topics_root = benchmark_dir / "episodes" / episode_id / "topics"
        if not topics_root.is_dir():
            logger.warning("No topics cache at %s — run 'evaluation extract' first.", topics_root)
            return None

        # Determine robot namespace from the first robot subdirectory
        robot_dirs = [d for d in topics_root.iterdir() if d.is_dir() and (d / "odom.parquet").exists()]
        if not robot_dirs:
            logger.warning("No robot odom data found in %s", topics_root)
            return None
        robot_dir = robot_dirs[0]

        frames = []
        # Robot ground-truth position (prefer tf_gt, fall back to odom)
        tf_gt_path = robot_dir / "tf_gt.parquet"
        odom_path = robot_dir / "odom.parquet"
        if tf_gt_path.exists():
            frames.append(pl.scan_parquet(tf_gt_path).select(["time_ns", "pos_x_gt", "pos_y_gt"]))
        elif odom_path.exists():
            frames.append(
                pl.scan_parquet(odom_path)
                .select(["time_ns", pl.col("pos_x").alias("pos_x_gt"), pl.col("pos_y").alias("pos_y_gt")])
            )
        else:
            return None

        # Source level from acoustics topic
        acoustics_path = robot_dir / "acoustics.parquet"
        if acoustics_path.exists():
            frames.append(
                pl.scan_parquet(acoustics_path)
                .select(["time_ns", pl.col("total_level_af_dba").alias("source_dba")])
            )
        else:
            # Fallback: look for acoustics in a differently-named file
            alt_path = robot_dir / "acoustic.parquet"
            if alt_path.exists():
                frames.append(
                    pl.scan_parquet(alt_path)
                    .select(["time_ns", pl.col("total_level_af_dba").alias("source_dba")])
                )

        # Pedestrian positions
        peds_path = topics_root / "peds.parquet"
        if peds_path.exists():
            frames.append(
                pl.scan_parquet(peds_path).select(["time_ns", "peds_positions"])
            )

        # Collision events (for 100 dBA impulse visualisation)
        collision_path = robot_dir / "collision_events.parquet"
        if collision_path.exists():
            frames.append(
                pl.scan_parquet(collision_path)
                .select(["time_ns", pl.col("collision_event").alias("has_collision")])
            )

        if not frames:
            return None

        # Align all topics on time_ns via forward-fill ASOF join
        df = frames[0].collect()
        for f in frames[1:]:
            right = f.collect()
            df = df.join_asof(right, on="time_ns", strategy="forward")

        return df

    def compute_field_timeseries(
        self,
        df: pl.DataFrame,
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        doors: dict,
        state_timeline=None,
        downsample: int = 1,
        stride: int = 1,
        max_frames: int = 120,
    ):
        """Compute the 2D acoustic field for a sequence of episode frames.

        Returns a list of ``(field_dba, eff_res, (h, w), open_set_frozen, source_dba)``
        tuples, one per rendered frame.
        """
        from ...processing.acoustics.door_state import DoorStateTimeline

        if compute_attenuations is None:
            logger.warning("C++ solver not available for timeseries.")
            return []

        if downsample > 1:
            grid = downsample_occupancy(grid, downsample)
            resolution = resolution * downsample

        h, w = grid.shape

        # Downsample door masks to match the solver grid (trim to grid dims:
        # stride slicing can leave one extra row vs the floor-divided grid)
        doors_ds = {}
        if doors:
            for name, (mask, tl_db) in doors.items():
                m_ds = mask[::downsample, ::downsample] if downsample > 1 else mask
                m_ds = m_ds[:h, :w]
                doors_ds[name] = (m_ds, tl_db)
        else:
            doors_ds = doors

        total_rows = len(df)

        # Determine frame indices. The stride samples the episode at its own
        # rate; when the cap kicks in the indices are spread EVENLY across the
        # whole episode (not taken from the start), so the animation covers
        # the full run rather than only its first seconds.
        indices = list(range(0, total_rows, stride))
        if len(indices) > max_frames:
            step = len(indices) / max_frames
            indices = [indices[int(i * step)] for i in range(max_frames)]

        logger.info(
            "compute_field_timeseries: %d frames (stride=%d, max=%d) on %dx%d grid",
            len(indices), stride, max_frames, w, h,
        )

        yy, xx = np.mgrid[0:h, 0:w]
        tx = np.ascontiguousarray(xx.flatten().astype(np.float32))
        ty = np.ascontiguousarray(yy.flatten().astype(np.float32))

        tl_cache: dict[tuple, np.ndarray] = {}
        results: list = []

        rows = df.rows(named=True)
        for frame_idx in indices:
            row = rows[frame_idx]

            rx_m = row.get("pos_x_gt")
            ry_m = row.get("pos_y_gt")
            source_dba = row.get("source_dba", _FIELD_VMIN_DBA)
            time_ns = int(row.get("time_ns", 0))

            if rx_m is None or ry_m is None or np.isnan(rx_m) or np.isnan(ry_m):
                results.append(None)
                continue

            rx_px = (float(rx_m) - ox) / resolution
            ry_px = (float(ry_m) - oy) / resolution

            # Collision impulse: boost source to 100 dBA so the field shows the crash
            is_collision = bool(row.get("has_collision", False))
            if is_collision:
                source_dba = 100.0

            # Look up door state at this timestamp
            open_set: frozenset = frozenset()
            if state_timeline is not None:
                open_set = state_timeline.open_doors_at(time_ns)

            tl_key = tuple(sorted(open_set))
            pixel_tl = tl_cache.get(tl_key)
            if pixel_tl is None:
                pixel_tl = build_pixel_tl(grid, doors_ds, open_doors=set(open_set))
                tl_cache[tl_key] = pixel_tl

            if source_dba is None or np.isnan(source_dba):
                source_dba = _FIELD_VMIN_DBA

            attenuations = compute_attenuations(
                occupancy_grid=grid,
                resolution=resolution,
                start_x_px=rx_px,
                start_y_px=ry_px,
                target_xs_px=tx,
                target_ys_px=ty,
                wall_tl=47.0,
                mic_distance=1.0,
                pixel_tl=pixel_tl,
            )

            att_grid = attenuations.reshape((h, w))
            field_dba = float(source_dba) - att_grid
            field_dba = np.clip(field_dba, 0, None)
            # (field, eff_res, (h, w), open_set, source_dba)
            results.append((field_dba, resolution, (h, w), open_set, float(source_dba)))

        logger.info("compute_field_timeseries: %d frames computed.", len([r for r in results if r is not None]))
        return results

    def render_animation(
        self,
        df: pl.DataFrame,
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        doors: dict,
        state_timeline=None,
        out_path: pathlib.Path | None = None,
        downsample: int = 1,
        stride: int = 1,
        max_frames: int = 120,
        fps: int = 10,
        dpi: int = 150,
        vmin: float | None = None,
        vmax: float | None = None,
        robot_trail: int = 0,
        show_doors: bool = True,
        fmt: str = "gif",
    ) -> pathlib.Path | None:
        """Render an animated GIF/MP4/PNG-sequence of the acoustic field timeseries.

        Returns the output path on success, ``None`` on failure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if vmin is None:
            vmin = _FIELD_VMIN_DBA

        fields = self.compute_field_timeseries(
            df, grid, resolution, ox, oy, doors,
            state_timeline=state_timeline,
            downsample=downsample, stride=stride, max_frames=max_frames,
        )

        # Filter out None frames (missing data)
        valid = [(i, f) for i, f in enumerate(fields) if f is not None]
        if not valid:
            logger.warning("render_animation: no valid frames to animate.")
            return None

        # Determine vmax from data if not specified. Use the 99th percentile
        # so a single outlier frame (e.g. an idle start at 42 dBA everywhere,
        # or a collision impulse) does not skew the whole colormap.
        if vmax is None:
            field_maxima = np.array([np.nanmax(f[0]) for _, f in valid], dtype=float)
            all_max = float(np.percentile(field_maxima, 99)) if len(field_maxima) > 0 else vmin
            vmax = float(np.ceil(all_max / 10.0) * 10.0)
            if vmax <= vmin:
                vmax = vmin + 20.0

        first_field, eff_res, (h, w), _, _ = valid[0][1]

        # Downsample grid once (same method as compute_field_timeseries)
        if downsample > 1:
            grid_ds = downsample_occupancy(grid, downsample)
        else:
            grid_ds = grid

        # Build downsampled door dict matching grid_ds dimensions (trim to
        # grid dims: stride slicing can leave one extra row vs the
        # floor-divided grid)
        doors_ds = {}
        door_mask_all = np.zeros((h, w), dtype=bool)
        if doors:
            for _name, (m, _tl) in doors.items():
                m_ds = m[::downsample, ::downsample] if downsample > 1 else m
                m_ds = m_ds[:h, :w]
                doors_ds[_name] = (m_ds, _tl)
                door_mask_all |= m_ds

        mask_grid = grid_ds  # already at the correct resolution

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_facecolor("black")
        ax.grid(False)
        extent = [ox, ox + w * eff_res, oy, oy + h * eff_res]

        # Pre-compute per-frame render grids
        render_grids = []
        open_door_masks = []
        for _, field_data in valid:
            field_dba, _, _, open_set, _ = field_data

            open_dm = np.zeros((h, w), dtype=bool)
            if doors_ds:
                pixel_tl = build_pixel_tl(grid_ds, doors_ds, open_doors=set(open_set))
                for _name, (m_ds, _tl) in doors_ds.items():
                    open_dm |= m_ds & (pixel_tl == 0.0)

            wall_or_closed = (mask_grid == 1) & ~open_dm
            rg = np.where(wall_or_closed | np.isinf(field_dba), np.nan, field_dba)
            rg = np.flipud(rg)
            render_grids.append(rg)
            open_door_masks.append(open_dm)

        # Plot the first frame
        im = ax.imshow(render_grids[0], cmap="inferno", origin="upper", extent=extent,
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="dBA", fraction=0.046, pad=0.04)

        # Subtle wall outline so room structure is always visible
        cy = np.linspace(extent[2], extent[3], mask_grid.shape[0])
        cx = np.linspace(extent[0], extent[1], mask_grid.shape[1])
        wall_outline = (mask_grid == 1).astype(np.uint8)
        ax.contour(cx, cy, wall_outline, levels=[0.5], colors=["#ffffff"],
                   linewidths=0.3, alpha=0.25)

        # Door contours — initial draw
        door_contour = None
        open_door_contour = None
        if show_doors and door_mask_all.any():
            door_contour = ax.contour(cx, cy, door_mask_all.astype(np.uint8), levels=[0.5],
                                      colors=["#00ffd5"], linewidths=1.2, alpha=0.85)
            if open_door_masks[0].any():
                open_door_contour = ax.contour(cx, cy, open_door_masks[0].astype(np.uint8),
                                               levels=[0.5], colors=["#00ff00"],
                                               linewidths=2.0, alpha=0.9)

        # Robot marker
        robot_dot, = ax.plot([], [], "g*", markersize=8, label="Robot")

        # Pedestrian markers
        ped_dots, = ax.plot([], [], "ro", markersize=4, label="Peds")

        # Robot trail
        trail_line = None
        if robot_trail > 0:
            trail_line, = ax.plot([], [], "g-", linewidth=0.8, alpha=0.4)

        title_text = ax.set_title("", fontsize=9)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Collect robot positions for trail
        robot_xs = []
        robot_ys = []
        rows = df.rows(named=True)
        total_rows = len(df)

        indices = list(range(0, total_rows, stride))
        if len(indices) > max_frames:
            step = len(indices) / max_frames
            indices = [indices[int(i * step)] for i in range(max_frames)]

        # Pre-compute pedestrian positions and collision flags per rendered
        # frame, keyed by the field's actual episode row index
        peds_per_frame = []
        has_collision_col = "has_collision" in df.columns
        collision_per_frame = []
        for orig_idx, _ in valid:
            row_idx = indices[orig_idx]
            row = rows[min(row_idx, total_rows - 1)]
            # Parse pedestrian positions
            peds_raw = row.get("peds_positions", [])
            peds = self._parse_pedestrian_positions(peds_raw)
            peds_per_frame.append(peds)
            # Collision flag
            coll = bool(row.get("has_collision", False)) if has_collision_col else False
            collision_per_frame.append(coll)

        # Red flash rectangle for collision frames (hidden initially)
        collision_flash = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         facecolor="red", alpha=0.0, zorder=10)
        ax.add_patch(collision_flash)
        # Store previous collision state to avoid redundant patch updates
        prev_collision = [False]  # mutable container for nonlocal

        def update(enum_tuple):
            nonlocal open_door_contour
            rg_idx, (orig_idx, field_data) = enum_tuple
            field_dba, _, _, open_set, source_dba = field_data
            rg = render_grids[rg_idx]

            im.set_data(rg)

            # Update robot position from the field's actual episode row
            row_idx = indices[orig_idx]
            row = rows[min(row_idx, total_rows - 1)]
            rx_m = float(row.get("pos_x_gt", 0) or 0)
            ry_m = float(row.get("pos_y_gt", 0) or 0)
            robot_dot.set_data([rx_m], [ry_m])
            robot_xs.append(rx_m)
            robot_ys.append(ry_m)

            if trail_line is not None:
                trail = robot_xs[-robot_trail:] if len(robot_xs) > robot_trail else robot_xs
                trail_y = robot_ys[-robot_trail:] if len(robot_ys) > robot_trail else robot_ys
                trail_line.set_data(trail, trail_y)

            # Update pedestrian positions
            peds = peds_per_frame[rg_idx]
            if peds:
                px = [p[0] for p in peds]
                py = [p[1] for p in peds]
                ped_dots.set_data(px, py)
            else:
                ped_dots.set_data([], [])

            # Collision indicator
            is_collision = collision_per_frame[rg_idx]
            if is_collision != prev_collision[0]:
                collision_flash.set_alpha(0.25 if is_collision else 0.0)
                prev_collision[0] = is_collision

            # Update door contours — remove old, redraw current
            if show_doors and open_door_masks[rg_idx].any():
                if open_door_contour is not None:
                    open_door_contour.remove()
                open_door_contour = ax.contour(cx, cy, open_door_masks[rg_idx].astype(np.uint8),
                                               levels=[0.5], colors=["#00ff00"],
                                               linewidths=2.0, alpha=0.9)

            collision_label = " COLLISION 100dBA!" if is_collision else ""
            title_text.set_text(
                f"t={row.get('time_ns', 0) / 1e9:.1f}s  |  "
                f"source={source_dba:.0f} dBA  |  "
                f"{len(open_set)} doors open"
                f"{collision_label}"
            )

            artists = [im, robot_dot, ped_dots, title_text]
            if trail_line is not None:
                artists.append(trail_line)
            return artists

        ani = animation.FuncAnimation(
            fig, update, frames=list(enumerate(valid)), interval=1000 // fps, blit=True
        )

        if out_path is None:
            out_path = pathlib.Path("plots") / "acoustic_animation.gif"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if fmt == "gif":
                ani.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
            elif fmt == "mp4":
                ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
            elif fmt == "frames":
                frames_dir = out_path.with_suffix("")
                frames_dir.mkdir(parents=True, exist_ok=True)
                for fi, (_, field_data) in enumerate(valid):
                    frame_path = frames_dir / f"frame_{fi:04d}.png"
                    field_dba, _, _, open_set, source_dba = field_data
                    open_dm = open_door_masks[fi] if show_doors else np.zeros((h, w), dtype=bool)
                    wall_or_closed = (mask_grid == 1) & ~open_dm
                    rg = np.where(wall_or_closed | np.isinf(field_dba), np.nan, field_dba)
                    rg = np.flipud(rg)
                    fig_frame, ax_frame = plt.subplots(figsize=(5, 4))
                    ax_frame.set_facecolor("black")
                    ax_frame.imshow(rg, cmap="inferno", origin="upper", extent=extent,
                                    vmin=vmin, vmax=vmax)
                    ax_frame.contour(cx, cy, wall_outline, levels=[0.5], colors=["#ffffff"],
                                     linewidths=0.3, alpha=0.25)
                    ax_frame.plot(rx_m, ry_m, "g*", markersize=8)
                    if show_doors and door_mask_all.any():
                        ax_frame.contour(cx, cy, door_mask_all.astype(np.uint8), levels=[0.5],
                                         colors=["#00ffd5"], linewidths=1.2, alpha=0.85)
                        if open_dm.any():
                            ax_frame.contour(cx, cy, open_dm.astype(np.uint8), levels=[0.5],
                                             colors=["#00ff00"], linewidths=2.0, alpha=0.9)
                    ax_frame.set_title(f"Frame {fi}", fontsize=9)
                    plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig_frame)
                logger.info("Saved %d frames to %s", len(valid), frames_dir)
            else:
                logger.warning("Unknown format %r — falling back to gif.", fmt)
                ani.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
        except Exception as e:
            logger.warning("Failed to save animation: %s", e)
            plt.close(fig)
            return None

        plt.close(fig)
        logger.info("Animation saved to %s (%d frames, %d fps)", out_path, len(valid), fps)
        return out_path


class AcousticFieldAnimationRenderer(AcousticFieldRenderer):
    """Manifest-driven acoustic field animation (GIF during report)."""

    PLOT_TYPE = "acoustic_field_animation"

    # ── Manifest-driven options (inherited + animation-specific) ──────────────
    #  options.fps:         output frame rate (default 10)
    #  options.max_frames:  cap on rendered frames (default 120)
    #  options.stride:      render every Nth data frame (default auto)
    #  options.downsample:  solver grid stride (default 2 for performance)
    #  options.format:      "gif" (default) | "mp4" | "frames"

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Generate acoustic-field animations.

        Default: one GIF of the worst episode. With ``per_episode: true`` in
        the spec options: one GIF per episode, saved as
        ``plots/{spec.id}_episode_XXX.{fmt}``.
        """
        if compute_attenuations is None:
            logger.warning("AcousticFieldAnimationRenderer: C++ solver not available.")
            return

        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            return

        map_name = work_df["map"][0] if "map" in work_df.columns else None
        if not map_name:
            return

        result = self._load_grid_and_meta(map_name, run_dir=self.run_dir)
        if result is None:
            return
        grid, meta = result
        resolution = meta["resolution"]
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

        downsample = int(self.spec.options.get("downsample", 2))
        fps = int(self.spec.options.get("fps", 10))
        max_frames = int(self.spec.options.get("max_frames", 120))
        stride = int(self.spec.options.get("stride", 1))
        fmt = str(self.spec.options.get("format", "gif"))
        vmin = float(self.spec.options.get("vmin", _FIELD_VMIN_DBA))

        doors = door_segments(map_name, grid, resolution, (ox, oy, 0.0), run_dir=self.run_dir)
        benchmark_dir = self.run_dir if self.run_dir else pathlib.Path(".")

        self._rendered_gifs: list[str] = []

        # ── Per-episode mode: one GIF per episode ──
        per_episode = bool(self.spec.options.get("per_episode", False))
        if per_episode and "episode" in work_df.columns:
            for ep_id in sorted(int(v) for v in work_df["episode"].unique().to_list()):
                gif_path = self._render_episode_gif(
                    ep_id, benchmark_dir, out_path,
                    grid, resolution, ox, oy, doors,
                    downsample, fps, max_frames, stride, fmt, vmin,
                    per_episode=True,
                )
                if gif_path is not None:
                    self._rendered_gifs.append(f"plots/{gif_path.name}")
            return

        # ── Worst-episode mode (default) ──
        worst = self._pick_worst_row(work_df)
        if worst is None:
            return
        episode_id = None
        if "episode" in work_df.columns:
            max_idx = work_df["ped_max_exposure_dba"].arg_max()
            if max_idx is not None:
                episode_id = int(work_df.row(max_idx, named=True).get("episode", 0))
        if episode_id is None:
            logger.warning("Cannot determine episode ID for animation.")
            return
        gif_path = self._render_episode_gif(
            episode_id, benchmark_dir, out_path,
            grid, resolution, ox, oy, doors,
            downsample, fps, max_frames, stride, fmt, vmin,
            per_episode=False,
        )
        if gif_path is not None:
            self._rendered_gifs.append(f"plots/{gif_path.name}")

    def _render_episode_gif(
        self, episode_id: int, benchmark_dir: pathlib.Path, out_path: pathlib.Path,
        grid: np.ndarray, resolution: float, ox: float, oy: float,
        doors: dict, downsample: int, fps: int, max_frames: int, stride: int,
        fmt: str, vmin: float, per_episode: bool = False,
    ) -> pathlib.Path | None:
        """Render one episode's acoustic-field animation; returns the GIF path."""
        episode_dir_name = f"episode_{int(episode_id):03d}"

        # run_dir may be the output dir; the benchmark may be a parent
        bdir = benchmark_dir
        if not (bdir / "episodes" / episode_dir_name).is_dir():
            for parent in bdir.parents:
                if (parent / "episodes" / episode_dir_name).is_dir():
                    bdir = parent
                    break

        episode_df = self._load_episode_data(bdir, episode_dir_name)
        if episode_df is None:
            logger.warning("No episode data for %s", episode_dir_name)
            return None

        from ...processing.acoustics.door_state import DoorStateTimeline

        semantic_path = bdir / "episodes" / episode_dir_name / "topics" / "semantic_snapshot.parquet"
        state_timeline = None
        if semantic_path.exists():
            semantic_df = pl.read_parquet(semantic_path)
            state_timeline = DoorStateTimeline.from_semantic_frame(semantic_df)

        if per_episode:
            gif_path = out_path.with_name(f"{out_path.stem}_{episode_dir_name}.{fmt}")
        else:
            gif_path = out_path.with_suffix(f".{fmt}" if fmt != "frames" else "")
            if fmt == "frames":
                gif_path = out_path.with_suffix("")

        try:
            self.render_animation(
                episode_df, grid, resolution, ox, oy, doors,
                state_timeline=state_timeline,
                out_path=gif_path,
                downsample=downsample, stride=stride, max_frames=max_frames,
                fps=fps, vmin=vmin, vmax=None,
                show_doors=bool(doors),
                fmt=fmt,
            )
        except Exception as e:
            logger.warning("Failed to render animation for %s: %s", episode_dir_name, e)
            return None
        logger.info("Animation saved to %s", gif_path)
        return gif_path

    def render_plotly(self, df: pl.DataFrame) -> str | list[str]:
        """Return HTML snippet(s) embedding the animated GIF(s) in the report.

        One snippet per GIF produced by ``render_seaborn`` (worst episode by
        default, one per episode with ``per_episode: true``).
        """
        work_df = self._prepared_df(df)
        if len(work_df) == 0 or compute_attenuations is None:
            return ""

        # The render_seaborn call already saved the GIF(s) during the report build;
        # we reference the files it produced (plots/<name>.gif).
        fmt = str(self.spec.options.get("format", "gif"))
        ext = "gif" if fmt in ("gif", "mp4") else ""
        if not ext:
            return ""

        per_episode = bool(self.spec.options.get("per_episode", False))
        if per_episode and "episode" in work_df.columns:
            chunks = []
            for ep_id in sorted(int(v) for v in work_df["episode"].unique().to_list()):
                gif_rel = f"plots/{self.spec.id}_episode_{ep_id:03d}.{ext}"
                caption = f"{self.spec.title} — episode_{ep_id:03d}"
                chunks.append(
                    f'<div style="text-align:center;">'
                    f'<img src="{gif_rel}" style="max-width:100%;border-radius:4px;" '
                    f'alt="{self.spec.title}">'
                    f'<br><span style="font-size:0.78em;color:#475569;">'
                    f'{caption}'
                    f'</span></div>'
                )
            return chunks

        gif_rel = f"plots/{self.spec.id}.{ext}"
        return (
            f'<div style="text-align:center;">'
            f'<img src="{gif_rel}" style="max-width:100%;border-radius:4px;" '
            f'alt="{self.spec.title}">'
            f'<br><span style="font-size:0.78em;color:#475569;">'
            f'{self.spec.title}'
            f'</span></div>'
        )

