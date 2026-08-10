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

try:
    from ...processing.acoustics.impedance_grid import compute_attenuations
except ImportError:
    compute_attenuations = None

logger = logging.getLogger(__name__)

_CELL_FIGSIZE = (5, 4)
_CELL_DPI = 150

# Hospital ambient sound level (dBA) — color-scale floor. Matches
# _ACOUSTIC_DEFAULTS["L_base_0"] in the characterization calculator.
_HOSPITAL_AMBIENT_DBA = 42.0


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
                            rx_m, ry_m, source_dba, downsample=1):
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
        )

        att_grid = attenuations.reshape((h, w))
        field_dba = source_dba - att_grid
        field_dba = np.clip(field_dba, 0, None)
        logger.info("AcousticFieldRenderer: full-field done.")
        return field_dba, resolution, (h, w)

    def _render_cell_png(self, grid, resolution, ox, oy,
                         rx_m, ry_m, source_dba, peds, title, out_path,
                         downsample=1, vmin=None, vmax=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._compute_full_field(grid, resolution, ox, oy,
                                          rx_m, ry_m, source_dba,
                                          downsample=downsample)
        if result is None:
            return False
        field_dba, eff_res, (h, w) = result

        if downsample > 1:
            mask_grid = grid[::downsample, ::downsample]
        else:
            mask_grid = grid

        render_grid = np.where((mask_grid == 1) | np.isinf(field_dba), np.nan, field_dba)
        render_grid = np.flipud(render_grid)

        plt.figure(figsize=_CELL_FIGSIZE)
        ax = plt.gca()
        ax.set_facecolor("black")
        ax.grid(False)
        extent = [ox, ox + w * eff_res, oy, oy + h * eff_res]

        im = plt.imshow(render_grid, cmap="inferno", origin="upper", extent=extent,
                        vmin=vmin, vmax=vmax)
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
        vmin = float(self.spec.options.get("vmin", _HOSPITAL_AMBIENT_DBA))
        mode = str(self.spec.options.get("mode", "grid")).lower()

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

            vmax = float(np.ceil(worst["ped_max_exposure_dba"] / 10.0) * 10.0)
            if vmax <= vmin:
                vmax = vmin + 20.0

            ok = self._render_cell_png(
                grid, resolution, ox, oy,
                worst["robot_x"], worst["robot_y"], worst["source_dba"],
                worst["pedestrians"],
                title=f"{row_label} / {col_label}  |  {worst['ped_max_exposure_dba']:.0f} dBA",
                out_path=png_path,
                downsample=downsample,
                vmin=vmin,
                vmax=vmax,
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

        global_max = max((w["ped_max_exposure_dba"] for w, _, _ in entries), default=80.0)
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

            ok = self._render_cell_png(
                grid, resolution, ox, oy,
                worst["robot_x"], worst["robot_y"], worst["source_dba"],
                worst["pedestrians"],
                title=f"{row_label} / {col_label}  |  {worst['ped_max_exposure_dba']:.0f} dBA",
                out_path=png_path,
                downsample=downsample,
                vmin=vmin,
                vmax=global_max,
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
        vmin = float(self.spec.options.get("vmin", _HOSPITAL_AMBIENT_DBA))

        worst = self._pick_worst_row(work_df)
        if worst is None:
            return

        vmax = float(np.ceil(worst["ped_max_exposure_dba"] / 10.0) * 10.0)
        if vmax <= vmin:
            vmax = vmin + 20.0

        self._render_cell_png(
            grid, resolution, ox, oy,
            worst["robot_x"], worst["robot_y"], worst["source_dba"],
            worst["pedestrians"],
            title=f"{self.spec.title} (Robot: {worst['source_dba']:.1f} dBA)",
            out_path=out_path,
            downsample=downsample,
            vmin=vmin,
            vmax=vmax,
        )
