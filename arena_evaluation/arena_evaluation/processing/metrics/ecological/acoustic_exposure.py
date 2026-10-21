from __future__ import annotations

import logging
import pathlib
import typing

import numpy as np
import polars as pl
from PIL import Image

from arena_evaluation.processing.acoustics.door_map import (
    _entity_matches_door,
    build_pixel_tl,
    door_segments,
)
from arena_evaluation.processing.acoustics.door_state import (
    ACOUSTIC_OPEN_PROGRESS_THRESHOLD,
    DoorStateTimeline,
)
from arena_evaluation.processing.map_registry import MapRegistry
from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.characterization import _ACOUSTIC_DEFAULTS
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

try:
    from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations
except (ImportError, OSError, RuntimeError):  # C++ extension unavailable
    compute_attenuations = None

logger = logging.getLogger(__name__)

_PED_THRESHOLD_BANDS: tuple[tuple[float, float], ...] = (
    (2.0, 0.05),
    (5.0, 0.50),
    (10.0, 1.00),
    (20.0, 1.50),
)
_PED_THRESHOLD_REMOTE = 2.00


def _ped_thresholds(p_dist: np.ndarray) -> np.ndarray:
    """Recompute threshold per pedestrian, finer when the pedestrian is close."""
    return np.select(
        [p_dist < limit for limit, _ in _PED_THRESHOLD_BANDS],
        [threshold for _, threshold in _PED_THRESHOLD_BANDS],
        default=_PED_THRESHOLD_REMOTE,
    )


class AcousticExposureCalculator(BaseMetricCalculator):
    """Computes pedestrian exposure to robotic ego-noise over the 2D impedance map."""

    NAME = "acoustic_exposure"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["proxemics"]
    REQUIRED_TOPICS = ["tf_gt", "peds"]

    UNITS = {
        "ped_max_exposure_dba": "dBA",
        "ped_leq_exposure_dba": "dBA",
        "timeseries_acoustic_exposure_dba": "dBA",
        "timeseries_acoustic_attenuation_db": "dB",
    }

    world: str | None = None
    maps_cache: pathlib.Path | None = None

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ped_max_exposure_dba",
            "ped_leq_exposure_dba",
            "ped_max_startle_factor",
            "timeseries_acoustic_exposure_dba",
            "timeseries_acoustic_attenuation_db",
            "worst_case_acoustic_frame",
        ]

    @staticmethod
    def _parse_pedestrian_positions(row: object) -> list[tuple[float, float]]:
        """Parse a single frame's pedestrian positions (flat or nested schema)."""
        pts: list[tuple[float, float]] = []
        if isinstance(row, str):
            import json

            try:
                row = json.loads(row)
            except Exception:
                row = []
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) == 0:
            return pts
        if isinstance(row[0], dict):
            for item in row:
                if isinstance(item, dict) and "x" in item and "y" in item:
                    pts.append((float(item["x"]), float(item["y"])))
        elif isinstance(row[0], (list, tuple, np.ndarray)):
            for item in row:
                if len(item) >= 2 and not np.isnan(item[0]) and not np.isnan(item[1]):
                    pts.append((float(item[0]), float(item[1])))
        else:
            for j in range(0, len(row), 3):
                if j + 1 < len(row):
                    if not np.isnan(row[j]) and not np.isnan(row[j + 1]):
                        pts.append((float(row[j]), float(row[j + 1])))
        return pts

    def _get_map_occupancy(self, map_name: str, run_dir: pathlib.Path | None = None) -> tuple[np.ndarray, float, tuple[float, float, float]] | None:
        """Load the map PNG as a binary occupancy grid, flipped so row 0 = bottom (y = origin_y)."""
        meta = MapRegistry.get_map(map_name, cache_dir=self.maps_cache, run_dir=run_dir)
        if not meta or "png_path" not in meta:
            return None
        try:
            img = Image.open(meta["png_path"]).convert("L")
            img_data = np.array(img)
            grid = np.ascontiguousarray(np.flipud((img_data < 200).astype(np.uint8)))
            return grid, meta["resolution"], meta["origin"]
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load map image for acoustics: {e}")
            return None

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        nulls = {k: None for k in self.output_keys()}

        if episode.is_reference or (episode.run is not None and episode.run.is_reference):
            logger.info("Skipping acoustic calculation for reference episode %s", episode.episode_id)
            return nulls

        if compute_attenuations is None:
            logger.warning("Acoustic solver unavailable (C++ extension failed to build); skipping episode %s", episode.episode_id)
            return nulls

        df = episode.data
        if df is None or len(df) == 0:
            logger.debug("No episode data for episode %s", episode.episode_id)
            return nulls

        map_val = episode.map

        map_name = prior_results.get("map", map_val)
        if not map_name:
            # registry seeds calc.world from the benchmark metadata
            map_name = self.world
        if not map_name:
            logger.warning("No map name available for episode %s - skipping acoustics", episode.episode_id)
            return nulls

        run_dir = None
        if episode.folder_manager:
            if episode.run is not None and episode.run.benchmark_id:
                run_dir = episode.folder_manager.data_root / episode.run.benchmark_id

        map_data = self._get_map_occupancy(map_name, run_dir=run_dir)
        if not map_data:
            logger.warning("Failed to load map occupancy for '%s' (episode %s)", map_name, episode.episode_id)
            return nulls

        grid, resolution, origin = map_data
        ox, oy = origin[0], origin[1]

        doors = door_segments(map_name, grid, resolution, origin, run_dir=run_dir)
        tl_cache: dict[tuple[str, ...], np.ndarray] = {}
        state_timeline = DoorStateTimeline.from_semantic_frame(
            episode.semantic_snapshot,
            progress_threshold=ACOUSTIC_OPEN_PROGRESS_THRESHOLD,
        )
        if doors:
            logger.info(
                "AcousticExposureCalculator: %d semantic doors loaded; semantic timeline %s",
                len(doors),
                "present" if state_timeline is not None else "ABSENT (doors default closed)",
            )

        if "pos_x_gt" not in df.columns or "pos_y_gt" not in df.columns:
            logger.warning("Ground-truth position columns missing for episode %s", episode.episode_id)
            return nulls

        # tf_gt publishes at ~10 Hz; the aligner tolerates 100 ms, so many frames carry a null pose.
        rx_m = df["pos_x_gt"].cast(pl.Float64).fill_null(strategy="forward").fill_null(0.0).to_numpy()
        ry_m = df["pos_y_gt"].cast(pl.Float64).fill_null(strategy="forward").fill_null(0.0).to_numpy()

        # Source level (from acoustics topic, or fallback)
        if "total_level_af_dba" in df.columns:
            source_dba = df["total_level_af_dba"].cast(pl.Float64).fill_null(strategy="forward").fill_null(_ACOUSTIC_DEFAULTS["L_base_0"]).to_numpy()
        else:
            source_dba = np.full(len(rx_m), _ACOUSTIC_DEFAULTS["L_base_0"])

        # Pedestrian positions
        if "peds_positions" not in df.columns:
            logger.warning("Pedestrian position data missing for episode %s", episode.episode_id)
            return nulls

        peds_pos = df["peds_positions"].to_list()

        ts_exposure: list[list[float]] = []
        ts_attenuation: list[list[float]] = []

        last_eval_doors: frozenset[str] | None = None
        last_eval_robot = np.zeros((0, 2), dtype=np.float64)
        last_eval_peds = np.zeros((0, 2), dtype=np.float64)
        current_attenuations = np.array([], dtype=np.float32)

        total_frames = len(rx_m)
        time_ns = df["time_ns"].to_numpy()

        solver_evals = 0
        scored_frames = 0

        for i in range(total_frames):
            pts = self._parse_pedestrian_positions(peds_pos[i])

            if not pts:
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            px_m = np.array([p[0] for p in pts], dtype=np.float32)
            py_m = np.array([p[1] for p in pts], dtype=np.float32)

            current_source = source_dba[i]

            open_set = state_timeline.open_doors_at(int(time_ns[i])) if state_timeline is not None else frozenset()
            num_peds = len(pts)

            if len(current_attenuations) != num_peds:
                current_attenuations = np.full(num_peds, np.inf, dtype=np.float32)
                last_eval_robot = np.zeros((num_peds, 2), dtype=np.float64)
                last_eval_peds = np.zeros((num_peds, 2), dtype=np.float64)
                indices_to_eval = list(range(num_peds))
            elif last_eval_doors is None or open_set != last_eval_doors:
                indices_to_eval = list(range(num_peds))
            else:
                p_dist = np.hypot(px_m - rx_m[i], py_m - ry_m[i])
                thresh = _ped_thresholds(p_dist)
                r_disp = np.hypot(rx_m[i] - last_eval_robot[:, 0], ry_m[i] - last_eval_robot[:, 1])
                p_disp = np.hypot(px_m - last_eval_peds[:, 0], py_m - last_eval_peds[:, 1])
                indices_to_eval = np.flatnonzero((r_disp > thresh) | (p_disp > thresh)).tolist()
                if i == total_frames - 1 and not indices_to_eval:
                    indices_to_eval = list(range(num_peds))

            if indices_to_eval:
                rx_px = (rx_m[i] - ox) / resolution
                ry_px = (ry_m[i] - oy) / resolution

                sub_px = ((px_m[indices_to_eval] - ox) / resolution).astype(np.float32)
                sub_py = ((py_m[indices_to_eval] - oy) / resolution).astype(np.float32)

                tl_key = tuple(sorted(open_set))
                pixel_tl = tl_cache.get(tl_key)
                if pixel_tl is None:
                    pixel_tl = build_pixel_tl(grid, doors, open_doors=set(open_set))
                    tl_cache[tl_key] = pixel_tl

                sub_atts = compute_attenuations(
                    occupancy_grid=grid,
                    resolution=resolution,
                    start_x_px=rx_px,
                    start_y_px=ry_px,
                    target_xs_px=sub_px,
                    target_ys_px=sub_py,
                    wall_tl=47.0,
                    mic_distance=1.0,
                    pixel_tl=pixel_tl,
                )

                current_attenuations[indices_to_eval] = sub_atts
                last_eval_robot[indices_to_eval] = (rx_m[i], ry_m[i])
                last_eval_peds[indices_to_eval] = np.column_stack((px_m[indices_to_eval], py_m[indices_to_eval]))

                last_eval_doors = open_set
                solver_evals += 1

            attenuations = current_attenuations

            # Filter out infinity (unreachable)
            valid = ~np.isinf(attenuations)
            if not np.any(valid):
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            att_valid = attenuations[valid]
            # SPL received = instantaneous source level - geometric attenuation
            exp_valid = current_source - att_valid

            ts_attenuation.append(att_valid.tolist())
            ts_exposure.append(exp_valid.tolist())

            scored_frames += 1
            if scored_frames % 50 == 0:
                logger.info(
                    "AcousticExposureCalculator: Evaluated %d frames... (%d/%d total frames processed)",
                    scored_frames,
                    i + 1,
                    total_frames,
                )

        logger.info(
            "AcousticExposureCalculator: Finished episode %s with %d unique solver evaluations.",
            episode.episode_id,
            solver_evals,
        )

        # Post-process for scalar metrics
        time_s = time_ns / 1e9

        n_frames = len(ts_exposure)
        frame_dt = np.zeros(n_frames, dtype=np.float64)
        if n_frames > 1:
            frame_dt[:-1] = np.maximum(np.diff(time_s), 0.0)
            frame_dt[-1] = frame_dt[-2]

        all_exp: list[float] = []
        all_w: list[float] = []
        for i, frame_exps in enumerate(ts_exposure):
            if not frame_exps:
                continue
            all_exp.extend(frame_exps)
            all_w.extend([float(frame_dt[i])] * len(frame_exps))

        if not all_exp:
            logger.info("No acoustic exposure values produced for episode %s", episode.episode_id)
            return {
                **nulls,
                "timeseries_acoustic_exposure_dba": ts_exposure,
                "timeseries_acoustic_attenuation_db": ts_attenuation,
            }

        exp_arr = np.array(all_exp)
        all_w_arr = np.array(all_w)

        max_exp = float(np.max(exp_arr))

        lin_exp = 10 ** (exp_arr / 10.0)
        weight_sum = float(np.sum(all_w_arr))
        if weight_sum > 0.0:
            leq_exp = float(10 * np.log10(np.sum(all_w_arr * lin_exp) / weight_sum))
        else:
            leq_exp = float(10 * np.log10(np.mean(lin_exp)))

        startle_rates: list[float] = []
        for i in range(1, len(ts_exposure)):
            prev = ts_exposure[i - 1]
            curr = ts_exposure[i]
            dt = time_s[i] - time_s[i - 1]
            if dt > 0 and len(prev) == len(curr) and len(curr) > 0:
                diffs = np.array(curr) - np.array(prev)
                rates = diffs / dt
                startle_rates.extend(rates.tolist())

        max_startle = float(max([r for r in startle_rates if r > 0.0], default=0.0))

        # Find worst-case frame (highest pedestrian exposure)
        max_idx = 0
        max_val = -1.0
        for i, frame_exps in enumerate(ts_exposure):
            if frame_exps:
                fm = max(frame_exps)
                if fm > max_val:
                    max_val = fm
                    max_idx = i

        # Re-parse the pedestrians for the worst-case frame
        worst_pts = self._parse_pedestrian_positions(peds_pos[max_idx])
        worst_frame = {
            "robot_x": float(rx_m[max_idx]),
            "robot_y": float(ry_m[max_idx]),
            "source_dba": float(source_dba[max_idx]),
            "pedestrians": [[float(p[0]), float(p[1])] for p in worst_pts],
            "door_states": {name: ("open" if state_timeline is not None and any(_entity_matches_door(name, e) for e in state_timeline.open_doors_at(int(df["time_ns"][max_idx]))) else "closed") for name in doors},
        }

        logger.info(
            "AcousticExposureCalculator: episode %s, max_exp=%.1f dBA, leq=%.1f dBA, startle=%.2f dBA/s",
            episode.episode_id,
            max_exp,
            leq_exp,
            max_startle,
        )

        return {
            "ped_max_exposure_dba": max_exp,
            "ped_leq_exposure_dba": leq_exp,
            "ped_max_startle_factor": max_startle,
            "timeseries_acoustic_exposure_dba": ts_exposure,
            "timeseries_acoustic_attenuation_db": ts_attenuation,
            "worst_case_acoustic_frame": worst_frame,
        }
