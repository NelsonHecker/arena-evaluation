# SESSION SNAPSHOT (2026-08-10) — current acoustic_exposure.py as of end of session.
# Path: src/Arena/arena_evaluation/arena_evaluation/arena_evaluation/processing/metrics/ecological/acoustic_exposure.py
# NOTE: this is the version WITH today's fixes (ffill positions+source, per-pedestrian collision impulse).
from __future__ import annotations
import typing
import logging
import numpy as np
import polars as pl
from PIL import Image

from ..base import BaseMetricCalculator
from ...map_registry import MapRegistry
from ..ecological.characterization import _ACOUSTIC_DEFAULTS

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle

# Import the C++ solver wrapper
try:
    from ...acoustics.impedance_grid import compute_attenuations
except ImportError:
    compute_attenuations = None

logger = logging.getLogger(__name__)


class AcousticExposureCalculator(BaseMetricCalculator):
    """Computes pedestrian exposure to robotic ego-noise using a multi-criteria
    Acoustic Dijkstra solver over the 2D impedance map."""
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
    def _parse_pedestrian_positions(row) -> list[tuple[float, float]]:
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

    def _get_map_occupancy(self, map_name: str, run_dir=None) -> tuple[np.ndarray, float, tuple[float, float, float]] | None:
        """Load the map PNG as a binary occupancy grid, flipped so row 0 = bottom (y = origin_y)."""
        meta = MapRegistry.get_map(map_name, run_dir=run_dir)
        if not meta or "png_path" not in meta:
            return None
        try:
            img = Image.open(meta["png_path"]).convert("L")
            img_data = np.array(img)
            # PIL row 0 = top of map; flip so row 0 = bottom (y = origin_y).
            # ascontiguousarray required: flipud returns a view that is not C_CONTIGUOUS.
            grid = np.ascontiguousarray(np.flipud((img_data < 200).astype(np.uint8)))
            return grid, meta["resolution"], meta["origin"]
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load map image for acoustics: {e}")
            return None

    def calculate(self, episode: "AlignedEpisodeBundle", prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        nulls = {k: None for k in self.output_keys()}

        # Skip heavy calculation for reference runs
        if getattr(episode.run, "is_reference", False):
            logger.info("Skipping acoustic calculation for reference episode %s", episode.episode_id)
            return nulls

        if compute_attenuations is None:
            logger.warning("Acoustics C++ solver not available.")
            return nulls

        df = episode.data
        if df is None or len(df) == 0:
            logger.debug("No episode data for episode %s", episode.episode_id)
            return nulls

        map_val = None
        if hasattr(episode, "map"):
            map_val = episode.map
        elif hasattr(episode, "run") and hasattr(episode.run, "map"):
            map_val = episode.run.map

        map_name = prior_results.get("map", map_val)
        if not map_name:
            logger.warning("No map name available for episode %s — skipping acoustics", episode.episode_id)
            return nulls

        run_dir = None
        if hasattr(episode, "folder_manager") and episode.folder_manager:
            if hasattr(episode.run, "benchmark_id") and episode.run.benchmark_id:
                run_dir = episode.folder_manager.data_root / episode.run.benchmark_id

        map_data = self._get_map_occupancy(map_name, run_dir=run_dir)
        if not map_data:
            logger.warning("Failed to load map occupancy for '%s' (episode %s)", map_name, episode.episode_id)
            return nulls

        grid, resolution, origin = map_data
        ox, oy = origin[0], origin[1]

        if "pos_x_gt" not in df.columns or "pos_y_gt" not in df.columns:
            logger.warning("Ground-truth position columns missing for episode %s", episode.episode_id)
            return nulls

        # tf_gt publishes at ~10 Hz with jitter, so ~40% of aligned frames have
        # null ground-truth pose (aligner tolerance = 100 ms == tf_gt period).
        # Forward-fill so the solver never sees NaN positions (which caused
        # all-inf fields / blank visualizations).
        rx_m = df["pos_x_gt"].cast(pl.Float64).fill_null(strategy="forward").fill_null(0.0).to_numpy()
        ry_m = df["pos_y_gt"].cast(pl.Float64).fill_null(strategy="forward").fill_null(0.0).to_numpy()

        # Source level (from acoustics topic, or fallback)
        if "total_level_af_dba" in df.columns:
            # Forward-fill source dropouts; leading nulls fall back to the idle
            # baseline. fill_null(0.0) previously made gap frames "silent"
            # (exposure ~= -attenuation), dragging Leq down.
            source_dba = (
                df["total_level_af_dba"].cast(pl.Float64)
                .fill_null(strategy="forward")
                .fill_null(_ACOUSTIC_DEFAULTS["L_base_0"])
                .to_numpy()
            )
        else:
            # Fallback to constant idle noise if acoustics topic missing
            source_dba = np.full(len(rx_m), _ACOUSTIC_DEFAULTS["L_base_0"])

        # Pedestrian positions
        if "peds_positions" not in df.columns:
            logger.warning("Pedestrian position data missing for episode %s", episode.episode_id)
            return nulls

        peds_pos = df["peds_positions"].to_list()

        ts_exposure: list[list[float]] = []
        ts_attenuation: list[list[float]] = []

        last_eval_rx = None
        last_eval_ry = None
        last_eval_peds = None
        last_eval_source = None

        POS_THRESHOLD = 0.1  # meters
        SRC_THRESHOLD = 1.0  # dB

        total_frames = len(rx_m)

        # Pre-compute collision frame mask (used for per-pedestrian impulse).
        COLLISION_IMPULSE_DBA = 100.0
        collision_mask = None
        if "collision_event" in df.columns:
            col_events = df["collision_event"].to_numpy()
            col_clean = np.nan_to_num(col_events.astype(float), nan=0.0)
            collision_mask = col_clean > 0
            n_collisions = int(np.sum(collision_mask))
            if n_collisions > 0:
                logger.info(
                    "AcousticExposureCalculator: %d collision frames detected -- "
                    "will apply %.0f dB(A) per-pedestrian impulse.",
                    n_collisions, COLLISION_IMPULSE_DBA,
                )

        logger.info(
            "AcousticExposureCalculator: Starting Dijkstra propagation for %d frames (episode %s)",
            total_frames, episode.episode_id,
        )
        eval_count = 0

        for i in range(total_frames):
            pts = self._parse_pedestrian_positions(peds_pos[i])

            if not pts:
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            px_m = np.array([p[0] for p in pts], dtype=np.float32)
            py_m = np.array([p[1] for p in pts], dtype=np.float32)

            current_source = source_dba[i]

            # Check if we should re-evaluate (caching based on movement / source level)
            should_eval = False
            if i == 0 or i == total_frames - 1:
                should_eval = True
            elif last_eval_rx is None:
                should_eval = True
            else:
                if np.hypot(rx_m[i] - last_eval_rx, ry_m[i] - last_eval_ry) > POS_THRESHOLD:
                    should_eval = True
                elif abs(current_source - last_eval_source) > SRC_THRESHOLD:
                    should_eval = True
                else:
                    if len(px_m) != len(last_eval_peds[0]):
                        should_eval = True
                    else:
                        ped_dists = np.hypot(px_m - last_eval_peds[0], py_m - last_eval_peds[1])
                        if np.any(ped_dists > POS_THRESHOLD):
                            should_eval = True

            if not should_eval:
                ts_exposure.append(ts_exposure[-1])
                ts_attenuation.append(ts_attenuation[-1])
                continue

            last_eval_rx = rx_m[i]
            last_eval_ry = ry_m[i]
            last_eval_peds = (px_m, py_m)
            last_eval_source = current_source

            rx_px = (rx_m[i] - ox) / resolution
            ry_px = (ry_m[i] - oy) / resolution

            px_px = (px_m - ox) / resolution
            py_px = (py_m - oy) / resolution

            # Run C++ solver
            attenuations = compute_attenuations(
                occupancy_grid=grid,
                resolution=resolution,
                start_x_px=rx_px,
                start_y_px=ry_px,
                target_xs_px=px_px,
                target_ys_px=py_px,
                wall_tl=47.0,  # single configurable TL
                mic_distance=1.0,
            )

            # Filter out infinity (unreachable)
            valid = ~np.isinf(attenuations)
            if not np.any(valid):
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            att_valid = attenuations[valid]
            # SPL received = source - attenuation
            exp_valid = source_dba[i] - att_valid

            # Per-pedestrian collision impulse: crash events add a ~100 dB(A)
            # penalty directly to each pedestrian's exposure at that frame.
            # This keeps the field visualization clean (no source-level bloom)
            # while the scalar metrics still capture the acoustic startle.
            if collision_mask is not None and collision_mask[i]:
                exp_valid = np.maximum(exp_valid, COLLISION_IMPULSE_DBA)

            ts_attenuation.append(att_valid.tolist())
            ts_exposure.append(exp_valid.tolist())

            eval_count += 1
            if eval_count % 50 == 0:
                logger.info(
                    "AcousticExposureCalculator: Evaluated %d frames... (%d/%d total frames processed)",
                    eval_count, i, total_frames,
                )

        logger.info(
            "AcousticExposureCalculator: Finished episode %s with %d unique solver evaluations.",
            episode.episode_id, eval_count,
        )

        # ── Post-process for scalar metrics ──────────────────────────────────────

        all_exp: list[float] = []
        for frame_exps in ts_exposure:
            all_exp.extend(frame_exps)

        if not all_exp:
            logger.info("No acoustic exposure values produced for episode %s", episode.episode_id)
            return {
                "ped_max_exposure_dba": None,
                "ped_leq_exposure_dba": None,
                "ped_max_startle_factor": None,
                "timeseries_acoustic_exposure_dba": ts_exposure,
                "timeseries_acoustic_attenuation_db": ts_attenuation,
                "worst_case_acoustic_frame": None,
            }

        all_exp = np.array(all_exp)

        # Max exposure
        max_exp = float(np.max(all_exp))

        # Leq (Equivalent Continuous Sound Level)
        # L_eq = 10 * log10( (1/N) * sum(10^(L_i / 10)) )
        lin_exp = 10 ** (all_exp / 10.0)
        leq_exp = float(10 * np.log10(np.mean(lin_exp)))

        # Startle Factor: Max positive rate of change (dBA/s) for any pedestrian
        time_s = df["time_ns"].to_numpy() / 1e9
        startle_rates: list[float] = []
        for i in range(1, len(ts_exposure)):
            prev = ts_exposure[i-1]
            curr = ts_exposure[i]
            dt = time_s[i] - time_s[i-1]
            if dt > 0 and len(prev) == len(curr) and len(curr) > 0:
                diffs = np.array(curr) - np.array(prev)
                rates = diffs / dt
                startle_rates.extend(rates.tolist())

        max_startle = float(np.max(startle_rates)) if startle_rates else 0.0

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
        }

        logger.info(
            "AcousticExposureCalculator: episode %s — max_exp=%.1f dBA, leq=%.1f dBA, startle=%.2f dBA/s",
            episode.episode_id, max_exp, leq_exp, max_startle,
        )

        return {
            "ped_max_exposure_dba": max_exp,
            "ped_leq_exposure_dba": leq_exp,
            "ped_max_startle_factor": max_startle,
            "timeseries_acoustic_exposure_dba": ts_exposure,
            "timeseries_acoustic_attenuation_db": ts_attenuation,
            "worst_case_acoustic_frame": worst_frame,
        }
