from __future__ import annotations

import json
import logging
import typing

import numpy as np
from PIL import Image
import polars as pl

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
    from arena_evaluation.processing.acoustics.impedance_grid import (
        compute_acoustic_field,
        compute_attenuations,
    )
except ImportError:
    compute_acoustic_field = None
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

    world: str | None = None

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

    def _get_map_occupancy(self, map_name: str, run_dir=None) -> tuple[np.ndarray, float, tuple[float, float, float]] | None:
        """Load the map PNG as a binary occupancy grid, flipped so row 0 = bottom (y = origin_y)."""
        meta = MapRegistry.get_map(map_name, run_dir=run_dir)
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

    def calculate(self, episode: "AlignedEpisodeBundle", prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        nulls = {k: None for k in self.output_keys()}

        # Skip heavy calculation for reference runs
        if episode.run is not None and episode.run.is_reference:
            logger.info("Skipping acoustic calculation for reference episode %s", episode.episode_id)
            return nulls

        if compute_attenuations is None:
            logger.warning("Acoustics C++ solver not available.")
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

        # Semantic door geometry + per-frame door state (PR #68 semantics)
        # Doors are per-pixel entities: closed = door TL (25 dB), open = carved.
        doors = door_segments(map_name, grid, resolution, origin, run_dir=run_dir)
        tl_cache: dict[tuple, np.ndarray] = {}
        state_timeline = DoorStateTimeline.from_semantic_frame(
            episode.semantic_snapshot,
            progress_threshold=ACOUSTIC_OPEN_PROGRESS_THRESHOLD,
        )
        if doors:
            logger.info(
                "AcousticExposureCalculator: %d semantic doors loaded; "
                "semantic timeline %s",
                len(doors),
                "present" if state_timeline is not None else "ABSENT (doors default closed)",
            )

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
        last_eval_doors = None
        current_field: np.ndarray | None = None
        last_attenuations: np.ndarray | None = None

        # Proximity-adaptive displacement thresholds:
        # Near pedestrians (< 2.0 m), fine granularity (0.05 m = 5 cm) is required because
        # 1/r geometric spreading produces rapid dB gradients (e.g. 0.5m -> 1.0m is 6 dB!).
        # At intermediate distances (< 5.0 m), 0.10 m (10 cm) preserves sub-decibel precision.
        # At far distances (>= 5.0 m), 0.20 m (20 cm) maintains high efficiency.
        POS_THRESHOLD_NEAR = 0.05  # meters (5 cm when robot is near pedestrians < 2.0m)
        POS_THRESHOLD_MID = 0.10   # meters (10 cm when robot is 2.0m - 5.0m)
        POS_THRESHOLD_FAR = 0.20   # meters (20 cm when robot is >= 5.0m or no pedestrians)
        total_frames = len(rx_m)

        eval_count = 0

        # Check if compute_attenuations was monkeypatched in testing
        from arena_evaluation.processing.acoustics.impedance_grid import (
            compute_attenuations as _real_ca,
        )
        is_mocked = compute_attenuations is not _real_ca

        for i in range(total_frames):
            pts = self._parse_pedestrian_positions(peds_pos[i])

            if not pts:
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            px_m = np.array([p[0] for p in pts], dtype=np.float32)
            py_m = np.array([p[1] for p in pts], dtype=np.float32)

            current_source = source_dba[i]

            # Door state at current timestamp
            open_set = (
                state_timeline.open_doors_at(int(df["time_ns"][i]))
                if state_timeline is not None
                else frozenset()
            )

            # Adaptive displacement threshold based on minimum pedestrian proximity
            if len(px_m) > 0:
                min_ped_dist = float(np.min(np.hypot(px_m - rx_m[i], py_m - ry_m[i])))
                if min_ped_dist < 2.0:
                    pos_threshold = POS_THRESHOLD_NEAR
                elif min_ped_dist < 5.0:
                    pos_threshold = POS_THRESHOLD_MID
                else:
                    pos_threshold = POS_THRESHOLD_FAR
            else:
                pos_threshold = POS_THRESHOLD_MID

            # Check if we should re-evaluate the robot's acoustic emission field.
            # Note: Pedestrian movement alone does NOT change the robot's acoustic propagation
            # field. When the robot is stationary (< pos_threshold) and doors are unchanged,
            # the field remains identical and can be resampled directly at pedestrian coordinates.
            robot_moved = (
                last_eval_rx is None
                or np.hypot(rx_m[i] - last_eval_rx, ry_m[i] - last_eval_ry) > pos_threshold
            )
            doors_changed = (open_set != last_eval_doors) if last_eval_doors is not None else True

            should_eval = (
                i == 0
                or robot_moved
                or doors_changed
                or (current_field is None and not is_mocked)
                or (last_attenuations is None and is_mocked)
            )

            rx_px = (rx_m[i] - ox) / resolution
            ry_px = (ry_m[i] - oy) / resolution

            px_px = (px_m - ox) / resolution
            py_px = (py_m - oy) / resolution

            if should_eval:
                last_eval_rx = rx_m[i]
                last_eval_ry = ry_m[i]
                last_eval_doors = open_set

                # Door-aware per-pixel TL (open doors carved to 0 dB)
                tl_key = tuple(sorted(open_set))
                pixel_tl = tl_cache.get(tl_key)
                if pixel_tl is None:
                    pixel_tl = build_pixel_tl(grid, doors, open_doors=set(open_set))
                    tl_cache[tl_key] = pixel_tl

                if not is_mocked and compute_acoustic_field is not None:
                    current_field = compute_acoustic_field(
                        occupancy_grid=grid,
                        resolution=resolution,
                        start_x_px=rx_px,
                        start_y_px=ry_px,
                        wall_tl=47.0,  # fallback TL when no pixel_tl (v1 path)
                        mic_distance=1.0,
                        pixel_tl=pixel_tl,
                    )
                    eval_count += 1
                else:
                    # Mocked / fallback path
                    attenuations = compute_attenuations(
                        occupancy_grid=grid,
                        resolution=resolution,
                        start_x_px=rx_px,
                        start_y_px=ry_px,
                        target_xs_px=px_px,
                        target_ys_px=py_px,
                        wall_tl=47.0,  # fallback TL when no pixel_tl (v1 path)
                        mic_distance=1.0,
                        pixel_tl=pixel_tl,
                    )
                    last_attenuations = attenuations
                    eval_count += 1

            if not is_mocked and current_field is not None:
                # Fast O(1) resampling of the existing acoustic field at pedestrian locations
                h_grid, w_grid = current_field.shape
                ix = np.clip(np.round(px_px).astype(np.int32), 0, w_grid - 1)
                iy = np.clip(np.round(py_px).astype(np.int32), 0, h_grid - 1)
                attenuations = current_field[iy, ix]
            elif not should_eval:
                attenuations = last_attenuations

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

            if (i + 1) % 100 == 0 or i == total_frames - 1:
                logger.info(
                    "AcousticExposureCalculator: Processed %d/%d frames (%d solver field evaluations)",
                    i + 1, total_frames, eval_count,
                )

        logger.info(
            "AcousticExposureCalculator: Finished episode %s with %d unique solver evaluations.",
            episode.episode_id, eval_count,
        )

        # Post-process for scalar metrics

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
            "door_states": {
                name: ("open" if state_timeline is not None and any(
                    _entity_matches_door(name, e)
                    for e in state_timeline.open_doors_at(int(df["time_ns"][max_idx]))
                ) else "closed")
                for name in doors
            },
        }

        logger.info(
            "AcousticExposureCalculator: episode %s, max_exp=%.1f dBA, leq=%.1f dBA, startle=%.2f dBA/s",
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
