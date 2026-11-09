from __future__ import annotations

import dataclasses
import logging
import pathlib
import typing
from typing import Final

import numpy as np
import polars as pl
from PIL import Image

from arena_evaluation.processing.acoustics.door_map import (
    _entity_matches_door,
    build_pixel_tl,
    door_segments,
)
from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations
from arena_evaluation.processing.map_registry import MapRegistry
from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.characterization import _ACOUSTIC_DEFAULTS
from arena_evaluation.processing.pedestrian_frames import pedestrians_by_time
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

logger = logging.getLogger(__name__)

TRACK_ALIGN_TOLERANCE_NS: Final[int] = 100_000_000
POSITION_EVALUATION_THRESHOLD_M: Final[float] = 0.5

SERIES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "run_id": pl.String,
    "episode_id": pl.Int64,
    "robot_id": pl.String,
    "t_ns": pl.Int64,
    "ped_id": pl.String,
    "exposure_dba": pl.Float64,
    "attenuation_db": pl.Float64,
}


@dataclasses.dataclass(frozen=True, slots=True)
class AcousticExposureResult:
    """Scalar exposure summaries plus the per-frame series the long table is built from."""

    scalars: dict[str, object]
    frame_times_ns: np.ndarray
    frame_ped_ids: list[list[str]]
    frame_exposure_dba: list[list[float]]
    frame_attenuation_db: list[list[float]]


class AcousticExposureCalculator(BaseMetricCalculator):
    """Pedestrian exposure to robotic ego-noise from a multi-criteria acoustic Dijkstra over the 2D impedance map."""

    NAME = "acoustic_exposure"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["proxemics"]
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_tracks"]

    UNITS = {
        "ped_max_exposure_dba": "dBA",
        "ped_leq_exposure_dba": "dBA",
        "ped_max_startle_factor": "dBA/s",
    }

    world: str | None = None
    maps_cache: pathlib.Path | None = None

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ped_max_exposure_dba",
            "ped_leq_exposure_dba",
            "ped_max_startle_factor",
            "worst_case_acoustic_frame",
        ]

    def _get_map_occupancy(self, map_name: str, run_dir: pathlib.Path | None = None) -> tuple[np.ndarray, float, tuple[float, float, float]] | None:
        """Load the map PNG as a binary occupancy grid, flipped so row 0 = bottom (y = origin_y)."""
        meta = MapRegistry.get_map(map_name, cache_dir=self.maps_cache, run_dir=run_dir)
        if not meta or "png_path" not in meta:
            return None
        try:
            img = Image.open(meta["png_path"]).convert("L")
            image = np.array(img)
            grid = np.ascontiguousarray(np.flipud((image < 200).astype(np.uint8)))
            return grid, meta["resolution"], meta["origin"]
        except (OSError, ValueError) as error:
            logger.warning(f"Failed to load map image for acoustics: {error}")
            return None

    def _ground_truth_pose(self, episode: AlignedEpisodeBundle, sample_times_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Map-frame /tf ground-truth robot x/y at the aligned sample times, None without ground truth.

        Wheel odometry is never a substitute; samples without a ground-truth fix stay NaN and produce no exposure.
        """
        pos_x_native, pos_y_native, _yaw, native_times_ns = self.resolve_native_pose(episode)
        if len(native_times_ns) == 0:
            return None
        reference = (
            pl.DataFrame({"t_ns": native_times_ns, "_rx": pos_x_native, "_ry": pos_y_native})
            .drop_nulls()
            .sort("t_ns")
        )
        aligned = (
            pl.DataFrame({"time_ns": sample_times_ns.astype(np.int64)})
            .join_asof(reference, left_on="time_ns", right_on="t_ns", strategy="backward", tolerance=TRACK_ALIGN_TOLERANCE_NS)
        )
        return aligned["_rx"].to_numpy().astype(np.float64), aligned["_ry"].to_numpy().astype(np.float64)

    def _compute(self, episode: AlignedEpisodeBundle) -> AcousticExposureResult | None:
        """Scalar summaries and per-frame pedestrian exposure series, None when the episode cannot be evaluated."""
        null_result = AcousticExposureResult(
            scalars={key: None for key in self.output_keys()},
            frame_times_ns=np.array([], dtype=np.int64),
            frame_ped_ids=[],
            frame_exposure_dba=[],
            frame_attenuation_db=[],
        )
        if episode.run is not None and episode.run.is_reference:
            logger.info("Skipping acoustic calculation for reference episode %s", episode.episode_id)
            return null_result

        df = episode.data
        if df is None or len(df) == 0:
            logger.debug("No episode data for episode %s", episode.episode_id)
            return null_result

        map_name = episode.map or self.world
        if not map_name:
            logger.warning("No map name available for episode %s - skipping acoustics", episode.episode_id)
            return null_result

        run_dir = episode.run_dir
        if run_dir is None and episode.folder_manager and episode.run is not None and episode.run.benchmark_id:
            run_dir = episode.folder_manager.data_root / episode.run.benchmark_id

        map_data = self._get_map_occupancy(map_name, run_dir=run_dir)
        if not map_data:
            logger.warning("Failed to load map occupancy for '%s' (episode %s)", map_name, episode.episode_id)
            return null_result

        grid, resolution, origin = map_data
        origin_x, origin_y = origin[0], origin[1]

        # Semantic door geometry + per-frame door state (closed = door TL, open = carved).
        doors = door_segments(map_name, grid, resolution, origin, run_dir=run_dir)
        tl_cache: dict[tuple, np.ndarray] = {}
        state_timeline = DoorStateTimeline.from_semantic_frame(episode.semantic_snapshot)
        if doors:
            logger.info(
                "AcousticExposureCalculator: %d semantic doors loaded; semantic timeline %s",
                len(doors),
                "present" if state_timeline is not None else "ABSENT (doors default closed)",
            )

        time_ns = df["time_ns"].to_numpy().astype(np.int64)
        pose = self._ground_truth_pose(episode, time_ns)
        if pose is None:
            logger.warning("Ground-truth robot_tracks missing for episode %s", episode.episode_id)
            return null_result
        rx_m, ry_m = pose
        robot_valid = ~np.isnan(rx_m) & ~np.isnan(ry_m)

        source_dba = (
            df["total_level_af_dba"].cast(pl.Float64).fill_null(strategy="forward").fill_null(_ACOUSTIC_DEFAULTS["L_base_0"]).to_numpy()
            if "total_level_af_dba" in df.columns
            else np.full(len(time_ns), _ACOUSTIC_DEFAULTS["L_base_0"])
        )

        peds_df = self.native_ped_frame(episode)
        if peds_df is None or "x" not in peds_df.columns:
            logger.warning("Pedestrian position data missing for episode %s", episode.episode_id)
            return null_result

        peds_at_time = pedestrians_by_time(time_ns, peds_df)

        frame_ped_ids: list[list[str]] = []
        ts_exposure: list[list[float]] = []
        ts_attenuation: list[list[float]] = []

        last_eval_rx = None
        last_eval_ry = None
        last_eval_peds = None
        last_eval_tl_key = None
        last_attenuations: np.ndarray | None = None
        solver_evals = 0
        scored_frames = 0
        total_frames = len(time_ns)

        for index in range(total_frames):
            frame = peds_at_time.get(int(time_ns[index]))
            if not robot_valid[index] or frame is None or len(frame) == 0:
                frame_ped_ids.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            px_m = frame["x"].to_numpy().astype(np.float32)
            py_m = frame["y"].to_numpy().astype(np.float32)
            if len(px_m) == 0:
                frame_ped_ids.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            current_source = source_dba[index]
            open_set = state_timeline.open_doors_at(int(time_ns[index])) if state_timeline is not None else frozenset()
            tl_key = tuple(sorted(open_set))

            should_eval = index == 0 or index == total_frames - 1 or last_eval_rx is None or last_eval_peds is None or last_attenuations is None
            if not should_eval and tl_key != last_eval_tl_key:
                should_eval = True
            if not should_eval:
                robot_moved = np.hypot(rx_m[index] - last_eval_rx, ry_m[index] - last_eval_ry) > POSITION_EVALUATION_THRESHOLD_M
                pedestrians_changed = len(px_m) != len(last_eval_peds[0]) or np.any(
                    np.hypot(px_m - last_eval_peds[0], py_m - last_eval_peds[1]) > POSITION_EVALUATION_THRESHOLD_M
                )
                should_eval = bool(robot_moved or pedestrians_changed)

            if should_eval:
                last_eval_rx = rx_m[index]
                last_eval_ry = ry_m[index]
                last_eval_peds = (px_m, py_m)
                last_eval_tl_key = tl_key

                rx_px = (rx_m[index] - origin_x) / resolution
                ry_px = (ry_m[index] - origin_y) / resolution
                px_px = (px_m - origin_x) / resolution
                py_px = (py_m - origin_y) / resolution

                pixel_tl = tl_cache.get(tl_key)
                if pixel_tl is None:
                    pixel_tl = build_pixel_tl(grid, doors, open_doors=set(open_set))
                    tl_cache[tl_key] = pixel_tl

                attenuations = compute_attenuations(
                    occupancy_grid=grid,
                    resolution=resolution,
                    start_x_px=rx_px,
                    start_y_px=ry_px,
                    target_xs_px=px_px,
                    target_ys_px=py_px,
                    wall_tl=47.0,
                    mic_distance=1.0,
                    pixel_tl=pixel_tl,
                )
                last_attenuations = attenuations
                solver_evals += 1
            else:
                assert last_attenuations is not None
                attenuations = last_attenuations

            valid = ~np.isinf(attenuations)
            if not np.any(valid):
                frame_ped_ids.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            valid_attenuation = attenuations[valid]
            valid_exposure = current_source - valid_attenuation
            frame_ped_ids.append([str(name) for name in frame["ped_id"].to_numpy()[valid]])
            ts_attenuation.append(valid_attenuation.tolist())
            ts_exposure.append(valid_exposure.tolist())

            scored_frames += 1
            if scored_frames % 50 == 0:
                logger.info(
                    "AcousticExposureCalculator: Evaluated %d frames... (%d/%d total frames processed)",
                    scored_frames,
                    index + 1,
                    total_frames,
                )

        logger.info(
            "AcousticExposureCalculator: Finished episode %s with %d unique solver evaluations.",
            episode.episode_id,
            solver_evals,
        )

        if not any(ts_exposure):
            logger.info("No acoustic exposure values produced for episode %s", episode.episode_id)
            return dataclasses.replace(null_result, scalars={**null_result.scalars, "worst_case_acoustic_frame": None})

        all_exposure = np.concatenate([np.asarray(frame, dtype=np.float64) for frame in ts_exposure if frame])
        max_exposure_dba = float(np.max(all_exposure))
        linear_exposure = 10.0 ** (all_exposure / 10.0)
        leq_exposure_dba = float(10.0 * np.log10(np.mean(linear_exposure)))

        time_s = time_ns / 1e9
        startle_rates: list[float] = []
        for index in range(1, len(ts_exposure)):
            previous = ts_exposure[index - 1]
            current = ts_exposure[index]
            dt = time_s[index] - time_s[index - 1]
            if dt > 0 and len(previous) == len(current) and len(current) > 0:
                startle_rates.extend(((np.asarray(current) - np.asarray(previous)) / dt).tolist())
        max_startle = float(np.max(startle_rates)) if startle_rates else 0.0

        worst_index = 0
        worst_value = -1.0
        for index, frame_exposure in enumerate(ts_exposure):
            if frame_exposure:
                frame_max = max(frame_exposure)
                if frame_max > worst_value:
                    worst_value = frame_max
                    worst_index = index

        worst_frame_peds = peds_at_time.get(int(time_ns[worst_index]))
        worst_points = (
            list(zip(worst_frame_peds["x"].cast(pl.Float64).to_list(), worst_frame_peds["y"].cast(pl.Float64).to_list(), strict=True))
            if worst_frame_peds is not None and len(worst_frame_peds) > 0
            else []
        )
        worst_frame = {
            "robot_x": float(rx_m[worst_index]),
            "robot_y": float(ry_m[worst_index]),
            "source_dba": float(source_dba[worst_index]),
            "pedestrians": [[float(point[0]), float(point[1])] for point in worst_points],
            "door_states": {name: ("open" if state_timeline is not None and any(_entity_matches_door(name, entity) for entity in state_timeline.open_doors_at(int(time_ns[worst_index]))) else "closed") for name in doors},
        }

        logger.info(
            "AcousticExposureCalculator: episode %s, max_exp=%.1f dBA, leq=%.1f dBA, startle=%.2f dBA/s",
            episode.episode_id,
            max_exposure_dba,
            leq_exposure_dba,
            max_startle,
        )

        return AcousticExposureResult(
            scalars={
                "ped_max_exposure_dba": max_exposure_dba,
                "ped_leq_exposure_dba": leq_exposure_dba,
                "ped_max_startle_factor": max_startle,
                "worst_case_acoustic_frame": worst_frame,
                "worst_case_acoustic_t_ns": int(time_ns[worst_index]),
            },
            frame_times_ns=time_ns,
            frame_ped_ids=frame_ped_ids,
            frame_exposure_dba=ts_exposure,
            frame_attenuation_db=ts_attenuation,
        )

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        result = self._compute(episode)
        assert result is not None
        return dict(result.scalars)

    def series_table(self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = "") -> pl.DataFrame:
        """1NF flat pedestrian acoustic exposure series, one row per (sample, pedestrian); empty with the same schema without data."""
        result = self._compute(episode)
        if result is None or not any(result.frame_exposure_dba):
            return pl.DataFrame(schema=SERIES_SCHEMA)

        frame_index: list[int] = []
        t_ns: list[int] = []
        ped_ids: list[str] = []
        exposure: list[float] = []
        attenuation: list[float] = []
        for index, (ped_id_row, exposure_row, attenuation_row) in enumerate(
            zip(result.frame_ped_ids, result.frame_exposure_dba, result.frame_attenuation_db, strict=True)
        ):
            for ped_id, exposure_dba, attenuation_db in zip(ped_id_row, exposure_row, attenuation_row, strict=True):
                frame_index.append(index)
                t_ns.append(int(result.frame_times_ns[index]))
                ped_ids.append(ped_id)
                exposure.append(float(exposure_dba))
                attenuation.append(float(attenuation_db))

        return pl.DataFrame(
            {
                "run_id": [run_id] * len(t_ns),
                "episode_id": [int(episode.episode_id)] * len(t_ns),
                "robot_id": [robot_id or (episode.robot_name or "robot")] * len(t_ns),
                "t_ns": t_ns,
                "ped_id": ped_ids,
                "exposure_dba": exposure,
                "attenuation_db": attenuation,
            },
            schema=SERIES_SCHEMA,
        )