from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.acoustics.door_map import (
    _entity_matches_door,
    barrier_door_tl,
    build_pixel_tl,
    load_barrier_map,
)
from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations
from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.ecological.characterization import _ACOUSTIC_DEFAULTS
from arena_evaluation.processing.pedestrian_frames import pedestrians_by_time
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

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
    "robot_x_m": pl.Float64,
    "robot_y_m": pl.Float64,
    "source_dba": pl.Float64,
    "ped_x_m": pl.Float64,
    "ped_y_m": pl.Float64,
    "is_worst_frame": pl.Boolean,
    "open_doors": pl.String,
}


@dataclasses.dataclass(frozen=True, slots=True)
class AcousticExposureResult:
    """Scalar exposure summaries plus the per-frame series the long table is built from."""

    scalars: dict[str, object]
    frame_times_ns: np.ndarray
    frame_robot_x_m: np.ndarray
    frame_robot_y_m: np.ndarray
    frame_source_dba: np.ndarray
    frame_ped_ids: list[list[str]]
    frame_ped_x_m: list[list[float]]
    frame_ped_y_m: list[list[float]]
    frame_exposure_dba: list[list[float]]
    frame_attenuation_db: list[list[float]]
    frame_open_doors: list[str]
    worst_frame_index: int


class AcousticExposureCalculator(BaseMetricCalculator):
    """Pedestrian exposure to robotic ego-noise from the barrier-map acoustic Dijkstra solver.

    Runs per episode and per pedestrian. Walls and doors are the only acoustic barriers, the
    solver charges a per-cell transmission loss, and a door counts as open from its semantic
    ``progress`` field. Per-sample values leave through :meth:`series_table` (including the
    worst-case frame marker); only scalars enter the metric row.
    """

    NAME = "acoustic_exposure"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["proxemics"]
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_tracks"]

    SCOPE = MetricScope.EPISODE
    SCOPES: typing.ClassVar[frozenset[MetricScope]] = frozenset({MetricScope.EPISODE, MetricScope.PEDESTRIAN})

    UNITS = {
        "ped_max_exposure_dba": "dBA",
        "ped_leq_exposure_dba": "dBA",
        "ped_max_startle_factor": "dBA/s",
    }

    world: str | None = None

    def __init__(self, robot_params: RobotParams) -> None:
        """One acoustic solve per episode is expensive, so its frame series is memoised for the pedestrian rollups."""
        super().__init__(robot_params)
        self._episode_frame_cache: dict[tuple, AcousticExposureResult] = {}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ped_max_exposure_dba",
            "ped_leq_exposure_dba",
            "ped_max_startle_factor",
            "worst_case_acoustic_frame",
        ]

    def _source_level(self, episode: AlignedEpisodeBundle, time_ns: np.ndarray) -> np.ndarray:
        """Emitted A-weighted level at the sample times, from the aligned row or the native acoustics topic."""
        default_dba = _ACOUSTIC_DEFAULTS["L_base_0"]
        df = episode.data
        if df is not None and len(df) == len(time_ns) and "total_level_af_dba" in df.columns:
            return df["total_level_af_dba"].cast(pl.Float64).fill_null(strategy="forward").fill_null(default_dba).to_numpy().astype(np.float64)
        acoustics = self.native_topics(episode).get("robot_acoustics")
        if acoustics is None or len(acoustics) == 0 or "total_level_af_dba" not in acoustics.columns:
            return np.full(len(time_ns), default_dba, dtype=np.float64)
        reference = acoustics.select(pl.col("t_ns").cast(pl.Int64), pl.col("total_level_af_dba").cast(pl.Float64)).drop_nulls().sort("t_ns")
        if len(reference) == 0:
            return np.full(len(time_ns), default_dba, dtype=np.float64)
        joined = pl.DataFrame({"t_ns": time_ns.astype(np.int64)}).join_asof(reference, on="t_ns", strategy="backward", tolerance=TRACK_ALIGN_TOLERANCE_NS).with_columns(pl.col("total_level_af_dba").fill_null(strategy="forward").fill_null(default_dba))
        return joined["total_level_af_dba"].to_numpy().astype(np.float64)

    def _time_base(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Sample times, ground-truth robot x/y and validity on the scope's clock, None without a clock or ground truth."""
        df = episode.data
        if df is not None and len(df) > 0 and "time_ns" in df.columns:
            time_ns = df["time_ns"].to_numpy().astype(np.int64)
            pos_x_native, pos_y_native, _yaw, native_times_ns = self.resolve_native_pose(episode)
            if len(native_times_ns) == 0:
                return None
            reference = pl.DataFrame({"t_ns": native_times_ns, "_rx": pos_x_native, "_ry": pos_y_native}).drop_nulls().sort("t_ns")
            aligned = pl.DataFrame({"time_ns": time_ns}).join_asof(reference, left_on="time_ns", right_on="t_ns", strategy="backward", tolerance=TRACK_ALIGN_TOLERANCE_NS)
            rx_m = aligned["_rx"].to_numpy().astype(np.float64)
            ry_m = aligned["_ry"].to_numpy().astype(np.float64)
            return time_ns, rx_m, ry_m, ~np.isnan(rx_m) & ~np.isnan(ry_m)

        # A scoped window carries no aligned frame; its native /tf track is the clock.
        pos_x, pos_y, _yaw, time_ns = self.resolve_native_pose(episode)
        if len(time_ns) == 0:
            return None
        return time_ns, pos_x.astype(np.float64), pos_y.astype(np.float64), np.ones(len(time_ns), dtype=bool)

    @staticmethod
    def _null_result() -> AcousticExposureResult:
        empty = np.array([], dtype=np.float64)
        return AcousticExposureResult(
            scalars=dict.fromkeys(AcousticExposureCalculator.output_keys(), None),
            frame_times_ns=np.array([], dtype=np.int64),
            frame_robot_x_m=empty,
            frame_robot_y_m=empty.copy(),
            frame_source_dba=empty.copy(),
            frame_ped_ids=[],
            frame_ped_x_m=[],
            frame_ped_y_m=[],
            frame_exposure_dba=[],
            frame_attenuation_db=[],
            frame_open_doors=[],
            worst_frame_index=0,
        )

    def _summarize(
        self,
        frame_times_ns: np.ndarray,
        frame_robot_x_m: np.ndarray,
        frame_robot_y_m: np.ndarray,
        frame_source_dba: np.ndarray,
        frame_ped_ids: list[list[str]],
        frame_ped_x_m: list[list[float]],
        frame_ped_y_m: list[list[float]],
        frame_exposure_dba: list[list[float]],
        frame_attenuation_db: list[list[float]],
        frame_open_doors: list[str],
        door_names: typing.Iterable[str],
    ) -> AcousticExposureResult:
        """Reduce one frame series to the scalar summaries and the worst-case frame."""
        if not any(frame_exposure_dba):
            logger.info("No acoustic exposure values produced")
            return self._null_result()

        all_exposure = np.concatenate([np.asarray(frame, dtype=np.float64) for frame in frame_exposure_dba if frame])
        max_exposure_dba = float(np.max(all_exposure))
        leq_exposure_dba = float(10.0 * np.log10(np.mean(10.0 ** (all_exposure / 10.0))))

        time_s = frame_times_ns / 1e9
        startle_rates: list[float] = []
        for index in range(1, len(frame_exposure_dba)):
            previous = frame_exposure_dba[index - 1]
            current = frame_exposure_dba[index]
            dt = time_s[index] - time_s[index - 1]
            if dt > 0 and len(previous) == len(current) and len(current) > 0:
                startle_rates.extend(((np.asarray(current) - np.asarray(previous)) / dt).tolist())
        max_startle = float(np.max(startle_rates)) if startle_rates else 0.0

        worst_index = 0
        worst_value = -1.0
        for index, frame_exposure in enumerate(frame_exposure_dba):
            if frame_exposure:
                frame_max = max(frame_exposure)
                if frame_max > worst_value:
                    worst_value = frame_max
                    worst_index = index

        worst_open_doors = {name for name in frame_open_doors[worst_index].split(",") if name}
        worst_frame = {
            "robot_x": float(frame_robot_x_m[worst_index]),
            "robot_y": float(frame_robot_y_m[worst_index]),
            "source_dba": float(frame_source_dba[worst_index]),
            "pedestrians": [[x_m, y_m] for x_m, y_m in zip(frame_ped_x_m[worst_index], frame_ped_y_m[worst_index], strict=True)],
            "door_states": {name: ("open" if name in worst_open_doors else "closed") for name in door_names},
        }

        logger.info(
            "AcousticExposureCalculator: max_exp=%.1f dBA, leq=%.1f dBA, startle=%.2f dBA/s",
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
                "worst_case_acoustic_t_ns": int(frame_times_ns[worst_index]),
            },
            frame_times_ns=frame_times_ns,
            frame_robot_x_m=frame_robot_x_m,
            frame_robot_y_m=frame_robot_y_m,
            frame_source_dba=frame_source_dba,
            frame_ped_ids=frame_ped_ids,
            frame_ped_x_m=frame_ped_x_m,
            frame_ped_y_m=frame_ped_y_m,
            frame_exposure_dba=frame_exposure_dba,
            frame_attenuation_db=frame_attenuation_db,
            frame_open_doors=frame_open_doors,
            worst_frame_index=worst_index,
        )

    def _compute(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> AcousticExposureResult:
        """Scalar summaries and per-frame pedestrian exposure series for the scope window."""
        if episode.run is not None and episode.run.is_reference:
            logger.info("Skipping acoustic calculation for reference episode %s", episode.episode_id)
            return self._null_result()

        map_name = prior_results.get("map") or episode.map or self.world
        if not map_name:
            logger.warning("No map name available for episode %s - skipping acoustics", episode.episode_id)
            return self._null_result()

        time_base = self._time_base(episode)
        if time_base is None:
            logger.warning("Ground-truth robot_tracks missing for episode %s", episode.episode_id)
            return self._null_result()

        peds_df = self.native_ped_frame(episode)
        if peds_df is None or "x" not in peds_df.columns:
            logger.warning("Pedestrian position data missing for episode %s", episode.episode_id)
            return self._null_result()

        run_dir = episode.run_dir
        if run_dir is None and episode.folder_manager and episode.run is not None and episode.run.benchmark_id:
            run_dir = episode.folder_manager.data_root / episode.run.benchmark_id

        # A missing world raises AcousticsError: a metric input error, never a silent null.
        barrier = load_barrier_map(map_name, run_dir=run_dir)
        grid, resolution = barrier.occupancy, barrier.resolution
        origin_x, origin_y = barrier.origin

        doors = barrier_door_tl(barrier)
        tl_cache: dict[tuple, np.ndarray] = {}
        state_timeline = DoorStateTimeline.from_semantic_frame(episode.semantic_snapshot)
        if doors and state_timeline is None:
            logger.warning(
                "AcousticExposureCalculator: %d doors but no door-state timeline for episode %s; all doors stay closed",
                len(doors),
                episode.episode_id,
            )

        time_ns, rx_m, ry_m, robot_valid = time_base
        source_dba = self._source_level(episode, time_ns)
        peds_at_time = pedestrians_by_time(time_ns, peds_df)

        frame_ped_ids: list[list[str]] = []
        frame_ped_x_m: list[list[float]] = []
        frame_ped_y_m: list[list[float]] = []
        ts_exposure: list[list[float]] = []
        ts_attenuation: list[list[float]] = []
        frame_open_doors: list[str] = []

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
            raw_open = state_timeline.open_doors_at(int(time_ns[index])) if state_timeline is not None else frozenset()
            # Canonicalise the semantic entity names to the barrier map's door keys once per frame.
            open_doors_here = tuple(sorted(name for name in doors if any(_entity_matches_door(name, entity) for entity in raw_open)))
            frame_open_doors.append(",".join(open_doors_here))
            if not robot_valid[index] or frame is None or len(frame) == 0:
                frame_ped_ids.append([])
                frame_ped_x_m.append([])
                frame_ped_y_m.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            px_m = frame["x"].to_numpy().astype(np.float32)
            py_m = frame["y"].to_numpy().astype(np.float32)
            if len(px_m) == 0:
                frame_ped_ids.append([])
                frame_ped_x_m.append([])
                frame_ped_y_m.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            current_source = source_dba[index]
            tl_key = open_doors_here

            should_eval = index == 0 or index == total_frames - 1 or last_eval_rx is None or last_eval_peds is None or last_attenuations is None
            if not should_eval and tl_key != last_eval_tl_key:
                should_eval = True
            if not should_eval:
                robot_moved = np.hypot(rx_m[index] - last_eval_rx, ry_m[index] - last_eval_ry) > POSITION_EVALUATION_THRESHOLD_M
                pedestrians_changed = len(px_m) != len(last_eval_peds[0]) or np.any(np.hypot(px_m - last_eval_peds[0], py_m - last_eval_peds[1]) > POSITION_EVALUATION_THRESHOLD_M)
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
                    pixel_tl = build_pixel_tl(grid, doors, open_doors=set(open_doors_here))
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
                frame_ped_x_m.append([])
                frame_ped_y_m.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            valid_attenuation = attenuations[valid]
            valid_exposure = current_source - valid_attenuation
            frame_ped_ids.append([str(name) for name in frame["ped_id"].to_numpy()[valid]])
            frame_ped_x_m.append([float(value) for value in px_m[valid]])
            frame_ped_y_m.append([float(value) for value in py_m[valid]])
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
        return self._summarize(time_ns, rx_m, ry_m, source_dba, frame_ped_ids, frame_ped_x_m, frame_ped_y_m, ts_exposure, ts_attenuation, frame_open_doors, doors.keys())

    def _episode_result(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> AcousticExposureResult:
        """The episode's exposure frame series, solved once and reused by the pedestrian rollups and the series table."""
        key = (int(episode.episode_id), str(episode.robot_name or ""), str(episode.map or ""), str(episode.run_dir or ""), str(episode.robot_name))
        cached = self._episode_frame_cache.get(key)
        if cached is None:
            cached = self._compute(episode, prior_results)
            self._episode_frame_cache[key] = cached
        return cached

    def _restrict_to_pedestrian(self, result: AcousticExposureResult, ped_id: str, door_names: typing.Iterable[str]) -> AcousticExposureResult:
        """The same solved frame series reduced to the rows of one pedestrian."""
        frame_ped_ids: list[list[str]] = []
        frame_ped_x_m: list[list[float]] = []
        frame_ped_y_m: list[list[float]] = []
        frame_exposure_dba: list[list[float]] = []
        frame_attenuation_db: list[list[float]] = []
        for index, ped_ids in enumerate(result.frame_ped_ids):
            keep = [position for position, name in enumerate(ped_ids) if name == ped_id]
            frame_ped_ids.append([ped_ids[position] for position in keep])
            frame_ped_x_m.append([result.frame_ped_x_m[index][position] for position in keep])
            frame_ped_y_m.append([result.frame_ped_y_m[index][position] for position in keep])
            frame_exposure_dba.append([result.frame_exposure_dba[index][position] for position in keep])
            frame_attenuation_db.append([result.frame_attenuation_db[index][position] for position in keep])
        return self._summarize(
            result.frame_times_ns,
            result.frame_robot_x_m,
            result.frame_robot_y_m,
            result.frame_source_dba,
            frame_ped_ids,
            frame_ped_x_m,
            frame_ped_y_m,
            frame_exposure_dba,
            frame_attenuation_db,
            result.frame_open_doors,
            door_names,
        )

    def _door_names(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> tuple[str, ...]:
        """Door keys of the episode's barrier map, for the worst-case frame's door states."""
        map_name = prior_results.get("map") or episode.map or self.world
        if not map_name:
            return ()
        run_dir = episode.run_dir
        if run_dir is None and episode.folder_manager and episode.run is not None and episode.run.benchmark_id:
            run_dir = episode.folder_manager.data_root / episode.run.benchmark_id
        return tuple(barrier_door_tl(load_barrier_map(map_name, run_dir=run_dir)).keys())

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        return dict(self._episode_result(episode, prior_results).scalars)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.PEDESTRIAN:
            # One solve per episode: the pedestrian slice reduces the episode frame series to its own rows.
            full_result = self._episode_result(scope_slice.full_episode, prior_results)
            ped_id = str(scope_slice.identity.get("ped_id") or scope_slice.slice_id)
            restricted = self._restrict_to_pedestrian(full_result, ped_id, self._door_names(scope_slice.full_episode, prior_results))
            return dict(restricted.scalars)
        raise ValueError(f"acoustic_exposure does not serve the {scope_slice.scope.value} scope")

    def series_table(self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = "") -> pl.DataFrame:
        """1NF flat pedestrian acoustic exposure series, one row per (sample, pedestrian).

        ``is_worst_frame`` marks the rows of the worst-case frame, and ``open_doors`` names the
        doors open at that sample, so the legacy worst-case frame struct lives here instead of
        in a nested fact column.
        """
        result = self._episode_result(episode, {})
        if not any(result.frame_exposure_dba):
            return pl.DataFrame(schema=SERIES_SCHEMA)

        t_ns: list[int] = []
        ped_ids: list[str] = []
        exposure: list[float] = []
        attenuation: list[float] = []
        robot_x: list[float] = []
        robot_y: list[float] = []
        source: list[float] = []
        ped_x: list[float] = []
        ped_y: list[float] = []
        is_worst: list[bool] = []
        open_doors: list[str] = []
        for index, (ped_id_row, ped_x_row, ped_y_row, exposure_row, attenuation_row) in enumerate(
            zip(
                result.frame_ped_ids,
                result.frame_ped_x_m,
                result.frame_ped_y_m,
                result.frame_exposure_dba,
                result.frame_attenuation_db,
                strict=True,
            )
        ):
            for ped_id, x_m, y_m, exposure_dba, attenuation_db in zip(ped_id_row, ped_x_row, ped_y_row, exposure_row, attenuation_row, strict=True):
                t_ns.append(int(result.frame_times_ns[index]))
                ped_ids.append(ped_id)
                exposure.append(float(exposure_dba))
                attenuation.append(float(attenuation_db))
                robot_x.append(float(result.frame_robot_x_m[index]))
                robot_y.append(float(result.frame_robot_y_m[index]))
                source.append(float(result.frame_source_dba[index]))
                ped_x.append(x_m)
                ped_y.append(y_m)
                is_worst.append(index == result.worst_frame_index)
                open_doors.append(result.frame_open_doors[index])

        return pl.DataFrame(
            {
                "run_id": [run_id] * len(t_ns),
                "episode_id": [int(episode.episode_id)] * len(t_ns),
                "robot_id": [robot_id or (episode.robot_name or "robot")] * len(t_ns),
                "t_ns": t_ns,
                "ped_id": ped_ids,
                "exposure_dba": exposure,
                "attenuation_db": attenuation,
                "robot_x_m": robot_x,
                "robot_y_m": robot_y,
                "source_dba": source,
                "ped_x_m": ped_x,
                "ped_y_m": ped_y,
                "is_worst_frame": is_worst,
                "open_doors": open_doors,
            },
            schema=SERIES_SCHEMA,
        )
