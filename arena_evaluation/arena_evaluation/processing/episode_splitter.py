from __future__ import annotations

import typing
import yaml
import polars as pl

from ..storage.schemas import TopicBundle, AlignedEpisodeBundle
from .topic_aligner import TopicAligner

class EpisodeSplitter:
    """
    Splits continuous topic data into discrete episodes using EpisodeRecord messages.
    """
    def __init__(self, aligner: TopicAligner, min_episode_frames: int = 5):
        self.aligner = aligner
        self.min_episode_frames = min_episode_frames

    def split(self, bundle: TopicBundle, robot_name: str | None = None) -> list[AlignedEpisodeBundle]:
        """
        Split the raw bundle into aligned episodes.
        """
        if bundle.odom is None:
            return []

        def _to_df(lf_or_df):
            if isinstance(lf_or_df, pl.LazyFrame):
                return lf_or_df.collect()
            return lf_or_df

        odom_is_empty = False
        if isinstance(bundle.odom, pl.LazyFrame):
            odom_is_empty = bundle.odom.limit(1).collect().height == 0
        else:
            odom_is_empty = len(bundle.odom) == 0

        if odom_is_empty:
            return []

        record_df = _to_df(bundle.episode_record)
        initialpose_df = _to_df(bundle.initialpose)
        plan_df = _to_df(bundle.plan)
            
        episodes = []
        
        if record_df is None or len(record_df) == 0:
            aligned_df = self.aligner.align(bundle)
            aligned_df = _to_df(aligned_df)
            if aligned_df is not None and len(aligned_df) >= self.min_episode_frames:
                episodes.append(
                    AlignedEpisodeBundle(
                        episode_id=0,
                        data=aligned_df,
                        start_pos=[],
                        goal_pos=[],
                        num_pedestrians=self._estimate_peds(aligned_df),
                        robot_name=robot_name
                    )
                )
            return episodes


        raw_rows = list(record_df.iter_rows(named=True))
        rows = list(raw_rows)
        rows.sort(key=lambda r: r["time_ns"])

        def _odom_bounds() -> tuple[int, int]:
            if isinstance(bundle.odom, pl.LazyFrame):
                df_min = bundle.odom.select(pl.col("time_ns").min()).collect().item()
                df_max = bundle.odom.select(pl.col("time_ns").max()).collect().item()
            else:
                df_min = bundle.odom.select(pl.col("time_ns").min()).item()
                df_max = bundle.odom.select(pl.col("time_ns").max()).item()
            return int(df_min), int(df_max)

        odom_min, odom_max = _odom_bounds()

        windows: list[tuple[dict, int | None]] = []
        if "outcome_state" in record_df.columns:
            current_start_row: dict | None = None
            for row in rows:
                outcome = row.get("outcome_state", 0)
                t = row["time_ns"]
                if outcome in (0, 1):  # QUEUED / RUNNING → start marker
                    if current_start_row is not None:
                        windows.append((current_start_row, t - 1))
                    current_start_row = row
                else:  # terminal outcome → closes the current window
                    if current_start_row is not None:
                        windows.append((current_start_row, t))
                        current_start_row = None
                    else:
                        # Terminal without a start marker: the episode began before
                        # the recording window — use the recording start.
                        windows.append((row, t))

            if current_start_row is not None:
                windows.append((current_start_row, None))
        else:
            from collections import defaultdict
            ep_map: dict[int, list[dict]] = defaultdict(list)
            for row in raw_rows:
                ep_map[row["episode_id"]].append(row)
            sorted_ep_ids = sorted(ep_map.keys(), key=lambda k: ep_map[k][0]["time_ns"])
            for idx, ep_id in enumerate(sorted_ep_ids):
                ep_rows = ep_map[ep_id]
                if len(ep_rows) == 1:
                    next_end = (ep_map[sorted_ep_ids[idx + 1]][0]["time_ns"] - 1) if (idx + 1 < len(sorted_ep_ids)) else None
                    windows.append((ep_rows[0], next_end))
                else:
                    windows.append((ep_rows[0], ep_rows[-1]["time_ns"]))

        for start_row, end_time in windows:
            if end_time is None:
                end_time = odom_max
                start_time = start_row["time_ns"]
            elif start_row.get("outcome_state", 0) in (2, 3, 4, 5):
                start_time = odom_min  # terminal-only window
            else:
                start_time = start_row["time_ns"]

            if start_time > end_time:
                continue

            aligned_df = self.aligner.align(bundle, start_time, end_time)
            aligned_df = _to_df(aligned_df)

            if aligned_df is None or len(aligned_df) < self.min_episode_frames:
                continue

            row = start_row
            start_pos = []
            goal_pos = []

            try:
                params_str = row["robots_params"]
                if params_str:
                    params_dict = yaml.safe_load(params_str)
                    for robot_params in params_dict.values():
                        if "start" in robot_params:
                            start_pos = robot_params["start"]
                        if "goal" in robot_params:
                            goal_pos = robot_params["goal"]
                        break
            except Exception:
                pass

            if not start_pos and initialpose_df is not None and len(initialpose_df) > 0:
                df_init = initialpose_df.filter(pl.col("time_ns") >= start_time)
                if len(df_init) > 0:
                    row_init = df_init.row(0, named=True)
                else:
                    row_init = initialpose_df.row(-1, named=True)
                start_pos = [row_init["pos_x"], row_init["pos_y"], row_init["yaw"]]

            if start_pos and len(start_pos) == 3 and plan_df is not None and len(plan_df) > 0:
                df_plan = plan_df.filter(pl.col("time_ns") >= start_time)
                if len(df_plan) > 0:
                    row_plan = df_plan.row(0, named=True)
                    if "poses_yaw" in row_plan and len(row_plan["poses_yaw"]) > 0:
                        plan_yaw = row_plan["poses_yaw"][0]
                        if plan_yaw != 0.0 and abs(start_pos[2] - plan_yaw) > 1.0:
                            start_pos[2] = plan_yaw

            if not start_pos and aligned_df is not None and len(aligned_df) > 0:
                first_row = aligned_df.row(0, named=True)
                if "pos_x" in first_row and "pos_y" in first_row:
                    start_pos = [first_row["pos_x"], first_row["pos_y"], first_row.get("yaw", 0.0)]

            if not goal_pos and plan_df is not None and len(plan_df) > 0:
                df_plan_ep = plan_df.filter((pl.col("time_ns") >= start_time) & (pl.col("time_ns") <= end_time))
                if len(df_plan_ep) > 0:
                    row_plan = df_plan_ep.row(0, named=True)
                    if "poses_x" in row_plan and len(row_plan["poses_x"]) > 0:
                        goal_pos = [row_plan["poses_x"][-1], row_plan["poses_y"][-1], row_plan["poses_yaw"][-1]]

            if not goal_pos and aligned_df is not None and len(aligned_df) > 0:
                last_row = aligned_df.row(-1, named=True)
                if "pos_x" in last_row and "pos_y" in last_row:
                    goal_pos = [last_row["pos_x"], last_row["pos_y"], last_row.get("yaw", 0.0)]
            episodes.append(
                AlignedEpisodeBundle(
                    episode_id=row["episode_id"],
                    data=aligned_df,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    num_pedestrians=self._estimate_peds(aligned_df),
                    robot_name=robot_name
                )
            )
        return episodes

    def _estimate_peds(self, df: pl.DataFrame) -> int:
        if "num_pedestrians" in df.columns:
            peds = df["num_pedestrians"].drop_nulls()
            if len(peds) > 0:
                return int(peds.max())
        return 0
