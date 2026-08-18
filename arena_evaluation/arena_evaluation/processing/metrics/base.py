from __future__ import annotations

from abc import ABC, abstractmethod
import typing
import numpy as np

if typing.TYPE_CHECKING:
    from ...storage.schemas import AlignedEpisodeBundle, RobotParams


class BaseMetricCalculator(ABC):
    """Abstract Base Class for all metric calculators."""

    NAME: str = ""
    CATEGORY: str = "general"
    REQUIRES_PEDSIM: bool = False
    DEPENDS_ON: list[str] = []
    REQUIRED_TOPICS: list[str | list[str] | tuple[str, ...] | set[str]] = []
    UNITS: dict[str, str] = {}
    PRIMARY_OUTPUTS: list[str] = []
    OUTPUT_DIRECTIONS: dict[str, str] = {}

    def __init__(self, robot_params: RobotParams):
        """Initializes the calculator with robot parameters."""
        self.robot_params = robot_params

    def resolve_robot_pose(self, episode: "AlignedEpisodeBundle") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract and resolve the robot's pose (pos_x, pos_y, yaw) in the map frame."""
        import numpy as np
        import polars as pl
        
        if episode.data is not None and len(episode.data) > 0:
            use_gt = "pos_x_gt" in episode.data.columns
            if use_gt:
                episode.data = episode.data.filter(
                    pl.col("pos_x_gt").is_not_null() & ~pl.col("pos_x_gt").is_nan() &
                    pl.col("pos_y_gt").is_not_null() & ~pl.col("pos_y_gt").is_nan() &
                    pl.col("yaw_gt").is_not_null() & ~pl.col("yaw_gt").is_nan()
                )
            else:
                episode.data = episode.data.filter(
                    pl.col("pos_x").is_not_null() & ~pl.col("pos_x").is_nan() &
                    pl.col("pos_y").is_not_null() & ~pl.col("pos_y").is_nan() &
                    pl.col("yaw").is_not_null() & ~pl.col("yaw").is_nan()
                )
        
        if episode.data is not None and len(episode.data) > 0:
            if "pos_x" in episode.data.columns:
                odom_x = episode.data["pos_x"].to_numpy().copy()
                odom_y = episode.data["pos_y"].to_numpy().copy()
                odom_yaw = episode.data["yaw"].to_numpy().copy()
            elif "pos_x_gt" in episode.data.columns:
                odom_x = episode.data["pos_x_gt"].to_numpy().copy()
                odom_y = episode.data["pos_y_gt"].to_numpy().copy()
                odom_yaw = episode.data["yaw_gt"].to_numpy().copy()
            else:
                odom_x = np.array([], dtype=np.float64)
                odom_y = np.array([], dtype=np.float64)
                odom_yaw = np.array([], dtype=np.float64)

            raw_odom_x0 = odom_x[0] if len(odom_x) > 0 else 0.0
            raw_odom_y0 = odom_y[0] if len(odom_y) > 0 else 0.0
            raw_odom_yaw0 = odom_yaw[0] if len(odom_yaw) > 0 else 0.0
            if len(odom_x) > 1:
                dists = np.sqrt(np.diff(odom_x)**2 + np.diff(odom_y)**2)
                jumps = np.where(dists > 0.5)[0]
                
                if len(jumps) > 0:
                    split_indices = jumps + 1
                    segments_x = np.split(odom_x, split_indices)
                    segments_y = np.split(odom_y, split_indices)
                    
                    best_seg_idx = -1
                    best_len = -1.0
                    
                    for i in range(len(segments_x)):
                        seg_x = segments_x[i]
                        seg_y = segments_y[i]
                        if len(seg_x) < 2:
                            seg_len = 0.0
                        else:
                            seg_len = np.sum(np.sqrt(np.diff(seg_x)**2 + np.diff(seg_y)**2))
                        
                        if seg_len >= 0.2 and seg_len > best_len:
                            best_len = seg_len
                            best_seg_idx = i
                    
                    if best_seg_idx != -1:
                        start_idx = 0 if best_seg_idx == 0 else int(split_indices[best_seg_idx - 1])
                        end_idx = int(split_indices[best_seg_idx]) if best_seg_idx < len(split_indices) - 1 else len(odom_x)
                        
                        odom_x = odom_x[start_idx:end_idx]
                        odom_y = odom_y[start_idx:end_idx]
                        odom_yaw = odom_yaw[start_idx:end_idx]
                        episode.data = episode.data.slice(start_idx, end_idx - start_idx)

        else:
            odom_x = np.array([], dtype=np.float64)
            odom_y = np.array([], dtype=np.float64)
            odom_yaw = np.array([], dtype=np.float64)
        
        if episode.start_pos and len(episode.start_pos) >= 2 and len(odom_x) > 0:
            start_x, start_y = episode.start_pos[0], episode.start_pos[1]
            start_yaw = episode.start_pos[2] if len(episode.start_pos) >= 3 else 0.0
            
            odom_x0 = raw_odom_x0
            odom_y0 = raw_odom_y0
            odom_yaw0 = raw_odom_yaw0
            
            theta = start_yaw - odom_yaw0
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            dx = odom_x - odom_x0
            dy = odom_y - odom_y0
            
            odom_x_trans = start_x + dx * cos_t - dy * sin_t
            odom_y_trans = start_y + dx * sin_t + dy * cos_t
            odom_yaw_trans = odom_yaw + theta
            odom_yaw_trans = (odom_yaw_trans + np.pi) % (2 * np.pi) - np.pi
        else:
            odom_x_trans, odom_y_trans, odom_yaw_trans = odom_x, odom_y, odom_yaw

        use_gt = episode.data is not None and "pos_x_gt" in episode.data.columns
        if use_gt:
            pos_x = episode.data["pos_x_gt"].to_numpy().copy()
            pos_y = episode.data["pos_y_gt"].to_numpy().copy()
            yaw = episode.data["yaw_gt"].to_numpy().copy()
        else:
            pos_x = odom_x_trans
            pos_y = odom_y_trans
            yaw = odom_yaw_trans
            
        return pos_x, pos_y, yaw, odom_x_trans, odom_y_trans, odom_yaw_trans

    # ------------------------------------------------------------------
    # Native-rate helpers (full multi-rate standard, 2026-08-12)
    # ------------------------------------------------------------------

    def native_topics(self, episode: "AlignedEpisodeBundle") -> dict:
        """Raw native-rate topic frames keyed by topic name; {} when absent."""
        topics = getattr(episode, "topics", None)
        return topics if isinstance(topics, dict) else {}

    def resolve_native_pose(
        self, episode: "AlignedEpisodeBundle"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ground-truth-first robot pose on the native odom time axis.

        Returns (pos_x, pos_y, yaw, time_ns) float64 arrays.
        - tf_gt (world frame, offset-corrected at extraction) is used when
          recorded — it is the trusted pose source.
        - Fallback: raw odom transformed from the robot-local odom origin to
          the world frame via the episode start pose (same transform as
          ``resolve_robot_pose``).
        """
        import polars as pl

        topics = self.native_topics(episode)
        odom = topics.get("odom")
        tf_gt = topics.get("tf_gt")

        if tf_gt is not None and "pos_x_gt" in tf_gt.columns:
            t_df = tf_gt.drop_nulls(["pos_x_gt", "pos_y_gt", "yaw_gt"]).sort("time_ns")
            if len(t_df) > 0:
                return (
                    t_df["pos_x_gt"].to_numpy().astype(np.float64),
                    t_df["pos_y_gt"].to_numpy().astype(np.float64),
                    t_df["yaw_gt"].to_numpy().astype(np.float64),
                    t_df["time_ns"].to_numpy().astype(np.int64),
                )

        if odom is None or "pos_x" not in odom.columns:
            return np.array([]), np.array([]), np.array([]), np.array([])

        o_df = odom.drop_nulls(["pos_x", "pos_y", "yaw"]).sort("time_ns")
        if len(o_df) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        ox = o_df["pos_x"].to_numpy().astype(np.float64)
        oy = o_df["pos_y"].to_numpy().astype(np.float64)
        oyaw = o_df["yaw"].to_numpy().astype(np.float64)
        time_ns = o_df["time_ns"].to_numpy().astype(np.int64)

        # Transform odom frame (robot-local origin) to the world frame using
        # the episode start pose, mirroring resolve_robot_pose.
        if episode.start_pos and len(episode.start_pos) >= 2:
            start_x, start_y = float(episode.start_pos[0]), float(episode.start_pos[1])
            start_yaw = float(episode.start_pos[2]) if len(episode.start_pos) >= 3 else 0.0
            theta = start_yaw - float(oyaw[0])
            c, s = np.cos(theta), np.sin(theta)
            dx = ox - ox[0]
            dy = oy - oy[0]
            ox = start_x + dx * c - dy * s
            oy = start_y + dx * s + dy * c
            oyaw = (oyaw + theta + np.pi) % (2 * np.pi) - np.pi

        return ox, oy, oyaw, time_ns

    def native_ped_frame(self, episode: "AlignedEpisodeBundle") -> pl.DataFrame | None:
        """Raw pedestrian frame (native rate) sorted by time_ns, or None."""
        topics = self.native_topics(episode)
        peds = topics.get("peds")
        if peds is None:
            return None
        if "time_ns" not in peds.columns:
            return None
        return peds.sort("time_ns")

    def pose_at_times(
        self,
        times_ns: np.ndarray,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        yaw: np.ndarray,
        time_ns: np.ndarray,
        tolerance_ns: int = 100_000_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward-asof sample pose arrays onto ``times_ns`` (native-rate join).

        Returns (pos_x, pos_y, yaw) at the query times; 0/0/0 for query times
        with no pose within tolerance.
        """
        import polars as pl

        if len(time_ns) == 0:
            return np.zeros(len(times_ns)), np.zeros(len(times_ns)), np.zeros(len(times_ns))
        q = pl.DataFrame({"time_ns": times_ns}).lazy()
        ref = pl.DataFrame({
            "time_ns": time_ns,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "yaw": yaw,
        }).lazy().sort("time_ns")
        joined = (
            q.join_asof(ref, on="time_ns", strategy="backward", tolerance=tolerance_ns)
            .select(["time_ns", "pos_x", "pos_y", "yaw"])
            .collect()
        )
        px = joined["pos_x"].to_numpy()
        py = joined["pos_y"].to_numpy()
        ya = joined["yaw"].to_numpy()
        if px.dtype != np.float64:
            px = px.astype(np.float64)
        if py.dtype != np.float64:
            py = py.astype(np.float64)
        if ya.dtype != np.float64:
            ya = ya.astype(np.float64)
        return (
            np.nan_to_num(px, nan=0.0),
            np.nan_to_num(py, nan=0.0),
            np.nan_to_num(ya, nan=0.0),
        )

    def values_at_times(
        self,
        values: np.ndarray,
        values_time_ns: np.ndarray,
        query_times_ns: np.ndarray,
        tolerance_ns: int = 100_000_000,
    ) -> np.ndarray:
        """Backward-asof sample an arbitrary per-frame array onto query times.

        Returns NaN where no value within tolerance (caller decides fill).
        """
        import polars as pl

        if len(values_time_ns) == 0 or len(query_times_ns) == 0:
            return np.full(len(query_times_ns), np.nan)
        q = pl.DataFrame({"time_ns": query_times_ns}).lazy()
        ref = pl.DataFrame({"time_ns": values_time_ns, "v": values}).lazy().sort("time_ns")
        joined = (
            q.join_asof(ref, on="time_ns", strategy="backward", tolerance=tolerance_ns)
            .select("v")
            .collect()
        )
        out = joined["v"].to_numpy()
        if out.dtype.kind != "f":
            out = out.astype(np.float64)
        return out

    @staticmethod
    def speed_from_pose(
        pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray
    ) -> np.ndarray:
        """GT speed (m/s) by finite-differencing GT positions."""
        n = len(pos_x)
        speed = np.zeros(n)
        if n < 2:
            return speed
        dt = np.diff(time_ns) / 1e9
        with np.errstate(divide="ignore", invalid="ignore"):
            speed[1:] = np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2) / dt
        dt_safe = np.where(dt <= 0.0, 1e-6, dt)
        speed[1:] = np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2) / dt_safe
        return np.nan_to_num(speed, nan=0.0)

    @staticmethod
    def velocity_from_pose(
        pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """GT velocity vector (m/s) by finite-differencing GT positions."""
        n = len(pos_x)
        vx = np.zeros(n)
        vy = np.zeros(n)
        if n < 2:
            return vx, vy
        dt = np.diff(time_ns) / 1e9
        dt_safe = np.where(dt <= 0.0, 1e-6, dt)
        vx[1:] = np.diff(pos_x) / dt_safe
        vy[1:] = np.diff(pos_y) / dt_safe
        return np.nan_to_num(vx, nan=0.0), np.nan_to_num(vy, nan=0.0)

    @classmethod
    @abstractmethod
    def output_keys(cls) -> list[str]:
        """Returns a list of keys that this calculator will output."""
        pass

    @abstractmethod
    def calculate(
        self,
        episode: "AlignedEpisodeBundle",
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Calculate metrics for a single episode."""
        pass
