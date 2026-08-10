from __future__ import annotations

from abc import ABC, abstractmethod
import typing

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
        
<<<<<<< HEAD
        # 1. Extract raw odom
        anchor_x = anchor_y = anchor_yaw = None
        if episode.data is not None and len(episode.data) > 0:
            odom_x = episode.data["pos_x"].to_numpy().copy()
            odom_y = episode.data["pos_y"].to_numpy().copy()
            odom_yaw = episode.data["yaw"].to_numpy().copy()
            anchor_x, anchor_y, anchor_yaw = odom_x[0], odom_y[0], odom_yaw[0]

            # Detect teleport jumps in the episode
=======
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

            
>>>>>>> origin-fork/jazzy
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

            odom_x0 = anchor_x if anchor_x is not None else odom_x[0]
            odom_y0 = anchor_y if anchor_y is not None else odom_y[0]
            odom_yaw0 = anchor_yaw if anchor_yaw is not None else odom_yaw[0]
            
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
