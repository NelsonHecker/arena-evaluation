from __future__ import annotations

from abc import ABC, abstractmethod
import typing

if typing.TYPE_CHECKING:
    from ...storage.schemas import AlignedEpisodeBundle, RobotParams


class BaseMetricCalculator(ABC):
    """
    Abstract Base Class for all metric calculators.
    
    A metric calculator takes an aligned episode bundle and any previously
    calculated metrics it depends on, and returns a dictionary of scalar
    or array results.
    
    To implement a new calculator:
    1. Subclass `BaseMetricCalculator`.
    2. Set the `NAME`, `CATEGORY`, `REQUIRES_PEDSIM`, and `DEPENDS_ON` class attributes.
    3. Implement `output_keys()` to declare what keys this calculator provides.
    4. Implement `calculate()` to perform the actual computation.
    
    The registry will automatically discover your subclass if it is located
    in the `arena_evaluation.processing.metrics` package hierarchy.
    """

    # The unique name of this calculator (used as a key in the registry and dependencies)
    NAME: str = ""
    
    # Category of the metric (e.g., "performance", "social", "naturalness")
    CATEGORY: str = "general"
    
    # Whether this metric requires pedestrian simulation data (arena_peds)
    REQUIRES_PEDSIM: bool = False
    
    # List of calculator NAMEs that must be run before this one.
    DEPENDS_ON: list[str] = []
    
    # List of topics required by this calculator. Each entry can be a string
    # (strictly required topic) or a collection/tuple of strings (any of the listed topics).
    REQUIRED_TOPICS: list[str | list[str] | tuple[str, ...] | set[str]] = []
    
    # Dictionary mapping output keys to their SI units.
    UNITS: dict[str, str] = {}

    def __init__(self, robot_params: RobotParams):
        """
        Initializes the calculator with robot parameters.
        """
        self.robot_params = robot_params

    def resolve_robot_pose(self, episode: "AlignedEpisodeBundle") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and resolve the robot's pose (pos_x, pos_y, yaw) in the map frame.
        Handles both absolute/relative TF ground truth (tf_gt) and fallback to raw odom.
        Returns:
            (pos_x, pos_y, yaw, odom_x_trans, odom_y_trans, odom_yaw_trans)
        """
        import numpy as np
        import polars as pl
        
        # Filter out rows with null or nan in active coordinates to disregard them from calculations
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
        
        # 1. Extract raw odom
        if episode.data is not None and len(episode.data) > 0:
            odom_x = episode.data["pos_x"].to_numpy().copy()
            odom_y = episode.data["pos_y"].to_numpy().copy()
            odom_yaw = episode.data["yaw"].to_numpy().copy()
            
            # Detect teleport jumps in the episode
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
        
        # 2. Transform raw odom to map frame if start_pos is available
        if episode.start_pos and len(episode.start_pos) >= 2 and len(odom_x) > 0:
            start_x, start_y = episode.start_pos[0], episode.start_pos[1]
            start_yaw = episode.start_pos[2] if len(episode.start_pos) >= 3 else 0.0
            
            odom_x0 = odom_x[0]
            odom_y0 = odom_y[0]
            odom_yaw0 = odom_yaw[0]
            
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

        # 3. Check if we should use ground truth TF
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
        """
        Returns a list of keys that this calculator will output.
        This is used to construct the final Parquet schema.
        """
        pass

    @abstractmethod
    def calculate(
        self,
        episode: "AlignedEpisodeBundle",
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """
        Calculate metrics for a single episode.
        
        Args:
            episode: The aligned bundle of topic DataFrames for this episode.
            prior_results: Results from calculators this one depends on.
            
        Returns:
            A dictionary mapping output keys to their calculated values.
            Values should be python scalars, lists, or numpy arrays.
            If calculation fails, return a dictionary of None/NaN values
            so the schema remains consistent.
        """
        pass
