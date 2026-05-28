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

    def __init__(self, robot_params: RobotParams):
        """
        Initializes the calculator with robot parameters.
        """
        self.robot_params = robot_params

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
