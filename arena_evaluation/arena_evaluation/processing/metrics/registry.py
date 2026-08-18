from __future__ import annotations

import importlib
import inspect
import pkgutil
import typing
from collections import defaultdict, deque

from .base import BaseMetricCalculator
from ...storage.exceptions import CircularDependencyError

if typing.TYPE_CHECKING:
    from ...storage.schemas import RobotParams, AlignedEpisodeBundle


class MetricRegistry:
    """Discovers, validates, and executes metric calculators in topological order."""
    def __init__(self, robot_params: RobotParams, world: str | None = None):
        self.robot_params = robot_params
        self.world = world
        self.calculators: dict[str, BaseMetricCalculator] = {}
        self.discover_calculators_cls()
        self._register_calculators()
        self.execution_stages = self._compute_execution_order()

    @classmethod
    def discover_calculators_cls(cls) -> None:
        """Recursively discover all BaseMetricCalculator subclasses in the metrics package."""
        import arena_evaluation.processing.metrics as metrics_pkg
        
        def iter_namespace(pkg):
            return pkgutil.iter_modules(pkg.__path__, pkg.__name__ + ".")

        for _, name, _ in iter_namespace(metrics_pkg):
            importlib.import_module(name)
            
        # Recursive discovery for subpackages
        for _, name, ispkg in iter_namespace(metrics_pkg):
            if ispkg:
                sub_pkg = importlib.import_module(name)
                for _, sub_name, _ in iter_namespace(sub_pkg):
                    importlib.import_module(sub_name)

    def _register_calculators(self) -> None:
        """Register all non-abstract subclasses"""
        for calc_cls in BaseMetricCalculator.__subclasses__():
            if not inspect.isabstract(calc_cls):
                if not calc_cls.NAME:
                    raise ValueError(f"Calculator {calc_cls.__name__} must define a NAME")
                if calc_cls.NAME in self.calculators:
                    raise ValueError(f"Duplicate calculator NAME: {calc_cls.NAME}")
                calc = calc_cls(self.robot_params)
                calc.world = self.world
                self.calculators[calc_cls.NAME] = calc

    @classmethod
    def get_all_units(cls) -> dict[str, str]:
        """Returns a combined dictionary of all units from all metrics."""
        cls.discover_calculators_cls()
        units = {}
        for calc_cls in BaseMetricCalculator.__subclasses__():
            if not inspect.isabstract(calc_cls):
                units.update(calc_cls.UNITS)
        return units

    def _compute_execution_order(self) -> list[list[str]]:
        """Compute topological sort of calculators using Kahn's algorithm."""
        in_degree = {name: 0 for name in self.calculators}
        adj_list = defaultdict(list)

        for name, calc in self.calculators.items():
            for dep in calc.DEPENDS_ON:
                if dep not in self.calculators:
                    raise ValueError(f"Calculator '{name}' depends on unknown calculator '{dep}'")
                adj_list[dep].append(name)
                in_degree[name] += 1

        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        stages = []
        processed_count = 0

        while queue:
            stage_size = len(queue)
            current_stage = []
            
            for _ in range(stage_size):
                u = queue.popleft()
                current_stage.append(u)
                processed_count += 1
                
                for v in adj_list[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
            
            stages.append(current_stage)

        if processed_count != len(self.calculators):
            raise CircularDependencyError("Circular dependency detected among metric calculators")

        return stages

    def list_metrics(self) -> list[dict[str, typing.Any]]:
        """Returns metadata about all registered metrics."""
        return [
            {
                "name": name,
                "category": calc.CATEGORY,
                "requires_pedsim": calc.REQUIRES_PEDSIM,
                "depends_on": calc.DEPENDS_ON,
                "required_topics": calc.REQUIRED_TOPICS,
                "outputs": calc.output_keys(),
                "primary_outputs": list(calc.PRIMARY_OUTPUTS),
                "output_directions": dict(calc.OUTPUT_DIRECTIONS),
            }
            for name, calc in self.calculators.items()
        ]

    def execution_order(self) -> list[list[str]]:
        """Returns the ordered stages of calculation."""
        return self.execution_stages

    def run(
        self,
        episode: AlignedEpisodeBundle,
        pedsim_available: bool = True,
        available_topics: set[str] | None = None
    ) -> dict[str, typing.Any]:
        """Executes all calculators in topological order."""
        results = {}

        if available_topics is None:
            available_topics = set(["odom"])
            if episode.data is not None:
                cols = set(episode.data.columns)
                if "scan_ranges" in cols or "scan_min" in cols:
                    available_topics.add("scan")
                if "linear_x" in cols or "cmd_linear" in cols or "cmd_vel" in cols:
                    available_topics.add("cmd_vel")
                if "joint_vel_left" in cols or "joint_vel_right" in cols or "joint_velocities" in cols:
                    available_topics.add("joint_states")
                if "peds_positions" in cols:
                    available_topics.add("peds")
                if "collision_event" in cols:
                    available_topics.add("collision_events")
                if "action_type" in cols:
                    available_topics.add("collision_monitor_state")
                if "pos_x_gt" in cols:
                    available_topics.add("tf_gt")
                if "total_power_w" in cols:
                    available_topics.add("power")
                if "total_energy_consumed_wh" in cols or "battery_soc_percent" in cols:
                    available_topics.add("energy")
                if "total_level_af_dba" in cols:
                    available_topics.add("acoustics")
        
        for stage in self.execution_stages:
            for calc_name in stage:
                calc = self.calculators[calc_name]
                
                # Check topic dependencies
                skip_due_to_topics = False
                for req in calc.REQUIRED_TOPICS:
                    if isinstance(req, str):
                        if req not in available_topics:
                            skip_due_to_topics = True
                            break
                    elif isinstance(req, (list, tuple, set)):
                        if not any(t in available_topics for t in req):
                            skip_due_to_topics = True
                            break

                if skip_due_to_topics:
                    # Fill with None for schema consistency
                    for key in calc.output_keys():
                        results[key] = None
                    continue

                if calc.REQUIRES_PEDSIM and not pedsim_available:
                    # Fill with None for schema consistency
                    for key in calc.output_keys():
                        results[key] = None
                    continue
                
                try:
                    # Pass a read-only view of prior results
                    calc_out = calc.calculate(episode, dict(results))
                    
                    # Validate output keys
                    expected_keys = set(calc.output_keys())
                    actual_keys = set(calc_out.keys())
                    if not expected_keys.issubset(actual_keys):
                        missing = expected_keys - actual_keys
                        raise ValueError(f"Calculator {calc_name} missing output keys: {missing}")
                        
                    results.update(calc_out)
                    
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Calculator {calc_name} failed on episode {episode.episode_id}: {e}"
                    )
                    # Fill with None for schema consistency on failure
                    for key in calc.output_keys():
                        results[key] = None

        return results
