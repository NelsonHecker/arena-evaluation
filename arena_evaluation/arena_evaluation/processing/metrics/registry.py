from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import types
import typing
from collections import defaultdict, deque
from collections.abc import Mapping

import polars as pl

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    MetricScope,
    ScopeExecutionOutcome,
    ScopeSlice,
)
from arena_evaluation.storage.exceptions import CircularDependencyError, MetricInputError
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


_SERIES_STEM_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _series_table_stem(class_name: str) -> str:
    """Snake-case calculator class name without its Calculator suffix, the 1NF series table stem."""
    return _SERIES_STEM_BOUNDARY.sub("_", class_name.removesuffix("Calculator")).lower()


def _topic_row_count(table: pl.DataFrame | pl.LazyFrame | None) -> int:
    """Row count of a materialized or lazy topic table, 0 when None."""
    if table is None:
        return 0
    if isinstance(table, pl.LazyFrame):
        return int(table.select(pl.len()).collect().item())
    return len(table)


class MetricRegistry:
    """Discovers, validates, and executes metric calculators across all 5 evaluation scopes."""

    def __init__(self, robot_params: RobotParams, world: str | None = None, metrics_config: dict[str, typing.Any] | None = None):
        self.robot_params = robot_params
        self.world = world
        self.metrics_config = dict(metrics_config or {})
        self.calculators: dict[str, BaseMetricCalculator] = {}
        self.calculators_by_scope: dict[MetricScope, dict[str, BaseMetricCalculator]] = {scope: {} for scope in MetricScope}
        self.execution_stages_by_scope: dict[MetricScope, list[list[str]]] = {scope: [] for scope in MetricScope}
        self.execution_stages: list[list[str]] = []
        self.discover_calculators_cls()
        self._register_calculators()
        for name, calc in self.calculators.items():
            for scope in calc.effective_scopes():
                if name not in self.calculators_by_scope[scope]:
                    self.calculators_by_scope[scope][name] = calc

        self.execution_stages_by_scope = {scope: self._compute_execution_order(scope) for scope in MetricScope}
        self.execution_stages = self.execution_stages_by_scope[MetricScope.EPISODE]

    @classmethod
    def discover_calculators_cls(cls) -> None:
        """Recursively discover all BaseMetricCalculator subclasses in the metrics package."""
        import arena_evaluation.processing.metrics as metrics_pkg

        def iter_namespace(pkg: types.ModuleType) -> typing.Iterator[pkgutil.ModuleInfo]:
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
        """Register all non-abstract subclasses, indexed by name and by scope."""

        def _get_leaf_subclasses(cls: type[BaseMetricCalculator]) -> list[type[BaseMetricCalculator]]:
            leaves: list[type[BaseMetricCalculator]] = []
            for sub in cls.__subclasses__():
                if inspect.isabstract(sub):
                    leaves.extend(_get_leaf_subclasses(sub))
                else:
                    leaves.append(sub)
                    leaves.extend(_get_leaf_subclasses(sub))
            return leaves

        for calc_cls in set(_get_leaf_subclasses(BaseMetricCalculator)):
            if not inspect.isabstract(calc_cls) and "<locals>" not in calc_cls.__qualname__:
                if not calc_cls.NAME:
                    raise ValueError(f"Calculator {calc_cls.__name__} must define a NAME")
                if calc_cls.NAME in self.calculators:
                    raise ValueError(f"Duplicate calculator NAME: {calc_cls.NAME}")
                self.register(calc_cls)

    def register(self, calculator: type[BaseMetricCalculator] | BaseMetricCalculator) -> None:
        """Register a calculator class or instance across all of its effective scopes."""
        calc = calculator(self.robot_params) if isinstance(calculator, type) else calculator
        calc.world = self.world
        calc.metrics_config = self.metrics_config
        self.calculators[calc.NAME] = calc
        for scope in calc.effective_scopes():
            self.calculators_by_scope[scope][calc.NAME] = calc
            self.execution_stages_by_scope[scope] = self._compute_execution_order(scope)
        self.execution_stages = self.execution_stages_by_scope[MetricScope.EPISODE]

    def series_calculators(self) -> list[tuple[str, BaseMetricCalculator]]:
        """Registered calculators publishing a 1NF series table, with their table stem, ordered by stem."""
        return sorted(
            (
                (calc.SERIES_TABLE_STEM or _series_table_stem(type(calc).__name__), calc)
                for calc in self.calculators.values()
                if type(calc).series_table is not BaseMetricCalculator.series_table
            ),
            key=lambda item: item[0],
        )

    @classmethod
    def get_all_units(cls) -> dict[str, str]:
        """Returns a combined dictionary of all units from all metrics."""
        cls.discover_calculators_cls()
        units = {}
        for calc_cls in BaseMetricCalculator.__subclasses__():
            if not inspect.isabstract(calc_cls):
                units.update(calc_cls.UNITS)
        return units

    def _compute_execution_order(self, scope: MetricScope = MetricScope.EPISODE) -> list[list[str]]:
        """Compute topological sort of calculators in `scope` using Kahn's algorithm."""
        scope_calcs = self.calculators_by_scope[scope]
        in_degree = {name: 0 for name in scope_calcs}
        adj_list = defaultdict(list)

        for name, calc in scope_calcs.items():
            for dep in calc.DEPENDS_ON:
                if dep in scope_calcs:
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

        if processed_count != len(scope_calcs):
            raise CircularDependencyError(f"Circular dependency detected among metric calculators in scope '{scope}'")

        return stages

    def get_execution_order_for_scope(self, scope: MetricScope) -> list[list[str]]:
        """Return topological execution stages of calculators registered for `scope`."""
        return self.execution_stages_by_scope[scope]

    def list_metrics(self) -> list[dict[str, typing.Any]]:
        """Returns metadata about all registered metrics."""
        return [
            {
                "name": name,
                "category": calc.CATEGORY,
                "scope": calc.SCOPE,
                "scopes": sorted(s.value for s in calc.effective_scopes()),
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

    def run_scope(
        self,
        scope: MetricScope,
        episode: AlignedEpisodeBundle,
        *,
        available_topics: set[str] | None = None,
        prior_results: dict[str, typing.Any] | None = None,
        pedsim_available: bool = True,
        progress_callback: typing.Callable[[str, int, int], None] | None = None,
        scope_slice: ScopeSlice | None = None,
    ) -> ScopeExecutionOutcome:
        """Executes all calculators registered for `scope` in topological order."""
        slice_episode = scope_slice.window_bundle if scope_slice is not None else episode
        full_episode = scope_slice.full_episode if scope_slice is not None else episode
        full_topics = full_episode.topics if isinstance(full_episode.topics, Mapping) else {}
        slice_topics = slice_episode.topics if isinstance(slice_episode.topics, Mapping) else {}

        if available_topics is not None:
            present_topics = {topic for topic in available_topics if full_topics.get(topic, True) is not None}
        else:
            present_topics = {name for name, table in full_topics.items() if table is not None}

        results = dict(prior_results or {})
        provenance: dict[str, str] = {}
        stages = self.execution_stages_by_scope[scope]
        total_calcs = sum(len(stage) for stage in stages)
        calc_idx = 0

        for stage in stages:
            for calc_name in stage:
                calc_idx += 1
                calc = self.calculators_by_scope[scope][calc_name]
                scope_keys = calc.output_keys_for_scope(scope)

                missing_topic: str | None = None
                empty_slice_topic = False
                for req in calc.required_topics_for_scope(scope):
                    if isinstance(req, str):
                        if req not in present_topics:
                            missing_topic = req
                            break
                        if req in slice_topics and _topic_row_count(slice_topics[req]) == 0:
                            empty_slice_topic = True
                    else:
                        if not any(t in present_topics for t in req):
                            missing_topic = req[0] if req else "unknown"
                            break
                        if all(t in slice_topics for t in req) and all(_topic_row_count(slice_topics[t]) == 0 for t in req):
                            empty_slice_topic = True

                if missing_topic is not None:
                    for key in scope_keys:
                        if key not in results:
                            results[key] = None
                        if results.get(key) is None:
                            provenance[key] = f"{key}:topic_missing:{missing_topic}"
                    continue

                if empty_slice_topic:
                    for key in scope_keys:
                        if key not in results:
                            results[key] = None
                        if results.get(key) is None:
                            provenance[key] = f"{key}:no_samples"
                    continue

                if calc.REQUIRES_PEDSIM and not pedsim_available:
                    for key in scope_keys:
                        if key not in results:
                            results[key] = None
                        if results.get(key) is None:
                            provenance[key] = f"{key}:pedsim_unavailable"
                    continue

                if progress_callback is not None:
                    try:
                        progress_callback(calc_name, calc_idx, total_calcs)
                    except Exception:
                        pass

                try:
                    calc_out = calc.calculate_for_scope(slice_episode, dict(results), scope_slice)
                    expected_keys = set(scope_keys)
                    actual_keys = set(calc_out.keys())
                    if not expected_keys.issubset(actual_keys):
                        missing = expected_keys - actual_keys
                        raise ValueError(f"Calculator {calc_name} missing output keys: {missing}")

                    results.update(calc_out)
                    for key in scope_keys:
                        if calc_out.get(key) is not None:
                            provenance.pop(key, None)
                        elif key not in provenance:
                            provenance[key] = f"{key}:no_samples"

                except MetricInputError:
                    raise
                except Exception as exc:
                    import logging

                    logging.getLogger(__name__).warning(f"Calculator {calc_name} failed on episode {slice_episode.episode_id}: {exc}")
                    exc_name = type(exc).__name__
                    for key in scope_keys:
                        if key not in results:
                            results[key] = None
                        provenance[key] = f"{key}:exception:{exc_name}"

        return ScopeExecutionOutcome(scope=scope, results=results, provenance=provenance)

    def run(
        self,
        episode: AlignedEpisodeBundle,
        *,
        available_topics: set[str] | None = None,
        pedsim_available: bool = True,
        progress_callback: typing.Callable[[str, int, int], None] | None = None,
    ) -> dict[str, typing.Any]:
        """Executes EPISODE scope calculators in topological order."""
        outcome = self.run_scope(
            MetricScope.EPISODE,
            episode,
            available_topics=available_topics,
            pedsim_available=pedsim_available,
            progress_callback=progress_callback,
        )
        return outcome.results
