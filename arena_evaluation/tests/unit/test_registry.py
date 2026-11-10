import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, MetricScope
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.exceptions import CircularDependencyError
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


# Mock calculators for testing topological sort
class Calc1(BaseMetricCalculator):
    NAME = "calc1"
    DEPENDS_ON = []

    @classmethod
    def output_keys(cls):
        return ["o1"]

    def calculate(self, ep, prior):
        return {"o1": 1}


class Calc2(BaseMetricCalculator):
    NAME = "calc2"
    DEPENDS_ON = ["calc1"]

    @classmethod
    def output_keys(cls):
        return ["o2"]

    def calculate(self, ep, prior):
        return {"o2": prior["o1"] + 1}


class Calc3(BaseMetricCalculator):
    NAME = "calc3"
    DEPENDS_ON = ["calc2"]

    @classmethod
    def output_keys(cls):
        return ["o3"]

    def calculate(self, ep, prior):
        return {"o3": prior["o2"] + 1}


def test_registry_ordering(monkeypatch):
    # Mock discovery/registration to only find our mocks
    monkeypatch.setattr(MetricRegistry, "discover_calculators_cls", classmethod(lambda cls: None))
    monkeypatch.setattr(
        MetricRegistry,
        "_register_calculators",
        lambda self: setattr(
            self,
            "calculators",
            {
                "calc3": Calc3(self.robot_params),
                "calc1": Calc1(self.robot_params),
                "calc2": Calc2(self.robot_params),
            },
        ),
    )

    params = RobotParams(0.2, 0.0, 10.0)
    registry = MetricRegistry(params)

    stages = registry.execution_order()
    assert stages == [["calc1"], ["calc2"], ["calc3"]]


class CustomNullReasonCalc(BaseMetricCalculator):
    NAME = "custom_null_reason"
    DEPENDS_ON = []

    @classmethod
    def output_keys(cls):
        return ["custom_null"]

    def calculate(self, ep, prior):
        return {"custom_null": None, "custom_null_null_reason": "custom_reason"}


def test_registry_uses_a_calculators_companion_null_reason(monkeypatch):
    monkeypatch.setattr(MetricRegistry, "discover_calculators_cls", classmethod(lambda cls: None))
    monkeypatch.setattr(
        MetricRegistry,
        "_register_calculators",
        lambda self: setattr(self, "calculators", {"custom_null_reason": CustomNullReasonCalc(self.robot_params)}),
    )
    registry = MetricRegistry(RobotParams(robot_radius=0.25))
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[])

    outcome = registry.run_scope(MetricScope.EPISODE, episode, available_topics=set(), pedsim_available=True)

    assert outcome.results["custom_null"] is None
    assert "custom_null_null_reason" not in outcome.results
    assert outcome.provenance["custom_null"] == "custom_null:custom_reason"
