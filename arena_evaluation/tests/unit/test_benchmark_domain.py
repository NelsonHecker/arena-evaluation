"""ROS domain selection for benchmark runs."""

from __future__ import annotations

import collections.abc
import dataclasses
import os

import pytest

from arena_evaluation.benchmark import domain as domain_module
from arena_evaluation.benchmark.domain import (
    DomainBusyError,
    DomainChoice,
    apply_domain,
    choose_domain,
    describe,
    gz_partition,
)
from arena_evaluation.benchmark.runner import _run_domain
from arena_evaluation.benchmark.state import Manifest


def _probe(busy: set[int]) -> collections.abc.Callable[[int], bool]:
    def probe(domain_id: int) -> bool:
        return domain_id in busy

    return probe


def test_free_domain_takes_the_lowest_clean_candidate():
    choice = choose_domain(probe=_probe({1, 2}), candidates=[1, 2, 3, 4])
    assert choice == DomainChoice(domain_id=3, source="probed", probed=(1, 2))
    assert not choice.explicit


def test_no_clean_candidate_names_the_fix():
    with pytest.raises(DomainBusyError, match="ros_domain_id"):
        choose_domain(probe=_probe({1, 2}), candidates=[1, 2])


def test_explicit_domain_is_kept_when_free():
    choice = choose_domain(57, probe=_probe(set()), candidates=[1])
    assert choice == DomainChoice(domain_id=57, source="explicit", probed=(57,))
    assert choice.explicit


def test_explicit_domain_that_already_has_a_runtime_names_domain_and_fix():
    with pytest.raises(DomainBusyError) as caught:
        choose_domain(1, probe=_probe({1}), candidates=[1, 2])
    message = str(caught.value)
    assert "domain 1" in message
    assert "ros_domain_id" in message
    assert "arena runtime" in message


def test_gz_partition_follows_the_domain():
    assert gz_partition(57) == "arena_57"


def test_apply_domain_exports_domain_and_gz_partition(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(domain_module.ROS_DOMAIN_ID_ENV, raising=False)
    monkeypatch.delenv(domain_module.GZ_PARTITION_ENV, raising=False)
    apply_domain(DomainChoice(domain_id=57, source="explicit", probed=(57,)))
    assert os.environ[domain_module.ROS_DOMAIN_ID_ENV] == "57"
    assert os.environ[domain_module.GZ_PARTITION_ENV] == "arena_57"


def test_describe_reports_where_the_domain_came_from():
    assert "domain 57 (requested" in describe(DomainChoice(domain_id=57, source="explicit", probed=(57,)))
    assert "no arena runtime" in describe(DomainChoice(domain_id=8, source="probed", probed=(1, 2, 3, 4, 5, 6, 7)))


def test_run_domain_keeps_a_lane_domain_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "21")
    monkeypatch.setattr("arena_evaluation.benchmark.runner.choose_domain", lambda explicit=None: choose_domain(explicit, probe=_probe(set()), candidates=[9]))
    choice = _run_domain(None, stored=None, lane="p1")
    assert choice.domain_id == 21


def test_run_domain_prefers_an_explicit_domain(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    monkeypatch.setattr("arena_evaluation.benchmark.runner.choose_domain", lambda explicit=None: choose_domain(explicit, probe=_probe(set()), candidates=[9]))
    assert _run_domain(57, stored=None, lane=None).source == "explicit"


def test_run_domain_reuses_a_recorded_domain_on_resume(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    monkeypatch.setattr("arena_evaluation.benchmark.runner.runtime_on_domain", lambda domain_id: False)
    assert _run_domain(None, stored=42, lane=None) == DomainChoice(domain_id=42, source="resume", probed=(42,))


def test_run_domain_picks_a_domain_when_a_resumed_one_is_busy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    monkeypatch.setattr("arena_evaluation.benchmark.runner.runtime_on_domain", lambda domain_id: True)
    monkeypatch.setattr("arena_evaluation.benchmark.runner.choose_domain", lambda explicit=None: choose_domain(explicit, probe=_probe(set()), candidates=[7]))
    assert _run_domain(None, stored=42, lane=None).domain_id == 7


def test_the_manifest_records_the_domain_and_defaults_it_for_old_runs():
    field = next(entry for entry in dataclasses.fields(Manifest) if entry.name == "ros_domain_id")
    assert field.default is None
