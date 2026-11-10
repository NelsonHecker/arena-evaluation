"""Contest preflight for planner availability, against the real registries."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_mcp")

from arena_evaluation.mcp import authoring


def _availability(registry: list[str], rosnav_rl: list[str] | None = None) -> dict[str, list[str]]:
    return {"registry": registry, "rosnav_rl": rosnav_rl or [], "nav2": []}


def test_contest_rejects_a_drl_driver_without_the_training_feature(monkeypatch):
    monkeypatch.setattr(authoring, "_training_available", lambda: False)
    monkeypatch.setattr("arena_planners.resolver.list_available", lambda **_: _availability(["cadrl"], ["cadrl"]))
    verdict = authoring.check_contest([{"name": "cadrl", "mobile": {"driver": "drl", "planner": "cadrl"}}])
    assert not verdict["valid"]
    assert any("cadrl" in problem and "arena feature training install" in problem for problem in verdict["problems"])


def test_contest_rejects_a_drl_planner_that_is_not_checked_out(monkeypatch):
    monkeypatch.setattr(authoring, "_training_available", lambda: True)
    monkeypatch.setattr("arena_planners.resolver.list_available", lambda **_: _availability(["sicnav"]))
    verdict = authoring.check_contest([{"name": "cadrl", "mobile": {"driver": "drl", "planner": "cadrl"}}])
    assert not verdict["valid"]
    assert any("cadrl" in problem and "arena feature planners add cadrl" in problem for problem in verdict["problems"])


def test_contest_accepts_a_drl_planner_that_is_checked_out(monkeypatch):
    monkeypatch.setattr(authoring, "_training_available", lambda: True)
    monkeypatch.setattr("arena_planners.resolver.list_available", lambda **_: _availability(["cadrl"]))
    verdict = authoring.check_contest([{"name": "cadrl", "mobile": {"driver": "drl", "planner": "cadrl"}}])
    assert verdict["valid"], verdict["problems"]


def test_a_nav2_contest_without_a_planner_block_passes():
    verdict = authoring.check_contest([{"name": "nav2", "mobile": {"driver": "nav2"}}])
    assert verdict["valid"], verdict["problems"]


def test_plugins_finds_nested_plugin_classes():
    config = {"controller_server": {"ros__parameters": {"FollowPath": {"plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"}}}}
    assert authoring._plugins(config) == ["nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"]
