from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_mcp.core import ToolError

from arena_evaluation.mcp.findings import (
    FindingStatus,
    evaluation_finding_attach,
    evaluation_finding_create,
    evaluation_finding_status,
    evaluation_findings,
)

from mcp_fixtures import RUN_ID, arena_data, build_run


@pytest.fixture
def run_dir(tmp_path):
    path = build_run(tmp_path / "data" / "benchmarks", RUN_ID)
    with arena_data(tmp_path):
        yield path


@pytest.mark.parametrize(
    ("start", "target", "allowed"),
    (
        ("draft", "draft", True),
        ("draft", "accepted", True),
        ("draft", "rejected", True),
        ("accepted", "accepted", True),
        ("accepted", "rejected", True),
        ("accepted", "draft", False),
        ("rejected", "rejected", True),
        ("rejected", "draft", False),
        ("rejected", "accepted", False),
    ),
)
def test_status_transition_matrix(run_dir, start, target, allowed):
    finding = evaluation_finding_create(RUN_ID, f"claim moving {start} to {target}")
    if start != "draft":
        evaluation_finding_status(RUN_ID, finding.finding_id, start)

    if not allowed:
        with pytest.raises(ToolError, match="cannot become"):
            evaluation_finding_status(RUN_ID, finding.finding_id, target)
        assert evaluation_findings(RUN_ID).findings[0].status.value == start
        return

    moved = evaluation_finding_status(RUN_ID, finding.finding_id, target)
    assert moved.status.value == target
    assert evaluation_findings(RUN_ID).findings[0].status.value == target


def test_an_unknown_status_is_refused_with_the_valid_names(run_dir):
    finding = evaluation_finding_create(RUN_ID, "claim")

    with pytest.raises(ToolError) as caught:
        evaluation_finding_status(RUN_ID, finding.finding_id, "archived")

    assert "unknown finding status" in caught.value.message
    assert "draft, accepted or rejected" in caught.value.hint


def test_attach_needs_an_accepted_finding(run_dir):
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    finding = evaluation_finding_create(RUN_ID, "claim")

    with pytest.raises(ToolError) as caught:
        evaluation_finding_attach(RUN_ID, finding.finding_id)

    assert "only an accepted finding" in caught.value.message
    assert "evaluation_finding_status" in caught.value.hint
    assert evaluation_findings(RUN_ID).findings[0].report is None

    evaluation_finding_status(RUN_ID, finding.finding_id, "accepted")
    attached = evaluation_finding_attach(RUN_ID, finding.finding_id)
    assert attached.finding.status is FindingStatus.ACCEPTED
    assert attached.finding.report == "report.html"


def test_two_concurrent_appenders_lose_nothing(run_dir):
    claims = [f"concurrent claim {index}" for index in range(16)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        created = list(pool.map(lambda claim: evaluation_finding_create(RUN_ID, claim), claims))

    stored = evaluation_findings(RUN_ID).findings
    assert sorted(finding.claim for finding in stored) == sorted(claims)
    assert len({finding.finding_id for finding in stored}) == len(claims)
    assert sorted(finding.finding_id for finding in created) == sorted(finding.finding_id for finding in stored)


def test_creating_a_finding_also_locks_the_document(run_dir):
    finding = evaluation_finding_create(RUN_ID, "claim")

    assert (run_dir / ".findings.yaml.lock").is_file()
    assert not list(run_dir.glob("*.tmp"))
    assert evaluation_findings(RUN_ID).findings[0].finding_id == finding.finding_id
