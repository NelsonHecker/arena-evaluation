"""MCP WP5 finding tools: the empirical claim lifecycle of one benchmark run.

A finding is the analyst's recorded claim ("the blind crowd made dwb cut the corner"), the
query or pinned evidence it rests on, the numbers it quotes and the figure it points at.
Findings live in ``<run>/findings.yaml`` written exactly like ``notes.yaml``; attaching one
to a report also records a ``notes.yaml`` row, so the claim renders next to the plot through
the existing report builder instead of a second store.
"""

from __future__ import annotations

import datetime
import math
import pathlib
import uuid
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

import polars as pl
import pydantic
import yaml
from arena_mcp.core import ToolError

from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.analytics.queries import QueryCatalog

from .base import RunId, component, domain, run_path
from .runs import Note, write_notes

__all__ = [
    "Finding",
    "FindingAttachResponse",
    "FindingStatus",
    "FindingsResponse",
    "evaluation_finding_attach",
    "evaluation_finding_create",
    "evaluation_finding_status",
    "evaluation_findings",
]

FINDINGS_FILENAME: Final[str] = "findings.yaml"
DEFAULT_REPORT: Final[str] = "report.html"
NOTE_LABEL_PREFIX: Final[str] = "finding"


class FindingStatus(StrEnum):
    """Review state of one recorded claim."""

    DRAFT = "draft"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Finding(pydantic.BaseModel):
    """One empirical claim, the evidence it rests on and its review state."""

    finding_id: str
    claim: str
    status: FindingStatus = FindingStatus.DRAFT
    created_at: str
    episode_id: int | None = None
    encounter_id: str | None = None
    contact_id: str | None = None
    pedestrian_id: str | None = None
    query: str | None = None
    stored_query: str | None = None
    evidence_id: str | None = None
    numbers: dict[str, float] = pydantic.Field(default_factory=dict)
    figure_path: str | None = None
    report: str | None = None


class FindingsResponse(pydantic.BaseModel):
    """Every finding recorded for one run, newest last."""

    run_id: str
    n_findings: int
    findings: list[Finding]


class FindingAttachResponse(pydantic.BaseModel):
    """A finding attached to a report, with the rendered note row that carries it."""

    run_id: str
    finding: Finding
    note_label: str
    findings_path: str
    notes_path: str


def findings_path(run_id: str) -> pathlib.Path:
    """The findings document of a run."""
    return run_path(run_id) / FINDINGS_FILENAME


def load_findings(path: pathlib.Path) -> list[Finding]:
    """The findings document as typed findings, empty when it does not exist yet."""
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    rows = raw.get("findings", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ToolError(f"{path.name} is not a findings document", hint="delete it or fix it as a list under 'findings:'")
    try:
        return [Finding.model_validate(row) for row in rows]
    except pydantic.ValidationError as exc:
        raise ToolError(f"{path.name} carries an invalid finding: {exc}") from exc


def write_findings(path: pathlib.Path, findings: list[Finding]) -> None:
    """Persist the findings document with the same load/write pattern as ``notes.yaml``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"findings": [finding.model_dump(mode="json") for finding in findings]}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _finding(path: pathlib.Path, finding_id: str) -> tuple[list[Finding], Finding]:
    """The findings document and the one finding it holds, a ToolError when the id is unknown."""
    findings = load_findings(path)
    for finding in findings:
        if finding.finding_id == finding_id:
            return findings, finding
    raise ToolError(f"no finding {finding_id!r} in this run", hint="evaluation_findings lists them")


def _new_finding_id(findings: list[Finding]) -> str:
    """A fresh finding id, unique inside the run."""
    taken = {finding.finding_id for finding in findings}
    for _ in range(100):
        candidate = f"fnd-{uuid.uuid4().hex[:8]}"
        if candidate not in taken:
            return candidate
    raise ToolError("could not allocate a finding id", hint="the run holds an implausible number of findings")


def _checked_claim(claim: str) -> str:
    text = claim.strip()
    if not text:
        raise ToolError("a finding needs a claim")
    return text


def _checked_numbers(numbers: dict[str, float] | None) -> dict[str, float]:
    """The quoted numbers, a ToolError when one is not finite."""
    if not numbers:
        return {}
    non_finite = sorted(key for key, value in numbers.items() if not math.isfinite(value))
    if non_finite:
        raise ToolError(f"finding numbers must be finite, got {', '.join(non_finite)}")
    return dict(numbers)


def _checked_evidence(run_dir: pathlib.Path, evidence_id: str) -> str:
    """The evidence id when the run's pinned evidence index carries it."""
    store = EvidenceStore(run_dir / "evidence")
    index = store.scan_index().filter(pl.col("evidence_id") == evidence_id).collect()
    if len(index) == 0:
        raise ToolError(f"no pinned evidence {evidence_id!r} in this run", hint="evaluation_query pin_evidence=true pins one, evaluation_evidence lists them")
    return evidence_id


def _checked_stored_query(run_dir: pathlib.Path, name: str) -> str:
    """The stored query name when the run's query declarations resolve it."""
    if not QueryCatalog([run_dir / "query_declarations"]).has_query(name):
        raise ToolError(f"no stored query {name!r} in this run", hint="evaluation_queries lists the run's query declarations")
    return name


def _checked_figure(run_dir: pathlib.Path, figure_path: str) -> str:
    """The figure path relative to the run when it stays inside, absolute otherwise; must exist."""
    candidate = pathlib.Path(figure_path)
    resolved = candidate if candidate.is_absolute() else run_dir / candidate
    if not resolved.is_file():
        raise ToolError(f"no figure at {figure_path!r}", hint="evaluation_render_plot writes plot artifacts into the run's plots directory")
    try:
        return str(resolved.relative_to(run_dir))
    except ValueError:
        return str(resolved)


@domain.tool()
def evaluation_finding_create(
    run_id: RunId,
    claim: Annotated[str, pydantic.Field(description="the empirical claim, one sentence")],
    episode_id: Annotated[int | None, pydantic.Field(description="episode the claim is about")] = None,
    encounter_id: Annotated[str | None, pydantic.Field(description="encounter the claim is about")] = None,
    contact_id: Annotated[str | None, pydantic.Field(description="contact the claim is about")] = None,
    pedestrian_id: Annotated[str | None, pydantic.Field(description="pedestrian the claim is about")] = None,
    query: Annotated[str | None, pydantic.Field(description="the SQL the claim rests on")] = None,
    stored_query: Annotated[str | None, pydantic.Field(description="a stored query declaration the claim rests on")] = None,
    evidence_id: Annotated[str | None, pydantic.Field(description="pinned evidence the claim rests on")] = None,
    numbers: Annotated[dict[str, float] | None, pydantic.Field(description="numbers the claim quotes, metric name to value")] = None,
    figure_path: Annotated[str | None, pydantic.Field(description="figure the claim points at, relative to the run or absolute")] = None,
) -> Finding:
    """Record one empirical claim as a draft finding, validating the evidence, stored query and figure it names.

    The claim is persisted in the run's ``findings.yaml`` and can later be listed, reviewed
    (draft/accepted/rejected) and attached to a report with evaluation_finding_attach.
    """
    path = findings_path(run_id)
    run_dir = path.parent
    if query is not None and stored_query is not None:
        raise ToolError("pass the SQL or a stored query name, not both", hint="evaluation_queries lists stored queries")
    findings = load_findings(path)
    finding = Finding(
        finding_id=_new_finding_id(findings),
        claim=_checked_claim(claim),
        status=FindingStatus.DRAFT,
        created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
        episode_id=episode_id,
        encounter_id=encounter_id,
        contact_id=contact_id,
        pedestrian_id=pedestrian_id,
        query=query,
        stored_query=None if stored_query is None else _checked_stored_query(run_dir, stored_query),
        evidence_id=None if evidence_id is None else _checked_evidence(run_dir, evidence_id),
        numbers=_checked_numbers(numbers),
        figure_path=None if figure_path is None else _checked_figure(run_dir, figure_path),
    )
    write_findings(path, [*findings, finding])
    return finding


@domain.tool(read_only=True)
def evaluation_findings(run_id: RunId) -> FindingsResponse:
    """Every finding recorded for a run, draft/accepted/rejected included."""
    findings = load_findings(findings_path(run_id))
    return FindingsResponse(run_id=run_id, n_findings=len(findings), findings=findings)


@domain.tool()
def evaluation_finding_status(
    run_id: RunId,
    finding_id: Annotated[str, pydantic.Field(description="finding to review, from evaluation_findings")],
    status: Annotated[Literal["draft", "accepted", "rejected"], pydantic.Field(description="new review state")],
) -> Finding:
    """Move one finding between draft, accepted and rejected."""
    path = findings_path(run_id)
    findings, finding = _finding(path, finding_id)
    reviewed = finding.model_copy(update={"status": FindingStatus(status)})
    write_findings(path, [reviewed if item.finding_id == finding_id else item for item in findings])
    return reviewed


@domain.tool()
def evaluation_finding_attach(
    run_id: RunId,
    finding_id: Annotated[str, pydantic.Field(description="finding to attach, from evaluation_findings")],
    report: Annotated[str, pydantic.Field(description="report.html or the run-local manifest name the finding belongs to")] = DEFAULT_REPORT,
) -> FindingAttachResponse:
    """Attach a finding to a report: record the report on the finding and append its note so the report renders it.

    The note row goes into the existing ``notes.yaml`` (label ``finding:<id>``), the same store the
    report builder already renders next to its tables, so no second presentation store is invented.
    """
    run_dir = findings_path(run_id).parent
    report_file = run_dir / report
    manifest = run_dir / f"{component(report.removesuffix('.yaml'), 'report')}.yaml"
    if report_file.is_file():
        report_token = report_file.name
    elif manifest.is_file():
        report_token = manifest.name
    else:
        raise ToolError(f"run {run_id!r} has no report {report!r}", hint="evaluation_report writes report.html; evaluation_create_manifest writes <name>.yaml in the run")
    path = findings_path(run_id)
    findings, finding = _finding(path, finding_id)
    attached = finding.model_copy(update={"report": report_token})
    write_findings(path, [attached if item.finding_id == finding_id else item for item in findings])
    note_label = f"{NOTE_LABEL_PREFIX}:{finding_id}"
    notes_path = run_dir / "notes.yaml"
    write_notes(notes_path, [Note(label=note_label, value=f"{attached.claim} [{attached.status.value}]")], "merge")
    return FindingAttachResponse(
        run_id=run_id,
        finding=attached,
        note_label=note_label,
        findings_path=str(path),
        notes_path=str(notes_path),
    )


@domain.resource("arena://evaluation/runs/{run_id}/findings", mime_type="application/json", description="A run's findings.yaml as typed findings")
def run_findings_resource(run_id: str) -> list[dict[str, Any]]:
    return [finding.model_dump(mode="json") for finding in load_findings(findings_path(run_id))]
