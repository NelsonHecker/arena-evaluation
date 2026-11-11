from __future__ import annotations

import typing

import pydantic
import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_evaluation.mcp.base import domain
from arena_evaluation.mcp.findings import evaluation_finding_attach, evaluation_finding_create, evaluation_finding_status, evaluation_findings
from arena_evaluation.mcp.query import evaluation_query
from arena_evaluation.mcp.slices import evaluation_slice, evaluation_worst_encounters

OVERWRITING_TOOLS = (
    "evaluation_benchmark",
    "evaluation_preflight",
    "evaluation_stop",
    "evaluation_kill",
    "evaluation_record_start",
    "evaluation_record_stop",
    "evaluation_write_notes",
    "evaluation_finding_status",
    "evaluation_create_suite",
    "evaluation_create_contest",
    "evaluation_create_manifest",
    "evaluation_query_save",
    "evaluation_save_plot_declaration",
    "evaluation_process",
    "evaluation_extract",
    "evaluation_report",
    "evaluation_plot",
    "evaluation_pipeline",
    "evaluation_acoustic",
)

PydanticTool = typing.Callable[..., pydantic.BaseModel]


def _annotations() -> dict[str, object]:
    return {spec.name: spec.annotations for spec in domain.tools}


def test_every_overwriting_tool_is_advertised_as_destructive():
    annotations = _annotations()

    for name in OVERWRITING_TOOLS:
        assert annotations[name].destructive_hint is True, name
        assert annotations[name].read_only_hint is False, name


def test_read_only_tools_are_idempotent_and_not_destructive():
    for spec in domain.tools:
        if spec.annotations.read_only_hint:
            assert spec.annotations.idempotent_hint is True, spec.name
            assert spec.annotations.destructive_hint is False, spec.name


def test_a_query_is_a_writer_only_when_it_pins_evidence():
    annotations = _annotations()["evaluation_query"]

    assert annotations.read_only_hint is False
    assert annotations.destructive_hint is False


def test_append_insight_is_gone_and_the_finding_tools_replace_it():
    names = {spec.name for spec in domain.tools}

    assert "evaluation_append_insight" not in names
    assert {"evaluation_finding_create", "evaluation_finding_status", "evaluation_finding_attach", "evaluation_findings"} <= names


@pytest.mark.parametrize("tool", (evaluation_findings, evaluation_finding_create, evaluation_finding_status, evaluation_finding_attach, evaluation_slice, evaluation_worst_encounters))
def test_a_model_returning_tool_stays_typed_for_its_output_schema(tool: PydanticTool):
    annotation = typing.get_type_hints(tool)["return"]

    assert isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel)


def test_the_capped_query_contract_keeps_its_inline_fields():
    annotation = typing.get_type_hints(evaluation_query)["return"]

    assert typing.get_origin(annotation) is dict
