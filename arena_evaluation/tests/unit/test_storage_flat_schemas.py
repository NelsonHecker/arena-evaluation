"""R8: every registered star-schema and telemetry schema is flat.

``Array`` is included with ``List``/``Struct``/``Object`` because the 1NF rule bans it too.
The evidence index is not a lakehouse dataset; its ``columns`` list is a report artifact.
"""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.analytics.catalog import TELEMETRY_TABLE_SCHEMAS
from arena_evaluation.storage.declarations import FACT_SCHEMAS
from arena_evaluation.storage.star_schema import DIMENSION_SCHEMAS

_NESTED_DTYPES = (pl.List, pl.Struct, pl.Object, pl.Array)


def _registered_schemas() -> list[tuple[str, pl.Schema]]:
    schemas = [(f"dim:{name}", schema) for name, schema in DIMENSION_SCHEMAS.items()]
    schemas += [(f"fact:{name}", schema) for name, schema in FACT_SCHEMAS.items()]
    schemas += [(f"telemetry:{name}", pl.Schema(schema)) for name, schema in TELEMETRY_TABLE_SCHEMAS.items()]
    return schemas


@pytest.mark.parametrize(("name", "schema"), _registered_schemas())
def test_a_registered_schema_has_no_nested_dtype(name: str, schema: pl.Schema) -> None:
    nested = {column: dtype for column, dtype in schema.items() if isinstance(dtype, _NESTED_DTYPES)}
    assert not nested, f"{name} has nested columns {nested}"
