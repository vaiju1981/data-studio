"""What a query written for the wrong database is told.

Found in a live bank run: the model wrote MySQL's `DATE_SUB(MAX(day), INTERVAL
'3' MONTH)` and DuckDB replied with the signature of *its* `date_sub`, which takes
three arguments and computes a difference. The reply described a function the
model had not asked for, so the next attempt was wrong in a new way and the
question ran out of rounds.

An error that steers wrongly costs every round that follows it, which is why this
is checked rather than left to the model to work out.
"""

from __future__ import annotations

import json

import pytest

from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.tools import AnalysisTools, dialect_hint


@pytest.fixture
def tools():
    dataset = Dataset.load(
        [CsvSource.from_upload("t.csv", b"day,amount\n2026-01-01,10\n2026-02-01,20\n")]
    )
    try:
        yield AnalysisTools(dataset)
    finally:
        dataset.close()


def test_the_query_that_prompted_this_is_answered_with_its_duckdb_spelling(tools) -> None:
    """The exact shape from the bank run."""
    payload = json.loads(
        tools.run_sql(
            "SELECT * FROM t WHERE day >= DATE_SUB((SELECT max(day) FROM t), INTERVAL '3' MONTH)"
        )
    )
    assert "error" in payload
    assert "INTERVAL '3' MONTH" in payload["error"]
    assert "- INTERVAL" in payload["error"], payload["error"]


@pytest.mark.parametrize(
    ("label", "sql", "expected"),
    [
        ("mysql date format", "SELECT DATE_FORMAT(day, '%Y-%m') FROM t", "STRFTIME"),
        ("sql server top", "SELECT TOP 1 day FROM t", "LIMIT"),
        ("snowflake dateadd", "SELECT DATEADD(month, -3, day) FROM t", "INTERVAL"),
    ],
)
def test_other_dialects_are_translated_too(tools, label, sql, expected) -> None:
    """One dialect would have fixed one bank question. The model reaches for
    whichever it was trained on."""
    payload = json.loads(tools.run_sql(sql))
    if "error" not in payload:
        pytest.skip(f"{label}: DuckDB accepts this as written")
    assert expected in payload["error"], payload["error"]


def test_an_ordinary_mistake_gets_no_rewrite(tools) -> None:
    """A mistyped column is not a dialect problem, and a rewrite offered for one
    is exactly the wrong steer this was written to stop."""
    payload = json.loads(tools.run_sql("SELECT nosuchcolumn FROM t"))
    assert "error" in payload
    assert "another dialect" not in payload["error"], payload["error"]


def test_a_query_that_means_the_same_everywhere_gets_no_rewrite() -> None:
    """The gate: the hint is only offered when reading the SQL as another dialect
    actually changes it."""
    assert dialect_hint("SELECT sum(amount) FROM t", "Binder Error: no function matches") is None


def test_a_failure_that_is_not_about_spelling_is_left_alone() -> None:
    """Out of memory, a timeout, a permission refusal — none of them want a
    rewrite, however the SQL reads."""
    assert dialect_hint("SELECT TOP 1 day FROM t", "Out of Memory Error: exceeded limit") is None
