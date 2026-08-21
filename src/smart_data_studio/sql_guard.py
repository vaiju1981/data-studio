"""Structural validation for model-authored SQL."""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

from smart_data_studio.config import MAX_QUERY_DEPTH, MAX_QUERY_TABLES


class UnsafeQuery(ValueError):
    """Raised when a query is not a single SELECT over loaded tables."""


# UNION, EXCEPT and INTERSECT parse to their own root node rather than a Select,
# and all three only read. Requiring a Select root refused `SELECT id FROM a
# EXCEPT SELECT id FROM b` — stacking period files and set-differencing two files
# are the two most natural multi-CSV questions — while the same query wrapped in a
# subquery passed, so the rule was arbitrary as well as wrong. INSERT ... SELECT
# is an Insert and stays refused.
READ_ONLY_ROOTS = (exp.Select, exp.SetOperation)


def validate_select(sql: str, allowed_tables: set[str]) -> str:
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as error:
        raise UnsafeQuery(f"SQL could not be parsed: {error}") from error

    # Split from the shape test below because the two are different mistakes, and
    # telling a model it wrote two statements when it wrote one set operation sent
    # it looking for a semicolon it had never typed.
    if len(statements) != 1:
        raise UnsafeQuery("Only one statement is allowed")
    if not isinstance(statements[0], READ_ONLY_ROOTS):
        raise UnsafeQuery(
            "Only a SELECT is allowed, optionally combined with UNION, EXCEPT or INTERSECT"
        )

    statement = statements[0]
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    referenced = {
        table.name.lower()
        for table in statement.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_names
    }
    unknown = referenced - {table.lower() for table in allowed_tables}
    if unknown:
        raise UnsafeQuery(f"Unknown table(s): {', '.join(sorted(unknown))}")

    # The timeout contains a runaway after the fact; these refuse the obvious ones
    # before any work starts. Both bounds sit well above real analytics.
    sources = len(list(statement.find_all(exp.Table)))
    if sources > MAX_QUERY_TABLES:
        raise UnsafeQuery(
            f"This query joins {sources} tables; the limit is {MAX_QUERY_TABLES}. "
            "Aggregate in steps instead."
        )
    depth = _depth(statement)
    if depth > MAX_QUERY_DEPTH:
        raise UnsafeQuery(
            f"This query nests {depth} levels deep; the limit is {MAX_QUERY_DEPTH}. "
            "Flatten it or use a CTE."
        )
    return statement.sql(dialect="duckdb")


def _depth(node: exp.Expression, level: int = 0) -> int:
    children = [child for child in node.args.values() if isinstance(child, exp.Expression)]
    nested = [item for value in node.args.values() if isinstance(value, list) for item in value]
    children += [item for item in nested if isinstance(item, exp.Expression)]
    return max((_depth(child, level + 1) for child in children), default=level)
