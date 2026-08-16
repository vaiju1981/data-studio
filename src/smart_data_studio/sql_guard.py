"""Structural validation for model-authored SQL."""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp


class UnsafeQuery(ValueError):
    """Raised when a query is not a single SELECT over loaded tables."""


def validate_select(sql: str, allowed_tables: set[str]) -> str:
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as error:
        raise UnsafeQuery(f"SQL could not be parsed: {error}") from error

    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise UnsafeQuery("Only one SELECT statement is allowed")

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
    return statement.sql(dialect="duckdb")
