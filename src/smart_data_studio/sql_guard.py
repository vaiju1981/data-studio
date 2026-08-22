"""Structural validation for model-authored SQL."""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

from smart_data_studio.config import MAX_QUERY_DEPTH, MAX_QUERY_TABLES


class UnsafeQuery(ValueError):
    """Raised when a query is not a single SELECT over loaded tables."""


# UNION, EXCEPT and INTERSECT parse to their own root node rather than a Select,
# and all three only read. INSERT ... SELECT is an Insert and stays refused.
READ_ONLY_ROOTS = (exp.Select, exp.SetOperation)


def validate_select(sql: str, allowed_tables: set[str], withheld: set[str] | None = None) -> str:
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as error:
        raise UnsafeQuery(f"SQL could not be parsed: {error}") from error

    # Split from the shape test below: they are different mistakes, and a model
    # told it wrote two statements looks for a semicolon it never typed.
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

    # Checked here rather than on the way out, because an alias renames a column
    # and no amount of looking at the result would recognise it afterwards.
    if withheld:
        named = sorted(
            {
                column.name
                for column in statement.find_all(exp.Column)
                if column.name.lower() in withheld
            }
        )
        if named:
            raise UnsafeQuery(
                f"Column(s) withheld as sensitive on this deployment: {', '.join(named)}. "
                "They are not in the schema and cannot be selected, filtered or grouped on."
            )
        _refuse_whole_row_values(statement)

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
            f"This query nests {depth} queries deep; the limit is {MAX_QUERY_DEPTH}. "
            "Flatten it or use a CTE."
        )
    return statement.sql(dialect="duckdb")


def _refuse_whole_row_values(statement: exp.Expression) -> None:
    """Refuse a table used as a value, which is every column at once under one name.

    DuckDB reads a bare table name in an expression as a STRUCT of the whole row,
    so `SELECT people FROM people` returns the withheld columns inside a column
    called `people` — naming nothing the guard above looks for, and carrying a
    name nothing on the way out recognises. `to_json(people)` and a cast to
    VARCHAR do the same and lose the struct type on the way, so this cannot be
    caught by inspecting the result either.

    Only reachable where the operator withheld something, and only for an
    unqualified name matching a table in this same query. A column genuinely
    sharing its table's name is refused with it; qualifying it says which was
    meant, which DuckDB needs in that case anyway.
    """
    rows = {table.name.lower() for table in statement.find_all(exp.Table) if table.name}
    rows |= {table.alias.lower() for table in statement.find_all(exp.Table) if table.alias}
    used = sorted(
        {
            column.name
            for column in statement.find_all(exp.Column)
            if not column.table and column.name.lower() in rows
        }
    )
    if used:
        raise UnsafeQuery(
            f"{', '.join(used)} names a table, not a column, and DuckDB reads that as the "
            "whole row — including the columns withheld as sensitive. Select the columns "
            "you want by name, or qualify it if you meant a column of the same name."
        )


def _depth(node: exp.Expression, level: int = 0) -> int:
    """How many queries deep the deepest branch goes.

    Only a query counts as a level. Counting every AST child instead measured
    expression shape: ten flat AND predicates parse as a leaning binary tree and
    were refused as thirteen levels of nesting, while three genuinely nested
    subqueries came to twelve and passed. The bound exists to refuse a generated
    query that nests without end, so it counts the thing that nests.
    """
    inner = level + isinstance(node, (exp.Select, exp.SetOperation))
    children = [child for child in node.args.values() if isinstance(child, exp.Expression)]
    nested = [item for value in node.args.values() if isinstance(value, list) for item in value]
    children += [item for item in nested if isinstance(item, exp.Expression)]
    return max((_depth(child, inner) for child in children), default=inner)
