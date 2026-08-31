"""Structural validation for model-authored SQL."""

from __future__ import annotations

from itertools import count

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

    # The columns are not in the workspace at all — they were dropped as the table
    # was built — so this is not what keeps them safe. It is what turns DuckDB's
    # "column not found" into a reason, which otherwise reads like a typo and gets
    # the same column asked for three more times in three more spellings.
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
                "They were not loaded and no query can reach them — do not ask again."
            )

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


def redact_literals(sql: str) -> str:
    """The query's shape, with everything a cell value can ride in on removed.

    The SQL is logged because it is the evidence behind an answer — but a
    generated filter carries real cell values, and `WHERE email = 'ada@x.com'`
    logged verbatim is a cell value in a stream that leaves the host, which is
    exactly what this app promises not to do. The shape is what diagnoses a slow
    or wrong query; the value is on the user's own screen beside the answer.

    Three ways in, not one. Masking the literals alone left the other two:

    - A comment is free text the model wrote, `/* ada@x.com */` included, and it
      survives serialization untouched. It carries no shape, so it simply goes.
    - A select alias is very often a value: `SUM(CASE WHEN region = 'North' ...)
      AS North` is the ordinary way to write a pivot, and the label is the cell.
      Numbered instead, which costs a name and keeps every table, column,
      function and operator that makes the query diagnosable.

    A table alias is left alone: it is the model's own short name, the column
    references depend on it, and it never comes from the data.
    """
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.ParseError:
        return "unparseable"

    labels = count()

    def mask_values(node: exp.Expression) -> exp.Expression:
        node.comments = None
        if isinstance(node, exp.Literal):
            return exp.Literal.string("?") if node.is_string else exp.Literal.number(0)
        return node

    def number_aliases(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Alias):
            return exp.alias_(node.this, f"c{next(labels)}", quoted=False)
        return node

    # Two passes, because transform walks parents before children and a replaced
    # node is not descended into. Renaming the alias in the same pass therefore
    # returned its expression *unvisited*, and every literal inside it survived —
    # in exactly the pivot the docstring above calls the reason for masking
    # aliases at all: `SUM(CASE WHEN region = 'North' ...) AS North` came out
    # with the label gone and the value still in it.
    return (
        tree.transform(mask_values).transform(number_aliases).sql(dialect="duckdb", comments=False)
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
