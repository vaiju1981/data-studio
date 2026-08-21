"""Refusing a query whose join would multiply what it totals.

The guard between a model-written join and a wrong number. It runs before the
query does, so an explosion is priced rather than paid for, and it refuses only
what it can prove: an aggregate it cannot trace to one side is treated as unsafe
rather than waved through.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from smart_data_studio.dataset import Dataset
from smart_data_studio.facts import Verified, verify, verify_key
from smart_data_studio.proposals import JoinCandidate, Ref

# DuckDB names some aggregates that sqlglot parses as ordinary functions, so a
# class test alone lets total() past and skips the guard entirely.
ANONYMOUS_AGGREGATES = frozenset(
    {"total", "list", "histogram", "arg_max", "arg_min", "product", "geomean", "entropy"}
)


def aggregates_in(tree: exp.Expression) -> list[exp.Expression]:
    """Every aggregate, by class where sqlglot knows one and by name where not."""

    def is_aggregate(node: exp.Expression) -> bool:
        if isinstance(node, exp.AggFunc):
            return True
        return isinstance(node, exp.Anonymous) and str(node.this).lower() in ANONYMOUS_AGGREGATES

    return [node for node in tree.walk() if is_aggregate(node)]


AGGREGATES = (exp.Sum, exp.Avg, exp.Count, exp.Min, exp.Max)


@dataclass(frozen=True)
class Source:
    """What a FROM or JOIN item resolves to.

    `unique_on` is set for a derived relation that visibly reduces its own grain —
    a subquery with DISTINCT or GROUP BY. Base-table facts must not be used for one
    of those: `a JOIN (SELECT DISTINCT k FROM b)` is safe even though `b.k` repeats.
    """

    table: str | None  # None for a derived relation
    unique_on: frozenset[str] | None = None


def _grain_of(select: exp.Expression) -> frozenset[str] | None:
    """The columns a SELECT reduces itself to one row per, if it visibly does."""
    if not isinstance(select, exp.Select):
        return None
    if select.args.get("distinct"):
        return frozenset(item.alias_or_name.lower() for item in select.expressions)
    group = select.args.get("group")
    if group:
        return frozenset(item.alias_or_name.lower() for item in group.expressions)
    return None


def sources_in(tree: exp.Expression, tables: set[str]) -> dict[str, Source]:
    """Alias, or name where there is no alias, mapped to what it stands for.

    A CTE is a derived relation like any subquery. Treated as a bare name, a CTE
    over table t joined back to t looks like t joined to itself.
    """
    found: dict[str, Source] = {}
    for cte in tree.find_all(exp.CTE):
        found[cte.alias_or_name.lower()] = Source(table=None, unique_on=_grain_of(cte.this))
    for node in list(tree.find_all(exp.Table)) + list(tree.find_all(exp.Subquery)):
        alias = (node.alias or getattr(node, "name", "") or "").lower()
        if isinstance(node, exp.Table):
            name = node.name.lower()
            if name in tables and (alias or name) not in found:
                found[alias or name] = Source(table=name)
            continue
        inner = node.this
        if isinstance(inner, exp.Select) and alias:
            found[alias] = Source(table=None, unique_on=_grain_of(inner))
    return found


def _is_distinct(node: exp.Expression) -> bool:
    """Whether this aggregate reads distinct values, which repetition cannot alter."""
    if node.args.get("distinct"):
        return True
    return isinstance(node.this, exp.Distinct)


def _column_alias(column: exp.Column, sources: dict[str, Source], dataset: Dataset) -> str | None:
    """Which relation in this query a column reads from, by alias.

    Alias rather than table: a self-join names one table twice, and asking which
    *table* a column came from cannot tell the two sides apart.
    """
    qualifier = (column.table or "").lower()
    if qualifier:
        return qualifier if qualifier in sources else None
    owners = [
        alias
        for alias, source in sources.items()
        if source.table
        and any(name.lower() == column.name.lower() for name, _ in dataset.schema(source.table))
    ]
    return owners[0] if len(owners) == 1 else None


def column_owner(column: exp.Column, sources: dict[str, Source], dataset: Dataset) -> str | None:
    """Which loaded table a column reads from, following its alias when it has one."""
    qualifier = (column.table or "").lower()
    if qualifier:
        source = sources.get(qualifier)
        return source.table if source else None
    # Unqualified: resolvable only when exactly one source could supply it.
    owners = {
        source.table
        for source in sources.values()
        if source.table
        and any(name.lower() == column.name.lower() for name, _ in dataset.schema(source.table))
    }
    return owners.pop() if len(owners) == 1 else None


def _join_refs(
    condition: exp.Expression, sources: dict[str, Source], dataset: Dataset
) -> tuple[dict[str, list[str]], bool]:
    """Columns each side contributes, and whether every predicate was an equality."""
    per_source: dict[str, list[str]] = {}
    stack, simple = [condition], True
    while stack:
        node = stack.pop()
        if isinstance(node, exp.And):
            stack.extend([node.left, node.right])
        elif isinstance(node, exp.Paren):
            stack.append(node.this)
        elif (
            isinstance(node, exp.EQ)
            and isinstance(node.left, exp.Column)
            and isinstance(node.right, exp.Column)
        ):
            for column in (node.left, node.right):
                key = (column.table or "").lower() or (column_owner(column, sources, dataset) or "")
                per_source.setdefault(key, []).append(column.name)
        else:
            simple = False
    return per_source, simple


def preflight(
    dataset: Dataset, sql: str, cache: dict | None = None
) -> tuple[str | None, str | None]:
    """(refusal, note): why this must not run, and what to say if it may.

    Not every aggregate over a repeated row is wrong. SUM always double counts.
    MIN and MAX cannot change however often a row appears. AVG changes its
    weighting, which is sometimes what was asked for, so it is noted rather than
    refused.

    Called before the query executes, so an explosion is refused rather than paid
    for.
    """
    # Gated on the dataset rather than the query shape: fan-out within one table is
    # the grain guard's job, and gating on shape let a CTE over a single table reach
    # a guard meant for cross-table fan-out.
    if len(dataset.tables) < 2:
        return None, None

    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception:
        return None, None  # the SQL guard reports malformed SQL; this is not its job

    joins = list(tree.find_all(exp.Join))
    if not joins:
        # Nothing is combined, so nothing can be multiplied.
        return None, None

    aggregates = aggregates_in(tree)
    if not aggregates:
        # Nothing is being totalled, so fan-out changes the row count but no figure.
        return None, None

    tables = {name.lower() for name in dataset.tables}
    sources = sources_in(tree, tables)
    unsupported = _unsupported(tree, joins, sources)
    if unsupported:
        return (
            f"This join cannot be checked for double counting ({unsupported}), and it "
            "aggregates, so it is not run. Express it as inner or left joins on "
            "AND-ed column equalities, or aggregate each side to one row per key first."
        ), None

    multiplying: dict[str, str] = {}
    for join in joins:
        note = _join_multiplication(dataset, join, sources, cache)
        if note is None:
            # Undetermined is not the same as safe.
            return (
                "This join's output grain cannot be established, and the query "
                "aggregates, so it is not run. Give each side a provable grain — "
                "join on the full key, or reduce a side with DISTINCT or GROUP BY "
                "over the join columns."
            ), None
        multiplying.update(note)

    dropped = _dropped_note(dataset, joins, sources, cache)
    if not multiplying:
        return None, dropped

    weighted: str | None = None
    for node in aggregates:
        if isinstance(node, (exp.Min, exp.Max)):
            continue  # unaffected by how often a row appears
        if _is_distinct(node):
            # Repeating rows cannot change a count or sum of distinct values.
            continue
        owners = [_column_alias(column, sources, dataset) for column in node.find_all(exp.Column)]
        hit = next((owner for owner in owners if owner in multiplying), None)
        # A column reaching us through a derived relation is renamed, not traced:
        # the alias resolves while what it selects is never examined.
        traced = [
            owner
            for owner in owners
            if owner and (source := sources.get(owner)) and source.table is not None
        ]
        if hit is None and traced and len(traced) == len(owners):
            continue  # every column traced to a base relation, none repeated
        if hit is None:
            # Nothing to trace: COUNT(*), SUM(1), or a column projected out of a
            # subquery. All read the joined output, which is what fan-out inflates.
            first = next(iter(multiplying.values()))
            return (
                f"{first} This aggregate reads the joined output rather than a column "
                f"that can be traced to one side, so it carries that multiplication. "
                f"Aggregate a named column, or fix the grain first."
            ), None
        if isinstance(node, exp.Avg):
            weighted = (
                f"{hit} is averaged over a join that repeats its rows, so each value "
                f"counts once per matching row. That is a weighted average — right if "
                f"the weighting was intended, and not the same as the average over "
                f"{hit} alone."
            )
            continue
        # Everything else is refused, including aggregates this cannot name.
        return multiplying[hit], None

    # Both can be true: nothing was refused, rows repeat *and* rows were dropped.
    return None, "  ".join(note for note in (weighted, dropped) if note) or None


def _unsupported(tree: exp.Expression, joins: list, sources: dict[str, Source]) -> str | None:
    """The shapes whose output grain cannot be proved."""
    # A window function is deliberately absent: it adds columns and never multiplies
    # rows, so it cannot cause fan-out. It merely cannot prove a grain, and the
    # grain logic never credits one.
    for join in joins:
        # RIGHT and FULL change which unmatched rows survive, not which rows
        # repeat, and repetition is the whole question here.
        if (join.args.get("kind") or "").upper() == "CROSS":
            return "cross join"
        if join.args.get("using"):
            continue
        condition = join.args.get("on")
        if condition is None:
            return "a join with no condition"
        problem = _predicate_problem(condition)
        if problem:
            return problem
    return None


def _predicate_problem(condition: exp.Expression) -> str | None:
    """Anything in an ON clause that is not an AND-ed equality of two columns.

    Checked here rather than left to the measurer, where an unmeasurable predicate
    falls through as "nothing found to multiply".
    """
    stack = [condition]
    while stack:
        node = stack.pop()
        if isinstance(node, exp.And):
            stack.extend([node.left, node.right])
        elif isinstance(node, exp.Paren):
            stack.append(node.this)
        elif isinstance(node, exp.EQ):
            if not (isinstance(node.left, exp.Column) and isinstance(node.right, exp.Column)):
                return "a join on an expression rather than two columns"
        else:
            return f"a {type(node).__name__.lower()} join predicate rather than an equality"
    return None


def _join_multiplication(
    dataset: Dataset, join: exp.Join, sources: dict[str, Source], cache: dict | None
) -> dict[str, str] | None:
    """Tables whose rows this join repeats, or None when that cannot be decided.

    None and an empty dict mean different things: nothing repeats, versus nothing
    could be measured. Collapsing them lets the second pass as the first.
    """
    condition = join.args.get("on")
    if condition is None:
        using = [item.name for item in join.args.get("using") or []]
        if not using:
            return None
        # USING names the same column on both sides.
        left, right = _using_sides(join, sources, using)
        if left is None or right is None:
            return None
        pairs = {left: using, right: using}
    else:
        pairs, simple = _join_refs(condition, sources, dataset)
        if not simple or len(pairs) != 2:
            return None

    resolved = {}
    for alias, columns in pairs.items():
        source = sources.get(alias)
        if source is None:
            return None
        resolved[alias] = (source, columns)

    (left_alias, (left_source, left_columns)), (right_alias, (right_source, right_columns)) = (
        resolved.items()
    )
    # A derived relation already reduced to one row per key cannot multiply the
    # other side, whatever its base table does. One that has not proved its grain
    # cannot be reasoned about either way.
    for source, columns in (
        (left_source, left_columns),
        (right_source, right_columns),
    ):
        if source.table is None and not _derived_is_unique(source, columns):
            return None
    if left_source.table is None or right_source.table is None:
        # A derived side proved unique cannot multiply the other — but the other
        # can multiply *it*: a derived row is repeated once per base row sharing
        # its key, so a measure computed in the subquery is summed once per match.
        if left_source.table is None:
            derived, base, base_columns = left_alias, right_source, right_columns
        else:
            derived, base, base_columns = right_alias, left_source, left_columns
        if base.table is None:
            return None  # two derived relations: not something to reason about yet
        try:
            facts = verify_key(dataset, Ref(base.table, tuple(base_columns)))
        except Exception:
            return None
        if facts.unique:
            return {}
        return {
            derived: (
                f"This join repeats rows of {derived}: {base.table} is not unique on "
                f"{', '.join(base_columns)} — {facts.distinct:,} values across "
                f"{facts.complete:,} rows — so each {derived} row is counted once per "
                f"match. Aggregate {base.table} to one row per key first, or take the "
                f"measure from {base.table} instead."
            )
        }

    candidate = JoinCandidate(
        Ref(left_source.table, tuple(left_columns)),
        Ref(right_source.table, tuple(right_columns)),
    )
    key = (candidate.left, candidate.right)
    if cache is not None and key in cache:
        measured = cache[key]
    else:
        try:
            measured = verify(dataset, candidate)
        except Exception:
            return None
        if cache is not None:
            cache[key] = measured

    found: dict[str, str] = {}
    for alias, side in ((left_alias, "left"), (right_alias, "right")):
        if measured.multiplies_side(side):
            found[alias] = _explain(measured, side, alias)
    return found


def _dropped_note(
    dataset: Dataset, joins: list, sources: dict[str, Source], cache: dict | None
) -> str | None:
    """Say when an inner join silently leaves rows out.

    Nothing multiplies, so nothing double counts — but a total over what matched is
    quietly short, and reads exactly as reasonable as a correct one.
    """
    for join in joins:
        if (join.args.get("side") or "").upper() in {"LEFT", "RIGHT", "FULL"}:
            continue
        key = _cached_key(join, sources, dataset)
        measured = cache.get(key) if cache and key else None
        if measured is None or not measured.partial:
            continue
        return (
            f"This inner join leaves rows out: {measured.left.unmatched:,} rows of "
            f"{measured.left.ref.table} and {measured.right.unmatched:,} of "
            f"{measured.right.ref.table} match nothing, so any total covers only what "
            f"matched. Use a LEFT join if the unmatched rows should still count."
        )
    return None


def _cached_key(join: exp.Join, sources: dict[str, Source], dataset: Dataset):
    """The cache key for this join, or None when it is not a simple base pair."""
    condition = join.args.get("on")
    if condition is None:
        using = [item.name for item in join.args.get("using") or []]
        left, right = _using_sides(join, sources, using) if using else (None, None)
        if not using or left is None or right is None:
            return None
        pairs = {left: using, right: using}
    else:
        pairs, simple = _join_refs(condition, sources, dataset)
        if not simple or len(pairs) != 2:
            return None
    refs = []
    for alias, columns in pairs.items():
        source = sources.get(alias)
        if source is None or source.table is None:
            return None
        refs.append(Ref(source.table, tuple(columns)))
    return (refs[0], refs[1])


def _using_sides(join: exp.Join, sources: dict[str, Source], using: list[str]):
    """The two aliases a USING clause relates, in query order."""
    names = [alias for alias in sources]
    joined = (join.this.alias or getattr(join.this, "name", "") or "").lower()
    others = [alias for alias in names if alias != joined]
    return (others[0] if others else None), (joined if joined in sources else None)


def _derived_is_unique(source: Source, columns: list[str]) -> bool:
    """Whether joining on these columns meets one row of this derived relation.

    The subset runs this way round: a subquery grouped by (a, b) is one row per
    pair, and joining on a alone still meets many of them.
    """
    return bool(source.unique_on) and source.unique_on <= {c.lower() for c in columns}


def _name(alias: str, ref: Ref) -> str:
    """The alias the query used, and the table behind it when they differ."""
    return alias if alias == ref.table else f"{alias} ({ref.table})"


def _explain(measured: Verified, side: str, alias: str) -> str:
    """What repeats, by how much, and the key that would stop it."""
    mine, other = (
        (measured.left, measured.right) if side == "left" else (measured.right, measured.left)
    )
    alias = _name(alias, mine.ref)
    return (
        f"This join repeats rows of {alias}: one of its rows matches up to "
        f"{mine.max_partners:,} rows of {other.ref.table}, producing "
        f"{measured.joined_rows:,} rows from {mine.rows:,}. Totalling a {alias} column "
        f"over that counts it once per match. {other.ref.table} is not unique on "
        f"{', '.join(other.ref.columns)} — add the rest of its key to the join, or "
        f"aggregate {other.ref.table} to one row per key first."
    )
