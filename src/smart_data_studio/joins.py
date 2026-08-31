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
# class test alone lets total() past and skips the guard entirely. Twenty-three of
# DuckDB's eighty-eight aggregates land here, and the list was written by hand at
# eight — mean, fsum, favg and sum_no_overflow among the missing, every one of
# them a way to total a fanned-out join with no refusal and no note.
#
# Enumerated from duckdb_functions() rather than remembered; test_relating_tables
# re-derives it and fails when a DuckDB upgrade adds one.
ANONYMOUS_AGGREGATES = frozenset(
    {
        "arbitrary",
        "arg_max",
        "arg_max_null",
        "arg_max_nulls_last",
        "arg_min",
        "arg_min_null",
        "arg_min_nulls_last",
        "bitstring_agg",
        "count_star",
        "entropy",
        "favg",
        "fill",
        "fsum",
        "geomean",
        "histogram",
        "histogram_exact",
        "kahan_sum",
        "kurtosis_pop",
        "list",
        "mad",
        "mean",
        "product",
        "rank_dense",
        "reservoir_quantile",
        "sem",
        "sum_no_overflow",
        "sumkahan",
        "total",
    }
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
    # The one base table a derived relation reads, where it reads exactly one.
    # A DISTINCT or GROUP BY over a table already unique on the join key cannot
    # repeat that key, and saying so costs a key check rather than running the
    # subquery to find out.
    reads: str | None = None


def _grain_of(select: exp.Expression) -> frozenset[str] | None:
    """The columns a SELECT reduces itself to one row per, if it visibly does.

    The empty set is a real answer and not a missing one: a select that aggregates
    without grouping is exactly one row, which no join can fail to meet.
    """
    if not isinstance(select, exp.Select):
        return None
    if select.args.get("distinct"):
        return frozenset(item.alias_or_name.lower() for item in select.expressions)
    group = select.args.get("group")
    if group:
        return _group_names(select, group.expressions)
    # No GROUP BY, so an aggregate here collapses the whole relation to one row.
    return frozenset() if aggregates_in(select) else None


def _group_names(select: exp.Select, grouped: list[exp.Expression]) -> frozenset[str] | None:
    """The GROUP BY as column names, with ordinals resolved against the select list.

    `GROUP BY 1` and `GROUP BY customer_id` are the same query, and reading the
    first as `alias_or_name` gives the string "1" — a column no join is ever on, so
    the relation could never prove its grain and every join onto it was refused.
    Ordinals are how this SQL is usually written.
    """
    names = []
    for item in grouped:
        if isinstance(item, exp.Literal) and item.is_int:
            index = int(item.name) - 1
            if not 0 <= index < len(select.expressions):
                return None  # out of range: DuckDB will reject it, and we prove nothing
            names.append(select.expressions[index].alias_or_name.lower())
        else:
            names.append(item.alias_or_name.lower())
    return frozenset(names)


def sources_in(tree: exp.Expression, tables: set[str]) -> dict[str, Source]:
    """Alias, or name where there is no alias, mapped to what it stands for.

    A CTE is a derived relation like any subquery. Treated as a bare name, a CTE
    over table t joined back to t looks like t joined to itself.
    """
    found: dict[str, Source] = {}
    for cte in tree.find_all(exp.CTE):
        found[cte.alias_or_name.lower()] = Source(
            table=None, unique_on=_grain_of(cte.this), reads=_single_table(cte.this, tables)
        )
    for node in list(tree.find_all(exp.Table)) + list(tree.find_all(exp.Subquery)):
        alias = (node.alias or getattr(node, "name", "") or "").lower()
        if isinstance(node, exp.Table):
            name = node.name.lower()
            # `FROM firsts f` is the CTE under a second name. Registered only by its
            # definition name, the alias the join actually uses resolved to nothing
            # and the side came back unknown.
            if name in found and alias and alias != name:
                found[alias] = found[name]
                continue
            if name in tables and (alias or name) not in found:
                found[alias or name] = Source(table=name)
            continue
        inner = node.this
        if isinstance(inner, exp.Select) and alias:
            found[alias] = Source(
                table=None, unique_on=_grain_of(inner), reads=_single_table(inner, tables)
            )
    return found


def owning_select(node: exp.Expression) -> exp.Expression | None:
    """The SELECT this node belongs to, which is the scope a join can inflate."""
    while node is not None and not isinstance(node, exp.Select):
        node = node.parent
    return node


def _reduces(select: exp.Expression) -> bool:
    """Whether this SELECT gives its output a grain of its own.

    A GROUP BY, a DISTINCT or an aggregate of its own all collapse the rows it
    read, so a fan-out underneath cannot be totalled again above it. Anything else
    merely renames and projects, and the repetition passes straight through.
    """
    if select.args.get("group") or select.args.get("distinct"):
        return True
    return any(owning_select(node) is select for node in aggregates_in(select))


def scopes_affected(join: exp.Join) -> set[int]:
    """The selects a fan-out here can inflate.

    Its own, and every enclosing one that reads those rows without reducing them
    first. Stopping at the join's own select let `SELECT sum(v) FROM (SELECT a.fee
    AS v FROM s JOIN a ...)` through: the subquery only renames a column, so the
    multiplication is still there when the outer sum reads it.
    """
    select = owning_select(join)
    found: set[int] = set()
    while select is not None:
        found.add(id(select))
        if _reduces(select):
            break
        select = owning_select(select.parent) if select.parent is not None else None
    return found


def scope_aliases(join: exp.Join) -> list[str]:
    """The aliases in the same FROM as this join.

    Its two sides are here and nothing else is. Searching every source in the tree
    instead let a table named only inside a CTE body be picked as a side — which
    refused correct queries when that table repeated the key, and allowed
    double-counting ones when it did not.
    """
    return select_aliases(join.parent)


def select_aliases(select: exp.Expression | None) -> list[str]:
    """The relations this SELECT reads directly, by the name the query calls them."""
    if not isinstance(select, exp.Select):
        return []
    # "from_" is what this sqlglot spells it; "from" is kept for an older one, and
    # reading only the missing name made every scope look like joins alone.
    source = select.args.get("from_") or select.args.get("from")
    items = [item.this for item in [source] if item is not None]
    items += [item.this for item in select.args.get("joins") or []]
    return [(item.alias or getattr(item, "name", "") or "").lower() for item in items]


def _single_table(select: exp.Expression, tables: set[str]) -> str | None:
    """The one loaded table this SELECT reads, where it reads exactly one."""
    if not isinstance(select, exp.Select):
        return None
    named = [node for node in select.find_all(exp.Table) if owning_select(node) is select]
    if len(named) != 1:
        return None
    name = named[0].name.lower()
    return name if name in tables else None


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

    # A name this workspace does not have cannot be reasoned about, and every
    # shape below then reads as an unprovable grain. A typo'd table came back as
    # "join on the full key, or reduce a side with DISTINCT" — advice about a join
    # that is not the problem, and a steer the model spends its remaining rounds
    # following. The SQL guard names the table; leave the message to it.
    known = {name.lower() for name in dataset.tables}
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    named = {table.name.lower() for table in tree.find_all(exp.Table) if table.name}
    if named - ctes - known:
        return None, None

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

    # Kept per select rather than in one pile: an aggregate inside a CTE body is
    # computed before the outer join runs, so no join out there can inflate it.
    # Pooled, the sum() that builds a totals CTE read as though it totalled the
    # joined output, and the ordinary share-of-total query was refused.
    by_scope: dict[int, dict[str, str]] = {}
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
        if note:
            for scope in scopes_affected(join):
                by_scope.setdefault(scope, {}).update(note)

    dropped = _dropped_note(dataset, joins, sources, cache)
    if not by_scope:
        return None, dropped

    weighted: str | None = None
    for node in aggregates:
        multiplying = by_scope.get(id(owning_select(node)), {})
        if not multiplying:
            continue
        if isinstance(node, (exp.Min, exp.Max)):
            continue  # unaffected by how often a row appears
        if _is_distinct(node):
            # Repeating rows cannot change a count or sum of distinct values.
            continue
        owners = [_column_alias(column, sources, dataset) for column in node.find_all(exp.Column)]
        hit = next((owner for owner in owners if owner in multiplying), None)
        # A base relation is traceable as it always was. A derived one is traceable
        # only while every multiplication in this scope belongs to this scope: once
        # a fan-out has leaked up from inside a subquery, the alias resolves and
        # what it selects has still never been examined, which is how
        # `sum(w.v) FROM (SELECT a.fee AS v FROM s JOIN a ...) w` launders one.
        own = set(select_aliases(owning_select(node)))
        leaked = any(alias not in own for alias in multiplying)
        traced = [
            owner
            for owner in owners
            if owner
            and (source := sources.get(owner)) is not None
            and (source.table is not None or (not leaked and owner in own))
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
            if _one_row_side(join, sources):
                continue
            return "cross join"
        if join.args.get("using"):
            continue
        condition = join.args.get("on")
        if condition is None:
            # `FROM per, tot` where tot is one row is how a share of a total is
            # written. It carries no condition because it needs none.
            if _one_row_side(join, sources):
                continue
            return "a join with no condition"
        problem = _predicate_problem(condition)
        if problem:
            return problem
    return None


def _one_row_side(join: exp.Join, sources: dict[str, Source]) -> str | None:
    """The alias in this join's scope that is provably a single row, if any.

    A relation of one row cannot multiply the other side — but the other side
    multiplies *it*, once per row, which is why the alias is returned rather than
    a yes.
    """
    for alias in scope_aliases(join):
        source = sources.get(alias)
        if source is not None and source.table is None and source.unique_on == frozenset():
            return alias
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
            single = _one_row_side(join, sources)
            if single is None:
                return None
            # Every row of the other side meets that one row, so it appears once
            # per row of the output. Totalling a column of it counts it that often.
            return {
                single: (
                    f"This join repeats the single row of {single} once per row it is "
                    f"joined to, so a total taken over {single} counts it that many "
                    f"times. Read its columns as values rather than aggregating them."
                )
            }
        # USING names the same column on both sides.
        left, right = _using_sides(join, sources, using, dataset)
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
    if left_source.table is None or right_source.table is None:
        # One side at least is derived, so there is no pair of base tables to
        # measure against each other. Each side is judged on its own instead: it
        # multiplies the other exactly when its own rows repeat the join key.
        found: dict[str, str] = {}
        for alias, source, columns, other in (
            (left_alias, left_source, left_columns, right_alias),
            (right_alias, right_source, right_columns, left_alias),
        ):
            repeats, evidence = _repeats_key(dataset, source, columns)
            if repeats is None:
                return None  # undetermined is not the same as safe
            if repeats:
                found[other] = (
                    f"This join repeats rows of {other}: {alias} is not unique on "
                    f"{', '.join(columns)} ({evidence}), so each {other} row is counted "
                    f"once per match. Aggregate {alias} to one row per key first, or "
                    f"take the measure from {alias} instead."
                )
        return found

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


def _repeats_key(dataset: Dataset, source: Source, columns: list[str]) -> tuple[bool | None, str]:
    """Whether this relation holds more than one row per these columns.

    True, False, or None for cannot be decided — and the evidence, because a
    refusal that does not say what it measured is a steer the model cannot act on.

    A derived relation grouped by more than the join columns is treated as
    repeating them. It may not in this data, but proving that would mean running
    the subquery, and the aggregate stage still lets MIN, MAX and the distinct
    counts through — which is what a retention query is made of.
    """
    if source.table is None:
        if _derived_is_unique(source, columns):
            return False, ""
        if source.unique_on is None:
            return None, ""
        # Grouped by more than the join key, which usually means it repeats it —
        # but not when it reduces a table that is already unique on that key. A
        # dimension read as SELECT DISTINCT id, label is the everyday case, and
        # refusing it costs the ordinary join onto a dimension.
        if source.reads is not None:
            base, _ = _repeats_key(dataset, Source(table=source.reads), columns)
            if base is False:
                return False, ""
        return True, f"it is one row per {', '.join(sorted(source.unique_on))}"
    try:
        facts = verify_key(dataset, Ref(source.table, tuple(columns)))
    except Exception:
        return None, ""
    if facts.unique:
        return False, ""
    return True, f"{facts.distinct:,} values across {facts.complete:,} rows"


def _dropped_note(
    dataset: Dataset, joins: list, sources: dict[str, Source], cache: dict | None
) -> str | None:
    """Say when a join silently leaves rows out.

    Nothing multiplies, so nothing double counts — but a total over what matched is
    quietly short, and reads exactly as reasonable as a correct one.

    LEFT and FULL keep the unmatched rows, so there is nothing to say. RIGHT drops
    every unmatched row on the left, which is the same silence as an inner join
    wearing a different word, and was exempted here with them.
    """
    for join in joins:
        if (join.args.get("side") or "").upper() in {"LEFT", "FULL"}:
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
        left, right = _using_sides(join, sources, using, dataset) if using else (None, None)
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


def _using_sides(join: exp.Join, sources: dict[str, Source], using: list[str], dataset: Dataset):
    """The two aliases a USING clause relates.

    The far side is whichever earlier source actually carries the column, not
    whichever came first. In `a JOIN b USING (k) JOIN c USING (m)` the second
    clause relates b to c whenever m lives on b, and taking the first alias
    measured a against c — a pair with no column in common, so the grain came back
    unknown and the whole query was refused. Fail-closed, and still the wrong
    answer to the question that was asked.
    """
    joined = (join.this.alias or getattr(join.this, "name", "") or "").lower()
    scope = [alias for alias in scope_aliases(join) if alias in sources]
    others = [alias for alias in scope or sources if alias != joined]
    wanted = {column.lower() for column in using}

    def carries(alias: str) -> bool:
        table = sources[alias].table
        if table is None:
            return False
        try:
            return wanted <= {name.lower() for name, _ in dataset.schema(table)}
        except Exception:
            return False

    # Falling back to the first keeps the old behaviour where the columns cannot be
    # read — a derived relation whose shape was never established.
    named = [alias for alias in others if carries(alias)]
    far = (named or others or [None])[0]
    return far, (joined if joined in sources else None)


def _derived_is_unique(source: Source, columns: list[str]) -> bool:
    """Whether joining on these columns meets one row of this derived relation.

    The subset runs this way round: a subquery grouped by (a, b) is one row per
    pair, and joining on a alone still meets many of them.
    """
    return source.unique_on is not None and source.unique_on <= {c.lower() for c in columns}


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
