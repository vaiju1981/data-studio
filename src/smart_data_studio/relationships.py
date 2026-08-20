"""Candidate table keys and joins across loaded tables.

The model proposes these from names, profiles and samples; it is good at that and
better than any name-and-overlap heuristic. A proposal is a hypothesis, so nothing
here trusts one: every reference is resolved against the loaded schema before a
verification query is built, and a proposal that names something withheld, unknown
or malformed is rejected rather than repaired.

Metadata is keyed by table *and* column, never by column alone — two files
commonly hold the same column name meaning different things.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import sqlglot
from sqlglot import exp

from smart_data_studio.config import (
    MAX_JOIN_CANDIDATES,
    MAX_KEY_CANDIDATES,
    MAX_KEY_COLUMNS,
    MAX_KEY_SEARCH_COLUMNS,
)
from smart_data_studio.dataset import Dataset, is_sensitive, quote_identifier


@dataclass(frozen=True)
class Ref:
    """One side of a candidate: a table and an ordered tuple of its columns."""

    table: str
    columns: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.table}({', '.join(self.columns)})"


# reason is annotation, not identity: the same join proposed twice with different
# wording is one candidate, and counting it twice spends the budget on prose.
@dataclass(frozen=True)
class KeyCandidate:
    ref: Ref
    reason: str = field(default="", compare=False)


@dataclass(frozen=True)
class JoinCandidate:
    left: Ref
    right: Ref
    reason: str = field(default="", compare=False)

    def __str__(self) -> str:
        return f"{self.left} = {self.right}"


@dataclass
class Proposals:
    keys: list[KeyCandidate] = field(default_factory=list)
    joins: list[JoinCandidate] = field(default_factory=list)
    # Why each rejected proposal was refused, for the log and the panel. A silent
    # drop would look identical to the model never having proposed anything.
    rejected: list[str] = field(default_factory=list)


class Invalid(ValueError):
    """A proposal that cannot be resolved against the loaded data."""


def _resolve(dataset: Dataset, table: str, columns: list[str] | tuple[str, ...]) -> Ref:
    """Match a proposal against the schema, returning the schema's own spelling.

    Identifiers are compared without case so a proposal saying `playerid` resolves,
    but what is stored is what the file actually calls the column — anything else
    produces SQL that does not run and diagnostics nobody can grep for.
    """
    known = {name.lower(): name for name in dataset.tables}
    actual_table = known.get(str(table).strip().lower())
    if actual_table is None:
        raise Invalid(f"unknown table {table!r}")

    schema = {name.lower(): name for name, _ in dataset.schema(actual_table)}
    if not columns:
        raise Invalid(f"{actual_table}: no columns given")
    if len(columns) > MAX_KEY_COLUMNS:
        raise Invalid(f"{actual_table}: {len(columns)} columns, more than {MAX_KEY_COLUMNS}")

    resolved: list[str] = []
    for column in columns:
        name = schema.get(str(column).strip().lower())
        if name is None:
            raise Invalid(f"{actual_table} has no column {column!r}")
        if is_sensitive(name):
            # Naming it in a rejection message would defeat withholding it.
            raise Invalid(f"{actual_table}: a withheld column was proposed")
        if name in resolved:
            raise Invalid(f"{actual_table}: {name} repeated")
        resolved.append(name)
    return Ref(actual_table, tuple(resolved))


def validate(dataset: Dataset, raw: list[dict]) -> Proposals:
    """Turn whatever the model proposed into candidates that resolve, or reasons.

    Bounded on purpose: a model that proposes forty joins between two tables would
    otherwise cost forty verification queries for no more insight than four.
    """
    found = Proposals()
    per_table: dict[str, int] = {}
    per_pair: dict[tuple[str, str], int] = {}

    for item in raw:
        try:
            kind = str(item.get("kind", "")).strip().lower()
            if kind == "key":
                ref = _resolve(dataset, item.get("table", ""), item.get("columns") or [])
                if per_table.get(ref.table, 0) >= MAX_KEY_CANDIDATES:
                    raise Invalid(f"{ref.table}: more than {MAX_KEY_CANDIDATES} key candidates")
                candidate = KeyCandidate(ref, str(item.get("reason", "")).strip())
                if candidate not in found.keys:
                    per_table[ref.table] = per_table.get(ref.table, 0) + 1
                    found.keys.append(candidate)
            elif kind == "join":
                left_raw, right_raw = item.get("left") or {}, item.get("right") or {}
                left = _resolve(dataset, left_raw.get("table", ""), left_raw.get("columns") or [])
                right = _resolve(
                    dataset, right_raw.get("table", ""), right_raw.get("columns") or []
                )
                if left.table == right.table:
                    raise Invalid(f"{left.table} joined to itself")
                if len(left.columns) != len(right.columns):
                    raise Invalid(f"{left} and {right} have different numbers of columns")
                pair = tuple(sorted((left.table, right.table)))
                if per_pair.get(pair, 0) >= MAX_JOIN_CANDIDATES:
                    raise Invalid(
                        f"{pair[0]}/{pair[1]}: more than {MAX_JOIN_CANDIDATES} join candidates"
                    )
                candidate = JoinCandidate(left, right, str(item.get("reason", "")).strip())
                if candidate not in found.joins:
                    per_pair[pair] = per_pair.get(pair, 0) + 1
                    found.joins.append(candidate)
            else:
                raise Invalid(f"unknown proposal kind {item.get('kind')!r}")
        except Invalid as error:
            found.rejected.append(str(error))
        except (AttributeError, TypeError) as error:
            found.rejected.append(f"malformed proposal: {type(error).__name__}: {error}")
    return found


@dataclass(frozen=True)
class KeyFacts:
    """What a candidate key does on the rows actually loaded."""

    ref: Ref
    rows: int
    complete: int  # rows with no NULL in any key column
    distinct: int
    # Unique, but made of measures rather than identifiers — set only on the
    # fallback path, which is the one place that knows the difference. A unique
    # text column like sku or email is a real key and is not marked.
    coincidental: bool = False

    @property
    def unique(self) -> bool:
        return bool(self.complete) and self.distinct == self.complete

    @property
    def has_nulls(self) -> bool:
        return self.complete < self.rows

    def describe(self) -> str:
        columns = ", ".join(self.ref.columns)
        if not self.unique:
            return (
                f"({columns}) comes closest to identifying a row, at {self.distinct:,} "
                f"values across {self.complete:,} rows — near enough to join on, and "
                f"the nearest thing this table has to a key"
            )
        nulls = f", though {self.rows - self.complete:,} rows have none" if self.has_nulls else ""
        if self.coincidental:
            # Calling this "one row per (jackpots)" reads as a key and invites a
            # join on it, when all it says is that these values happen not to
            # repeat in the rows loaded today.
            return (
                f"no column identifies a row; ({columns}) happens to be unique here, "
                f"but it measures rather than identifies, so it is a property of this "
                f"data rather than a key to join on"
            )
        return f"one row per ({columns}){nulls}"


def _measure_keys(
    dataset: Dataset, table: str, candidates: list[tuple[str, ...]]
) -> list[KeyFacts]:
    """Measure several candidate keys of one table in a single pass."""
    if not candidates:
        return []
    rows = dataset.row_count(table)
    projections = []
    for index, columns in enumerate(candidates):
        key, non_null = _key_sql(Ref(table, columns))
        projections += [
            f"count(*) FILTER (WHERE {non_null}) AS complete_{index}",
            f"count(DISTINCT CASE WHEN {non_null} THEN {key} END) AS distinct_{index}",
        ]
    with dataset._deadline():
        row = dataset.connection.execute(
            f"SELECT {', '.join(projections)} FROM {quote_identifier(table)}"
        ).fetchone()
    return [
        KeyFacts(
            ref=Ref(table, columns),
            rows=rows,
            complete=int(row[index * 2]),
            distinct=int(row[index * 2 + 1]),
        )
        for index, columns in enumerate(candidates)
    ]


def _looks_like_identifier(name: str) -> bool:
    """id, playerId, movie_id — an integer with a name like this is not a measure."""
    lowered = name.lower()
    return lowered == "id" or lowered.endswith(("id", "_id", "key", "code", "number", "no"))


def verify_key(dataset: Dataset, ref: Ref) -> KeyFacts:
    """Measure whether these columns identify a row, counting nulls apart."""
    key, non_null = _key_sql(ref)
    quoted = quote_identifier(ref.table)
    with dataset._deadline():
        complete, distinct = dataset.connection.execute(
            f"SELECT count(*), count(DISTINCT {key}) FROM {quoted} WHERE {non_null}"
        ).fetchone()
    return KeyFacts(
        ref=ref, rows=dataset.row_count(ref.table), complete=int(complete), distinct=int(distinct)
    )


def measure_columns(dataset: Dataset, table: str) -> set[str]:
    """Columns that are quantities rather than identifiers.

    Money and percentages carry plenty of distinct values and identify nothing.
    Ranked on distinct count alone an asset table offered (grossWin, ticketOut) as
    its key, and listed grossWin ahead of assetId as a join column — the same
    mistake twice, so the test lives in one place.
    """
    found = set()
    for name, kind in dataset.schema(table):
        upper = kind.upper()
        if any(part in upper for part in ("DOUBLE", "FLOAT", "DECIMAL", "REAL")):
            found.add(name)  # a float is never an identifier
        elif any(part in upper for part in ("INT", "HUGEINT")) and not _looks_like_identifier(name):
            # An integer can be either: movieId identifies, jackpots measures.
            found.add(name)
    return found


def discover_keys(
    dataset: Dataset, table: str, distinct_by_column: dict[str, float]
) -> list[KeyFacts]:
    """The narrowest column sets that identify a row, singles before pairs.

    Bounded rather than exhaustive: single columns first, and pairs only among the
    few with the most distinct values, since a key's parts are necessarily among
    the more various columns. Minimality falls out of the order — once a single
    column is a key, no pair containing it is offered, because a unique pair whose
    subset is already unique is not a composite key.

    This is what lets the profile say "one row per (assetId, day)" before anything
    joins on assetId alone, which is the difference between preventing the 439x
    error and correcting it afterwards.
    """
    rows = dataset.row_count(table)
    if rows < 2:
        return []
    # Floats only. An integer is a perfectly good key component — a day stored as
    # 20240101, an hour, a numbered stand — and excluding every integer measure
    # from the search hid composite keys made of them.
    measures = {
        name
        for name, kind in dataset.schema(table)
        if any(part in kind.upper() for part in ("DOUBLE", "FLOAT", "DECIMAL", "REAL"))
    }
    # Identifier-named columns first, then by distinct count. Ranking on count
    # alone offered jackpots ahead of assetId, because a measure carries plenty of
    # values and identifies nothing — but excluding every integer instead hid
    # composite keys made of them, so the name decides the order rather than
    # membership.
    ranked = sorted(
        ((name, count) for name, count in distinct_by_column.items() if name not in measures),
        key=lambda item: (not _looks_like_identifier(item[0]), -item[1]),
    )
    if not ranked:
        return []
    names = [name for name, _ in ranked[:MAX_KEY_SEARCH_COLUMNS]]
    # One scan for all the singles, then one for all the pairs. Measuring each
    # candidate on its own meant up to twenty-one passes over the same table.
    singles = _measure_keys(dataset, table, [(name,) for name in names])
    exact = [facts for facts in singles if facts.unique]
    # An identifier that is unique is a key. A measure that is unique is a
    # coincidence of the data — jackpots was offered as the key of an asset table
    # because its values happened not to repeat — so a composite of identifiers is
    # looked for first, and the coincidence kept only as a fallback.
    named = [facts for facts in exact if _looks_like_identifier(facts.ref.columns[0])]
    if named:
        return named[:MAX_KEY_CANDIDATES]

    # Minimality, as the plan requires: a pair containing an already-unique column
    # is unique for that reason alone and says nothing. Without this the search
    # offered (assetId, jackpots) — unique only because jackpots was.
    already = {facts.ref.columns[0] for facts in exact}
    open_names = [name for name in names if name not in already]
    combinations = [
        (first, second)
        for index, first in enumerate(open_names)
        for second in open_names[index + 1 :]
    ]
    pairs = _measure_keys(dataset, table, combinations)
    unique = [facts for facts in pairs if facts.unique]
    if unique:
        return unique[:MAX_KEY_CANDIDATES]
    if exact:
        # Kept, because saying nothing was worse — but marked, so describe() calls
        # it what it is rather than dressing a measure up as a key.
        quantities = measure_columns(dataset, table)
        return [
            replace(facts, coincidental=all(name in quantities for name in facts.ref.columns))
            for facts in exact[:MAX_KEY_CANDIDATES]
        ]

    # Nothing identifies a row exactly. The nearest set is still the thing worth
    # saying: (assetId, day) at 11,420 values across 11,421 rows is what a join
    # should use, and one duplicated row does not make assetId alone — 25 values
    # across 11,421 — any less wrong.
    best = max(pairs, key=lambda facts: facts.distinct, default=None)
    return [best] if best is not None and best.distinct > singles[0].distinct else []


@dataclass(frozen=True)
class Side:
    """What one side of a join looks like on the rows actually loaded."""

    ref: Ref
    rows: int
    joinable: int  # rows whose key has no NULL, the only ones equality can match
    distinct_keys: int
    unique: bool
    matched_rows: int  # joinable rows finding at least one partner
    max_partners: int

    @property
    def null_keys(self) -> int:
        return self.rows - self.joinable

    @property
    def unmatched(self) -> int:
        return self.joinable - self.matched_rows


@dataclass(frozen=True)
class Verified:
    """A measured join. Structural only — it says nothing about meaning."""

    left: Side
    right: Side
    joined_rows: int

    @property
    def cardinality(self) -> str:
        if self.left.unique and self.right.unique:
            return "1:1"
        if self.right.unique:
            return "N:1"
        if self.left.unique:
            return "1:N"
        return "N:N"

    @property
    def partial(self) -> bool:
        """Some row on either side finds no partner, whatever the cardinality."""
        return bool(self.left.unmatched or self.right.unmatched)

    def multiplies_side(self, side: str) -> bool:
        """Whether rows of the left or right side repeat. See multiplies()."""
        return (self.left if side == "left" else self.right).max_partners > 1

    def multiplies(self, table: str) -> bool:
        """Whether rows of this table actually repeat in the output.

        Measured, not inferred from the uniqueness flag. A single stray duplicate
        key elsewhere in a table makes it non-unique and the join N:N, while no row
        that participates is duplicated at all — the real asset file has exactly
        that shape. What decides whether a sum double-counts is whether any
        matched row finds more than one partner.
        """
        if table == self.left.ref.table:
            return self.left.max_partners > 1
        if table == self.right.ref.table:
            return self.right.max_partners > 1
        return False


def _key_sql(ref: Ref) -> tuple[str, str]:
    """The key expression and the predicate that its parts are all non-null."""
    parts = [quote_identifier(column) for column in ref.columns]
    key = f"({', '.join(parts)})" if len(parts) > 1 else parts[0]
    non_null = " AND ".join(f"{part} IS NOT NULL" for part in parts)
    return key, non_null


def verify(dataset: Dataset, candidate: JoinCandidate) -> Verified:
    """Measure a join by counting, never by building it.

    Grouping each side by its key first and joining those frequency tables gives
    the joined row count as sum(left_count * right_count) — the 439x explosion is
    measured at 25 rows rather than materialised at 12.5 million.

    NULL keys are counted apart. DuckDB counts NULL-bearing tuples in
    count(DISTINCT (a, b)) while an equality join matches none of them, so folding
    them together overstates containment on exactly the columns most likely to
    hold nulls.
    """
    sides = {}
    for name, ref in (("l", candidate.left), ("r", candidate.right)):
        key, non_null = _key_sql(ref)
        sides[name] = (
            f"SELECT {key} AS k, count(*) AS n FROM {quote_identifier(ref.table)} "
            f"WHERE {non_null} GROUP BY 1"
        )

    # Bounded like any other query. Grouping two large tables by their keys is
    # cheap next to the join it prices, but "cheap" is not "unbounded", and a
    # verification with no deadline could outlast the question that asked for it.
    with dataset._deadline():
        row = dataset.connection.execute(
            f"""
        WITH l AS ({sides["l"]}), r AS ({sides["r"]}),
        paired AS (SELECT l.k, l.n AS ln, r.n AS rn FROM l JOIN r ON l.k IS NOT DISTINCT FROM r.k)
        SELECT
          (SELECT count(*) FROM l) AS l_keys,
          (SELECT count(*) FROM r) AS r_keys,
          (SELECT coalesce(sum(n), 0) FROM l) AS l_joinable,
          (SELECT coalesce(sum(n), 0) FROM r) AS r_joinable,
          (SELECT coalesce(max(n), 0) FROM l) AS l_widest,
          (SELECT coalesce(max(n), 0) FROM r) AS r_widest,
          coalesce(sum(ln::HUGEINT * rn), 0) AS joined,
          coalesce(sum(ln), 0) AS l_matched,
          coalesce(sum(rn), 0) AS r_matched,
          coalesce(max(rn), 0) AS l_max_partners,
          coalesce(max(ln), 0) AS r_max_partners
        FROM paired
        """
        ).fetchone()
    (
        l_keys,
        r_keys,
        l_joinable,
        r_joinable,
        l_widest,
        r_widest,
        joined,
        l_matched,
        r_matched,
        l_partners,
        r_partners,
    ) = (int(value) for value in row)

    def side(ref: Ref, keys: int, joinable: int, widest: int, matched: int, partners: int) -> Side:
        return Side(
            ref=ref,
            rows=dataset.row_count(ref.table),
            joinable=joinable,
            distinct_keys=keys,
            # Unique among joinable rows: one row per key, nulls excluded.
            unique=bool(joinable) and widest == 1,
            matched_rows=matched,
            max_partners=partners,
        )

    return Verified(
        left=side(candidate.left, l_keys, l_joinable, l_widest, l_matched, l_partners),
        right=side(candidate.right, r_keys, r_joinable, r_widest, r_matched, r_partners),
        joined_rows=joined,
    )


# --- M3: refusing a join before it runs -----------------------------------------

# DuckDB names some aggregates that sqlglot parses as ordinary functions, so the
# class test alone let total() past — and a query whose only aggregate is unlisted
# skipped the guard entirely.
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
    a subquery with DISTINCT or GROUP BY. Base-table facts must not be used for
    one of those: `ratings JOIN (SELECT DISTINCT movieId FROM tags)` is safe even
    though `tags.movieId` repeats, and warning about it would refuse SQL the model
    writes correctly and unprompted.
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


def _sources(tree: exp.Expression, tables: set[str]) -> dict[str, Source]:
    """Alias, or name where there is no alias, mapped to what it stands for.

    A CTE is a derived relation like any subquery. Treating it as a bare name left
    `WITH top AS (SELECT playerId FROM t GROUP BY playerId) SELECT ... FROM t JOIN top`
    looking like t joined to itself — a correct and very common shape, refused.
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


def _column_owner(column: exp.Column, sources: dict[str, Source], dataset: Dataset) -> str | None:
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
                key = (column.table or "").lower() or (
                    _column_owner(column, sources, dataset) or ""
                )
                per_source.setdefault(key, []).append(column.name)
        else:
            simple = False
    return per_source, simple


def preflight(
    dataset: Dataset, sql: str, cache: dict | None = None
) -> tuple[str | None, str | None]:
    """(refusal, note): why this must not run, and what to say if it may.

    Not every aggregate over a repeated row is wrong. A SUM always double counts.
    MIN and MAX cannot change, however often a row appears. An AVG changes its
    weighting, which is sometimes exactly what was asked for — "for every session,
    look up that machine's utilisation that day, then average over sessions" is a
    session-weighted average by construction. Refusing all three cost a bank
    question its tool rounds for a query that was doing what it was told.

    Called before the query executes, not after: the incomplete asset join builds
    12.5 million rows on the way to a number that is 439 times too large, and there
    is no reason to pay for that before saying so.

    A query touching one table is returned untouched — table *references*, not
    distinct names, so a self-join is not mistaken for a single-table query and
    waved through.
    """
    # One loaded table, one code path — the plan's guarantee, and a property of the
    # dataset rather than of the query. Gating on query shape instead let a CTE
    # over the single table reach a guard that exists for cross-table fan-out, and
    # cost three bank questions their tool rounds. Fan-out within one table is the
    # grain guard's job, which already runs.
    if len(dataset.tables) < 2:
        return None, None

    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception:
        return None, None  # the SQL guard reports malformed SQL; this is not its job

    joins = list(tree.find_all(exp.Join))
    if not joins:
        # Nothing is combined, so nothing can be multiplied. Counting table
        # references instead made a CTE over one table look like a self-join.
        return None, None

    aggregates = aggregates_in(tree)
    if not aggregates:
        # Nothing is being totalled, so fan-out changes the row count but no figure.
        return None, None

    tables = {name.lower() for name in dataset.tables}
    sources = _sources(tree, tables)
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
            # Undetermined is not the same as safe. Letting it through was how an
            # unmeasurable join ran precisely because it was unmeasurable.
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
            # Counting or summing distinct values cannot be changed by repeating
            # rows — that is what DISTINCT is for, and refusing it turned the
            # standard way of writing a safe count into an error.
            continue
        owners = [_column_alias(column, sources, dataset) for column in node.find_all(exp.Column)]
        hit = next((owner for owner in owners if owner in multiplying), None)
        # A column reaching us through a derived relation is not traced, it is
        # merely renamed: sum(w.v) over a subquery that joins inside itself was
        # allowed because w resolved, while what w selects was never examined.
        traced = [
            owner
            for owner in owners
            if owner and (source := sources.get(owner)) and source.table is not None
        ]
        if hit is None and traced and len(traced) == len(owners):
            continue  # every column traced to a base relation, none repeated
        if hit is None:
            # Nothing to trace: COUNT(*), SUM(1), or a column projected out of a
            # subquery under an alias. All of them read the joined output, which is
            # the thing fan-out inflates — and treating "cannot tell" as "safe" is
            # how each of them walked past.
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
    """The shapes whose output grain the first version cannot prove."""
    # A window function is deliberately not refused. It adds columns and never
    # multiplies rows, so it cannot cause fan-out; what it cannot do is *prove* a
    # grain, and the grain logic simply never credits one. Refusing outright
    # blocked ordinary analysis — a LAG inside a CTE cost a bank question its
    # rounds — for a risk windows do not carry.
    for join in joins:
        # RIGHT and FULL change which unmatched rows survive, not which rows
        # repeat, and repetition is the whole question here. A self-join is
        # measurable too, now that each side is tracked by its alias.
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

    Checked here rather than left to the measurer: a predicate we cannot measure
    used to fall through as "nothing found to multiply", which let an unmeasurable
    join run precisely because it was unmeasurable.
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
    # A derived relation that already reduced itself to one row per key cannot
    # multiply the other side, whatever its base table does. One that has not
    # proved its grain is not something this version can reason about either way.
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

    Nothing multiplies, so nothing is double counted — but an inner join that
    drops 9,185 of one side's rows produces a total that is quietly short, and a
    short total looks exactly as reasonable as a correct one.
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

    The subset runs this way round. A subquery grouped by (assetId, day) is one
    row per pair, and joining on assetId alone still meets many of them — the
    reversed test called that unique and waved the join through.
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
