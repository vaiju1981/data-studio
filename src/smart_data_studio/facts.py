"""What a candidate key or join actually does to the rows that are loaded.

Structural only: this measures repetition, containment and cardinality. Whether
two columns *mean* the same thing is not a question the data can settle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from smart_data_studio.config import MAX_KEY_CANDIDATES, MAX_KEY_SEARCH_COLUMNS
from smart_data_studio.dataset import Dataset, quote_identifier
from smart_data_studio.proposals import JoinCandidate, Ref


@dataclass(frozen=True)
class KeyFacts:
    """What a candidate key does on the rows actually loaded."""

    ref: Ref
    rows: int
    complete: int  # rows with no NULL in any key column
    distinct: int
    # Unique, but made of measures rather than identifiers. Set only on the fallback
    # path, the one place that knows the difference; a unique sku or email is a real
    # key and is not marked.
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
            # "one row per (x)" reads as a key and invites a join on it, when all
            # this says is that the values happen not to repeat today.
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

    Money and percentages carry plenty of distinct values and identify nothing, so
    distinct count alone ranks them as keys. The test lives here, used by both the
    key search and the join ranking.
    """
    found = set()
    for name, kind in dataset.schema(table):
        upper = kind.upper()
        if any(part in upper for part in ("DOUBLE", "FLOAT", "DECIMAL", "REAL")):
            found.add(name)  # a float is never an identifier
        elif any(part in upper for part in ("INT", "HUGEINT")) and not _looks_like_identifier(name):
            # An integer can be either: orderId identifies, quantity measures.
            found.add(name)
    return found


def discover_keys(
    dataset: Dataset, table: str, distinct_by_column: dict[str, float]
) -> list[KeyFacts]:
    """The narrowest column sets that identify a row, singles before pairs.

    Bounded rather than exhaustive: singles first, then pairs among the few with the
    most distinct values, since a key's parts are necessarily among them. Minimality
    falls out of that order — a unique pair whose subset is already unique is not a
    composite key, so no pair containing a key is offered.
    """
    rows = dataset.row_count(table)
    if rows < 2:
        return []
    # Floats only. An integer is a perfectly good key component — a day stored as
    # 20240101, an hour, a numbered stand.
    measures = {
        name
        for name, kind in dataset.schema(table)
        if any(part in kind.upper() for part in ("DOUBLE", "FLOAT", "DECIMAL", "REAL"))
    }
    # Identifier-named first, then by distinct count: the name decides the order
    # rather than membership, since count alone ranks a measure above a key and
    # excluding integers outright hides composite keys made of them.
    ranked = sorted(
        ((name, count) for name, count in distinct_by_column.items() if name not in measures),
        key=lambda item: (not _looks_like_identifier(item[0]), -item[1]),
    )
    if not ranked:
        return []
    names = [name for name, _ in ranked[:MAX_KEY_SEARCH_COLUMNS]]
    # One scan for all the singles, then one for all the pairs.
    singles = _measure_keys(dataset, table, [(name,) for name in names])
    exact = [facts for facts in singles if facts.unique]
    # A unique identifier is a key; a unique measure is a coincidence of the rows
    # loaded today. So a composite of identifiers is looked for first, and the
    # coincidence kept only as a fallback.
    named = [facts for facts in exact if _looks_like_identifier(facts.ref.columns[0])]
    if named:
        return named[:MAX_KEY_CANDIDATES]

    # A pair containing an already-unique column is unique for that reason alone
    # and says nothing.
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
        # Kept, but marked, so describe() does not dress a measure up as a key.
        quantities = measure_columns(dataset, table)
        return [
            replace(facts, coincidental=all(name in quantities for name in facts.ref.columns))
            for facts in exact[:MAX_KEY_CANDIDATES]
        ]

    # Nothing identifies a row exactly, but the nearest set is still what a join
    # should use: one duplicated row does not make a far coarser column any better.
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

        Measured, not inferred from the uniqueness flag: one stray duplicate key
        anywhere makes a table non-unique and the join N:N while no participating
        row repeats at all. What decides whether a sum double counts is whether a
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

    Grouping each side by its key and joining those frequency tables gives the
    joined row count as sum(left_count * right_count), so an explosion is priced
    rather than materialised.

    NULL keys are counted apart: DuckDB counts NULL-bearing tuples in
    count(DISTINCT (a, b)) while an equality join matches none of them, so folding
    them together overstates containment.
    """
    sides = {}
    for name, ref in (("l", candidate.left), ("r", candidate.right)):
        key, non_null = _key_sql(ref)
        sides[name] = (
            f"SELECT {key} AS k, count(*) AS n FROM {quote_identifier(ref.table)} "
            f"WHERE {non_null} GROUP BY 1"
        )

    # Bounded like any other query: cheap next to the join it prices is not free.
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
