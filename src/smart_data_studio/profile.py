"""Fast profiles and deterministic observations powered by DuckDB."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from smart_data_studio import facts, logs
from smart_data_studio.config import (
    DICTIONARY_VALUES,
    MAX_CELL_CHARS_TO_MODEL,
    MAX_SHARED_COLUMNS,
    MAX_SUMMARIZE_QUERIES,
    MAX_VARYING_COLUMNS,
    MIN_SENTINEL_ROWS,
    PER_ROW_CHANGE_SHARE,
    SENTINEL_GAP_RATIO,
    SENTINEL_SHARE,
)
from smart_data_studio.dataset import (
    Dataset,
    OutOfQueries,
    looks_like_identifier,
    quote_identifier,
)


@dataclass
class TableProfile:
    table_name: str
    row_count: int
    stats: pd.DataFrame
    findings: list[str]
    # What the dimension columns actually contain, one line each.
    dictionary: list[str] = field(default_factory=list)
    # The same values keyed by column, for guards rather than for reading.
    values: dict[str, list[str]] = field(default_factory=dict)
    # How a row is identified, measured. Empty for a single loaded table.
    keys: list = field(default_factory=list)
    # Columns another loaded table also has, and how far each is from identifying
    # a row — exactly the columns a join reaches for.
    shared: list[str] = field(default_factory=list)
    # Measures another table also has under the same name. These do not join; they
    # get taken from the wrong table.
    shared_measures: list[str] = field(default_factory=list)
    # The column a row is a repeat of. A question about that entity has to be
    # aggregated to it, not counted by row.
    entity_key: str | None = None

    def prompt_text(self) -> str:
        columns = [
            "column_name",
            "column_type",
            "min",
            "max",
            "approx_unique",
            "avg",
            "q50",
            "null_percentage",
        ]
        available = [column for column in columns if column in self.stats.columns]
        # SUMMARIZE covers every column, so min and max carry real cell values. The
        # model-facing text is filtered here rather than the panel the owner reads.
        visible = self.stats
        rendered_stats = (
            visible[available]
            .map(lambda value: "—" if pd.isna(value) else value)
            .to_string(index=False)
        )
        rendered_findings = "\n".join(f"- {finding}" for finding in self.findings)
        # Stated before anything joins, so the right key is used first time.
        rendered_keys = (
            "\nHow a row is identified: " + "; ".join(facts.describe() for facts in self.keys) + "."
            if self.keys
            else ""
        )
        rendered_shared = (
            "\nColumns another table also has, and what each does here: "
            + "; ".join(self.shared)
            + ". Joining on one that repeats multiplies rows."
            if self.shared
            else ""
        )
        rendered_measures = (
            "\nMeasures another table also has under the same name, with this table's "
            "total: " + "; ".join(self.shared_measures) + ". They are different "
            "quantities — take each from the table the question is about."
            if self.shared_measures
            else ""
        )
        rendered_dictionary = (
            "\nValues held by the dimension columns:\n"
            + "\n".join(f"- {line}" for line in self.dictionary)
            if self.dictionary
            else ""
        )
        return (
            f"Table {self.table_name}: {self.row_count:,} rows\n"
            f"Profile (approx_unique is an estimate, not an exact count):\n{rendered_stats}\n"
            f"Findings:\n{rendered_findings}{rendered_keys}{rendered_shared}{rendered_measures}{rendered_dictionary}"
        )


@dataclass
class Allowance:
    """Fallback queries the profile may still spend, shared by all its tables.

    One allowance for the workspace, not one for every table: nine pathological
    files each given sixty attempts spend the whole session before the first
    question can be asked.

    It counts its own attempts rather than watching the workspace's query counter,
    which ordinary profiling advances too. Reading that counter, sixteen healthy
    tables were enough to use the allowance up, and the one bad table after them
    came back with nothing described at all — the common case paying for the
    pathological one.
    """

    left: int = MAX_SUMMARIZE_QUERIES
    exhausted: bool = False

    def spend(self) -> bool:
        if self.left <= 0:
            self.exhausted = True
            return False
        self.left -= 1
        return True


def profile_dataset(dataset: Dataset) -> list[TableProfile]:
    allowance = Allowance()
    return [profile_table(dataset, table, allowance) for table in dataset.tables]


def _summarize(
    dataset: Dataset, table_name: str, allowance: Allowance | None = None
) -> tuple[pd.DataFrame, list[str], bool]:
    """SUMMARIZE, falling back to column by column when one column defeats it.

    stddev over a NaN or infinity raises OutOfRange, which otherwise takes the
    whole table's profile with it — and the app is unusable for a file it cannot
    profile. One bad column costs its own statistics and nothing else.
    """
    quoted = quote_identifier(table_name)
    try:
        return dataset.run(f"SUMMARIZE {quoted}").fetchdf(), [], False
    except Exception:
        logs.failure("summarize.per_column")

    names = [name for name, _ in dataset.schema(table_name)]
    allowance = allowance if allowance is not None else Allowance()
    frames, refused = _summarize_in_halves(dataset, quoted, names, allowance)
    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Whether every refusal was proved, or the sweep gave up with columns untried.
    return stats, refused, allowance.exhausted


def _summarize_in_halves(
    dataset: Dataset, quoted: str, names: list[str], allowance: Allowance
) -> tuple[list[pd.DataFrame], list[str]]:
    """SUMMARIZE these columns together, halving to find whichever defeats it.

    Column by column keeps the promise — one bad column costs its own statistics
    and nothing else — at the price of a full scan each. On a 400-column file that
    is 400 scans, and once every scan is charged to the workspace, one bad column
    in a wide file spent most of the session's budget before the first question
    was asked. Halving isolates the same column in about a dozen queries.

    Halving is cheap per bad column and not free: sixty-seven of them in a
    400-column file still came to 482 queries. So the sweep spends an allowance
    and then stops, because a workspace that can describe most of its columns and
    answer nothing is worse than one that describes fewer and works.
    """
    if not allowance.spend():
        return [], list(names)

    projection = ", ".join(quote_identifier(name) for name in names)
    try:
        frame = dataset.run(f"SUMMARIZE (SELECT {projection} FROM {quoted})").fetchdf()
    except OutOfQueries:
        # Nothing further will succeed either, so stop and say what went undescribed
        # rather than retrying every remaining column against an empty budget.
        return [], list(names)
    except Exception:
        if len(names) == 1:
            return [], list(names)
    else:
        return [frame], []

    middle = len(names) // 2
    left, left_refused = _summarize_in_halves(dataset, quoted, names[:middle], allowance)
    right, right_refused = _summarize_in_halves(dataset, quoted, names[middle:], allowance)
    return left + right, left_refused + right_refused


def profile_table(
    dataset: Dataset, table_name: str, allowance: Allowance | None = None
) -> TableProfile:
    stats, unsummarised, gave_up = _summarize(dataset, table_name, allowance)
    row_count = dataset.row_count(table_name)
    exact_distinct = _exact_distinct(dataset, table_name, stats, row_count)
    findings = _findings(stats, row_count, exact_distinct)
    # Ahead of the rest: a meaningless average outranks a cardinality note.
    findings = list(_sentinels(dataset, table_name, stats).values()) + findings
    dictionary, values = _dictionary(dataset, table_name, stats)
    # Only with something to join to: one CSV adds no query and no output.
    keys = (
        facts.discover_keys(dataset, table_name, _distinct_by_column(stats))
        if len(dataset.tables) > 1
        else []
    )
    many = len(dataset.tables) > 1
    shared = _shared_columns(dataset, table_name, row_count) if many else []
    shared_measures = _shared_measures(dataset, table_name) if many else []
    grain, entity_key = _entity_grain(dataset, table_name, stats, row_count, exact_distinct)
    if grain:
        findings.insert(0, grain)
    if unsummarised:
        # Said rather than silently missing, or the column reads as one the file
        # does not have. Two different reasons, and claiming the wrong one would
        # tell the reader a column holds an infinity when nobody ever looked.
        findings.append(
            f"No statistics could be computed for {len(unsummarised)} column(s) — so many "
            "columns here defeat SUMMARIZE that the search for them was stopped. They are "
            "still queryable."
            if gave_up
            else f"No statistics could be computed for {', '.join(unsummarised)} — the values "
            "include NaN or infinity, which DuckDB cannot summarise. The column is still "
            "queryable."
        )
    if not findings:
        findings.append("No obvious constant, empty, high-null, or key-like columns were found.")
    return TableProfile(
        table_name=table_name,
        row_count=row_count,
        stats=stats,
        findings=findings,
        dictionary=dictionary,
        values=values,
        keys=keys,
        shared=shared,
        shared_measures=shared_measures,
        entity_key=entity_key,
    )


def _entity_grain(
    dataset: Dataset,
    table_name: str,
    stats: pd.DataFrame,
    row_count: int,
    exact_distinct: dict[str, int],
) -> tuple[str | None, str | None]:
    """Report which columns stay constant within the table's entity key.

    Grouping by an entity key together with a column that varies inside it splits
    one entity across several rows, each carrying a partial total. A column-by-column
    profile cannot reveal that, so it is worth one extra pass.

    The exact counts decide which column is the key. approx_unique comes from a
    sketch that drifts, and a row id reading a few short of the row count is picked
    as the entity ahead of the real one — after which the table looks like it is
    already at entity grain and nothing is reported at all.
    """
    records = stats.to_dict(orient="records")

    def distinct(row: dict) -> float:
        exact = exact_distinct.get(str(row["column_name"]))
        return float(exact) if exact is not None else _number(row.get("approx_unique"))

    candidates = [
        (distinct(row), str(row["column_name"]))
        for row in records
        if looks_like_identifier(str(row["column_name"])) and 1 < distinct(row) < row_count
    ]
    if not candidates:
        return None, None
    _, key = max(candidates)

    others = [str(row["column_name"]) for row in records if str(row["column_name"]) != key]
    if not others:
        return None, None
    # approx_count_distinct is exact at the low cardinalities the "> 1" test cares about.
    inner = ", ".join(
        f"approx_count_distinct({quote_identifier(name)}) AS d{index}"
        for index, name in enumerate(others)
    )
    # count_if rides along on the same pass: max says a column varies somewhere,
    # this says how widely.
    outer = ", ".join(
        f"max(d{index}) AS m{index}, count_if(d{index} > 1) AS c{index}"
        for index in range(len(others))
    )
    try:
        # repeated rides along on the same pass: a single-row entity cannot change
        # anything, so it is the wrong denominator for how widely a column moves.
        row = dataset.run(
            f"SELECT count(*) AS entities, count_if(n > 1) AS repeated, {outer} FROM ("
            f"SELECT {quote_identifier(key)}, count(*) AS n, {inner} "
            f"FROM {quote_identifier(table_name)} GROUP BY 1)"
        ).fetchone()
    except Exception:
        return None, None  # a profiling nicety must never stop the data from loading

    entities, repeated = int(row[0]), int(row[1])
    # Two values per column, after the two scalars in front of them.
    stable = [others[index] for index in range(len(others)) if _number(row[2 + index * 2]) <= 1]
    # Attributes only: a per-row id, timestamp or measure varies by definition, and
    # listing those buries the one that carries meaning.
    dimensions = _attribute_columns(stats, row_count, facts.measure_columns(dataset, table_name))
    ceiling = repeated * PER_ROW_CHANGE_SHARE
    moving = sorted(
        (
            (int(_number(row[3 + index * 2])), others[index])
            for index in range(len(others))
            if 0 < _number(row[3 + index * 2]) < ceiling and others[index] in dimensions
        ),
        reverse=True,
    )
    # Both ends, not the top. Ranked by count alone a rarely-changing column is
    # always the first thing the cap discards, and it is the one worth naming:
    # everywhere else it reads as a property of the entity.
    if len(moving) > MAX_VARYING_COLUMNS:
        edge = MAX_VARYING_COLUMNS // 2
        moving = moving[:edge] + moving[-edge:]
    if entities >= row_count:
        # One row per key: it is the grain already, not an entity to aggregate to.
        return None, None

    listed = ", ".join(stable[:12]) + (", …" if len(stable) > 12 else "") if stable else "none"
    changes = ", ".join(
        f"{name} for {count:,} ({_share(count, repeated)})" for count, name in moving
    )
    return (
        f"{key} repeats: {entities:,} values across {row_count:,} rows "
        f"(~{row_count / entities:.1f} rows each). Constant within it: {listed}. Every other "
        f"column varies, so adding one to GROUP BY {key} splits a single {key} across several "
        f"rows — aggregate those with MAX or SUM instead."
        + (
            f" Changes within a single {key}, as a share of the {repeated:,} with more "
            f"than one row: {changes}. One that changes for a few reads everywhere else as "
            f"a property of the {key} and is not, so filtering or grouping on it will not "
            f"do what it appears to."
            if changes
            else ""
        )
    ), key


def _share(count: int, total: int) -> str:
    """A readable proportion that does not round a rare one away to 0.0%."""
    portion = count / total if total else 0.0
    return f"{portion:.1%}" if portion >= 0.001 else "under 0.1%"


def _exact_distinct(
    dataset: Dataset, table_name: str, stats: pd.DataFrame, row_count: int
) -> dict[str, int]:
    """Exact distinct counts for the columns that could be keys.

    SUMMARIZE reports approx_unique from a sketch, which drifts either way and can
    exceed the row count, so it can neither confirm a key nor be quoted as a fact.
    Only near-unique columns are worth the exact pass.
    """
    if row_count == 0:
        return {}
    candidates = [
        str(row["column_name"])
        for row in stats.to_dict(orient="records")
        if _number(row.get("approx_unique")) >= row_count * 0.9
    ]
    if not candidates:
        return {}
    projections = ", ".join(f"COUNT(DISTINCT {quote_identifier(name)})" for name in candidates)
    row = dataset.run(f"SELECT {projections} FROM {quote_identifier(table_name)}").fetchone()
    return {name: int(count) for name, count in zip(candidates, row, strict=True)}


def _shared_columns(dataset: Dataset, table_name: str, row_count: int) -> list[str]:
    """How far each column shared with another table is from identifying a row.

    A join reaches for the column both files have. Saying whether it repeats before
    anything joins turns a refusal into a right answer first time.
    """
    measures = facts.measure_columns(dataset, table_name)
    mine = {name for name, _ in dataset.schema(table_name) if name not in measures}
    elsewhere = {
        name for other in dataset.tables if other != table_name for name, _ in dataset.schema(other)
    }
    together = sorted(mine & elsewhere)
    if not together or not row_count:
        return []
    counts = dataset.run(
        "SELECT "
        + ", ".join(f"count(DISTINCT {quote_identifier(name)})" for name in together)
        + f" FROM {quote_identifier(table_name)}"
    ).fetchone()

    measured = sorted(
        ((name, int(distinct)) for name, distinct in zip(together, counts, strict=True)),
        key=lambda item: -item[1],
    )
    # Bounded and prioritised: the most various first, because those are the
    # plausible keys. Constants are kept regardless, since joining on one pairs
    # every row with every row.
    constants = [item for item in measured if item[1] <= 1]
    plausible = [item for item in measured if item[1] > 1][:MAX_SHARED_COLUMNS]

    described = []
    for name, distinct in plausible:
        if distinct >= row_count:
            described.append(f"{name} identifies a row on its own")
        else:
            described.append(f"{name} repeats: {distinct:,} values across {row_count:,} rows")
    for name, _ in constants[:2]:
        described.append(f"{name} is the same for every row, so joining on it pairs everything")
    return described


def _shared_measures(dataset: Dataset, table_name: str) -> list[str]:
    """Measures another table also carries under the same name, with this total.

    A provenance problem, not a join one: nothing double counts and no join is
    made, so the fan-out guard has nothing to catch. Showing both totals is what
    stops the wrong one being used.
    """
    mine = facts.measure_columns(dataset, table_name)
    elsewhere = {
        name
        for other in dataset.tables
        if other != table_name
        for name in facts.measure_columns(dataset, other)
    }
    together = sorted(mine & elsewhere)[:MAX_SHARED_COLUMNS]
    if not together:
        return []
    quoted = quote_identifier(table_name)
    totals = dataset.run(
        "SELECT "
        + ", ".join(f"sum({quote_identifier(name)})" for name in together)
        + f" FROM {quoted}"
    ).fetchone()
    return [
        f"{name} here totals {float(total):,.0f}"
        for name, total in zip(together, totals, strict=True)
        if total is not None
    ]


def _distinct_by_column(stats: pd.DataFrame) -> dict[str, float]:
    """Approximate distinct counts, which is enough to rank a key search."""
    return {
        str(row["column_name"]): _number(row.get("approx_unique"))
        for row in stats.to_dict(orient="records")
    }


def _attribute_columns(
    stats: pd.DataFrame, row_count: int, measures: set[str] = frozenset()
) -> dict[str, float]:
    """Columns that describe an entity rather than identify or measure it.

    Two ends are useless: a constant says nothing, and a value unique to nearly
    every row identifies rather than describes. Type is not part of it — a tier or
    a store number stored as an integer describes an entity exactly as one stored
    as text does, and reading text alone made those invisible.

    Which leaves the measures, which vary per row by their nature and drown the
    rest. They go by name, because §8 of the plan settled that snapshot and
    measure cannot be told apart by cardinality: a numeric attribute whose name
    says nothing — a credit limit, a target — still reads here as a measure.
    """
    found = {}
    for row in stats.to_dict(orient="records"):
        name = str(row["column_name"])
        if name in measures:
            continue
        distinct = _number(row.get("approx_unique"))
        if distinct < 2 or distinct > max(row_count * 0.9, 2):
            continue
        found[name] = distinct
    return found


def _dimension_columns(stats: pd.DataFrame, row_count: int) -> dict[str, float]:
    """Attribute columns whose values are worth listing, which means the text ones.

    The dictionary quotes values back; a numeric attribute's commonest values say
    nothing a reader cannot get from min, max and the quartiles already shown.
    """
    kinds = {
        str(row["column_name"]): str(row["column_type"]).upper()
        for row in stats.to_dict(orient="records")
    }
    return {
        name: distinct
        for name, distinct in _attribute_columns(stats, row_count).items()
        if "VARCHAR" in kinds.get(name, "")
    }


def _dictionary(
    dataset: Dataset, table_name: str, stats: pd.DataFrame
) -> tuple[list[str], dict[str, list[str]]]:
    """The values each dimension column actually holds, not merely its name.

    The schema says an ageGroup column exists; it does not say the column runs from
    "21-25" to "81+". Without that, a question about segmentation reaches for
    whichever column the exploration happened to sample.

    approx_top_k gathers every column in one pass rather than one scan per column.
    """
    candidates = list(_dimension_columns(stats, dataset.row_count(table_name)).items())
    if not candidates:
        return [], {}

    projections = ", ".join(
        f"approx_top_k({quote_identifier(name)}, {DICTIONARY_VALUES}) AS v{index}"
        for index, (name, _) in enumerate(candidates)
    )
    values = (
        dataset.run(f"SELECT {projections} FROM {quote_identifier(table_name)}")
        .fetchdf()
        .iloc[0]
        .to_dict()
    )

    lines: list[str] = []
    # Kept structurally too, so a guard can compare a value named in a question
    # against the one the query filtered on.
    held: dict[str, list[str]] = {}
    for index, (name, distinct) in enumerate(candidates):
        # Explicit None test: this arrives as a numpy array, and `array or []`
        # raises rather than falling back.
        raw = values.get(f"v{index}")
        found = [str(item) for item in (raw if raw is not None else []) if not pd.isna(item)]
        if not found:
            continue
        shown = [
            item[:MAX_CELL_CHARS_TO_MODEL] + "…" if len(item) > MAX_CELL_CHARS_TO_MODEL else item
            for item in found
        ]
        held[name] = shown
        if distinct <= DICTIONARY_VALUES:
            lines.append(f"{name} ({len(shown)} values): {', '.join(sorted(shown))}")
        else:
            lines.append(f"{name} (~{int(distinct):,} values), most common: {', '.join(shown)}")
    return lines, held


def _sentinels(dataset: Dataset, table_name: str, stats: pd.DataFrame) -> dict[str, str]:
    """Extreme values that repeat, which are codes rather than measurements.

    Real datasets bury "missing" inside a numeric column as -200, -999 or 9999. The
    value parses, the column is numeric, the average is arithmetic, and the answer
    is silently wrong.

    The test is that a value is both repeated and *isolated*: the step from it to
    the next is far wider than the typical step, and it covers a real share of the
    rows. Comparing the step to the column's whole range instead scales with the
    range, so the same code is caught in a narrow column and missed in a wide one.

    Typical step is the inner range over the distinct values in it, so a binned
    column whose values sit evenly apart is not accused of hiding a code.
    """
    candidates = []
    for row in stats.to_dict(orient="records"):
        kind = str(row["column_type"]).upper()
        if not any(k in kind for k in ("INT", "DOUBLE", "DECIMAL", "FLOAT")):
            continue
        try:
            low, high = float(row["min"]), float(row["max"])
            distinct = float(row["approx_unique"])
        except (TypeError, ValueError):
            continue
        if low < high and distinct > 2:
            candidates.append((str(row["column_name"]), low, high, distinct))
    if not candidates:
        return {}

    table = quote_identifier(table_name)
    projections = []
    for index, (name, low, high, _) in enumerate(candidates):
        column = quote_identifier(name)
        projections += [
            f"count({column}) AS present_{index}",
            f"count_if({column} = {low!r}) AS lon_{index}",
            f"count_if({column} = {high!r}) AS hin_{index}",
            f"min({column}) FILTER (WHERE {column} > {low!r}) AS lo2_{index}",
            f"max({column}) FILTER (WHERE {column} < {high!r}) AS hi2_{index}",
        ]
    values = (
        dataset.run(f"SELECT {', '.join(projections)} FROM {table}").fetchdf().iloc[0].to_dict()
    )

    found: dict[str, str] = {}
    for index, (name, low, high, distinct) in enumerate(candidates):
        present = _number(values.get(f"present_{index}"))
        if present < MIN_SENTINEL_ROWS:
            continue
        inner_low, inner_high = values.get(f"lo2_{index}"), values.get(f"hi2_{index}")
        if inner_low is None or inner_high is None or pd.isna(inner_low) or pd.isna(inner_high):
            continue
        inner_low, inner_high = float(inner_low), float(inner_high)
        spread = inner_high - inner_low
        if spread <= 0:
            continue
        typical = spread / max(distinct - 1, 1)
        for edge, inner, count in (
            (low, inner_low, _number(values.get(f"lon_{index}"))),
            (high, inner_high, _number(values.get(f"hin_{index}"))),
        ):
            gap = abs(inner - edge)
            if count / present >= SENTINEL_SHARE and gap > typical * SENTINEL_GAP_RATIO:
                found[name] = (
                    f"{name} holds {_trim(edge)} in {count / present:.0%} of rows, standing "
                    f"{gap / typical:.0f} times further from the next value than values in this "
                    f"column normally sit apart. That is the shape of a missing-value code, not "
                    f"a measurement — exclude it before averaging, or the answer will be "
                    f"confidently wrong."
                )
                break
    return found


def _trim(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"


def _findings(stats: pd.DataFrame, row_count: int, exact_distinct: dict[str, int]) -> list[str]:
    findings: list[str] = []
    for row in stats.to_dict(orient="records"):
        name = str(row["column_name"])
        data_type = str(row["column_type"])
        exact = exact_distinct.get(name)
        unique = float(exact) if exact is not None else _number(row.get("approx_unique"))
        null_percentage = _number(row.get("null_percentage"))
        non_null = row_count * (1 - null_percentage / 100)

        if null_percentage >= 100:
            findings.append(f"{name} is entirely null.")
        elif unique <= 1:
            findings.append(f"{name} is constant with {int(unique)} distinct value.")
        elif null_percentage >= 30:
            findings.append(f"{name} is {null_percentage:.1f}% null.")

        if row_count > 1 and null_percentage == 0 and exact is not None and exact == row_count:
            findings.append(
                f"{name} is unique across all {row_count:,} rows, so it is a candidate key. "
                "Whether it identifies the entity is a semantic question the data cannot settle."
            )
        elif "VARCHAR" in data_type.upper() and non_null >= 20 and unique / non_null >= 0.8:
            qualifier = "" if exact is not None else "about "
            findings.append(
                f"{name} is high-cardinality text with {qualifier}{int(unique):,} distinct values."
            )

    return findings


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
