"""Fast profiles and deterministic observations powered by DuckDB."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smart_data_studio.config import (
    MIN_SENTINEL_ROWS,
    SENTINEL_GAP_RATIO,
    SENTINEL_SHARE,
)
from smart_data_studio.dataset import Dataset, quote_identifier


@dataclass
class TableProfile:
    table_name: str
    row_count: int
    stats: pd.DataFrame
    findings: list[str]

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
        rendered_stats = (
            self.stats[available]
            .map(lambda value: "—" if pd.isna(value) else value)
            .to_string(index=False)
        )
        rendered_findings = "\n".join(f"- {finding}" for finding in self.findings)
        return (
            f"Table {self.table_name}: {self.row_count:,} rows\n"
            f"Profile (approx_unique is an estimate, not an exact count):\n{rendered_stats}\n"
            f"Findings:\n{rendered_findings}"
        )


def profile_dataset(dataset: Dataset) -> list[TableProfile]:
    return [profile_table(dataset, table) for table in dataset.tables]


def profile_table(dataset: Dataset, table_name: str) -> TableProfile:
    stats = dataset.connection.execute(f"SUMMARIZE {quote_identifier(table_name)}").fetchdf()
    row_count = dataset.row_count(table_name)
    exact_distinct = _exact_distinct(dataset, table_name, stats, row_count)
    findings = _findings(stats, row_count, exact_distinct)
    # Ahead of the rest: a column whose average is meaningless is worth more than
    # a note that another column is high-cardinality.
    findings = list(_sentinels(dataset, table_name, stats).values()) + findings
    grain = _entity_grain(dataset, table_name, stats, row_count)
    if grain:
        findings.insert(0, grain)
    if not findings:
        findings.append("No obvious constant, empty, high-null, or key-like columns were found.")
    return TableProfile(
        table_name=table_name,
        row_count=row_count,
        stats=stats,
        findings=findings,
    )


def _entity_grain(
    dataset: Dataset, table_name: str, stats: pd.DataFrame, row_count: int
) -> str | None:
    """Report which columns stay constant within the table's entity key.

    Grouping by an entity key together with a column that varies inside it splits
    one entity across several rows, each carrying a partial total — a wrong number
    that looks entirely reasonable. Nothing in a column-by-column profile reveals
    this, so it is worth one extra pass.
    """
    records = stats.to_dict(orient="records")
    candidates = [
        (_number(row.get("approx_unique")), str(row["column_name"]))
        for row in records
        if _looks_like_key(str(row["column_name"]))
        and 1 < _number(row.get("approx_unique")) < row_count
    ]
    if not candidates:
        return None
    _, key = max(candidates)

    others = [str(row["column_name"]) for row in records if str(row["column_name"]) != key]
    if not others:
        return None
    # approx_count_distinct is exact at the low cardinalities the "> 1" test cares about.
    inner = ", ".join(
        f"approx_count_distinct({quote_identifier(name)}) AS d{index}"
        for index, name in enumerate(others)
    )
    outer = ", ".join(f"max(d{index}) AS m{index}" for index in range(len(others)))
    try:
        row = dataset.connection.execute(
            f"SELECT count(*) AS entities, {outer} FROM ("
            f"SELECT {quote_identifier(key)}, {inner} "
            f"FROM {quote_identifier(table_name)} GROUP BY 1)"
        ).fetchone()
    except Exception:
        return None  # a profiling nicety must never stop the data from loading

    entities = int(row[0])
    stable = [others[index] for index in range(len(others)) if _number(row[index + 1]) <= 1]
    if entities >= row_count:
        return None  # one row per key: it is the grain, not an entity to group within

    listed = ", ".join(stable[:12]) + (", …" if len(stable) > 12 else "") if stable else "none"
    return (
        f"{key} repeats: {entities:,} values across {row_count:,} rows "
        f"(~{row_count / entities:.1f} rows each). Constant within it: {listed}. Every other "
        f"column varies, so adding one to GROUP BY {key} splits a single {key} across several "
        f"rows — aggregate those with MAX or SUM instead."
    )


def _looks_like_key(name: str) -> bool:
    """id, player_id, playerId, playerID — but not paid, valid, PAID or PYRAMID.

    The separator is what counts. Testing only that the character before "id" is
    upper-case accepted every all-caps word ending in those letters, so the letter
    before the suffix has to be lower-case: the boundary in playerId, absent in PAID.
    """
    lowered = name.lower()
    if lowered == "id" or lowered.endswith("_id"):
        return True
    return len(name) > 2 and name.endswith(("Id", "ID")) and name[-3].islower()


def _exact_distinct(
    dataset: Dataset, table_name: str, stats: pd.DataFrame, row_count: int
) -> dict[str, int]:
    """Exact distinct counts for the columns that could be keys.

    SUMMARIZE reports approx_unique from a sketch, which drifts either way and can
    even exceed the row count — so it can neither confirm a key nor be quoted as a
    fact. Only near-unique columns are worth the exact pass, which keeps the cost
    off wide tables.
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
    row = dataset.connection.execute(
        f"SELECT {projections} FROM {quote_identifier(table_name)}"
    ).fetchone()
    return {name: int(count) for name, count in zip(candidates, row, strict=True)}


def _sentinels(dataset: Dataset, table_name: str, stats: pd.DataFrame) -> dict[str, str]:
    """Extreme values that repeat, which are codes rather than measurements.

    Real datasets bury "missing" inside a numeric column as -200, -999 or 9999.
    Nothing about the load is wrong — the value parses, the column is numeric, the
    average is arithmetic — and the answer is silently, confidently wrong. The UCI
    air quality file reports mean CO of -34.2 this way, against a true 2.15.

    The test is that a value is both repeated and *isolated*: the step from it to
    the next value is far wider than the typical step between values, and it
    accounts for a real share of the rows. Comparing the step to the column's whole
    range instead looks right and is not — it catches -200 in a column spanning 12,
    and misses the same -200 in a column spanning 1,476, which is the case that
    turned a true mean of 246.9 into 168.6.

    Typical step is the inner range over the distinct values in it, so a binned
    column whose values all sit 100 apart is not accused of hiding a code in its
    lowest bin.

    SUMMARIZE has already paid for min and max, so this costs one further scan.
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
        dataset.connection.execute(f"SELECT {', '.join(projections)} FROM {table}")
        .fetchdf()
        .iloc[0]
        .to_dict()
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
