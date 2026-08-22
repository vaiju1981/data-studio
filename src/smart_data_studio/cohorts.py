"""Following a population forward from the moment it starts.

Retention, account vintage, readmission within thirty days, repeat purchase and
warranty claims are one computation wearing five nouns: take the entities that
began in a period, then count how many of them come back in each period after.

The reason it is a tool rather than a query is the denominator. Asked how a
January cohort was doing, the model divided each later month by the entities
active in January — 6,780 — where the cohort is everyone who registered in
January, 7,349. Every figure it quoted was arithmetically correct and the
retention curve was wrong, because 569 of the cohort first appeared later and
were counted in the numerators while missing from the base. Nothing in the SQL
looks wrong; the query simply never asked how large the cohort was.

So the size is counted once, from the cohort itself, and travels with every rate.
"""

from __future__ import annotations

import pandas as pd

from smart_data_studio.config import MAX_COHORT_HORIZON, MAX_COHORTS
from smart_data_studio.dataset import Dataset, quote_identifier

# date_trunc and date_diff take these as a bare word, so they are never
# interpolated from what the model said — only chosen from here.
PERIODS = ("day", "week", "month", "quarter", "year")


class NotCohortable(ValueError):
    """Raised when the columns given cannot describe a cohort."""


def cohort_window(
    dataset: Dataset,
    table: str,
    entity_column: str,
    cohort_column: str,
    activity_column: str,
    period: str = "month",
    horizon: int = 12,
) -> dict[str, object]:
    """How much of each starting cohort is still active in the periods after it."""
    if table not in dataset.tables:
        raise NotCohortable(f"Unknown table: {table}")
    known = {name for name, _ in dataset.schema(table)}
    missing = [c for c in (entity_column, cohort_column, activity_column) if c not in known]
    if missing:
        raise NotCohortable(f"Column(s) not found in {table}: {', '.join(missing)}")
    if period not in PERIODS:
        raise NotCohortable(f"period must be one of {', '.join(PERIODS)}")
    horizon = max(1, min(int(horizon), MAX_COHORT_HORIZON))

    entity = quote_identifier(entity_column)
    # TRY_CAST because a date is very often stored as text, and a column that
    # will not cast should say so rather than return an empty cohort.
    started = f"date_trunc('{period}', TRY_CAST({quote_identifier(cohort_column)} AS TIMESTAMP))"
    acted = f"date_trunc('{period}', TRY_CAST({quote_identifier(activity_column)} AS TIMESTAMP))"

    frame = dataset.run(f"""
        WITH base AS (
            SELECT {entity} AS entity, {started} AS cohort, {acted} AS active
            FROM {quote_identifier(table)}
            WHERE {entity} IS NOT NULL AND {started} IS NOT NULL
        ),
        sized AS (
            -- Every entity that started in the period, whether or not it ever
            -- came back. This is the base, and counting it here is the point.
            SELECT cohort, count(DISTINCT entity) AS cohort_size FROM base GROUP BY 1
        ),
        seen AS (
            SELECT cohort, date_diff('{period}', cohort, active) AS offset,
                   count(DISTINCT entity) AS active_entities
            FROM base WHERE active IS NOT NULL GROUP BY 1, 2
        )
        SELECT sized.cohort, sized.cohort_size, seen.offset, seen.active_entities
        FROM sized JOIN seen USING (cohort)
        WHERE seen.offset BETWEEN 0 AND {horizon}
        ORDER BY sized.cohort, seen.offset
    """).fetchdf()
    if frame.empty:
        raise NotCohortable(
            f"No cohort could be built: {cohort_column} did not parse as a date in any row. "
            "Check the column, or convert it first."
        )

    # Activity dated before the entity's own cohort. Reported rather than dropped
    # in silence: it is a real property of the data, and left unexplained the
    # answer reaches for a cause it cannot know.
    early = dataset.run(f"""
        SELECT count(DISTINCT {entity}) FROM {quote_identifier(table)}
        WHERE {acted} IS NOT NULL AND {started} IS NOT NULL AND {acted} < {started}
    """).fetchone()[0]

    cohorts = []
    for start, rows in frame.groupby("cohort", sort=True):
        size = int(rows["cohort_size"].iloc[0])
        cohorts.append(
            {
                "cohort": str(pd.Timestamp(start).date()),
                "size": size,
                "retention": [
                    {
                        "offset": int(row.offset),
                        "active": int(row.active_entities),
                        "rate": round(int(row.active_entities) / size, 4),
                    }
                    for row in rows.itertuples()
                ],
            }
        )

    result: dict[str, object] = {
        "entity": entity_column,
        "period": period,
        "cohorts_found": len(cohorts),
        # The most recent, not the first. An old cohort has the complete curve and
        # is the one nobody can act on, and taking from the front hid the cohort
        # actually being asked about behind two years of finished ones.
        "cohorts": cohorts[-MAX_COHORTS:],
        "reading": (
            f"rate is active {entity_column} values divided by that cohort's own size — "
            "everyone who started in the period, including those who first appear later. "
            "It is not a share of the entities active at offset 0, which is smaller and "
            "gives a retention curve that is wrong in a way nothing else shows."
        ),
    }
    if len(cohorts) > MAX_COHORTS:
        result["note"] = (
            f"{len(cohorts)} cohorts found; the most recent {MAX_COHORTS} are shown. "
            "The earlier ones are complete and unchanging."
        )
    if early:
        result["activity_before_the_cohort_started"] = {
            "entities": int(early),
            "note": (
                f"{int(early):,} {entity_column} values have activity dated before their own "
                f"{cohort_column}. That is a property of the data, not a finding — say it is "
                "there, and do not offer a reason for it that no query establishes."
            ),
        }
    return result
