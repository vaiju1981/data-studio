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
            -- Every row of the entity, whether or not it carries a start date. An
            -- event table commonly writes the signup on the first row and leaves
            -- it blank after, and requiring it here dropped every later row: two
            -- users with January signups and February activity came back with
            -- offset 0 alone, a retention curve missing the retention.
            SELECT {entity} AS entity, {started} AS cohort, {acted} AS active
            FROM {quote_identifier(table)}
            WHERE {entity} IS NOT NULL
        ),
        starts AS (
            -- One cohort per entity, and the earliest one. A file carrying the
            -- start date on every activity row can disagree with itself, and the
            -- raw per-row value put a single entity in January and again in
            -- February — counted in two cohort sizes and two sets of numerators,
            -- which is the one thing a cohort must never be. An entity with no
            -- start date at all is in no cohort, which is what the join below does.
            SELECT entity, min(cohort) AS cohort FROM base
            WHERE cohort IS NOT NULL GROUP BY 1
        ),
        sized AS (
            -- Every entity that started in the period, whether or not it ever
            -- came back. This is the base, and counting it here is the point.
            SELECT cohort, count(*) AS cohort_size FROM starts GROUP BY 1
        ),
        seen AS (
            SELECT starts.cohort, date_diff('{period}', starts.cohort, base.active) AS offset,
                   count(DISTINCT base.entity) AS active_entities
            FROM base JOIN starts USING (entity)
            WHERE base.active IS NOT NULL GROUP BY 1, 2
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

    # Both measured against the cohort the entity was actually placed in — its
    # earliest — rather than against whatever start its own row happened to carry.
    # Reported rather than dropped in silence: each is a real property of the data,
    # and left unexplained the answer reaches for a cause it cannot know.
    early, conflicting = dataset.run(f"""
        WITH per_entity AS (
            SELECT {entity} AS entity, min({started}) AS first_start,
                   min({acted}) AS first_activity,
                   count(DISTINCT {started}) AS starts
            FROM {quote_identifier(table)}
            WHERE {entity} IS NOT NULL
            GROUP BY 1
        )
        SELECT count(*) FILTER (WHERE first_activity < first_start),
               count(*) FILTER (WHERE starts > 1)
        FROM per_entity
    """).fetchone()

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
    if conflicting:
        result["entities_with_more_than_one_start"] = {
            "entities": int(conflicting),
            "note": (
                f"{int(conflicting):,} {entity_column} value(s) carry more than one "
                f"{cohort_column} period. Each was placed in its earliest, so it appears in "
                f"one cohort only — but the later starts are still in the file, and if they "
                f"mean something (a re-registration, a second account) the cohort they belong "
                f"to is a choice this tool made rather than one the data settled."
            ),
        }
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
