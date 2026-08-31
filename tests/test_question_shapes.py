"""The SQL these question classes are actually written as, run through run_sql.

Written because a review claimed six question classes — period over period, mix
shift, funnels, what-if, segmentation and data quality — were out of reach of the
twelve tools, on the evidence of the tool list alone. Six of the eight shapes
below run today and return the right number, so that claim was wrong: `run_sql`
takes window functions, QUALIFY, NTILE, CTEs and set operations, and a class of
question is not out of reach for wanting a tool of its own.

What the same probes did find is four queries the join guard judges wrongly, and
those are the second half of this file: three correct queries it refuses, and one
double-counting query it allows. The last is the failure this repo exists to
prevent — a total inflated fourfold, returned with no warning on it.

No model anywhere here. Every query is one a person could type, and what is
pinned is what the machinery does with it — which is the layer that can be
asserted on rather than measured as a rate.
"""

from __future__ import annotations

import json

import pytest
from corpus import DOMAINS

from smart_data_studio.dataset import Dataset
from smart_data_studio.tools import AnalysisTools


@pytest.fixture(scope="module")
def workspaces():
    """One workspace per domain, built on first use and closed at the end."""
    built: dict[str, AnalysisTools] = {}
    try:
        yield built
    finally:
        for tools in built.values():
            tools.dataset.close()


def tools_for(workspaces: dict[str, AnalysisTools], domain: str) -> AnalysisTools:
    if domain not in workspaces:
        workspaces[domain] = AnalysisTools(Dataset.load(DOMAINS[domain]()))
    return workspaces[domain]


def run(tools: AnalysisTools, sql: str) -> dict:
    """The tool's own JSON, with a refusal turned into a readable failure."""
    payload = json.loads(tools.run_sql(sql))
    if "error" in payload:
        pytest.fail(f"run_sql refused this query:\n{payload['error']}\n\n{sql}")
    return payload


def value(tools: AnalysisTools, sql: str) -> float:
    """A figure proved directly against the workspace, for anchoring an answer."""
    return float(tools.dataset.query(sql).frame.iloc[0, 0])


# --- the shapes that already work ---------------------------------------------


def test_period_over_period_runs_as_a_window_over_a_cte(workspaces) -> None:
    """ "How did revenue move month on month" — LAG over a monthly CTE. No tool
    knows what a month is; run_sql does not need one to."""
    tools = tools_for(workspaces, "ecommerce")
    payload = run(
        tools,
        """
        WITH monthly AS (
            SELECT date_trunc('month', CAST(day AS DATE)) AS month, sum(value) AS revenue
            FROM orders GROUP BY month
        )
        SELECT month, revenue,
               lag(revenue) OVER (ORDER BY month) AS prior,
               100.0 * (revenue - lag(revenue) OVER (ORDER BY month))
                     / lag(revenue) OVER (ORDER BY month) AS pct_change
        FROM monthly ORDER BY month
        """,
    )
    assert payload["row_count"] == 8
    rows = payload["rows"]
    assert round(rows[0]["revenue"], 2) == 4184.60
    assert rows[0]["prior"] is None, "the first month has nothing to compare against"
    assert round(rows[1]["prior"], 2) == 4184.60
    assert round(rows[1]["pct_change"], 2) == round(100.0 * (4081.54 - 4184.60) / 4184.60, 2)


def test_a_funnel_runs_as_distinct_counts_down_the_steps(workspaces) -> None:
    """Opened, then transacted, then defaulted. A funnel is conditional distinct
    counts, and count(DISTINCT ...) is exactly what the join guard leaves alone."""
    tools = tools_for(workspaces, "finance")
    payload = run(
        tools,
        """
        SELECT count(DISTINCT a.account_id) AS opened,
               count(DISTINCT t.account_id) AS transacted,
               count(DISTINCT d.account_id) AS defaulted
        FROM accounts a
        LEFT JOIN transactions t USING (account_id)
        LEFT JOIN defaults d ON d.account_id = a.account_id
        """,
    )
    assert payload["rows"][0] == {"opened": 100, "transacted": 100, "defaulted": 19}


def test_a_what_if_runs_as_scenario_rows_over_a_base_cte(workspaces) -> None:
    """ "What if average order value rose 10%" — one CTE and a UNION ALL, which the
    SQL guard admits as a read-only root."""
    tools = tools_for(workspaces, "ecommerce")
    payload = run(
        tools,
        """
        WITH base AS (SELECT count(*) AS orders, avg(value) AS aov FROM orders)
        SELECT 'actual' AS scenario, orders * aov AS revenue FROM base
        UNION ALL SELECT 'aov +10%', orders * aov * 1.10 FROM base
        UNION ALL SELECT 'orders +10%', orders * 1.10 * aov FROM base
        """,
    )
    actual = value(tools, "SELECT sum(value) FROM orders")
    scenarios = {row["scenario"]: row["revenue"] for row in payload["rows"]}
    assert round(scenarios["actual"], 2) == round(actual, 2)
    assert round(scenarios["aov +10%"], 2) == round(actual * 1.10, 2)


def test_segmentation_runs_as_ntile_over_a_spend_cte(workspaces) -> None:
    """ "Split the customers into value quartiles" — NTILE, then a profile per band."""
    tools = tools_for(workspaces, "ecommerce")
    payload = run(
        tools,
        """
        WITH spend AS (SELECT customer_id, sum(value) AS total FROM orders GROUP BY customer_id),
             banded AS (SELECT customer_id, total, ntile(4) OVER (ORDER BY total) AS quartile
                        FROM spend)
        SELECT quartile, count(*) AS customers, round(sum(total), 2) AS revenue
        FROM banded GROUP BY quartile ORDER BY quartile
        """,
    )
    assert [row["quartile"] for row in payload["rows"]] == [1, 2, 3, 4]
    assert payload["rows"][0] == {"quartile": 1, "customers": 38, "revenue": 2570.56}
    assert round(sum(row["revenue"] for row in payload["rows"]), 2) == round(
        value(tools, "SELECT sum(value) FROM orders"), 2
    ), "the quartiles must partition the revenue, not sample it"


def test_top_n_per_group_runs_with_qualify(workspaces) -> None:
    """DuckDB's QUALIFY parses through sqlglot and survives the guard, so the
    largest order per customer needs no subquery."""
    tools = tools_for(workspaces, "ecommerce")
    payload = run(
        tools,
        """
        SELECT customer_id, value FROM orders
        QUALIFY row_number() OVER (PARTITION BY customer_id ORDER BY value DESC) = 1
        """,
    )
    assert payload["row_count"] == value(tools, "SELECT count(DISTINCT customer_id) FROM orders")


def test_data_quality_is_askable_as_an_ordinary_query(workspaces) -> None:
    """ "Is this data trustworthy" needs no tool either: nulls, unparseable dates and
    duplicate keys are counts."""
    tools = tools_for(workspaces, "operations")
    payload = run(
        tools,
        """
        SELECT count(*) AS rows,
               count(*) - count(latency_ms) AS missing_latency,
               count(*) - count(TRY_CAST(day AS DATE)) AS unparseable_day,
               count(DISTINCT request_id) AS distinct_ids
        FROM requests
        """,
    )
    assert payload["rows"][0] == {
        "rows": 900,
        "missing_latency": 0,
        "unparseable_day": 0,
        "distinct_ids": 900,
    }


# --- what the probes actually found: four queries the join guard judges wrongly -


REVENUE_BY_CHANNEL = {
    "direct": 11086.32,
    "email": 7247.79,
    "search": 10254.81,
    "social": 10001.61,
}


def test_a_group_by_ordinal_does_not_change_whether_a_query_is_allowed(workspaces) -> None:
    """`GROUP BY 1` and `GROUP BY customer_id` are the same query, and the second
    is admitted while the first is refused.

    The grain of a CTE is read off its GROUP BY as `alias_or_name`, which for an
    ordinal is the string "1" — a column no join is ever on, so the side never
    proves its grain. Ordinals are how this SQL is usually written.
    """
    tools = tools_for(workspaces, "ecommerce")
    named = run(
        tools,
        """
        WITH per AS (SELECT customer_id, sum(value) AS spend FROM orders GROUP BY customer_id)
        SELECT c.channel, round(sum(per.spend), 2) AS revenue
        FROM customers c JOIN per USING (customer_id) GROUP BY c.channel
        """,
    )
    ordinal = run(
        tools,
        """
        WITH per AS (SELECT customer_id, sum(value) AS spend FROM orders GROUP BY 1)
        SELECT c.channel, round(sum(per.spend), 2) AS revenue
        FROM customers c JOIN per USING (customer_id) GROUP BY 1
        """,
    )
    assert {row["channel"]: row["revenue"] for row in named["rows"]} == REVENUE_BY_CHANNEL
    assert ordinal["rows"] == named["rows"]


def test_a_join_between_two_grain_proved_ctes_runs(workspaces) -> None:
    """Mix shift, retention and share-of-total are all written this way: reduce
    each side to its own grain in a CTE, then join the two.

    Both sides here prove their grain — one row per customer, one row per customer
    and channel — so the output grain is provable. `_join_multiplication` returns
    None for any pair of derived relations before measuring either.
    """
    tools = tools_for(workspaces, "ecommerce")
    payload = run(
        tools,
        """
        WITH per AS (SELECT customer_id, sum(value) AS spend FROM orders GROUP BY customer_id),
             cust AS (SELECT DISTINCT customer_id, channel FROM customers)
        SELECT cust.channel, round(sum(per.spend), 2) AS revenue
        FROM cust JOIN per USING (customer_id) GROUP BY cust.channel
        """,
    )
    assert {row["channel"]: row["revenue"] for row in payload["rows"]} == REVENUE_BY_CHANNEL


def test_a_refusal_never_names_a_table_that_is_not_a_side_of_the_join(workspaces) -> None:
    """The same query as above, and the reason given for refusing it is about
    `orders` — a table inside a CTE body, not a side of the join at all.

    `sources_in` puts CTEs and the base tables *inside* them in one namespace, and
    `_using_sides` then picks the far side by which name carries the column,
    in text order. A steer this wrong costs every remaining round.
    """
    tools = tools_for(workspaces, "ecommerce")
    payload = json.loads(
        tools.run_sql(
            """
            WITH per AS (SELECT customer_id, sum(value) AS spend FROM orders GROUP BY customer_id),
                 cust AS (SELECT DISTINCT customer_id, channel FROM customers)
            SELECT cust.channel, sum(per.spend) AS revenue
            FROM cust JOIN per USING (customer_id) GROUP BY cust.channel
            """
        )
    )
    error = payload.get("error", "")
    assert "orders" not in error, (
        "the refusal blames a table that is not one of the two joined relations:\n" + error
    )


def test_a_cross_join_to_a_one_row_total_is_not_refused_as_unconditioned(workspaces) -> None:
    """Share of total, written the ordinary way: a comma join onto a relation of
    exactly one row. It carries no condition because it needs none — one row
    cannot multiply anything — and it is refused as "a join with no condition"."""
    tools = tools_for(workspaces, "ecommerce")
    payload = run(
        tools,
        """
        WITH per AS (SELECT c.channel, sum(o.value) AS revenue
                     FROM orders o JOIN customers c USING (customer_id) GROUP BY c.channel),
             tot AS (SELECT sum(revenue) AS total FROM per)
        SELECT per.channel, round(100.0 * per.revenue / tot.total, 2) AS pct
        FROM per, tot ORDER BY per.channel
        """,
    )
    assert round(sum(row["pct"] for row in payload["rows"]), 1) == 100.0


def test_a_fan_out_between_two_ctes_is_not_silently_totalled(workspaces) -> None:
    """The one that matters: a double-counting query the guard allows.

    Every customer's largest order, summed across a join that repeats each of
    those rows once per day that customer ordered. The total comes back 4.08x too
    big, with no warning attached — the guard measured this join against
    `customers`, which is unique on customer_id and is not a side of it.

    A refusal or a warning both pass here. Returning the number alone does not.
    """
    tools = tools_for(workspaces, "ecommerce")
    payload = json.loads(
        tools.run_sql(
            """
            WITH known AS (SELECT customer_id, channel FROM customers),
                 cust_max AS (SELECT customer_id, max(value) AS biggest
                              FROM orders GROUP BY customer_id),
                 per_day AS (SELECT customer_id, day, count(*) AS n
                             FROM orders GROUP BY customer_id, day)
            SELECT sum(cust_max.biggest) AS total_biggest
            FROM per_day JOIN cust_max USING (customer_id)
            """
        )
    )
    if "error" in payload:
        return  # refused before it ran, which is the guard working
    if any(key.endswith("_warning") for key in payload):
        return  # ran, and said the total is weighted by the join

    truth = value(
        tools,
        "SELECT sum(biggest) FROM (SELECT customer_id, max(value) AS biggest "
        "FROM orders GROUP BY customer_id)",
    )
    returned = payload["rows"][0]["total_biggest"]
    pytest.fail(
        f"a double-counted total was returned with nothing said about it: "
        f"{returned:,.2f} against a true {truth:,.2f}, inflated {returned / truth:.2f}x"
    )


def test_a_cte_referenced_under_an_alias_is_still_the_same_relation(workspaces) -> None:
    """`FROM firsts f JOIN acts a` — the aliases the join actually names.

    CTEs were registered only under their definition names, so `f` and `a`
    resolved to nothing, the near side came back unknown, and every retention
    query written this way was refused.
    """
    tools = tools_for(workspaces, "ecommerce")
    aliased = run(
        tools,
        """
        WITH firsts AS (SELECT customer_id, min(date_trunc('month', CAST(day AS DATE))) AS cohort
                        FROM orders GROUP BY customer_id),
             acts AS (SELECT DISTINCT customer_id, date_trunc('month', CAST(day AS DATE)) AS month
                      FROM orders)
        SELECT f.cohort, count(DISTINCT a.customer_id) AS active
        FROM firsts f JOIN acts a USING (customer_id) GROUP BY f.cohort ORDER BY f.cohort
        """,
    )
    plain = run(
        tools,
        """
        WITH firsts AS (SELECT customer_id, min(date_trunc('month', CAST(day AS DATE))) AS cohort
                        FROM orders GROUP BY customer_id),
             acts AS (SELECT DISTINCT customer_id, date_trunc('month', CAST(day AS DATE)) AS month
                      FROM orders)
        SELECT firsts.cohort, count(DISTINCT acts.customer_id) AS active
        FROM firsts JOIN acts USING (customer_id) GROUP BY firsts.cohort ORDER BY firsts.cohort
        """,
    )
    assert aliased["rows"] == plain["rows"]
    # Everyone who ever ordered is active in the month they first ordered.
    assert sum(row["active"] for row in aliased["rows"]) >= value(
        tools, "SELECT count(DISTINCT customer_id) FROM orders"
    )


def test_a_cohort_fan_out_counted_distinctly_is_not_refused(workspaces) -> None:
    """Retention is a fan-out by construction — one cohort row per customer against
    many active months — and its measure is a distinct count, which repetition
    cannot alter. The guard bailed before it ever looked at the aggregate."""
    tools = tools_for(workspaces, "ecommerce")
    payload = run(
        tools,
        """
        WITH firsts AS (SELECT customer_id, min(date_trunc('month', CAST(day AS DATE))) AS cohort
                        FROM orders GROUP BY customer_id),
             acts AS (SELECT DISTINCT customer_id, date_trunc('month', CAST(day AS DATE)) AS month
                      FROM orders)
        SELECT f.cohort, datediff('month', f.cohort, a.month) AS months_since,
               count(DISTINCT a.customer_id) AS active
        FROM firsts f JOIN acts a USING (customer_id)
        GROUP BY f.cohort, months_since ORDER BY f.cohort, months_since
        """,
    )
    month_zero = [row for row in payload["rows"] if row["months_since"] == 0]
    assert sum(row["active"] for row in month_zero) == value(
        tools, "SELECT count(DISTINCT customer_id) FROM orders"
    ), "every customer is active in their own first month, and each is counted once"


def test_a_share_of_a_period_total_runs_against_a_totals_cte(workspaces) -> None:
    """Mix shift, in the shape it is actually written: segment rows joined to the
    period totals they are a share of. `t` is one row per period and cannot
    multiply anything; `p` is many per period and multiplies `t`, which only MAX
    reads."""
    tools = tools_for(workspaces, "ecommerce")
    payload = run(
        tools,
        """
        WITH p AS (SELECT CASE WHEN CAST(o.day AS DATE) < DATE '2026-07-01'
                               THEN 'before' ELSE 'after' END AS period,
                          c.channel, count(o.order_id) AS orders
                   FROM orders o JOIN customers c USING (customer_id)
                   GROUP BY period, c.channel),
             t AS (SELECT period, sum(orders) AS total FROM p GROUP BY period)
        SELECT p.period, p.channel, p.orders,
               round(100.0 * p.orders / max(t.total), 2) AS share
        FROM p JOIN t USING (period)
        GROUP BY p.period, p.channel, p.orders ORDER BY p.period, p.channel
        """,
    )
    assert payload["row_count"] == 8, "four channels in each of two periods"
    for period in ("before", "after"):
        shares = [row["share"] for row in payload["rows"] if row["period"] == period]
        assert round(sum(shares), 1) == 100.0, f"the {period} shares must be of one total"
