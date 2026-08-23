"""Questions the five domain workspaces are built to be asked.

Each anchor carries the query that produced it, in the same entry, so the number
and its proof cannot drift apart — the single-CSV bank keeps them in two places
and needed a separate test to hold them together. Here `test_corpus.py` re-derives
every one of them in the fast suite, because these fixtures are small enough to
rebuild in milliseconds.

Anchored only where the figure is not a choice. A percentile is not: on this data
`quantile_cont` and `approx_quantile` differ by 2.5%, which is outside the
tolerance an answer is matched with, so latency questions are asked here and
checked by a test that reads what the answer did rather than which number it
landed on.
"""

from __future__ import annotations

# (domain, number, question, {sql that proves an anchor: its value})
BANK: list[tuple[str, int, str, dict[str, float]]] = [
    # --- healthcare: a patient has many encounters, so the denominator is a choice
    (
        "healthcare",
        1,
        "What share of encounters resulted in a readmission within 30 days?",
        {"SELECT avg(readmitted_30d) * 100 FROM encounters": 16.72},
    ),
    (
        "healthcare",
        2,
        "How many distinct patients were readmitted within 30 days at least once?",
        {
            "SELECT count(*) FROM (SELECT patient_id FROM encounters "
            "GROUP BY 1 HAVING max(readmitted_30d) = 1)": 39,
        },
    ),
    (
        "healthcare",
        3,
        "What is the total cost of all encounters?",
        {"SELECT sum(cost) FROM encounters": 1_553_008.70},
    ),
    # --- education: four sittings per student, so a mark is not an independent draw
    (
        "education",
        4,
        "What share of assessments were passed?",
        {"SELECT avg(passed) * 100 FROM assessments": 75.0},
    ),
    (
        "education",
        5,
        "What is the average assessment score for students on the honours programme?",
        {
            "SELECT avg(a.score) FROM assessments a JOIN students s USING (student_id) "
            "WHERE s.programme = 'honours'": 61.802,
        },
    ),
    # --- finance: the numerator and the denominator live on different tables
    (
        "finance",
        6,
        "What is the default rate among subprime accounts?",
        {
            "SELECT avg(CASE WHEN d.account_id IS NOT NULL THEN 100.0 ELSE 0 END) "
            "FROM accounts a LEFT JOIN defaults d USING (account_id) "
            "WHERE a.segment = 'subprime'": 43.75,
        },
    ),
    (
        "finance",
        7,
        "What is the total transaction amount for accounts on the overdraft product?",
        {
            "SELECT sum(t.amount) FROM accounts a JOIN transactions t USING (account_id) "
            "WHERE a.product = 'overdraft'": 101_754.51,
        },
    ),
    (
        "finance",
        8,
        "How many accounts are there in total, and how many of them defaulted?",
        {
            "SELECT count(*) FROM accounts": 100,
            "SELECT count(*) FROM defaults": 19,
        },
    ),
    # --- ecommerce: order value is skewed, and a refund rate has two denominators
    (
        "ecommerce",
        9,
        "What is the mean order value, and what is the median?",
        {
            "SELECT avg(value) FROM orders": 71.60,
            "SELECT quantile_cont(value, 0.5) FROM orders": 58.37,
        },
    ),
    (
        "ecommerce",
        10,
        "What share of orders from customers acquired through social were refunded?",
        {
            "SELECT count(r.order_id) * 100.0 / count(*) FROM orders o "
            "JOIN customers c USING (customer_id) LEFT JOIN refunds r USING (order_id) "
            "WHERE c.channel = 'social'": 17.33,
        },
    ),
    (
        "ecommerce",
        11,
        "How many customers placed more than one order?",
        {
            "SELECT count(*) FROM (SELECT customer_id FROM orders "
            "GROUP BY 1 HAVING count(*) > 1)": 125,
        },
    ),
    # --- operations: a long tail the mean sits well below
    (
        "operations",
        12,
        "What share of checkout requests failed?",
        {"SELECT avg(failed) * 100 FROM requests WHERE service = 'checkout'": 8.53},
    ),
    (
        "operations",
        13,
        "How many requests are in the data, and how many of them failed?",
        {
            "SELECT count(*) FROM requests": 900,
            "SELECT sum(failed) FROM requests": 29,
        },
    ),
]
