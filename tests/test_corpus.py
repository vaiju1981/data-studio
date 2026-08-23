"""The domain fixtures load, and still carry the hazard they were built for.

Loading is the easy half. The half worth a test is that each workspace stays
*shaped*: a generator tidied later could produce five perfectly ordinary files
that a bank runs green over while proving nothing, and nobody would notice,
because every question would still have an answer.

So each domain asserts its own trap — the one a careless answer falls into — and
the numbers here are the reason the fixture exists rather than decoration.
"""

from __future__ import annotations

import pytest
from corpus import DOMAINS

from smart_data_studio.dataset import Dataset
from smart_data_studio.profile import profile_dataset


def workspace(name: str) -> Dataset:
    return Dataset.load(DOMAINS[name]())


def one(dataset: Dataset, sql: str) -> float:
    return float(dataset.query(sql).frame.iloc[0, 0])


@pytest.mark.parametrize("name", sorted(DOMAINS), ids=sorted(DOMAINS))
def test_a_domain_loads_and_profiles_without_the_private_file(name: str) -> None:
    """The point of building these: a bank over them needs no 2.7GB CSV, no
    self-hosted runner, and no data anybody has to be given access to."""
    dataset = workspace(name)
    try:
        assert len(dataset.tables) >= 2, f"{name} is not a multi-table workspace"
        profiles = profile_dataset(dataset)
        assert len(profiles) == len(dataset.tables)
        for profile in profiles:
            assert profile.row_count > 0
            assert len(profile.stats) == len(dataset.schema(profile.table_name))
            assert profile.findings, f"{name}.{profile.table_name} produced no findings"
    finally:
        dataset.close()


@pytest.mark.parametrize("name", sorted(DOMAINS), ids=sorted(DOMAINS))
def test_a_domain_is_built_the_same_way_twice(name: str) -> None:
    """Seeded, because an anchor derived from a fixture that moves is not an anchor."""
    first = [source.content for source in DOMAINS[name]()]
    second = [source.content for source in DOMAINS[name]()]
    assert first == second


def test_a_readmission_rate_depends_on_whether_it_counts_patients_or_encounters() -> None:
    """16.7% of encounters against 32.5% of patients — the same file, and a factor
    of two between them. A rate quoted without saying which is not imprecise, it
    is a different number."""
    dataset = workspace("healthcare")
    try:
        per_encounter = one(dataset, "SELECT avg(readmitted_30d) FROM encounters")
        per_patient = one(
            dataset,
            "SELECT avg(ever) FROM (SELECT patient_id, max(readmitted_30d) AS ever "
            "FROM encounters GROUP BY 1)",
        )
        assert per_patient > per_encounter * 1.5, (
            "the fixture no longer separates the two denominators, so the trap is gone"
        )
        # And the clustering that causes it: patients return, so encounters are not
        # independent observations of a patient.
        encounters = one(dataset, "SELECT count(*) FROM encounters")
        patients = one(dataset, "SELECT count(DISTINCT patient_id) FROM encounters")
        assert encounters > patients * 2
    finally:
        dataset.close()


def test_a_pass_rate_depends_on_whether_it_counts_students_or_sittings() -> None:
    """The same shape in another domain, and the plainest clustering case there is:
    every student sits exactly four assessments."""
    dataset = workspace("education")
    try:
        per_sitting = one(dataset, "SELECT avg(passed) FROM assessments")
        per_student = one(
            dataset,
            "SELECT avg(ok) FROM (SELECT student_id, max(passed) AS ok "
            "FROM assessments GROUP BY 1)",
        )
        assert per_student > per_sitting * 1.1
        assert one(dataset, "SELECT count(*) FROM assessments") == 4 * one(
            dataset, "SELECT count(DISTINCT student_id) FROM assessments"
        )
    finally:
        dataset.close()


def test_a_default_rate_is_per_account_and_transactions_are_not_accounts() -> None:
    """A three-file chain where the denominator is on a different table from the
    numerator, so the join is where the rate goes wrong."""
    dataset = workspace("finance")
    try:
        accounts = one(dataset, "SELECT count(*) FROM accounts")
        defaulted = one(dataset, "SELECT count(*) FROM defaults")
        transactions = one(dataset, "SELECT count(*) FROM transactions")
        assert 0 < defaulted < accounts
        # Counting defaults over transactions instead is off by an order of
        # magnitude, which is the mistake worth having a fixture for.
        assert transactions > accounts * 10

        subprime = one(
            dataset,
            "SELECT avg(CASE WHEN d.account_id IS NOT NULL THEN 1.0 ELSE 0.0 END) "
            "FROM accounts a LEFT JOIN defaults d USING (account_id) WHERE a.segment = 'subprime'",
        )
        prime = one(
            dataset,
            "SELECT avg(CASE WHEN d.account_id IS NOT NULL THEN 1.0 ELSE 0.0 END) "
            "FROM accounts a LEFT JOIN defaults d USING (account_id) WHERE a.segment = 'prime'",
        )
        assert subprime > prime * 2, "there is nothing for a rate comparison to find"
    finally:
        dataset.close()


def test_order_value_is_skewed_so_the_mean_is_not_the_typical_order() -> None:
    dataset = workspace("ecommerce")
    try:
        mean = one(dataset, "SELECT avg(value) FROM orders")
        median = one(dataset, "SELECT quantile_cont(value, 0.5) FROM orders")
        assert mean > median * 1.2, "a symmetric fixture asks nothing of a distribution tool"

        refunds = one(dataset, "SELECT count(*) FROM refunds")
        orders = one(dataset, "SELECT count(*) FROM orders")
        customers = one(dataset, "SELECT count(DISTINCT customer_id) FROM orders")
        # Refund rate over orders and over customers are again two numbers.
        assert 0 < refunds < orders and customers < orders
    finally:
        dataset.close()


def test_latency_has_a_tail_the_mean_hides() -> None:
    """Mean 67.8ms, median 30.7ms, p99 1,024.8ms. Anyone paged at three in the
    morning is looking at the last of those and none of the first two."""
    dataset = workspace("operations")
    try:
        row = dataset.query(
            "SELECT avg(latency_ms) AS mean, quantile_cont(latency_ms, 0.5) AS p50, "
            "quantile_cont(latency_ms, 0.99) AS p99 FROM requests"
        ).frame.iloc[0]
        assert row["mean"] > row["p50"] * 1.5
        assert row["p99"] > row["mean"] * 5, "the tail is what the fixture is for"

        checkout = one(dataset, "SELECT avg(failed) FROM requests WHERE service = 'checkout'")
        rest = one(dataset, "SELECT avg(failed) FROM requests WHERE service <> 'checkout'")
        assert checkout > rest * 2
    finally:
        dataset.close()
