"""The live bank that needs no private data.

    USE_LLM=1 pytest tests/test_domain_bank.py -q

Unlike the other two banks there is no file to be missing: the workspaces are
built from a seed, so this runs on any machine with a model endpoint. That is the
point of it — the casino banks cannot be reproduced by anyone without the 2.7GB
file, so the only cross-domain evidence anybody could check was none.

Every anchor here is proved against its fixture in the fast suite by
`test_corpus.py`, so a failure below is the answer being wrong, never the number
in the bank having gone stale.
"""

from __future__ import annotations

import os

import pytest
from anchors import mentions
from corpus import DOMAINS
from domain_bank import BANK

from smart_data_studio.agent import DataAgent
from smart_data_studio.dataset import Dataset
from smart_data_studio.profile import profile_dataset

pytestmark = pytest.mark.skipif(
    os.environ.get("USE_LLM") != "1",
    reason="needs a live model; set USE_LLM=1 to run",
)


@pytest.fixture(scope="module")
def agents():
    """One workspace per domain, built on first use and closed at the end.

    Lazily, so `-k healthcare` explores one domain rather than five.
    """
    built: dict[str, DataAgent] = {}
    try:
        yield built
    finally:
        for agent in built.values():
            agent.dataset.close()


def agent_for(agents: dict[str, DataAgent], domain: str) -> DataAgent:
    if domain not in agents:
        dataset = Dataset.load(DOMAINS[domain]())
        agent = DataAgent(dataset, profile_dataset(dataset))
        agent.build_understanding()
        agents[domain] = agent
    return agents[domain]


@pytest.mark.parametrize(
    ("domain", "number", "question", "proofs"),
    BANK,
    ids=[f"{domain}{number:02d}-{domain}" for domain, number, _, _ in BANK],
)
def test_domain_bank(
    agents, domain: str, number: int, question: str, proofs: dict[str, float]
) -> None:
    agent = agent_for(agents, domain)
    answer = agent.ask(question, multi_turn=False, depth="never")

    assert answer.text.strip(), f"{domain} q{number}: empty answer"
    assert "could not finish" not in answer.text, f"{domain} q{number}: ran out of tool rounds"
    assert answer.results or answer.analyses, f"{domain} q{number}: answered with no evidence"
    for value in proofs.values():
        assert mentions(answer.text, value), (
            f"{domain} q{number}: expected {value:,.2f} in the answer\n\n{answer.text}"
        )


# --- the traps, which are why these fixtures are shaped the way they are --------


def test_a_readmission_rate_says_which_population_it_counted(agents) -> None:
    """16.7% of encounters or 32.5% of patients — both are right and they are not
    the same claim, so an answer that gives a figure and not its population has
    said something that cannot be checked."""
    agent = agent_for(agents, "healthcare")
    answer = agent.ask("What is the 30-day readmission rate?", multi_turn=False, depth="never")

    counted = []
    if mentions(answer.text, 16.72):
        counted.append("encounter")
    if mentions(answer.text, 32.5):
        counted.append("patient")
    assert counted, f"neither denominator's figure appears:\n{answer.text}"
    assert any(word in answer.text.lower() for word in counted), (
        "the rate is there and the population it counts is not:\n" + answer.text
    )


def test_a_significance_question_runs_a_test_rather_than_describing_the_gap(agents) -> None:
    """A baseline, and deliberately a weak one.

    Every student sits exactly four assessments, so the rows are not independent
    and any p-value from them is overstated. Nothing in the tool knows that yet —
    the entity half arrives with the grain contract — so what is pinned here is
    that asking whether a difference is real produces a test at all. When
    compare_groups learns about entities, this is where the stronger assertion
    goes.
    """
    agent = agent_for(agents, "education")
    answer = agent.ask(
        "Do honours students score higher than foundation students, and is the "
        "difference real or could it be noise?",
        multi_turn=False,
        depth="never",
    )
    comparisons = [record for record in answer.analyses if record.kind == "comparison"]
    assert comparisons, f"no statistical test was run:\n{answer.text}"
    assert "cliffs_delta" in comparisons[0].result["test"]


def test_typical_latency_is_not_answered_with_an_average(agents) -> None:
    """Mean 67.8ms, median 30.7ms, p99 1,024.8ms. An average is the one number
    that describes nobody's experience here, so the question is whether the answer
    asked for a quantile at all — checked in the SQL rather than in the prose,
    because which percentile it picks is its own business."""
    agent = agent_for(agents, "operations")
    answer = agent.ask(
        "What is typical latency for each service, and how bad does it get?",
        multi_turn=False,
        depth="never",
    )
    sql = " ".join(result.sql.lower() for result in answer.results)
    assert "quantile" in sql or "percentile" in sql or "median" in sql, (
        "typical latency was answered without ever asking for one:\n" + sql
    )


def test_a_yes_or_no_outcome_is_compared_as_a_rate(agents) -> None:
    """Measured before this tool existed: the model got the denominator right and
    cautioned in prose — "small sample size, which typically implies lower
    precision" — with no way to say 14 of 32 is anywhere from 28% to 61%. What it
    reached for was compare_groups, which reports Cliff's delta for a 0/1 column
    and called a sixfold difference in default risk medium.

    Asserted as "the tool was used, or the guard fired", because those are the two
    things this code controls. Whether the model follows the guard is measured by
    the bank rather than made into a probabilistic gate.
    """
    import sqlglot

    agent = agent_for(agents, "finance")
    question = "What is the default rate for subprime accounts, and how precise is that estimate?"
    answer = agent.ask(question, multi_turn=False, depth="never")
    assert mentions(answer.text, 43.75), f"the rate itself is wrong:\n{answer.text}"

    rates = [record for record in answer.analyses if record.kind == "rates"]
    if rates:
        subprime = next(
            group for group in rates[0].result["groups"] if group["group"] == "subprime"
        )
        assert (subprime["events"], subprime["observed"]) == (14, 32)
        low, high = subprime["interval_95_pct"]
        assert low < 30 and high > 58, f"the interval {low}-{high} carries no uncertainty"
        return

    warned = [
        result.sql
        for result in answer.results
        if agent.tools._rate_note(sqlglot.parse_one(result.sql, dialect="duckdb"))
    ]
    assert warned, (
        "The rate was calculated manually, but neither compare_rates nor the rate guard "
        "was reached:\n" + "\n".join(result.sql for result in answer.results)
    )
