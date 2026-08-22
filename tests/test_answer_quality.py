"""Whether an answer's prose survives its own evidence.

Both banks anchor on values, so they prove the arithmetic and the guards. They
cannot catch the failure that prompted this file: an answer whose every figure was
correct, whose retention was a share of the wrong population, and which named a
cause the data could not establish. Nothing in an anchor reads a sentence.

Two halves, and the order matters. Calibration goes first: planted answers whose
faults are known, so a judge that cannot separate a clean answer from a broken one
fails here rather than quietly grading real work. Only then are real answers
graded, on questions chosen because they invite the fault.

Skipped unless the CSV is present and USE_LLM=1 is set:

    USE_LLM=1 pytest tests/test_answer_quality.py -q
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from judge import describe, evidence_of, failures, grade

from smart_data_studio.agent import DataAgent
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_dataset

CSV = Path("~/ga_cache/training_data/STATION_playerVisits_REDROCK.csv").expanduser()

live = [
    pytest.mark.skipif(not CSV.is_file(), reason=f"visits file not present at {CSV}"),
    pytest.mark.skipif(
        os.environ.get("USE_LLM") != "1",
        reason="slow and needs a live model; set USE_LLM=1 to run",
    ),
]

pytestmark = live


# --- calibration: does the judge know a fault when it is handed one? -------------

# Deliberately domain-neutral. The judge is being tested, not the data, and a
# fixture written in one corpus's vocabulary would test how well it reads that.
EVIDENCE = """Query 1:
SELECT strftime('%Y-%m', order_date) AS month, count(DISTINCT customer_id) AS active
FROM orders WHERE signup_month = '2026-01' GROUP BY 1 ORDER BY 1
Returned 3 rows. First rows:
  month  active
2026-01    6780
2026-02     844
2026-03     492"""

CLEAN = (
    "Activity fell sharply after the first month. Of the 6,780 customers active in "
    "January, 844 were active in February and 492 in March. These counts are active "
    "customers per month, not a share of the cohort — the query never established how "
    "many customers signed up in January, so the drop is measured against January's "
    "actives rather than against the cohort itself."
)


def _said(quote: str, answer: str) -> bool:
    """Whether the judge quoted the answer rather than paraphrasing it."""
    squash = lambda text: re.sub(r"\s+", " ", text).strip().lower()  # noqa: E731
    return bool(quote.strip()) and squash(quote) in squash(answer)


# A claim one division away from the evidence. The judge faulted this as an
# invented cause on a real answer, reasoning that the evidence "does not
# explicitly state" the derived figure — which would fault an analyst for doing
# arithmetic. Kept as a fixture so the rubric cannot drift back.
DERIVED = (
    "The top cities dominate through frequency rather than spend: they take "
    "6,628,194 visits from 206,698 players against 1,228,904 from 255,527, so a "
    "player there visits several times more often than one elsewhere."
)

DERIVED_EVIDENCE = """Query 1:
SELECT segment, count(*) AS visits, count(DISTINCT customer_id) AS players
FROM trips GROUP BY 1
Returned 2 rows. All 2 are reproduced here:
  segment   visits  players
      top  6628194   206698
     rest  1228904   255527"""

PLANTED: list[tuple[str, str]] = [
    (
        "invented_cause",
        "The February collapse was driven by a pricing change that pushed customers to "
        "competitors, and the March recovery reflects the discount campaign taking hold.",
    ),
    (
        "unstated_base",
        "Retention dropped to 12.4% in February and settled near 7% by March, a steep "
        "fall by any standard.",
    ),
    (
        "unsupported_figure",
        "Activity fell from 6,780 in January to 844 in February, and lifetime value for "
        "the cohort averaged $431.20 per customer.",
    ),
    (
        "overstated_coverage",
        "Across all 2.4 million orders in the table, every customer who signed up in "
        "January followed the same pattern without exception.",
    ),
]


@pytest.mark.parametrize(("fault", "answer"), PLANTED, ids=[item[0] for item in PLANTED])
def test_the_judge_catches_a_planted_fault(fault: str, answer: str) -> None:
    """If the judge cannot find a fault written to be found, its verdict on a real
    answer means nothing.

    The quote is checked against the answer rather than against a keyword. Which
    words a judge picks out of an offending sentence is its own business; that the
    words are in the answer at all is the property worth holding it to, because a
    judge that paraphrases can invent a fault out of nothing.
    """
    graded = grade("How is the January cohort doing month over month?", answer, EVIDENCE)

    assert graded[fault].failed, f"{fault} was planted and not found: {describe(graded)}"
    assert _said(graded[fault].quote, answer), (
        f"{fault} was not quoted from the answer: {graded[fault].quote!r}"
    )


@pytest.mark.parametrize(
    ("label", "answer", "evidence"),
    [("plain", CLEAN, EVIDENCE), ("derived", DERIVED, DERIVED_EVIDENCE)],
    ids=["plain", "derived"],
)
def test_the_judge_leaves_a_sound_answer_alone(label: str, answer: str, evidence: str) -> None:
    """The other half of calibration, and the half that keeps it honest: a judge
    that fails everything catches every planted fault and is still useless.

    The derived case is the one that bit. A claim resting on arithmetic over two
    figures in the evidence is supported, and a judge demanding the arithmetic be
    spelled out faults the analyst for doing their job."""
    graded = grade("How do the segments compare?", answer, evidence)
    assert not failures(graded), f"{label}: a sound answer was faulted: {describe(graded)}"


# --- the real thing: questions chosen because they invite the fault -------------


@pytest.fixture(scope="module")
def agent():
    dataset = Dataset.load([CsvSource.from_path(CSV)])
    try:
        built = DataAgent(dataset, profile_dataset(dataset))
        built.build_understanding()
        yield built
    finally:
        dataset.close()


# Each of these tempts a specific fault. A "why" question invites a cause the data
# cannot show; a "share" question invites a denominator that is never named; a
# "compare" question invites describing part of a result as the whole.
TEMPTING: list[tuple[str, str]] = [
    ("cohort", "How are the players who first registered in January 2026 doing month over month?"),
    ("why", "Why did net win fall in the worst month of the data?"),
    ("share", "What share of players are high value, and how has that share moved?"),
    ("cause", "What is driving the difference in theo win between club levels?"),
    ("coverage", "Compare the top cities by coin in against the rest of the estate."),
]


@pytest.mark.parametrize(("label", "question"), TEMPTING, ids=[item[0] for item in TEMPTING])
def test_a_real_answer_survives_its_own_evidence(agent, label: str, question: str) -> None:
    answer = agent.ask(question, multi_turn=False, depth="never")
    assert answer.text.strip(), f"{label}: empty answer"

    graded = grade(question, answer.text, evidence_of(answer))
    assert not failures(graded), f"{label}: {describe(graded)}\n\nANSWER:\n{answer.text}"
