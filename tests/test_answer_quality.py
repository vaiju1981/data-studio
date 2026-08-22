"""Whether an answer's prose survives its own evidence.

Both banks anchor on values, so they prove the arithmetic and the guards. They
cannot catch the failure that prompted this file: an answer whose every figure was
correct, whose retention was a share of the wrong population, and which named a
cause the data could not establish. Nothing in an anchor reads a sentence.

Two halves, and the order matters. Calibration goes first: planted answers whose
faults are known, so a judge that cannot separate a clean answer from a broken one
fails here rather than quietly grading real work. Only then are real answers
graded, on questions chosen because they invite the fault — and graded as a rate,
because one answer is a coin toss and this bank is here to catch a fall, not to
police which side it lands on.

Skipped unless the CSV is present and USE_LLM=1 is set:

    USE_LLM=1 pytest tests/test_answer_quality.py -q
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest
from judge import JudgeUnusable, describe, evidence_of, failures, grade, quotes

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

# A decomposition wearing the word "because". Every figure it names is in the
# evidence and the two tiers account for the whole movement, so this says where
# the change sits rather than what caused it. The judge faulted it three times out
# of three until the rubric said the word does not decide the fault — what follows
# it does, and whether that is in the evidence.
DECOMPOSED = (
    "Revenue fell from 20,000,000 in January to 8,468,037 in April because the top tier "
    "fell from 16,000,000 to 4,468,037; mid was unchanged at 4,000,000."
)

DECOMPOSED_EVIDENCE = """Query 1:
SELECT month, sum(revenue) AS revenue FROM sales GROUP BY 1 ORDER BY 1
Returned 3 rows. All 3 are reproduced here:
   month   revenue
 2026-01  20000000
 2026-02  16353518
 2026-04   8468037

Query 2:
SELECT month, tier, sum(revenue) AS revenue FROM sales GROUP BY 1, 2 ORDER BY 1, 2
Returned 6 rows. All 6 are reproduced here:
   month tier   revenue
 2026-01  mid   4000000
 2026-01  top  16000000
 2026-02  mid   4000000
 2026-02  top  12353518
 2026-04  mid   4000000
 2026-04  top   4468037"""


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

    assert graded[fault].failed, f"{fault} was planted and not found: {describe(graded, answer)}"
    assert quotes(graded[fault].quote, answer), (
        f"{fault} was not quoted from the answer: {graded[fault].quote!r}"
    )


# The question travels with the fixture: a sound answer graded against a question
# it does not answer is a different test, and a confusing one to read when it fails.
SOUND: list[tuple[str, str, str, str]] = [
    ("plain", "How is the January cohort doing month over month?", CLEAN, EVIDENCE),
    ("derived", "How do the segments compare?", DERIVED, DERIVED_EVIDENCE),
    ("decomposed", "Why did revenue fall?", DECOMPOSED, DECOMPOSED_EVIDENCE),
]


@pytest.mark.parametrize(
    ("label", "question", "answer", "evidence"), SOUND, ids=[item[0] for item in SOUND]
)
def test_the_judge_leaves_a_sound_answer_alone(
    label: str, question: str, answer: str, evidence: str
) -> None:
    """The other half of calibration, and the half that keeps it honest: a judge
    that fails everything catches every planted fault and is still useless.

    The derived case is the one that bit. A claim resting on arithmetic over two
    figures in the evidence is supported, and a judge demanding the arithmetic be
    spelled out faults the analyst for doing their job.

    The decomposed case is the same mistake wearing a different word: "because the
    top tier collapsed from 16,000,000 to 4,468,037" names a group the evidence
    holds and quotes its figures. Saying where a change sits is the job too."""
    graded = grade(question, answer, evidence)
    assert not failures(graded, answer), (
        f"{label}: a sound answer was faulted: {describe(graded, answer)}"
    )


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


# Measured before it was chosen, over three sessions: 11/15, 2/2, 6/15 — 19 of 32
# clean, and a spread from 40% to 100%. The coverage question went 0/3 in one
# session and 2/2 in the next, an hour apart, with both later answers plainly
# good. A per-question pass/fail was reading that as a regression.
#
# The faults are real, which was worth checking rather than assuming: the phrases
# the bank objected to — "tend to be higher-value per trip", "driven by volume
# rather than the rate" — were replayed against evidence that supported them and
# the judge left all of them alone, four times out of four, while still catching
# an invented one four times out of four. These questions are built to tempt a
# fault and the model takes the bait about half the time.
#
# So the floor is a collapse detector and nothing more. At fifteen samples, with a
# true rate that moves between 40% and 73% session to session, no tighter bar is
# stable — 50% failed a run whose answers were ordinary. What you read is the rate
# and the breakdown, printed on every run; what fails the test is the system
# ceasing to work at all.
RUNS_PER_QUESTION = 3
CLEAN_FLOOR = 0.2
# Above this share of unquotable verdicts the judge is not grading, and a rate
# taken from what is left would be a number about nothing.
UNUSABLE_CEILING = 0.34


def test_real_answers_survive_their_evidence_at_a_rate(agent) -> None:
    clean: Counter[str] = Counter()
    graded_count: Counter[str] = Counter()
    found: list[str] = []
    broken: list[str] = []
    unusable: list[str] = []

    for label, question in TEMPTING:
        for run in range(RUNS_PER_QUESTION):
            answer = agent.ask(question, multi_turn=False, depth="never")
            # A turn that failed is not a bad answer, it is no answer, and it is
            # kept out of the rate rather than averaged into it: explain_failure
            # returns readable prose, prose with no claims in it grades clean, and
            # an outage would otherwise raise the score.
            reason = _did_not_answer(answer)
            if reason:
                broken.append(f"{label} run {run}: {reason}")
                continue
            try:
                graded = grade(question, answer.text, evidence_of(answer))
                faults = failures(graded, answer.text)
            except JudgeUnusable as error:
                # A fault the judge cannot quote is still never scored as clean —
                # that contract is the point of the layer. It is set aside rather
                # than allowed to end the run, because one paraphrased quote in
                # fifteen should not cost the other fourteen measurements.
                unusable.append(f"{label} run {run}: {error}")
                continue
            graded_count[label] += 1
            if faults:
                found.append(f"{label} run {run}: {describe(graded, answer.text)}")
            else:
                clean[label] += 1

    assert not broken, "turns that produced no answer at all:\n" + "\n".join(broken)

    total = sum(graded_count.values())
    attempted = total + len(unusable)
    # A judge that cannot quote most of what it finds is not grading, and no rate
    # taken from the remainder would mean anything.
    assert total and len(unusable) <= attempted * UNUSABLE_CEILING, (
        f"{len(unusable)} of {attempted} verdicts could not be quoted from the answer, "
        f"which is more than the judge is allowed to leave unusable:\n" + "\n".join(unusable)
    )

    rate = sum(clean.values()) / total
    # Printed rather than only asserted: a rate that holds while the faults move
    # is a change worth seeing, and `pytest -s` is where you see it.
    print(f"\nanswers surviving their evidence: {sum(clean.values())}/{total} ({rate:.0%})")
    for line in found:
        print(f"  {line}")
    breakdown = "\n".join(
        f"  {label}: {clean[label]}/{graded_count[label]}" for label, _ in TEMPTING
    )
    aside = (
        f"\n{len(unusable)} verdict(s) set aside as unquotable:\n" + "\n".join(unusable)
        if unusable
        else ""
    )
    assert rate >= CLEAN_FLOOR, (
        f"{sum(clean.values())} of {total} graded answers survived their evidence "
        f"({rate:.0%}, floor {CLEAN_FLOOR:.0%})\n{breakdown}\n\n" + "\n".join(found) + aside
    )


def _did_not_answer(answer) -> str:
    """Why this turn produced nothing to grade, or empty when it produced something."""
    if not answer.text.strip():
        return "empty answer"
    if "could not finish" in answer.text:
        return "ran out of tool rounds"
    if "could not be completed" in answer.text:
        return "the turn raised"
    if not (answer.results or answer.analyses):
        return "answered with no evidence"
    return ""
