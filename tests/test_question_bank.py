"""Opt-in regression run of the question bank against the real visits file.

Slow (loads 2.7GB and calls a live model), so it is skipped unless both the CSV
is present and USE_LLM=1 is set:

    USE_LLM=1 pytest tests/test_question_bank.py -v

Run it before shipping a change to prompts, tools or the profile. Assertions
anchor on numbers verified independently in SQL rather than on wording, so the
model is free to phrase an answer however it likes and still pass.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from smart_data_studio.agent import DataAgent
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_dataset

CSV = Path("~/ga_cache/training_data/STATION_playerVisits_REDROCK.csv").expanduser()

pytestmark = [
    pytest.mark.skipif(not CSV.is_file(), reason=f"visits file not present at {CSV}"),
    pytest.mark.skipif(
        os.environ.get("USE_LLM") != "1",
        reason="slow and needs a live model; set USE_LLM=1 to run",
    ),
]


@pytest.fixture(scope="module")
def agent():
    dataset = Dataset.load([CsvSource.from_path(CSV)])
    try:
        built = DataAgent(dataset, profile_dataset(dataset))
        built.build_understanding()
        yield built
    finally:
        dataset.close()


def mentions(text: str, value: float, tolerance: float = 0.01) -> bool:
    """Is this number in the answer, however it happens to be formatted?

    Matches 126.31, 126,310,000 and 126.31M alike, so a change in phrasing does
    not fail the run while a change in the arithmetic does.
    """
    found = [float(token) for token in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]
    scales = (value, value / 1e3, value / 1e6, value / 1e9)
    return any(
        abs(number - scale) <= abs(scale) * tolerance for number in found for scale in scales
    )


def test_understanding_covers_the_basics(agent) -> None:
    """Only the scale is guaranteed; which other facts make the summary varies by run."""
    text = agent.understanding
    assert mentions(text, 7_857_098) or mentions(text, 460_442)


@pytest.mark.parametrize(
    ("label", "question", "anchors"),
    [
        (
            "profile totals",
            "How many visits and how many distinct players are in the data?",
            [7_857_098, 460_442],
        ),
        (
            "ytd comparison",
            "How is theo win year to date this year compared to last year same period?",
            [126.31e6, 126.84e6],
        ),
        (
            "win-back",
            "Which players do we need for a win back campaign? Use a 90 day lapse.",
            [],
        ),
        (
            "forecast",
            "Based on past monthly theo win, what are the next 12 months predictions?",
            [22.08e6],
        ),
        (
            "trend",
            "Is total monthly theo win trending up or down across the available history?",
            [],
        ),
        (
            "daily anomalies",
            "Which individual days had unusual total theo win?",
            [],
        ),
        (
            "club level split",
            "How many visits are there at each club level?",
            [3_652_373],
        ),
    ],
)
def test_question_bank_answers_stay_correct(agent, label, question, anchors) -> None:
    answer = agent.ask(question)
    assert answer.text and "could not be completed" not in answer.text, label
    assert answer.results or answer.analyses, f"{label}: no evidence produced"
    for value in anchors:
        assert mentions(answer.text, value), f"{label}: expected {value:,.0f} in the answer"


def test_win_back_groups_by_player_alone(agent) -> None:
    """The original bug: GROUP BY playerId, lastVisit split one player across rows."""
    answer = agent.ask(
        "Which players do we need for a win back campaign? Use a 90 day lapse "
        "and rank by lifetime coin in."
    )
    sql = " ".join(result.sql for result in answer.results).lower()
    assert "group by" in sql
    assert "lastvisit" not in sql.split("group by")[-1].split("order by")[0]
    # The lapse window must be measured from the data, not from today.
    assert "current_date" not in sql


def test_ageGroup_trap_is_still_caught(agent) -> None:
    """Anchor on the verdict, not the route: it reaches this by more than one path."""
    answer = agent.ask(
        "Is the ageGroup field trustworthy for segmentation? Check the distinct "
        "values and their date ranges before answering."
    )
    # The SQL is the stable part; the wording of the verdict moves between runs.
    sql = " ".join(result.sql for result in answer.results).lower()
    assert "agegroup" in sql, "it never looked at the field"

    verdict = answer.text.lower()
    problems = (
        "not trustworthy",
        "untrustworthy",
        "not reliable",
        "unreliable",
        "inconsistent",
        "shift",
        "overlap",
        "changed",
        "caution",
    )
    assert any(word in verdict for word in problems), (
        f"the overlapping age bands were not flagged: {answer.text[:200]}"
    )


def test_partial_months_are_still_excluded_from_a_forecast(agent) -> None:
    answer = agent.ask("Forecast monthly theo win for the next 6 months.")
    forecasts = [item for item in answer.analyses if item.kind == "forecast"]
    assert forecasts, "no forecast analysis was recorded"
    result = forecasts[0].result
    assert result["periods_used"] <= 24, "a partial month slipped into the model"
    # The honesty check must always travel with the forecast.
    assert "verdict" in result["accuracy"] or "note" in result["accuracy"]
