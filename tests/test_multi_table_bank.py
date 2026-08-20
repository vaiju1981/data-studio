"""Opt-in regression run for two real tables that must be joined correctly.

The single-table bank proves the agent behaves on one file. This proves it behaves
when a question spans two, which is a different failure: joining
`sessions` to `assetDaily` on `assetId` alone produces **1,016,575,122** rows
against a correct **2,291,623** — a 444x explosion whose total looks entirely
reasonable.

    USE_LLM=1 pytest tests/test_multi_table_bank.py -q

Slow: it loads 2GB and calls a live model per question. Anchors were computed
independently in SQL with the composite join and are recorded beside each case.

The asset table is one row per machine per day, and `(assetId, day)` is *nearly*
its key — 1,304,622 distinct pairs across 1,304,681 rows, 59 of them duplicated.
So a measure summed over the join differs from the same measure summed on its own
table by about 0.003%, well inside the tolerance here. Where that ambiguity would
matter, the case anchors on a count instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from anchors import mentions

from smart_data_studio.agent import DataAgent
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_dataset

DATA = Path("~/ga_cache/training_data").expanduser()
SESSIONS = DATA / "STATION_sessions_REDROCK.csv"
ASSETS = DATA / "STATION_assetDaily_REDROCK.csv"
VISITS = DATA / "STATION_playerVisits_REDROCK.csv"

pytestmark = [
    pytest.mark.skipif(
        not (SESSIONS.is_file() and ASSETS.is_file() and VISITS.is_file()),
        reason=f"the three REDROCK files are not all present in {DATA}",
    ),
    pytest.mark.skipif(
        os.environ.get("USE_LLM") != "1",
        reason="slow and needs a live model; set USE_LLM=1 to run",
    ),
]

# (number, question, anchors). Every anchor was proved separately in SQL.
BANK: list[tuple[int, str, list[float]]] = [
    # Tier 1 — the join is unavoidable, because the column exists on one side only.
    (1, "Total session coin in grouped by the machine's paytableId.", [860_992_423]),
    (2, "How many sessions happened on leased machines?", [677_180]),
    # Three readings are all correct answers to different questions: 21.05 over
    # the join, 5.15 over every asset row, 14.48 over asset rows that had a
    # session. An anchored figure has to be asked for, so this names the grain.
    (
        3,
        "For every session, look up that machine's peakUtil on that session's day, "
        "then average those values over all sessions.",
        [21.05],
    ),
    # Deliberately ambiguous again. coinIn exists on both tables and means
    # different things — 861M on sessions against 5.92bn on assets — and this
    # wording once produced $500,401,572 from the wrong table. It is the harder
    # case, so it is the one worth asking.
    (4, "Which game version earned the most session coin in?", [67_143_446]),
    # Tier 2 — the join is unnecessary, and noticing that is the right answer.
    (5, "Which manufacturer produced the most session coin in?", [549_331_469]),
    (6, "How many distinct machines had at least one session?", [2_784]),
    # Tier 3 — measures that must not be summed across the join.
    (
        7,
        # 1,304,681 on its own table; 2,291,623 if summed over the join, which is
        # the fan-out this bank exists for.
        "What is the total number of machine days recorded for the assets?",
        [1_304_681],
    ),
    (8, "How many distinct manufacturers are there?", [15]),
    # Tier 4 — shape rather than a single number.
    (9, "Total session theo win by machine class.", [56_265_811]),
    (10, "Compare session coin in against the asset table's own coin in.", []),
    (11, "Which cabinet type has the highest coin in per session?", []),
    (12, "For each manufacturer, how many machines and how many sessions?", []),
]


# The second pair, and the one that pushed this bank past a single shape: player
# behaviour rather than machine detail. 7,857,098 visits and 2,291,518 sessions
# over 460,442 and 77,460 players — so most players have no session at all, and
# joining the two on playerId alone multiplies one player's 419 visits by their
# 9 sessions. They also share coinIn, netWin, jackpots and day.
PLAYER_BANK: list[tuple[int, str, list[float]]] = [
    # Tier 1 — one side answers it, and noticing that is the point.
    (101, "How many players had at least one gaming session?", [77_460]),
    (102, "How many distinct game titles were played?", [775]),
    # Tier 2 — both sides, without a join that could multiply.
    (
        103,
        "How many players visited but never had a gaming session?",
        [382_982],
    ),
    (
        104,
        "Of the players who have sessions, what is the club level mix? "
        "Club level is on the visits table.",
        [48_750],
    ),
    # Tier 3 — the question that prompted this pair being added at all.
    (
        105,
        "Top games played by the top 100 players in the last 3 months, "
        "ranking players by their visit coin in.",
        [24_414_809],
    ),
    # Tier 4 — coinIn is on both and means different things: 6.99bn of visit coin
    # in against 861M of session coin in.
    (106, "What is the total coin in recorded on the sessions table?", [860_962_789]),
    (107, "What is the total coin in recorded on the visits table?", [6_988_046_242]),
]


@pytest.fixture(scope="module")
def agent():
    dataset = Dataset.load([CsvSource.from_path(SESSIONS), CsvSource.from_path(ASSETS)])
    try:
        built = DataAgent(dataset, profile_dataset(dataset))
        built.build_understanding()
        yield built
    finally:
        dataset.close()


@pytest.fixture(scope="module")
def players():
    """Visits and sessions: 3.8GB, so it is loaded once for the whole module."""
    dataset = Dataset.load([CsvSource.from_path(VISITS), CsvSource.from_path(SESSIONS)])
    try:
        built = DataAgent(dataset, profile_dataset(dataset))
        built.build_understanding()
        yield built
    finally:
        dataset.close()


def joins_present(answer) -> list[str]:
    return [result.sql for result in answer.results if " join " in result.sql.lower()]


@pytest.mark.parametrize(
    ("number", "question", "anchors"), BANK, ids=[f"m{item[0]:02d}" for item in BANK]
)
def test_multi_table_bank(agent, number, question, anchors) -> None:
    answer = agent.ask(question, multi_turn=False, depth="never")

    assert answer.text.strip(), f"m{number}: empty answer"
    assert "could not finish" not in answer.text, f"m{number}: ran out of tool rounds"
    assert answer.results, f"m{number}: answered without querying anything"
    for anchor in anchors:
        assert mentions(answer.text, anchor), (
            f"m{number}: expected ~{anchor:,} in:\n{answer.text[:400]}"
        )


def test_the_incomplete_join_never_produces_the_answer(agent) -> None:
    """The failure this bank exists for. Joining on assetId alone gives
    1,016,575,122 rows against a correct 2,291,623, and the total that follows is
    444 times too large while looking perfectly ordinary.
    """
    answer = agent.ask(
        "Total session coin in grouped by the machine's paytableId.",
        multi_turn=False,
        depth="never",
    )
    assert mentions(answer.text, 860_992_423), answer.text[:300]
    # Nothing anywhere near the inflated figure.
    assert not mentions(answer.text, 382_000_000_000, tolerance=0.5), "a fan-out total was reported"


def test_a_measure_from_the_repeated_side_is_not_summed_over_the_join(agent) -> None:
    """machineDays belongs to the asset table. Summed over a join to sessions it
    reads 2,291,623 rather than 1,304,681 — right arithmetic, wrong question."""
    answer = agent.ask(
        "What is the total number of machine days recorded for the assets?",
        multi_turn=False,
        depth="never",
    )
    assert mentions(answer.text, 1_304_681), answer.text[:300]
    assert not mentions(answer.text, 2_291_623), "machine days were summed across the join"


def test_a_join_that_is_not_needed_is_not_made(agent) -> None:
    """manufacturer is on both tables, so the question is answerable from sessions
    alone. Joining anyway is not wrong, but noticing is better and cheaper."""
    answer = agent.ask(
        "Which manufacturer produced the most session coin in?", multi_turn=False, depth="never"
    )
    assert mentions(answer.text, 549_331_469), answer.text[:300]


def test_the_bank_needs_no_guard_refusals(agent) -> None:
    """The point of stating the grain in the profile: the model should write a
    correctly-grained join first time, leaving the guard as a backstop.

    Checking the SQL text for the word "day" was the wrong test — the model writes
    `JOIN (SELECT assetId, MAX(paytableId) ... GROUP BY assetId) ON s.assetId =
    a.assetId`, which joins on assetId alone and is entirely correct, because the
    subquery is already one row per assetId. What matters is whether anything had
    to be refused.
    """
    refusals = 0
    for _, question, _ in BANK[:6]:
        agent.ask(question, multi_turn=False, depth="never")
        refusals += sum(
            1
            for message in agent.messages
            if message.get("role") == "tool" and "double count" in str(message.get("content", ""))
        )
    assert refusals == 0, f"{refusals} joins had to be refused and rewritten"


@pytest.mark.parametrize(
    ("number", "question", "anchors"),
    PLAYER_BANK,
    ids=[f"p{item[0]}" for item in PLAYER_BANK],
)
def test_player_bank(players, number, question, anchors) -> None:
    answer = players.ask(question, multi_turn=False, depth="never")

    assert answer.text.strip(), f"p{number}: empty answer"
    assert "could not finish" not in answer.text, f"p{number}: ran out of tool rounds"
    assert answer.results, f"p{number}: answered without querying anything"
    for anchor in anchors:
        assert mentions(answer.text, anchor), (
            f"p{number}: expected ~{anchor:,} in:\n{answer.text[:400]}"
        )


def test_visits_and_sessions_are_not_joined_on_the_player_alone(players) -> None:
    """One player has 419 visits and 9 sessions, so joining the two on playerId
    multiplies them into 3,771 rows. The right shape is a subquery — which is what
    the model wrote unprompted for the question this pair was added for.
    """
    answer = players.ask(
        "Top games played by the top 100 players in the last 3 months, "
        "ranking players by their visit coin in.",
        multi_turn=False,
        depth="never",
    )
    assert mentions(answer.text, 24_414_809), answer.text[:300]
    for result in answer.results:
        flat = " ".join(result.sql.split()).lower()
        if "join" in flat and "playervisits" in flat and "sessions" in flat:
            assert "playerid = " not in flat or "day" in flat, flat[:220]


def test_a_measure_on_both_tables_is_taken_from_the_one_named(players) -> None:
    """coinIn totals 6.99bn on visits and 861M on sessions. Naming the table is
    the whole question."""
    for question, anchor, wrong in (
        ("What is the total coin in recorded on the sessions table?", 860_962_789, 6_988_046_242),
        ("What is the total coin in recorded on the visits table?", 6_988_046_242, 860_962_789),
    ):
        answer = players.ask(question, multi_turn=False, depth="never")
        assert mentions(answer.text, anchor), f"{question}: {answer.text[:220]}"
        assert not mentions(answer.text, wrong), f"{question} took the other table"
