"""The judge must never turn silence into a pass.

The whole layer is worth nothing if a judge that says nothing reads as a judge
that found nothing — and that is not hypothetical, it is what happened the first
time it ran: the endpoint ignored the requested format, answered in YAML, the
parser found no JSON, and every fault defaulted to clean. Ten graded answers, no
findings, and the harness looked like it was working.

So the contract is tested with a stubbed judge rather than a live one. No model,
no network, deterministic, and it runs in the fast suite — because "does a broken
judge fail loudly" is a property worth holding on every commit, not one worth
checking on the days somebody sets USE_LLM=1.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from judge import RUBRIC, Finding, JudgeUnusable, describe, failures, grade, quotes

ANSWER = "Revenue rose to $4.2M in March, driven by the new pricing tier."


class Stub:
    """A judge that says whatever it is told to say."""

    def __init__(self, text: str):
        self.text = text

    def chat(self, **_kwargs):
        return SimpleNamespace(message=SimpleNamespace(content=self.text))


def clean_reply() -> str:
    findings = ", ".join(
        f'{{"dimension": "{name}", "failed": false, "quote": "", "why": ""}}' for name in RUBRIC
    )
    return f'{{"findings": [{findings}]}}'


@pytest.mark.parametrize(
    ("label", "reply"),
    [
        ("prose instead of json", "The answer looks fine to me, no problems found."),
        ("yaml, which is what the endpoint actually returned", "invented_cause:\n  failed: false"),
        ("empty", ""),
        ("valid json, no findings key", '{"result": "ok"}'),
        ("findings present but empty", '{"findings": []}'),
        (
            "only some of the faults ruled on",
            '{"findings": [{"dimension": "invented_cause", "failed": false, '
            '"quote": "", "why": ""}]}',
        ),
        (
            "a dimension that is not in the rubric",
            '{"findings": [{"dimension": "vibes", "failed": false, "quote": "", "why": ""}]}',
        ),
    ],
)
def test_a_judge_that_says_nothing_usable_is_not_a_pass(label: str, reply: str) -> None:
    with pytest.raises(JudgeUnusable):
        grade("Why did revenue rise?", ANSWER, "Query 1: ...", client=Stub(reply))


def test_a_complete_verdict_is_accepted() -> None:
    """The other direction, so the guard above cannot pass by refusing everything."""
    graded = grade("Why did revenue rise?", ANSWER, "Query 1: ...", client=Stub(clean_reply()))
    assert set(graded) == set(RUBRIC)
    assert not failures(graded, ANSWER)


def test_a_fault_the_judge_cannot_quote_is_not_a_pass() -> None:
    """Dropping it hides a fault; counting it trusts a judge that may have invented
    one. Neither is honest, so it asks for a human instead."""
    for quote in ("", "   ", "the answer blamed a competitor"):
        graded = {name: Finding(name, False, "", "") for name in RUBRIC}
        graded["invented_cause"] = Finding("invented_cause", True, quote, "made up")
        with pytest.raises(JudgeUnusable):
            failures(graded, ANSWER)


def test_a_fault_quoted_from_the_answer_counts() -> None:
    graded = {name: Finding(name, False, "", "") for name in RUBRIC}
    graded["invented_cause"] = Finding(
        "invented_cause", True, "driven by the new pricing tier", "no query shows pricing"
    )
    found = failures(graded, ANSWER)
    assert [item.dimension for item in found] == ["invented_cause"]


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("driven by the new pricing tier", True),
        ("DRIVEN BY THE NEW PRICING TIER", True),  # case is the judge's business
        ("driven   by the\nnew pricing tier", True),  # so is whitespace
        ("driven by the old pricing tier", False),
        ("", False),
    ],
)
def test_a_quote_must_come_from_the_answer(quote: str, expected: bool) -> None:
    assert quotes(quote, ANSWER) is expected


def test_describe_never_throws_while_a_test_is_already_failing() -> None:
    """It runs inside assertion messages, so a describe() that raises replaces the
    finding you needed to read with a TypeError — and only when something failed,
    which is the one moment it matters."""
    graded = {name: Finding(name, False, "", "") for name in RUBRIC}
    assert describe(graded, ANSWER) == "clean"

    graded["invented_cause"] = Finding("invented_cause", True, "driven by the new pricing tier", "")
    assert "invented_cause" in describe(graded, ANSWER)

    # An unquotable fault would raise in failures(); describe still has to render.
    graded["unstated_base"] = Finding("unstated_base", True, "something never written", "")
    rendered = describe(graded, ANSWER)
    assert "NOT QUOTED" in rendered, rendered
