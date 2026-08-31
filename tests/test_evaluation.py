"""The machinery a rate is measured with, tested without a model.

The banks need an endpoint and cannot run on every commit. What can — and what
has to, because a silent bookkeeping bug turns a run into a number nobody can
trust — is everything around them: that an outcome is recorded with the two facts
that make runs comparable, that a failure is recorded as loudly as a pass, that
the report groups by what actually moves a rate, and that a person's report of a
wrong answer survives to be replayed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import results

from smart_data_studio import feedback
from smart_data_studio.config import MODEL_ID, PROMPT_VERSION


@pytest.fixture
def destination(tmp_path, monkeypatch):
    path = tmp_path / "runs.jsonl"
    monkeypatch.setenv("SDS_BANK_RESULTS", str(path))
    return path


def test_nothing_is_written_unless_a_destination_is_set(tmp_path, monkeypatch) -> None:
    """An ordinary local run of the bank should leave no file behind."""
    monkeypatch.delenv("SDS_BANK_RESULTS", raising=False)
    results.record("domain", "How many?", True, 1.0)
    assert not list(tmp_path.iterdir())


def test_an_outcome_carries_what_makes_two_runs_comparable(destination) -> None:
    """A rate filed against neither the model nor the prompt version says that
    something moved without saying what moved it."""
    results.record("domain", "How many patients?", True, 2.5, domain="healthcare")
    entry = json.loads(destination.read_text().splitlines()[0])
    assert entry["model"] == MODEL_ID
    assert entry["prompt_version"] == PROMPT_VERSION
    assert (entry["bank"], entry["passed"], entry["domain"]) == ("domain", True, "healthcare")


def test_a_failure_is_recorded_as_readily_as_a_pass(destination) -> None:
    """A rate built only from passes is not a rate. This is the assertion that
    keeps the bank honest about its own denominator."""
    results.record("domain", "one", True, 1.0)
    results.record("domain", "two", False, 1.0)
    outcomes = [json.loads(line)["passed"] for line in destination.read_text().splitlines()]
    assert outcomes == [True, False]


def test_a_destination_that_cannot_be_written_does_not_fail_the_bank(monkeypatch) -> None:
    """Bookkeeping must never be the reason a bank run dies."""
    monkeypatch.setenv("SDS_BANK_RESULTS", "/does/not/exist/anywhere/runs.jsonl")
    results.record("domain", "How many?", True, 1.0)  # must not raise


def test_the_report_groups_by_what_moves_a_rate(tmp_path) -> None:
    """Two models over the same bank are two rates, and averaging them into one
    hides exactly the thing the report exists to show."""
    path = tmp_path / "runs.jsonl"
    rows = [
        {
            "bank": "domain",
            "question": "a",
            "passed": True,
            "seconds": 1,
            "model": "m1",
            "prompt_version": "p1",
        },
        {
            "bank": "domain",
            "question": "b",
            "passed": False,
            "seconds": 3,
            "model": "m1",
            "prompt_version": "p1",
        },
        {
            "bank": "domain",
            "question": "a",
            "passed": True,
            "seconds": 2,
            "model": "m2",
            "prompt_version": "p1",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    report = subprocess.run(
        [sys.executable, str(Path("tools/bank_report.py")), str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "m1" in report and "m2" in report
    assert "1/2" in report and "50%" in report, report
    assert "1/1" in report and "100%" in report, report
    # The failed question is named, because a rate without them is not actionable.
    assert "[domain] b" in report


def test_a_damaged_results_line_does_not_stop_the_report(tmp_path) -> None:
    """These files are appended to by long runs that can be killed mid-write."""
    path = tmp_path / "runs.jsonl"
    path.write_text(
        '{"bank": "domain", "question": "a", "passed": true, "seconds": 1}\n{"bank": "dom\n'
    )
    report = subprocess.run(
        [sys.executable, str(Path("tools/bank_report.py")), str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "1/1" in report


# --- what a person reports ----------------------------------------------------


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("SDS_STATE_DIR", str(tmp_path))
    return tmp_path / "feedback.jsonl"


def test_a_reported_answer_keeps_what_it_takes_to_replay_it(state) -> None:
    """The question, the SQL and the answer. A report saying only "this was wrong"
    is a feeling; these three make it a test case."""
    feedback.record(
        feedback.Report(
            question="What is the readmission rate?",
            answer="16.7% of encounters.",
            sql=["SELECT avg(readmitted_30d) FROM encounters"],
            note="per encounter, I asked per patient",
        )
    )
    entry = feedback.recall()[0]
    assert entry["question"] == "What is the readmission rate?"
    assert entry["sql"] == ["SELECT avg(readmitted_30d) FROM encounters"]
    assert entry["note"].startswith("per encounter")
    assert entry["prompt_version"] == PROMPT_VERSION and entry["model"] == MODEL_ID


def test_reports_do_not_grow_without_bound(state, monkeypatch) -> None:
    """A file nobody prunes is a file nobody opens."""
    monkeypatch.setattr(feedback, "MAX_KEPT", 3)
    for number in range(5):
        feedback.record(feedback.Report(question=f"q{number}", answer="a", sql=[]))
    kept = [entry["question"] for entry in feedback.recall()]
    assert kept == ["q2", "q3", "q4"]


def test_delete_my_data_takes_the_reports_with_it(state) -> None:
    """It holds questions and SQL carrying real cell values, so it has to be one of
    the things that control clears."""
    feedback.record(feedback.Report(question="q", answer="a", sql=["SELECT 1"]))
    assert state.is_file()
    feedback.forget()
    assert not state.exists()
    assert feedback.recall() == []
