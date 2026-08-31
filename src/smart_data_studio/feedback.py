"""Answers a person marked wrong, kept so the bank can be built from real ones.

The banks are questions somebody imagined going wrong. This is the other kind:
questions that actually did. Every entry is written because a person pressed a
button on a specific answer, and it holds what would be needed to reproduce it —
the question, the SQL that ran, and the answer given.

That is a change from `recent.py`, which stores paths and never data: a recorded
SQL statement carries the cell values in its filters, and the answer carries
figures from the file. So it is written only on an explicit click, never
automatically, it stays on this machine, and **Delete my data** removes it with
everything else. The README says so too, because a promise that only holds in a
docstring is not a promise.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from smart_data_studio import logs
from smart_data_studio.config import MODEL_ID, PROMPT_VERSION, VERSION

# Enough to work through in a sitting. Older entries fall off the front rather
# than growing a file nobody prunes.
MAX_KEPT = 200


def _store() -> Path:
    root = os.environ.get("SDS_STATE_DIR") or str(Path.home() / ".smart-data-studio")
    return Path(root) / "feedback.jsonl"


@dataclass(frozen=True)
class Report:
    """One answer somebody marked wrong."""

    question: str
    answer: str
    sql: list[str]
    note: str = ""

    def as_entry(self) -> dict[str, object]:
        return {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "version": VERSION,
            # The two things that decide whether a replay is comparable: a bank
            # result filed against neither says a rate moved without saying what
            # moved it.
            "prompt_version": PROMPT_VERSION,
            "model": MODEL_ID,
            "question": self.question,
            "answer": self.answer,
            "sql": self.sql,
            "note": self.note,
        }


def record(report: Report) -> None:
    """Append one report, keeping the file to its last MAX_KEPT entries."""
    store = _store()
    store.parent.mkdir(parents=True, exist_ok=True)
    kept = recall()[-(MAX_KEPT - 1) :] if store.exists() else []
    kept.append(report.as_entry())
    store.write_text("\n".join(json.dumps(entry) for entry in kept) + "\n")
    # The event, never the content: the log leaves the host and this file does not.
    logs.event("feedback.recorded", kept=len(kept))


def recall() -> list[dict]:
    """Everything recorded, oldest first. A damaged line is skipped, not fatal."""
    try:
        lines = _store().read_text().splitlines()
    except (OSError, ValueError):
        return []
    found = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            found.append(entry)
    return found


def forget() -> None:
    """Delete the file. Part of what Delete my data means."""
    _store().unlink(missing_ok=True)
