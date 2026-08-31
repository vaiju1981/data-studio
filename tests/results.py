"""Where a bank run's outcome goes, so two runs can be compared.

A pass rate that is printed and then scrolls away answers "did it pass today".
The questions that matter are "is it worse than last month", "did that prompt
change cost anything", and "which model is the floor" — and none of them can be
asked of a number nobody kept.

One JSON object per question, appended to the file named by SDS_BANK_RESULTS.
Unset, nothing is written and the banks behave exactly as before, which is what
keeps this out of the way of an ordinary local run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from smart_data_studio.config import MODEL_ID, PROMPT_VERSION, VERSION


def record(bank: str, question: str, passed: bool, seconds: float, **extra: object) -> None:
    """Append one question's outcome. Never raises: a bank must not fail because
    its bookkeeping could not be written."""
    destination = os.environ.get("SDS_BANK_RESULTS")
    if not destination:
        return
    entry = {
        "bank": bank,
        "question": question,
        "passed": passed,
        "seconds": round(seconds, 2),
        # The three things that decide whether two runs are comparable at all.
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "version": VERSION,
        **extra,
    }
    try:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass
