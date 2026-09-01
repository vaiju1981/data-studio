"""Run the live-model banks on this machine, and report what they measured.

CI runs the fast suite, which needs no model. These need one, and the endpoint is
wherever `SDS_OLLAMA_HOST` points — in practice a laptop. That is the whole reason
they are not on GitHub's runners: a hosted runner can reach neither the endpoint
nor the private file, and exposing one so that it could would be a production
decision taken for the sake of a test.

    python tools/banks.py              # every bank that can run here
    python tools/banks.py domain       # just the reproducible one

Outcomes are appended to bank-results.jsonl with the model and prompt version that
produced them, so two runs a month apart can be compared rather than remembered.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
# Ordered cheapest first, and the reproducible one first of all: if the endpoint
# is wrong, four minutes says so rather than forty.
BANKS = {
    "domain": "tests/test_domain_bank.py",
    "single": "tests/test_question_bank.py",
    "multi": "tests/test_multi_table_bank.py",
    "quality": "tests/test_answer_quality.py",
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(BANKS)
    unknown = [name for name in wanted if name not in BANKS]
    if unknown:
        print(f"Unknown bank(s): {', '.join(unknown)}. Choose from: {', '.join(BANKS)}")
        return 2

    results = ROOT / "bank-results.jsonl"
    environment = {**os.environ, "USE_LLM": "1", "SDS_BANK_RESULTS": str(results)}
    host = environment.get("SDS_OLLAMA_HOST", "http://localhost:11434")
    print(f"Model {environment.get('SDS_MODEL_ID', '(default)')} at {host}\n")

    failed = []
    for name in wanted:
        print(f"--- {name} ---", flush=True)
        # Streamed rather than captured: these take minutes each, and a run with no
        # sign of life is one people kill.
        finished = subprocess.run(
            [sys.executable, "-m", "pytest", BANKS[name], "-q", "-s"], cwd=ROOT, env=environment
        )
        if finished.returncode:
            failed.append(name)

    print()
    subprocess.run([sys.executable, str(ROOT / "tools/bank_report.py"), str(results)], cwd=ROOT)
    if failed:
        # Named rather than left in the scrollback: a bank is allowed to fail a
        # question, and which bank it was decides whether that matters.
        print(f"\nBanks with failures: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
