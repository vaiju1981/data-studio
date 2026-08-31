"""What the recorded bank runs say, grouped by the things that change a rate.

    SDS_BANK_RESULTS=runs.jsonl USE_LLM=1 pytest tests/test_domain_bank.py -q
    python tools/bank_report.py runs.jsonl

Grouped by model and prompt version rather than only by date, because those are
what move a rate. A run that cannot say which model produced it cannot be
compared with anything.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    found = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and "bank" in entry:
            found.append(entry)
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"No results at {path}")
        return 1
    entries = load(path)
    if not entries:
        print(f"{path} holds no bank results")
        return 1

    tallies: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0.0])
    for entry in entries:
        key = (entry.get("model", "?"), entry.get("prompt_version", "?"), entry["bank"])
        tally = tallies[key]
        tally[0] += 1
        tally[1] += bool(entry.get("passed"))
        tally[2] += float(entry.get("seconds") or 0)

    width = max(len(f"{model} · {prompt}") for model, prompt, _ in tallies)
    print(f"{'model · prompt':{width}}  {'bank':22} {'rate':>12}  {'mean':>7}")
    for (model, prompt, bank), (total, passed, seconds) in sorted(tallies.items()):
        rate = passed / total
        print(
            f"{model + ' · ' + prompt:{width}}  {bank:22} "
            f"{passed:>4}/{total:<4} {rate:>3.0%}  {seconds / total:>6.1f}s"
        )

    failed = [entry for entry in entries if not entry.get("passed")]
    if failed:
        print(f"\n{len(failed)} failed, most recent first:")
        for entry in list(reversed(failed))[:15]:
            print(f"  [{entry['bank']}] {entry['question'][:88]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
