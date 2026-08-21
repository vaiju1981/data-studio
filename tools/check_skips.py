"""Fail if a test outside the opt-in live-model suite is skipping.

A count would have to be bumped every time a bank gains a question. The invariant
that actually matters is different: the fast suite must never go quiet. Only the
live-model banks may skip, and only because they need a model.

    pytest -q --junitxml=results.xml && python tools/check_skips.py results.xml
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import Path

# Both banks, not one. The multi-table bank was added opt-in like the first and
# never added here, so this gate has been failing CI on every commit since —
# reporting the second bank's own skips as the fast suite going quiet.
ALLOWED = ("test_question_bank", "test_multi_table_bank")


def main(report: str) -> int:
    root = ElementTree.parse(Path(report)).getroot()
    unexpected: Counter[str] = Counter()
    total = skipped = 0
    for case in root.iter("testcase"):
        total += 1
        if case.find("skipped") is None:
            continue
        skipped += 1
        source = f"{case.get('classname', '')}.{case.get('name', '')}"
        if not any(allowed in source for allowed in ALLOWED):
            unexpected[case.get("classname", "?")] += 1

    banks = " or ".join(ALLOWED)
    print(f"{skipped} skipped of {total}; {sum(unexpected.values())} outside {banks}")
    if unexpected:
        for where, count in unexpected.most_common():
            print(f"  {count} skipped in {where}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results.xml"))
