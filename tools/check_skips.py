"""Fail if a test outside the opt-in live-model suite is skipping.

A count would have to be bumped every time the bank gains a question. The
invariant that actually matters is different: the fast suite must never go quiet.
Only `test_question_bank.py` may skip, and only because it needs a model.

    pytest -q --junitxml=results.xml && python tools/check_skips.py results.xml
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import Path

ALLOWED = "test_question_bank"


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
        if ALLOWED not in source:
            unexpected[case.get("classname", "?")] += 1

    print(f"{skipped} skipped of {total}; {sum(unexpected.values())} outside {ALLOWED}")
    if unexpected:
        for where, count in unexpected.most_common():
            print(f"  {count} skipped in {where}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results.xml"))
