"""Finding a known number in an answer, however the model chose to write it.

Shared by both question banks. The model is free to phrase and format an answer
however it likes and still pass, which is the whole point of anchoring on values
proved separately in SQL — but that freedom includes writing "549.33 million"
where the anchor is 549,331,469, and a check that only reads digits scores that
as a miss on an answer that is exactly right.
"""

from __future__ import annotations

import re

SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "m": 1e6,
    "mn": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
}

# A number, then optionally a magnitude word attached to it.
_NUMBER = re.compile(
    r"(-?[\d,]*\.?\d+)\s*(" + "|".join(sorted(SCALES, key=len, reverse=True)) + r")?\b",
    re.IGNORECASE,
)


def numbers_in(text: str) -> list[float]:
    """Every number an answer states, scaled by any magnitude word beside it."""
    found: list[float] = []
    for digits, scale in _NUMBER.findall(text.replace("$", "")):
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue
        found.append(value)
        if scale:
            # Both readings are kept: "1.3m" is 1,300,000, and a bare "5m" in a
            # column name or an id is still worth matching literally.
            found.append(value * SCALES[scale.lower()])
    return found


def mentions(text: str, value: float, tolerance: float = 0.01) -> bool:
    """Whether a number close to this appears anywhere in the answer."""
    for number in numbers_in(text):
        if value == 0 and number == 0:
            return True
        if value and abs(number - value) <= abs(value) * tolerance:
            return True
    return False
