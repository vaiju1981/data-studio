"""How much of each bank actually checks a number.

Written because a pass rate was read as stronger than it was — "45 passed" for a
bank where six questions carried a value proved separately in SQL and the rest
asserted only that an answer arrived with a query behind it. Both are worth
running and they are not the same claim.

No model and no CSV: this reads the banks as data, so the shape of the evidence
is visible on every commit rather than on the days somebody runs the banks.
"""

from __future__ import annotations

import pytest
from test_multi_table_bank import BANK as MULTI_BANK
from test_multi_table_bank import PLAYER_BANK, TRIO_BANK
from test_question_bank import BANK as SINGLE_BANK


def anchored(bank: list[tuple]) -> list[tuple]:
    return [item for item in bank if any(isinstance(part, list) and part for part in item)]


BANKS = [
    ("single-CSV", SINGLE_BANK, 12),
    ("multi-CSV", MULTI_BANK, 9),
    ("player", PLAYER_BANK, 7),
    ("trio", TRIO_BANK, 3),
]


@pytest.mark.parametrize(("label", "bank", "floor"), BANKS, ids=[item[0] for item in BANKS])
def test_a_bank_never_verifies_less_than_it_did(label: str, bank: list[tuple], floor: int) -> None:
    """A ratchet, not a target. Anchors are expensive — each one is a figure proved
    independently in SQL — so they are added slowly, but a question that had one
    and lost it turns a checked answer into an unchecked one without the pass rate
    moving at all."""
    found = anchored(bank)
    assert len(found) >= floor, (
        f"{label}: {len(found)} of {len(bank)} questions carry an anchor, down from {floor}"
    )


def test_the_single_csv_bank_is_mostly_unanchored() -> None:
    """Stated as a test so it cannot be forgotten again.

    Twenty-six of its questions assert only that an answer arrived, was not a
    round-limit message, and had a query behind it. That is a real check and it is
    not a check that the number is right, so its pass rate should never be quoted
    as though every question verified a value.

    The twelve that are anchored are checked against the file itself by
    test_every_anchor_is_still_true_of_the_file, so the number in the bank is one
    a query reproduces rather than one somebody typed.
    """
    found = anchored(SINGLE_BANK)
    assert len(found) / len(SINGLE_BANK) < 0.5, (
        "the single-CSV bank is now mostly anchored — update this test and say so, "
        "because its pass rate finally means what it looks like it means"
    )
