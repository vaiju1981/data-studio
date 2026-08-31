"""Personal data found by what a value looks like, not by what a column is called.

`SDS_SENSITIVE_COLUMNS` matches names. It protects a column called `email` and
misses one called `notes`, and the second is the one that actually catches people
out — a free-text field somebody pasted addresses into. The endpoint may be
hosted, and it receives the profile, the samples and every query result, so the
gap between "we named the obvious columns" and "we looked" is the whole risk.

This does not withhold anything. Withholding on a guess would silently delete a
column somebody needs; naming it is the operator's decision, and noticing it is
this code's.
"""

from __future__ import annotations

import pytest

from smart_data_studio.dataset import CsvSource, Dataset


def warnings_for(body: bytes) -> list[str]:
    dataset = Dataset.load([CsvSource.from_upload("t.csv", body)])
    try:
        return list(dataset.lineage[0].warnings)
    finally:
        dataset.close()


def test_a_column_of_addresses_called_notes_is_noticed() -> None:
    """The case the name list cannot reach."""
    body = b"id,notes\n1,ada@example.com\n2,grace@example.org\n3,alan@example.net\n"
    found = [note for note in warnings_for(body) if "email addresses" in note]
    assert found, warnings_for(body)
    assert "notes" in found[0]
    assert "SDS_SENSITIVE_COLUMNS" in found[0], "the note has to say what to do about it"


def test_card_shaped_values_are_noticed() -> None:
    """Spaces and dashes included, since that is how anybody writes one down."""
    body = b"id,reference\n1,4111 1111 1111 1111\n2,5500-0000-0000-0004\n3,340000000000009\n"
    found = [note for note in warnings_for(body) if "payment card numbers" in note]
    assert found, warnings_for(body)


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("order ids of card length", b"id,order_ref\n1,7012345678901234\n2,7012345678901235\n"),
        ("plain long numbers", b"id,meter\n1,123456789012345\n2,223456789012345\n"),
        ("ordinary text", b"id,note\n1,delivered late\n2,left with neighbour\n"),
        ("a mention of email", b"id,note\n1,email the customer\n2,call them\n"),
    ],
)
def test_ordinary_columns_are_left_alone(label: str, body: bytes) -> None:
    """The rule's whole worth is that it is worth reading. A warning that fires on
    every long number is one nobody looks at by the second day — so the card shape
    requires an issuer prefix, and the email shape requires an address rather than
    the word."""
    noisy = [
        note
        for note in warnings_for(body)
        if "email addresses" in note or "payment card numbers" in note
    ]
    assert not noisy, f"{label}: {noisy}"


def test_a_column_already_withheld_is_not_also_warned_about() -> None:
    """Saying it twice helps nobody, and the second one reads as though the first
    did not work."""
    from smart_data_studio import dataset as dataset_module

    body = b"id,email\n1,ada@example.com\n2,grace@example.org\n"
    original = dataset_module.SENSITIVE_COLUMNS
    dataset_module.SENSITIVE_COLUMNS = ("email",)
    try:
        dataset = Dataset.load([CsvSource.from_upload("t.csv", body)])
        try:
            assert dataset.lineage[0].withheld == ["email"]
            assert not [n for n in dataset.lineage[0].warnings if "email addresses" in n]
        finally:
            dataset.close()
    finally:
        dataset_module.SENSITIVE_COLUMNS = original


def test_a_sprinkling_is_enough_to_be_worth_saying() -> None:
    """Personal data is not a majority phenomenon: one address in a hundred rows of
    notes is still an address sent to a hosted endpoint."""
    rows = [b"1,delivered late"] * 96 + [b"2,ada@example.com"] * 4
    body = b"id,note\n" + b"\n".join(rows) + b"\n"
    assert [note for note in warnings_for(body) if "email addresses" in note]
