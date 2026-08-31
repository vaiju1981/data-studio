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


# --- what a name says, where a value shape says nothing -----------------------


def test_a_date_of_birth_and_a_postcode_are_noticed_by_name() -> None:
    """Neither has a shape to detect: a date of birth is a date and a postcode is a
    code. Both identify somebody the moment they sit beside an id, and the real
    file that prompted this carries both."""
    body = b"playerId,birthDate,zipCode,coinIn\n1,1970-01-01,89101,10\n2,1980-02-02,89102,20\n"
    found = [note for note in warnings_for(body) if "the name says personal data" in note]
    assert found, warnings_for(body)
    assert "birthDate" in found[0] and "zipCode" in found[0]
    assert "SDS_SENSITIVE_COLUMNS" in found[0]


def test_they_are_named_in_one_warning_rather_than_several() -> None:
    """Five notes about one decision is five chances to stop reading."""
    body = b"email,phone,birthDate,postcode,amount\na@b.com,555,1970-01-01,SW1,10\n"
    found = [note for note in warnings_for(body) if "the name says personal data" in note]
    assert len(found) == 1, found


def test_a_column_already_withheld_is_not_named_again() -> None:
    """It is already out of everything the model sees."""
    from smart_data_studio import dataset as dataset_module

    original = dataset_module.SENSITIVE_COLUMNS
    dataset_module.SENSITIVE_COLUMNS = ("birthdate",)
    try:
        body = b"playerId,birthDate,zipCode\n1,1970-01-01,89101\n"
        found = [note for note in warnings_for(body) if "the name says personal data" in note]
        assert found and "zipCode" in found[0]
        assert "birthDate" not in found[0]
    finally:
        dataset_module.SENSITIVE_COLUMNS = original


@pytest.mark.parametrize(
    "column", ["gameName", "machineName", "hostName", "cabinetType", "assetNumber"]
)
def test_ordinary_columns_that_merely_contain_a_word_are_left_alone(column: str) -> None:
    """`name` is deliberately not on the list: this file's own data has gameName,
    machineName and host, and a warning that fires on those is one nobody
    finishes."""
    body = f"{column},amount\nx,10\n".encode()
    assert not [n for n in warnings_for(body) if "the name says personal data" in n]


# --- the proposal, which is what a person actually sees -----------------------


def test_the_word_list_is_a_floor_under_any_proposal() -> None:
    """A model that misses a date of birth cannot make it un-personal, so the
    deterministic rule is unioned in rather than consulted only on failure."""
    from smart_data_studio import sensitive

    assert sensitive.by_name(["playerId", "birthDate", "coinIn"]) == {"birthDate"}


def test_a_model_that_is_down_still_proposes_something(monkeypatch) -> None:
    """Loading a file must not depend on a model being up. It makes the proposal
    worse; it cannot make it absent."""
    from smart_data_studio import sensitive

    def refuse(*args, **kwargs):
        raise ConnectionError("no endpoint")

    monkeypatch.setattr(sensitive.ollama, "Client", refuse)
    schema = {"people": [("playerId", ""), ("birthDate", ""), ("coinIn", "")]}
    assert sensitive.propose(schema) == {"birthDate"}


def test_a_column_the_model_invented_is_not_withheld(monkeypatch) -> None:
    """It would withhold nothing while looking like it had."""
    from smart_data_studio import sensitive

    class Reply:
        @staticmethod
        def chat(**kwargs):
            return {"message": {"content": '["birthDate", "homeAddress", "notAColumn"]'}}

    monkeypatch.setattr(sensitive.ollama, "Client", lambda **kwargs: Reply())
    schema = {"people": [("playerId", ""), ("birthDate", ""), ("homeAddress", "")]}
    assert sensitive.propose(schema) == {"birthDate", "homeAddress"}


def test_the_proposal_is_asked_before_any_value_is_read(tmp_path) -> None:
    """The ordering is the point: names and types come from the header, so asking
    which columns are sensitive cannot itself disclose one."""
    from smart_data_studio.dataset import Dataset, source_from_path

    path = tmp_path / "people.csv"
    path.write_text("playerId,birthDate,coinIn\n1,1970-01-01,10\n")
    schema = Dataset.preview_columns([source_from_path(path)])
    assert list(schema) == ["people"]
    assert [name for name, _ in schema["people"]] == ["playerId", "birthDate", "coinIn"]


def test_a_chosen_column_is_never_loaded() -> None:
    """Not filtered on the way out — absent. The guard's own reasoning: a column
    in the table can be reshaped back out of it, and one that was never loaded
    cannot."""
    body = b"playerId,birthDate,coinIn\n1,1970-01-01,10\n2,1980-02-02,20\n"
    dataset = Dataset.load([CsvSource.from_upload("people.csv", body)], withhold=["birthDate"])
    try:
        assert [name for name, _ in dataset.schema("people")] == ["playerId", "coinIn"]
        assert dataset.lineage[0].withheld == ["birthDate"]
        with pytest.raises(Exception, match="birthDate"):
            dataset.query("SELECT birthDate FROM people")
    finally:
        dataset.close()
