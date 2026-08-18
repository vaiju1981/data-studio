"""Conventions that CSVs from other domains actually use.

Built from the hazards that published datasets exhibit rather than downloaded —
what matters is covering the failure modes, and each fixture here is one that
either broke ingestion or passed through silently wrong before it was written.
"""

from __future__ import annotations

import pytest

from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.sql_guard import UnsafeQuery


def warnings_for(name: str, body: bytes) -> tuple[list[str], int, str]:
    source = CsvSource.from_upload(name, body)
    dataset = Dataset.load([source])
    try:
        table = dataset.tables[0]
        rows = int(dataset.query(f"SELECT count(*) AS n FROM {table}").frame.iloc[0, 0])
        return [note for item in dataset.lineage for note in item.warnings], rows, source.encoding
    finally:
        dataset.close()


def test_a_windows_encoded_export_loads_with_its_accents_intact() -> None:
    notes, rows, encoding = warnings_for(
        "fr.csv", "nom,ville,montant\nRené,Montréal,100\nZoë,Genève,200\n".encode("latin-1")
    )
    assert encoding == "cp1252" and rows == 2 and not notes


def test_a_byte_order_mark_does_not_become_part_of_the_first_column() -> None:
    notes, rows, encoding = warnings_for("b.csv", "id,value\n1,10\n".encode("utf-8-sig"))
    assert encoding == "utf-8-sig" and rows == 1 and not notes


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("currency and percent", b'ticker,price\nAAPL,"$1,234.56"\nMSFT,"$987.65"\n'),
        # Semicolons, because a decimal comma requires them — see the pathological
        # case below for what a comma-separated file does with commas inside values.
        ("european decimals", b"datum;betrag\n01.02.2024;1.234,56\n03.04.2024;2.345,67\n"),
    ],
)
def test_decorated_numbers_are_flagged_rather_than_silently_left_as_text(label, body) -> None:
    """The old rule fired between 90% and 99% castable, so a column that parsed as
    text *entirely* — the common case — produced no warning at all."""
    notes, _, _ = warnings_for("n.csv", body)
    assert any("carry a currency symbol" in note for note in notes), label


def test_a_decimal_comma_in_a_comma_separated_file_is_not_silently_mangled() -> None:
    """This file is genuinely ambiguous — one column containing commas reads exactly
    like two columns — so what matters is that it complains rather than appearing
    to have parsed."""
    notes, _, _ = warnings_for("bad.csv", b"betrag\n1.234,56\n2.345,67\n")
    assert notes, "an unparseable file loaded without a word"


def test_missing_value_markers_are_flagged() -> None:
    notes, _, _ = warnings_for("m.csv", b"id,score\n1,10\n2,NA\n3,-\n4,?\n5,null\n")
    assert any("mean 'missing'" in note for note in notes)


def test_a_missing_header_row_is_reported_because_a_row_is_lost_with_it() -> None:
    notes, rows, _ = warnings_for("h.csv", b"1,alpha,10\n2,beta,20\n")
    assert rows == 1  # the first row became the header
    assert any("looks like data rather than headers" in note for note in notes)


def test_ragged_rows_that_load_as_nothing_say_so() -> None:
    notes, rows, _ = warnings_for("r.csv", b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n")
    assert rows == 0
    assert any("no rows at all" in note for note in notes)


def test_ambiguous_day_month_dates_are_flagged_but_unambiguous_ones_are_not() -> None:
    ambiguous, _, _ = warnings_for("d.csv", b"when,amount\n01/02/2024,10\n03/04/2024,20\n")
    assert any("day-first and month-first" in note for note in ambiguous)

    # A day past the twelfth settles it, so there is nothing to warn about.
    clear, _, _ = warnings_for("d.csv", b"when,amount\n01/02/2024,10\n12/31/2024,20\n")
    assert not any("day-first and month-first" in note for note in clear)


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("tab separated", b"id\tname\tqty\n1\twidget\t5\n"),
        ("semicolon separated", b"id;name;qty\n1;widget;5\n"),
        ("quoted newlines", b'id,comment\n1,"line one\nline two"\n'),
        ("escaped quotes", b'id,comment\n1,"has ""quotes"" inside"\n'),
    ],
)
def test_separators_and_quoting_are_handled_without_complaint(label, body) -> None:
    notes, rows, _ = warnings_for("s.csv", body)
    assert rows == 1, label
    assert not notes, f"{label}: {notes}"


def test_identifiers_that_must_keep_their_leading_zeros_stay_text() -> None:
    """A zip code read as a number is a zip code destroyed, so text is correct here."""
    dataset = Dataset.load([CsvSource.from_upload("z.csv", b"zip\n02134\n90210\n")])
    try:
        assert dataset.schema("z")[0][1] == "VARCHAR"
        assert dataset.query("SELECT zip FROM z").frame.iloc[0, 0] == "02134"
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("label", "body", "expected"),
    [
        ("us currency", b'id,price\n1,"$1,234.56"\n2,"$987.65"\n', [1234.56, 987.65]),
        ("european", b"id;betrag\n1;1.234,56\n2;2.345,67\n", [1234.56, 2345.67]),
        ("percent", b"id,rate\n1,2.5%\n2,10%\n", [2.5, 10.0]),
    ],
)
def test_a_text_column_can_be_read_as_a_number(label, body, expected) -> None:
    """Reporting that a column parsed as text and leaving the user to cast it in
    every query is half an answer. Which separator convention applies is read from
    the data — assuming the wrong one turns 1.234,56 into 1.23456."""
    dataset = Dataset.load([CsvSource.from_upload("t.csv", body)])
    try:
        column = dataset.text_columns("t")[0]
        note = dataset.convert_to_number("t", column)
        assert dataset.query(f"SELECT {column} FROM t").frame[column].tolist() == expected, label
        assert "converted to a number" in note
    finally:
        dataset.close()


def test_values_that_cannot_convert_are_counted_rather_than_hidden() -> None:
    dataset = Dataset.load([CsvSource.from_upload("t.csv", b"id,amount\n1,10\n2,n/a\n3,30\n")])
    try:
        note = dataset.convert_to_number("t", "amount")
        assert "1 value(s) would not convert" in note
        assert dataset.query("SELECT count(amount) AS n FROM t").frame.iloc[0, 0] == 2
    finally:
        dataset.close()


def test_a_genuinely_textual_column_is_not_emptied_by_a_stray_click() -> None:
    """Found by clicking it: the control defaulted to a region name and nulled all
    400 values. Converting is refused when it would destroy the column."""
    dataset = Dataset.load([CsvSource.from_upload("t.csv", b"region,price\nEast,5\nWest,6\n")])
    try:
        with pytest.raises(ValueError, match="not a number in disguise"):
            dataset.convert_to_number("t", "region")
        assert dataset.query("SELECT region FROM t").frame.region.tolist() == ["East", "West"]
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("column", "message"),
    [("price", "already"), ("nope", "no column")],
)
def test_conversion_refuses_what_it_cannot_convert(column: str, message: str) -> None:
    dataset = Dataset.load([CsvSource.from_upload("t.csv", b"region,price\nEast,5\nWest,6\n")])
    try:
        with pytest.raises(ValueError, match=message):
            dataset.convert_to_number("t", column)
    finally:
        dataset.close()


def test_one_unreadable_file_does_not_cost_the_others() -> None:
    """Uploading three CSVs and getting nothing because one repeats a column name
    is the wrong trade when the other two are fine."""
    dataset = Dataset.load(
        [
            CsvSource.from_upload("good.csv", b"id,value\n1,10\n"),
            CsvSource.from_upload("dupes.csv", b"id,id\n1,2\n"),
            CsvSource.from_upload("also_good.csv", b"id,name\n1,widget\n"),
        ]
    )
    try:
        assert dataset.tables == ("good", "also_good")
        assert dataset.query("SELECT value FROM good").frame.iloc[0, 0] == 10
        # The one that failed is named, not silently absent from the panel.
        assert len(dataset.rejected) == 1
        assert "repeats column name" in dataset.rejected[0]
        # Named once. The error already carries the filename, so prefixing it
        # unconditionally printed it twice.
        assert dataset.rejected[0].count("dupes.csv") == 1
    finally:
        dataset.close()


def test_a_half_built_table_is_not_left_queryable() -> None:
    """A file rejected after its table was created must leave nothing behind."""
    dataset = Dataset.load(
        [
            CsvSource.from_upload("good.csv", b"id,value\n1,10\n"),
            CsvSource.from_upload("dupes.csv", b"id,id\n1,2\n"),
        ]
    )
    try:
        loaded = dataset.connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        assert [row[0] for row in loaded] == ["good"]
        with pytest.raises(UnsafeQuery, match="Unknown table"):
            dataset.query("SELECT * FROM dupes")
    finally:
        dataset.close()


def test_nothing_loading_is_still_an_error() -> None:
    with pytest.raises(ValueError, match="repeats column name"):
        Dataset.load([CsvSource.from_upload("dupes.csv", b"id,id\n1,2\n")])
