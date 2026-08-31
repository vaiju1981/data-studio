"""Formats other than CSV, and the wrappers they arrive in.

CSV carries hazards no other format has — an encoding to guess, a delimiter to
sniff, a first row that might be data — and `tests/test_domains.py` covers those.
These formats carry their own column names and types instead, so what is worth
pinning here is different: that they load at all, that a workbook's sheets become
separate tables, that a compressed file is unwrapped rather than parsed as
gibberish, and that the ceilings and refusals a CSV meets are the same ones these
meet.

Built in memory from small frames, so nothing here needs a fixture file.
"""

from __future__ import annotations

import gzip
import io
import zipfile

import pandas as pd
import pytest

from smart_data_studio.dataset import (
    DataFileSource,
    Dataset,
    source_from_path,
    source_from_upload,
)

FRAME = pd.DataFrame({"region": ["north", "south", "north"], "amount": [10, 20, 30]})


def excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


def loaded(*sources) -> Dataset:
    return Dataset.load(list(sources))


def test_a_parquet_file_loads_with_its_own_types(tmp_path) -> None:
    """Parquet states its types, so nothing is inferred and nothing can be inferred
    wrongly — the reason to prefer it over an exported CSV."""
    path = tmp_path / "sales.parquet"
    FRAME.to_parquet(path)
    dataset = loaded(source_from_path(path))
    try:
        assert dataset.tables == ("sales",)
        assert dict(dataset.schema("sales"))["amount"].upper().startswith("BIGINT")
        assert dataset.query("SELECT sum(amount) AS total FROM sales").frame.iloc[0, 0] == 60
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("filename", "orient", "lines"),
    [("events.json", "records", False), ("events.ndjson", "records", True)],
    ids=["json array", "newline-delimited"],
)
def test_json_loads_either_way_it_is_written(filename: str, orient: str, lines: bool) -> None:
    """One array or one object per line. Both are called JSON by whoever exported
    them, and neither is worth making the user convert."""
    content = FRAME.to_json(orient=orient, lines=lines).encode()
    dataset = loaded(source_from_upload(filename, content))
    try:
        assert dataset.row_count(dataset.tables[0]) == 3
    finally:
        dataset.close()


def test_each_sheet_of_a_workbook_becomes_its_own_table() -> None:
    """A workbook is several tables in one file, which is the whole reason a source
    yields parts. Loading only the first sheet would lose the rest silently."""
    content = excel_bytes({"Orders": FRAME, "Regions": pd.DataFrame({"region": ["north"]})})
    dataset = loaded(source_from_upload("book.xlsx", content))
    try:
        assert set(dataset.tables) == {"book_orders", "book_regions"}
        assert dataset.row_count("book_orders") == 3
        assert dataset.row_count("book_regions") == 1
        # Each sheet is named in the lineage, or two tables from one file read as
        # one file loaded twice.
        assert {item.source for item in dataset.lineage} == {
            "book.xlsx (Orders)",
            "book.xlsx (Regions)",
        }
    finally:
        dataset.close()


def test_one_unreadable_sheet_does_not_cost_the_others() -> None:
    """The same promise a batch of files already gets, at the grain a workbook
    actually fails at: the duplicate-name refusal applies per sheet."""
    content = excel_bytes(
        {"Good": FRAME, "Bad": pd.DataFrame([[1, 2]], columns=["amount", "AMOUNT"])}
    )
    dataset = loaded(source_from_upload("book.xlsx", content))
    try:
        assert dataset.tables == ("book_good",)
        assert any("AMOUNT" in reason for reason in dataset.rejected)
    finally:
        dataset.close()


def test_a_gzipped_csv_is_unwrapped_rather_than_parsed_as_bytes() -> None:
    """Compressed exports are ordinary. Handed to the CSV reader as they sit, the
    gzip header is the first line and every column name is binary."""
    content = gzip.compress(FRAME.to_csv(index=False).encode())
    dataset = loaded(source_from_upload("sales.csv.gz", content))
    try:
        assert dataset.tables == ("sales",)
        assert [name for name, _ in dataset.schema("sales")] == ["region", "amount"]
    finally:
        dataset.close()


def test_a_zip_of_one_file_is_unwrapped_and_a_zip_of_many_says_so() -> None:
    """Unwrapping one is a convenience; guessing which of five was meant is not."""
    single = io.BytesIO()
    with zipfile.ZipFile(single, "w") as archive:
        archive.writestr("sales.csv", FRAME.to_csv(index=False))
    dataset = loaded(source_from_upload("sales.zip", single.getvalue()))
    try:
        assert dataset.tables == ("sales",)
    finally:
        dataset.close()

    many = io.BytesIO()
    with zipfile.ZipFile(many, "w") as archive:
        archive.writestr("a.csv", FRAME.to_csv(index=False))
        archive.writestr("b.csv", FRAME.to_csv(index=False))
    with pytest.raises(ValueError, match="exactly one"):
        source_from_upload("both.zip", many.getvalue())


def test_an_unsupported_extension_names_what_is_supported() -> None:
    """The refusal a user meets most often, so it says what to do rather than what
    went wrong."""
    with pytest.raises(ValueError) as refusal:
        source_from_upload("notes.docx", b"whatever")
    assert ".parquet" in str(refusal.value) and ".xlsx" in str(refusal.value)


def test_formats_mix_in_one_workspace_and_can_be_joined() -> None:
    """The reason this unit exists: a Parquet fact table beside a CSV dimension is
    an ordinary way for a person's data to sit, and it should need no conversion."""
    parquet = io.BytesIO()
    FRAME.to_parquet(parquet)
    dataset = loaded(
        DataFileSource(name="sales.parquet", kind="parquet", content=parquet.getvalue()),
        source_from_upload("regions.csv", b"region,manager\nnorth,ada\nsouth,grace\n"),
    )
    try:
        assert set(dataset.tables) == {"sales", "regions"}
        answer = dataset.query(
            "SELECT r.manager, sum(s.amount) AS total FROM sales s "
            "JOIN regions r USING (region) GROUP BY r.manager ORDER BY r.manager"
        ).frame
        assert answer["total"].tolist() == [40, 20]
    finally:
        dataset.close()


def test_a_sensitive_column_is_withheld_whatever_the_format_is() -> None:
    """The withholding runs after the table is built, so it should hold for every
    format — asserted rather than assumed, because it is the one guarantee that
    cannot be recovered if a format slips past it."""
    frame = pd.DataFrame({"email": ["a@x.com"], "amount": [10]})
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    from smart_data_studio import dataset as dataset_module

    original = dataset_module.SENSITIVE_COLUMNS
    dataset_module.SENSITIVE_COLUMNS = ("email",)
    try:
        dataset = loaded(
            DataFileSource(name="people.parquet", kind="parquet", content=buffer.getvalue())
        )
        try:
            assert [name for name, _ in dataset.schema("people")] == ["amount"]
            assert dataset.lineage[0].withheld == ["email"]
        finally:
            dataset.close()
    finally:
        dataset_module.SENSITIVE_COLUMNS = original


def test_nested_json_loads_and_its_columns_stay_reachable() -> None:
    """Nested objects are not flattened yet — that is §9c.4. What matters today is
    that they load rather than refuse, and that a field stays reachable, including
    through the SQL guard, which sees `customer.name` in the same shape as a
    table-qualified column."""
    content = b'[{"id": 1, "customer": {"name": "ada"}}, {"id": 2, "customer": {"name": "grace"}}]'
    dataset = loaded(source_from_upload("events.json", content))
    try:
        assert dataset.row_count("events") == 2
        names = dataset.query("SELECT customer.name AS who FROM events ORDER BY who").frame
        assert names["who"].tolist() == ["ada", "grace"]
    finally:
        dataset.close()
