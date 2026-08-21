from __future__ import annotations

import json

import duckdb
import pytest

from smart_data_studio.config import DIGEST_SAMPLE_ROWS, MAX_LLM_PAYLOAD_CHARS
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_dataset

SALES = b"region,amount,order_id,note\nNorth,10,1,alpha\nSouth,20,2,beta\nNorth,15,3,gamma\n"


def test_upload_loads_queries_profiles_and_locks_external_access() -> None:
    dataset = Dataset.load([CsvSource.from_upload("Sales 2026.csv", SALES)])
    try:
        assert dataset.tables == ("sales_2026",)
        assert dataset.row_count("sales_2026") == 3
        result = dataset.query(
            "SELECT region, SUM(amount) AS total FROM sales_2026 GROUP BY region ORDER BY region"
        )
        assert result.total_rows == 2
        assert result.frame.to_dict(orient="records") == [
            {"region": "North", "total": 25.0},
            {"region": "South", "total": 20.0},
        ]

        profiles = profile_dataset(dataset)
        assert profiles[0].row_count == 3
        assert "column_name" in profiles[0].stats.columns
        assert any("order_id is unique across all" in item for item in profiles[0].findings)
        assert "Profile" in profiles[0].prompt_text()
        assert "Sample rows from sales_2026" in dataset.sample_text()

        with pytest.raises(duckdb.Error):
            dataset.connection.execute("SELECT * FROM read_csv_auto('/etc/passwd')")
        with pytest.raises(duckdb.Error):
            dataset.connection.execute("SET enable_external_access = true")
    finally:
        dataset.close()


def test_local_paths_and_duplicate_names_load_as_separate_tables(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "data.csv"
    second = second_dir / "data.csv"
    first.write_bytes(b"id,value\n1,10\n")
    second.write_bytes(b"id,label\n1,A\n")

    dataset = Dataset.load([CsvSource.from_path(first), CsvSource.from_path(second)])
    try:
        assert dataset.tables == ("data", "data_2")
        result = dataset.query(
            "SELECT data.id, value, label FROM data JOIN data_2 ON data.id = data_2.id"
        )
        assert result.frame.to_dict(orient="records") == [{"id": 1, "value": 10, "label": "A"}]
    finally:
        dataset.close()


def test_query_result_is_capped_but_reports_full_row_count() -> None:
    rows = "value\n" + "\n".join(str(value) for value in range(12)) + "\n"
    dataset = Dataset.load([CsvSource.from_upload("numbers.csv", rows.encode())])
    try:
        result = dataset.query("SELECT * FROM numbers ORDER BY value", row_limit=5)
        assert len(result.frame) == 5
        assert result.total_rows == 12
        assert result.truncated
        assert result.rows_payload()["truncated"] is True
    finally:
        dataset.close()


def test_small_result_reaches_the_model_as_rows() -> None:
    dataset = Dataset.load([CsvSource.from_upload("sales.csv", SALES)])
    try:
        result = dataset.query("SELECT region, SUM(amount) AS total FROM sales GROUP BY region")
        payload = dataset.tool_payload(result)
        assert payload.get("returned") != "digest"
        assert len(payload["rows"]) == 2
    finally:
        dataset.close()


def test_oversized_result_becomes_a_digest_describing_every_row() -> None:
    rows = ["id,label,amount"]
    rows += [f"{index},{'x' * 60}-{index},{index}" for index in range(600)]
    dataset = Dataset.load([CsvSource.from_upload("wide.csv", ("\n".join(rows) + "\n").encode())])
    try:
        result = dataset.query("SELECT * FROM wide ORDER BY id")
        payload = dataset.tool_payload(result)
        assert payload["returned"] == "digest"
        assert payload["row_count"] == 600
        assert len(payload["sample_rows"]) == DIGEST_SAMPLE_ROWS
        # The stats come from SUMMARIZE over the whole result, so they describe all
        # 600 rows rather than only the handful in sample_rows.
        assert str(payload["columns"]["amount"]["max"]) == "599"
        # The whole point: the digest fits where the rows did not.
        rendered = json.dumps(payload, default=str)
        assert len(rendered) < MAX_LLM_PAYLOAD_CHARS
        assert len(json.dumps(result.rows_payload(), default=str)) > MAX_LLM_PAYLOAD_CHARS
    finally:
        dataset.close()


def test_a_result_column_named_like_the_counter_is_not_clobbered() -> None:
    """The row count is read positionally; by name it would pick up the user's column."""
    rows = b"a,__total_rows\n1,99\n2,98\n"
    dataset = Dataset.load([CsvSource.from_upload("clash.csv", rows)])
    try:
        result = dataset.query("SELECT * FROM clash")
        assert result.total_rows == 2  # not 99, the user's first value
        assert result.frame["__total_rows"].tolist() == [99, 98]
    finally:
        dataset.close()


def test_digest_stays_within_budget_when_the_statistics_alone_overflow() -> None:
    width = 400
    header = ",".join(f"some_longish_column_name_{index}" for index in range(width))
    row = ",".join(str(index) for index in range(width))
    rows = (f"{header}\n{row}\n{row}\n").encode()
    dataset = Dataset.load([CsvSource.from_upload("verywide.csv", rows)])
    try:
        payload = dataset.tool_payload(dataset.query("SELECT * FROM verywide"))
        assert payload["returned"] == "digest"
        assert not payload["sample_rows"]  # samples are given up first
        # Describing fewer columns is the last resort, and it says so.
        assert payload["columns_described"] == f"{len(payload['columns'])} of {width}"
        assert len(json.dumps(payload, default=str)) <= MAX_LLM_PAYLOAD_CHARS
    finally:
        dataset.close()


def test_columns_mentioned_in_definitions_surface_typos() -> None:
    dataset = Dataset.load([CsvSource.from_upload("sales.csv", SALES)])
    try:
        good = dataset.columns_mentioned_in("avgBet = amount / order_id, ignore note")
        assert good == ["amount", "note", "order_id"]
        # A misspelt column simply does not appear, which is how the user sees it.
        assert dataset.columns_mentioned_in("avgBet = amountt / orderid") == []
    finally:
        dataset.close()


def test_a_windows_export_loads_from_a_path_as_it_does_from_an_upload(tmp_path) -> None:
    """decode_csv normalises an upload; a path was handed to DuckDB as it sits on
    disk, so the same file loaded through the browser and failed through the box
    beside it. 0x92 is the smart quote DuckDB's own latin-1 mode also refuses."""
    path = tmp_path / "euro.csv"
    path.write_bytes("region,ventas\nCafé,10\nMünchen,20\n".encode("cp1252") + b"O\x92Brien,30\n")

    dataset = Dataset.load([CsvSource.from_path(path)])
    try:
        assert dataset.query("SELECT * FROM euro").frame.to_dict(orient="records") == [
            {"region": "Café", "ventas": 10},
            {"region": "München", "ventas": 20},
            {"region": "O’Brien", "ventas": 30},
        ]
    finally:
        dataset.close()


def test_a_utf8_path_is_not_transcoded_on_the_way_in(tmp_path) -> None:
    """The retry must stay on the failure path: a valid file is handed straight to
    DuckDB, which is the whole reason a path is cheaper than an upload."""
    path = tmp_path / "plain.csv"
    path.write_text("region,ventas\nCafé,10\n", encoding="utf-8")

    dataset = Dataset.load([CsvSource.from_path(path)])
    try:
        assert dataset.row_count("plain") == 1
    finally:
        dataset.close()


def test_a_binary_path_is_named_rather_than_parsed(tmp_path) -> None:
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"PK\x03\x04\x00\x00binary payload")
    with pytest.raises(ValueError, match="looks binary"):
        CsvSource.from_path(path)


def test_an_absurd_local_file_is_refused_before_it_is_parsed(tmp_path, monkeypatch) -> None:
    """MAX_INGEST_ROWS catches this only once the table already exists."""
    monkeypatch.setattr("smart_data_studio.dataset.MAX_LOCAL_FILE_BYTES", 10)
    path = tmp_path / "big.csv"
    path.write_text("region,amount\nNorth,10\nSouth,20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="the limit is"):
        CsvSource.from_path(path)


def test_repairing_a_column_type_leaves_the_column_order_alone() -> None:
    """Appending the converted column moved it to the end, quietly reordering
    SELECT *, the sample rows the model is shown, and the parsed-columns panel."""
    rows = b"".join(f'{index},"${index}.00",C\n'.encode() for index in range(1, 60))
    dataset = Dataset.load([CsvSource.from_upload("t.csv", b"id,price,city\n" + rows)])
    try:
        before = [name for name, _ in dataset.schema("t")]
        dataset.convert_to_number("t", "price")
        assert [name for name, _ in dataset.schema("t")] == before
        assert dict(dataset.schema("t"))["price"] == "DOUBLE"
    finally:
        dataset.close()
