from __future__ import annotations

import json

import duckdb
import pandas as pd
import pytest

from smart_data_studio.config import DIGEST_SAMPLE_ROWS, MAX_LLM_PAYLOAD_CHARS
from smart_data_studio.dataset import CsvSource, Dataset, defuse_formulas
from smart_data_studio.profile import profile_dataset
from smart_data_studio.sql_guard import redact_literals

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


def test_headers_differing_only_in_case_are_refused() -> None:
    """DuckDB matches names without regard to case and renames the second to A_1.
    Comparing them exactly let through the one collision the check exists for."""
    with pytest.raises(ValueError, match="repeats column name"):
        Dataset.load([CsvSource.from_upload("d.csv", b"a,A,b\n1,2,3\n")])


def test_a_duplicate_hidden_behind_a_quoted_newline_is_still_caught() -> None:
    """CSV allows a newline inside a quoted field; a header is a record, not a line.

    Cut at the first physical newline the header came back short, the two `a`
    columns never met each other, and DuckDB renamed the second to `a_1` without
    saying so — which is exactly what this check exists to prevent.
    """
    source = CsvSource.from_upload("dup.csv", b'a,"note\nwrapped",a\n1,2,3\n')
    assert source.header_names() == ["a", "note\nwrapped", "a"]
    with pytest.raises(ValueError, match="repeats column name"):
        Dataset.load([source])


def test_an_accented_header_survives_the_wider_read() -> None:
    """The chunk must not be decoded as cp1252, which never fails and mangles UTF-8."""
    source = CsvSource.from_upload("acc.csv", "région,montant\nA,1\n".encode())
    assert source.header_names() == ["région", "montant"]


def test_a_cell_a_spreadsheet_would_run_is_defused_on_the_way_out() -> None:
    """CSV quoting does not stop it: the value is a formula once it is a cell."""
    frame = pd.DataFrame(
        {"name": ["=cmd|' /C calc'!A1", "+1+1", "@SUM(A1)", "-5", "Ada"], "n": [1, 2, 3, 4, 5]}
    )
    defused = defuse_formulas(frame)["name"].tolist()
    assert defused[:3] == ["'=cmd|' /C calc'!A1", "'+1+1", "'@SUM(A1)"]
    # A negative number is not a formula, and prefixing it would corrupt the column.
    assert defused[3:] == ["-5", "Ada"]


def test_the_logged_query_carries_the_shape_and_not_the_values() -> None:
    """The SQL is logged as evidence; a generated filter carries real cell values,
    and the log is the one stream that leaves the host."""
    masked = redact_literals("SELECT * FROM people WHERE email = 'ada@example.com' AND age > 40")
    assert "ada@example.com" not in masked
    assert "PEOPLE" in masked.upper() and "EMAIL" in masked.upper()


def test_the_header_is_split_the_way_duckdb_will_split_it() -> None:
    """Deciding the delimiter separately is how the check and the reader disagree.

    `a;b;c;d,x,x` is four fields read as semicolons and three read as commas. The
    check picked semicolons and saw no duplicate; DuckDB picked commas and renamed
    the second `x` to `x_1` without a word.
    """
    ambiguous = CsvSource.from_upload("amb.csv", b"a;b;c;d,x,x\n1,2,3\n")
    with pytest.raises(ValueError, match="repeats column name"):
        Dataset.load([ambiguous])


def test_a_genuine_semicolon_file_is_still_read_as_one() -> None:
    """The counterpart: taking the reader's dialect must not break the files the
    widest-field guess got right."""
    for body, expected in (
        (b"a;b;c\n1;2;3\n", ["a", "b", "c"]),
        (b"a\tb\n1\t2\n", ["a", "b"]),
        (b"a,b\n1,2\n", ["a", "b"]),
    ):
        dataset = Dataset.load([CsvSource.from_upload("f.csv", body)])
        try:
            assert [name for name, _ in dataset.schema("f")] == expected
        finally:
            dataset.close()

    # And a real duplicate is still caught whatever the delimiter is.
    with pytest.raises(ValueError, match="repeats column name"):
        Dataset.load([CsvSource.from_upload("s.csv", b"a;b;b\n1;2;3\n")])


def test_a_comment_or_an_alias_cannot_carry_a_value_into_the_log() -> None:
    """Masking the literals left two other ways in.

    A comment is free text the model wrote and survives serialization untouched;
    a select alias is very often a cell value, since `SUM(CASE WHEN region =
    'North' ...) AS North` is the ordinary way to write a pivot.
    """
    masked = redact_literals(
        "SELECT sum(a) AS North, b AS region FROM t o /* ada@example.com */ WHERE x = 'y'"
    )
    assert "ada@example.com" not in masked and "North" not in masked
    # The shape survives: tables, columns, functions and the table alias remain.
    assert "SUM(a)" in masked and "FROM t AS o" in masked and "b AS c1" in masked


def test_unparseable_sql_is_logged_as_a_word_rather_than_verbatim() -> None:
    assert redact_literals("not sql at all (((") == "unparseable"


def test_a_literal_inside_an_aliased_expression_is_masked_too() -> None:
    """The transform walks parents before children and does not descend into a
    node it has replaced.

    Renaming the alias in the same pass therefore handed back its expression
    unvisited, and every literal inside it survived — in exactly the pivot that is
    the reason for masking aliases at all.
    """
    masked = redact_literals("SELECT SUM(CASE WHEN region = 'North' THEN 1 END) AS North FROM t")
    assert "North" not in masked, f"the value survived inside the alias: {masked}"
    assert "AS c0" in masked and "CASE WHEN" in masked


def test_a_local_file_is_sniffed_in_the_encoding_it_will_be_read_in() -> None:
    """DuckDB reads UTF-8 only, so sniffing a Windows-1252 file as it sits fails.

    The delimiter then fell back to a guess while the load transcoded and sniffed
    the converted file — the two disagreeing by a different route, and a;b;c;d,x,x
    arriving as x and x_1.
    """
    body = "a;b;c;d,x,x\n1,2,3\ncafé,5,6\n".encode("cp1252")
    with pytest.raises(ValueError, match="repeats column name"):
        Dataset.load([CsvSource.from_upload("amb.csv", body)])


def test_a_windows_1252_file_still_loads_with_its_accents() -> None:
    """The counterpart: normalising before the sniff must not refuse the files it
    was added to read correctly."""
    dataset = Dataset.load(
        [CsvSource.from_upload("ok.csv", "ville,montant\nMontréal,10\n".encode("cp1252"))]
    )
    try:
        assert [name for name, _ in dataset.schema("ok")] == ["ville", "montant"]
        assert dataset.query("SELECT ville FROM ok").frame.iloc[0, 0] == "Montréal"
    finally:
        dataset.close()
