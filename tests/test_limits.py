"""Resource budgets and hostile-input handling."""

from __future__ import annotations

import importlib

import duckdb
import pytest

from smart_data_studio import config
from smart_data_studio.dataset import CsvSource, Dataset


def reloaded(monkeypatch, **environment: str):
    """Re-import config and dataset with an environment a deployment might set."""
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    importlib.reload(config)
    return importlib.reload(importlib.import_module("smart_data_studio.dataset"))


@pytest.fixture(autouse=True)
def restore_modules():
    yield
    importlib.reload(config)
    importlib.reload(importlib.import_module("smart_data_studio.dataset"))


def test_an_oversized_upload_is_refused_before_it_is_parsed(monkeypatch) -> None:
    module = reloaded(monkeypatch, SDS_MAX_UPLOAD_BYTES="100")
    with pytest.raises(ValueError, match="the limit is"):
        module.CsvSource.from_upload("big.csv", b"a,b\n" + b"1,2\n" * 100)


def test_too_many_columns_is_refused(monkeypatch) -> None:
    module = reloaded(monkeypatch, SDS_MAX_INGEST_COLUMNS="3")
    header = ",".join(f"c{index}" for index in range(5))
    row = ",".join("1" for _ in range(5))
    with pytest.raises(ValueError, match="columns; the limit is"):
        module.Dataset.load([module.CsvSource.from_upload("w.csv", f"{header}\n{row}\n".encode())])


def test_too_many_rows_is_refused(monkeypatch) -> None:
    module = reloaded(monkeypatch, SDS_MAX_INGEST_ROWS="2")
    rows = "a\n" + "\n".join(str(index) for index in range(5)) + "\n"
    with pytest.raises(ValueError, match="rows; the limit is"):
        module.Dataset.load([module.CsvSource.from_upload("t.csv", rows.encode())])


def test_local_paths_can_be_disabled_for_hosted_use(monkeypatch, tmp_path) -> None:
    """Reading a server path is right for a local tool and disqualifying for a shared one."""
    csv = tmp_path / "s.csv"
    csv.write_bytes(b"a\n1\n")
    module = reloaded(monkeypatch, SDS_ALLOW_LOCAL_PATHS="false")
    with pytest.raises(PermissionError, match="disabled on this deployment"):
        module.CsvSource.from_path(csv)


def test_a_runaway_query_is_cancelled_and_the_session_survives(monkeypatch) -> None:
    """DuckDB has no statement timeout; the connection is interrupted from a timer."""
    module = reloaded(monkeypatch, SDS_QUERY_TIMEOUT_SECONDS="2")
    dataset = module.Dataset.load([module.CsvSource.from_upload("s.csv", b"a\n1\n2\n")])
    try:
        with pytest.raises(TimeoutError, match="was cancelled"):
            dataset.query("SELECT count(*) AS n FROM range(4000000000) t(i) WHERE i%7=0")
        # The point of cancelling rather than dying: the workspace still answers.
        assert dataset.query("SELECT count(*) AS n FROM s").frame.iloc[0, 0] == 2
    finally:
        dataset.close()


def test_the_duckdb_budget_is_applied_and_then_frozen() -> None:
    dataset = Dataset.load([CsvSource.from_upload("s.csv", b"a\n1\n")])
    try:
        settings = dict(
            dataset.connection.execute(
                "SELECT name, value FROM duckdb_settings() "
                "WHERE name IN ('memory_limit', 'threads')"
            ).fetchall()
        )
        assert settings["threads"] == str(config.DUCKDB_THREADS)
        assert settings["memory_limit"] not in ("", None)
        # Locked afterwards, so model-written SQL cannot lift its own budget.
        with pytest.raises(duckdb.Error):
            dataset.connection.execute("SET memory_limit='64GB'")
    finally:
        dataset.close()


def test_binary_and_mis_encoded_uploads_are_refused_with_the_fix() -> None:
    with pytest.raises(ValueError, match="looks binary"):
        CsvSource.from_upload("x.csv", b"a,b\n1,\x00\x00binary\n")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        CsvSource.from_upload("x.csv", "a,b\n1,café\n".encode("latin-1"))


def test_upload_names_are_sanitised_before_they_reach_a_log_or_a_filename() -> None:
    from smart_data_studio.dataset import safe_name

    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name('bad name\nwith"quotes.csv') == "bad_name_with_quotes.csv"
    assert safe_name("") == "upload"
    assert CsvSource.from_upload("../../evil.csv", b"a\n1\n").name == "evil.csv"


def test_duplicate_and_absurd_headers_are_refused() -> None:
    with pytest.raises(ValueError, match="repeats column name"):
        Dataset.load([CsvSource.from_upload("d.csv", b"a,a\n1,2\n")])


def test_sensitive_columns_are_kept_out_of_everything_the_model_sees(monkeypatch) -> None:
    module = reloaded(monkeypatch, SDS_SENSITIVE_COLUMNS="ssn,email")
    rows = b"name,ssn,email_address,amount\nAda,111-22-3333,ada@example.com,10\n"
    dataset = module.Dataset.load([module.CsvSource.from_upload("p.csv", rows)])
    try:
        schema, samples = dataset.schema_text(), dataset.sample_text()
        for secret in ("ssn", "email_address", "111-22-3333", "ada@example.com"):
            assert secret not in schema and secret not in samples
        assert "withheld as sensitive" in schema
        assert "amount" in schema and "10" in samples
    finally:
        dataset.close()


def test_a_session_runs_out_of_query_budget(monkeypatch) -> None:
    module = reloaded(monkeypatch, SDS_MAX_SESSION_QUERIES="2")
    dataset = module.Dataset.load([module.CsvSource.from_upload("s.csv", b"a\n1\n")])
    try:
        dataset.query("SELECT * FROM s")
        dataset.query("SELECT * FROM s")
        with pytest.raises(RuntimeError, match="has run its"):
            dataset.query("SELECT * FROM s")
    finally:
        dataset.close()


def test_absurdly_shaped_queries_are_refused_before_they_run() -> None:
    from smart_data_studio.sql_guard import UnsafeQuery, validate_select

    wide = "SELECT 1 FROM " + ", ".join(f"t{index}" for index in range(20))
    with pytest.raises(UnsafeQuery, match="joins 20 tables"):
        validate_select(wide, {f"t{index}" for index in range(20)})

    nested = "SELECT 1"
    for _ in range(20):
        nested = f"SELECT * FROM ({nested}) AS x"
    with pytest.raises(UnsafeQuery, match="nests"):
        validate_select(nested, set())


def test_long_cells_are_trimmed_on_the_way_to_the_model() -> None:
    rows = ("note\n" + "x" * 5000 + "\n").encode()
    dataset = Dataset.load([CsvSource.from_upload("n.csv", rows)])
    try:
        payload = dataset.query("SELECT note FROM n").rows_payload()
        assert "(truncated)" in payload["rows"][0]["note"]
        assert len(payload["rows"][0]["note"]) < 5000
    finally:
        dataset.close()
