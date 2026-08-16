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
