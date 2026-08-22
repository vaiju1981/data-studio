"""Resource budgets and hostile-input handling."""

from __future__ import annotations

import importlib
from pathlib import Path

import duckdb
import pytest

from smart_data_studio import config
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.sql_guard import UnsafeQuery


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


def test_binary_uploads_are_refused_but_legacy_encodings_are_not() -> None:
    """Refusing anything but UTF-8 turned away ordinary European and Excel exports."""
    with pytest.raises(ValueError, match="looks binary"):
        CsvSource.from_upload("x.csv", b"a,b\n1,\x00\x00binary\n")

    source = CsvSource.from_upload("x.csv", "nom,ville\nRené,Genève\n".encode("latin-1"))
    assert source.encoding == "cp1252"
    # Normalised on the way in, so nothing downstream has to know.
    assert "René" in source.content.decode("utf-8")


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


def test_every_query_that_scans_rows_goes_through_the_guarded_path() -> None:
    """A helper that reaches for the connection itself is not on a longer leash,
    it is on none: DuckDB has no statement timeout, so the interrupt Dataset.run
    arranges is the only way a query can be stopped at all. The cohort grid, the
    value lookup, the coverage check and the digest each ran straight against the
    connection, which is a full scan apiece that nothing could cancel."""
    package = Path("src/smart_data_studio")
    # dataset.py owns the guarded path; the healthcheck holds its own connection
    # and exists to prove DuckDB answers at all.
    exempt = {package / "dataset.py", package / "healthcheck.py"}
    offenders = [
        f"{path}:{number}"
        for path in package.rglob("*.py")
        if path not in exempt
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if "connection.execute(" in line
    ]
    assert not offenders, f"queries running outside Dataset.run: {offenders}"


def test_a_model_callable_helper_is_bounded_like_any_other_query(monkeypatch) -> None:
    """The structural test above says where the queries are written; this says the
    path they were moved onto is the one that carries the deadline and the budget."""
    from smart_data_studio.cohorts import cohort_window

    rows = b"player,joined,seen\n1,2026-01-05,2026-01-06\n1,2026-01-05,2026-02-08\n"
    dataset = Dataset.load([CsvSource.from_upload("visits.csv", rows)])
    try:
        deadlines = []
        original = dataset._deadline
        monkeypatch.setattr(dataset, "_deadline", lambda: (deadlines.append(True), original())[1])
        before = dataset.queries_run
        cohort_window(dataset, "visits", "player", "joined", "seen")
        assert deadlines, "the cohort grid ran without the deadline every query gets"
        assert dataset.queries_run > before, "its scans were not charged to the session"
    finally:
        dataset.close()


def test_a_withheld_column_is_not_in_the_workspace_to_be_reshaped(monkeypatch) -> None:
    """Filtering a sensitive column out of the result cannot be made to work.

    Hiding it from the schema left SELECT * returning it. Cutting it from the
    result left a bare table name, which DuckDB reads as a struct of the whole
    row. Refusing that left three more: a positional rename in a table alias, the
    same in a CTE, and packing every column into a list. None of them is doing
    anything wrong — they are reshaping a table that should not hold the column —
    so the column is not loaded, and there is nothing left to reshape.
    """
    module = reloaded(monkeypatch, SDS_SENSITIVE_COLUMNS="ssn,dob")
    rows = b"name,ssn,dob,amount\nAda,111-22-3333,1980-01-02,10\n"
    dataset = module.Dataset.load([module.CsvSource.from_upload("p.csv", rows)])
    try:
        assert [name for name, _ in dataset.schema("p")] == ["name", "amount"]
        assert dataset.lineage[0].withheld == ["ssn", "dob"]

        for sql in (
            "SELECT * FROM p",
            "SELECT COLUMNS(*) FROM p",
            "SELECT p FROM p",
            "SELECT to_json(p) AS j FROM p",
            "SELECT * FROM p AS t(w, x)",
            "WITH r(w, x) AS (SELECT * FROM p) SELECT * FROM r",
            "SELECT [COLUMNS(*)::VARCHAR] AS everything FROM p",
        ):
            shown = dataset.query(sql).frame.to_string()
            for secret in ("111-22-3333", "1980-01-02"):
                assert secret not in shown, f"{secret} came back from: {sql}"

        # Named directly it is refused with a reason, rather than with DuckDB's
        # "column not found", which reads like a typo worth retrying.
        with pytest.raises(UnsafeQuery, match="withheld as sensitive"):
            dataset.query("SELECT ssn FROM p")
    finally:
        dataset.close()


def test_a_file_of_nothing_but_withheld_columns_is_refused(monkeypatch) -> None:
    """DuckDB will not drop a table's last column, and a table of nothing is not
    something to hand back as though it loaded."""
    module = reloaded(monkeypatch, SDS_SENSITIVE_COLUMNS="ssn")
    with pytest.raises(ValueError, match="nothing left to analyse"):
        module.Dataset.load([module.CsvSource.from_upload("p.csv", b"ssn\n111-22-3333\n")])


def test_a_withheld_column_is_absent_from_a_digest_too(monkeypatch) -> None:
    """The digest summarises the SQL rather than the frame the columns were cut
    from, so a large SELECT * would report their range instead of their values."""
    module = reloaded(monkeypatch, SDS_SENSITIVE_COLUMNS="ssn")
    body = "name,ssn,amount\n" + "".join(f"n{i},111-22-{i:04d},{i}\n" for i in range(400))
    dataset = module.Dataset.load([module.CsvSource.from_upload("p.csv", body.encode())])
    try:
        digest = dataset._digest(dataset.query("SELECT * FROM p"))
        assert set(digest["columns"]) == {"name", "amount"}
    finally:
        dataset.close()


def test_a_wide_file_with_one_bad_column_does_not_spend_the_session() -> None:
    """SUMMARIZE refuses a table over one column holding an infinity, and the
    fallback used to describe every column on its own — a full scan each. On a
    512-column file that is 512 scans, which is more than the whole session budget,
    so profiling exhausted it and every remaining column came back undescribed
    with nothing said about why."""
    from smart_data_studio.profile import profile_dataset

    width = 400
    header = ",".join(f"c{index}" for index in range(width))
    rows = [
        ",".join(
            ("1e400" if (column == 7 and line % 2 == 0) else str(column)) for column in range(width)
        )
        for line in range(60)
    ]
    body = (header + "\n" + "\n".join(rows) + "\n").encode()

    dataset = Dataset.load([CsvSource.from_upload("wide.csv", body)])
    try:
        before = dataset.queries_run
        profiles = profile_dataset(dataset)
        spent = dataset.queries_run - before
        assert spent < 50, f"profiling one table spent {spent} of the session's queries"
        # And the promise it was paying for is kept: one bad column, and only it,
        # loses its statistics.
        assert len(profiles[0].stats) == width - 1
        assert any("c7" in finding for finding in profiles[0].findings)
    finally:
        dataset.close()


def test_running_out_of_queries_is_not_swallowed_column_by_column() -> None:
    """The sweep catches per attempt so one failure costs one column. An exhausted
    budget is not that kind of failure — retried against every remaining column it
    produces a table with no statistics and no reason given — so it has its own
    type and stops the search."""
    from smart_data_studio.dataset import OutOfQueries
    from smart_data_studio.profile import _summarize_in_halves

    dataset = Dataset.load([CsvSource.from_upload("s.csv", b"a,b\n1,2\n")])
    try:
        assert isinstance(OutOfQueries("x"), RuntimeError)
        dataset.queries_run = spent = 10**9
        frames, refused = _summarize_in_halves(dataset, '"s"', ["a", "b"])
        assert not frames and refused == ["a", "b"]
        assert dataset.queries_run == spent, "it tried again for each column"
    finally:
        dataset.close()
