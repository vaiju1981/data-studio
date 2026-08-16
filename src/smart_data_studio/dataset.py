"""CSV ingestion and the locked-down DuckDB workspace."""

from __future__ import annotations

import csv
import json
import re
import tempfile
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from smart_data_studio import logs
from smart_data_studio.config import (
    ALLOW_LOCAL_PATHS,
    DIGEST_SAMPLE_ROWS,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    MAX_CELL_CHARS_TO_MODEL,
    MAX_DISPLAY_ROWS,
    MAX_HEADER_LENGTH,
    MAX_INGEST_CELLS,
    MAX_INGEST_COLUMNS,
    MAX_INGEST_ROWS,
    MAX_LLM_PAYLOAD_CHARS,
    MAX_LLM_ROWS,
    MAX_SESSION_QUERIES,
    MAX_UPLOAD_BYTES,
    QUERY_TIMEOUT_SECONDS,
    SAMPLE_ROWS,
    SENSITIVE_COLUMNS,
    temp_directory,
)
from smart_data_studio.sql_guard import validate_select

TOTAL_ROWS_COLUMN = "__total_rows"


def safe_name(value: str) -> str:
    """A filename fit for a log line or a download header.

    Upload names arrive from a browser and can carry newlines, quotes or path
    separators; none of those belong in a log or a Content-Disposition header.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(value).name).strip("._-")
    return (cleaned or "upload")[:100]


def is_sensitive(column: str) -> bool:
    lowered = column.lower()
    return any(marker in lowered for marker in SENSITIVE_COLUMNS)


def quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


@dataclass(frozen=True)
class CsvSource:
    """One CSV supplied either as an upload or as a local path."""

    name: str
    path: Path | None = None
    content: bytes | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.content is None):
            raise ValueError("A CSV source needs exactly one of path or content")

    @classmethod
    def from_path(cls, path: str | Path) -> CsvSource:
        if not ALLOW_LOCAL_PATHS:
            raise PermissionError(
                "Loading from a server path is disabled on this deployment. Upload the file."
            )
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"CSV file not found: {resolved}")
        return cls(name=resolved.name, path=resolved)

    def header_names(self) -> list[str]:
        """The header row as written.

        Checked before loading because DuckDB silently renames a duplicate to
        `a_1`, so by the time the table exists the collision has been papered over
        and the model is reading a column nobody named.
        """
        if self.content is not None:
            first = self.content.split(b"\n", 1)[0].decode("utf-8", "replace")
        else:
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                first = handle.readline()
        return next(csv.reader([first.rstrip("\r")]), [])

    @classmethod
    def from_upload(cls, name: str, content: bytes) -> CsvSource:
        if not content:
            raise ValueError(f"{name} is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"{name} is {len(content) / 1e6:.0f}MB; the limit is "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f}MB."
            )
        _reject_unreadable(name, content)
        return cls(name=safe_name(name), content=content)


def _reject_unreadable(name: str, content: bytes) -> None:
    """Fail on binary or mis-encoded input with a message that says what to do.

    A NUL byte in the first block means this is not a CSV at all, and a decode
    error means the encoding is not UTF-8 — two different fixes, so two messages.
    """
    head = content[:65536]
    if b"\x00" in head:
        raise ValueError(f"{name} looks binary, not CSV. Export it as text and try again.")
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{name} is not valid UTF-8 (byte {error.start} of the first block). "
            "Re-save it as UTF-8 — most spreadsheets offer 'CSV UTF-8'."
        ) from error


@dataclass
class QueryResult:
    sql: str
    frame: pd.DataFrame
    total_rows: int

    @property
    def truncated(self) -> bool:
        return self.total_rows > len(self.frame)

    def rows_payload(self, limit: int = MAX_LLM_ROWS) -> dict[str, object]:
        head = self.frame.head(limit)
        rows = json.loads(head.to_json(orient="records", date_format="iso"))
        return {
            "columns": head.columns.tolist(),
            "rows": [_trim_cells(row) for row in rows],
            "row_count": self.total_rows,
            "truncated": self.total_rows > len(head),
        }


def _trim_cells(row: dict[str, object]) -> dict[str, object]:
    """Shorten long free text before it reaches the prompt.

    A notes field can be longer than the rest of the row put together, and the
    analysis never turns on its tail.
    """
    trimmed = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > MAX_CELL_CHARS_TO_MODEL:
            trimmed[key] = f"{value[:MAX_CELL_CHARS_TO_MODEL]}… (truncated)"
        else:
            trimmed[key] = value
    return trimmed


class Dataset:
    """An in-memory database that becomes read-only after all CSVs are loaded."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, tables: tuple[str, ...]):
        self.connection = connection
        self.tables = tables
        self.queries_run = 0

    @classmethod
    def load(cls, sources: Iterable[CsvSource]) -> Dataset:
        source_list = list(sources)
        if not source_list:
            raise ValueError("Choose at least one CSV file")

        connection = duckdb.connect(database=":memory:")
        table_names: list[str] = []
        try:
            cls._apply_budget(connection)
            for source in source_list:
                cls._check_header(source)
                table_name = cls._unique_table_name(source.name, table_names)
                with logs.timed("ingest", table=table_name) as fields:
                    cls._load_source(connection, table_name, source)
                    fields.update(cls._check_size(connection, table_name))
                table_names.append(table_name)
            connection.execute("SET enable_external_access = false")
            connection.execute("SET lock_configuration = true")
        except Exception:
            connection.close()
            raise
        return cls(connection, tuple(table_names))

    @staticmethod
    def _apply_budget(connection: duckdb.DuckDBPyConnection) -> None:
        """Bound memory, threads and spill before the connection is locked shut.

        Without this one careless join takes the whole process with it, and after
        the lock these settings can no longer be changed — which is the point.
        """
        Path(temp_directory()).mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        connection.execute(f"SET threads={int(DUCKDB_THREADS)}")
        connection.execute(f"SET temp_directory='{temp_directory()}'")

    @staticmethod
    def _check_size(connection: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, int]:
        quoted = quote_identifier(table_name)
        rows = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
        names = [row[0] for row in connection.execute(f"DESCRIBE {quoted}").fetchall()]
        columns = len(names)
        if rows > MAX_INGEST_ROWS:
            raise ValueError(f"{table_name} has {rows:,} rows; the limit is {MAX_INGEST_ROWS:,}.")
        if columns > MAX_INGEST_COLUMNS:
            raise ValueError(
                f"{table_name} has {columns:,} columns; the limit is {MAX_INGEST_COLUMNS:,}."
            )
        if rows * columns > MAX_INGEST_CELLS:
            raise ValueError(
                f"{table_name} holds {rows * columns:,} cells; the limit is "
                f"{MAX_INGEST_CELLS:,}. Filter or aggregate before loading."
            )
        return {"rows": rows, "columns": columns}

    @staticmethod
    def _check_header(source: CsvSource) -> None:
        names = source.header_names()
        overlong = [name for name in names if len(name) > MAX_HEADER_LENGTH]
        if overlong:
            raise ValueError(
                f"{source.name} has a column name longer than {MAX_HEADER_LENGTH} characters: "
                f"{overlong[0][:60]}… — the first row is probably data, not a header."
            )
        duplicates = sorted({name for name in names if names.count(name) > 1 and name})
        if duplicates:
            raise ValueError(
                f"{source.name} repeats column name(s): {', '.join(duplicates[:5])}. "
                "DuckDB would rename the second to name_1 without saying so — rename "
                "them yourself so the right one is read."
            )

    @staticmethod
    def _unique_table_name(filename: str, existing: list[str]) -> str:
        stem = Path(filename).stem
        base = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower() or "dataset"
        if base[0].isdigit():
            base = f"data_{base}"
        candidate = base
        suffix = 2
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _load_source(
        connection: duckdb.DuckDBPyConnection, table_name: str, source: CsvSource
    ) -> None:
        temporary_path: Path | None = None
        path = source.path
        if source.content is not None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary_file:
                temporary_file.write(source.content)
                temporary_path = Path(temporary_file.name)
            path = temporary_path

        try:
            connection.execute(
                f"CREATE TABLE {quote_identifier(table_name)} AS "
                "SELECT * FROM read_csv_auto(?, header = true, sample_size = -1)",
                [str(path)],
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def schema(self, table_name: str) -> list[tuple[str, str]]:
        self._require_table(table_name)
        rows = self.connection.execute(f"DESCRIBE {quote_identifier(table_name)}").fetchall()
        return [(row[0], row[1]) for row in rows]

    def row_count(self, table_name: str) -> int:
        self._require_table(table_name)
        return self.connection.execute(
            f"SELECT COUNT(*) FROM {quote_identifier(table_name)}"
        ).fetchone()[0]

    def schema_text(self) -> str:
        sections = []
        for table in self.tables:
            shown = [(name, kind) for name, kind in self.schema(table) if not is_sensitive(name)]
            hidden = len(self.schema(table)) - len(shown)
            columns = ", ".join(f"{quote_identifier(name)} {kind}" for name, kind in shown)
            note = f" — {hidden} column(s) withheld as sensitive" if hidden else ""
            sections.append(f"Table {quote_identifier(table)} ({columns}){note}")
        return "\n".join(sections)

    def tool_payload(self, result: QueryResult) -> dict[str, object]:
        """What the model sees: the rows themselves when small, a digest when not."""
        rows = result.rows_payload()
        if len(json.dumps(rows, default=str)) <= MAX_LLM_PAYLOAD_CHARS:
            return rows
        return self._digest(result)

    def _digest(self, result: QueryResult) -> dict[str, object]:
        """A compact stand-in for a result too large to put in the prompt.

        The statistics come from SUMMARIZE over the whole result rather than the
        rows on screen, so anything the model quotes from here holds for every row.
        """
        stats = self.connection.execute(f"SUMMARIZE ({result.sql})").fetchdf()
        columns: dict[str, dict[str, object]] = {}
        for row in stats.to_dict(orient="records"):
            entry = {
                "type": row["column_type"],
                "min": row["min"],
                "max": row["max"],
                "approx_distinct": row["approx_unique"],
                "null_percentage": row["null_percentage"],
            }
            if row.get("avg") is not None:
                entry["avg"] = row["avg"]
            columns[str(row["column_name"])] = entry

        head = result.frame.head(DIGEST_SAMPLE_ROWS)
        sample = json.loads(head.to_json(orient="records", date_format="iso"))
        note = (
            f"This result has {result.total_rows:,} rows, too many to include. The column "
            "statistics above cover every row; sample_rows shows only the first few, so do "
            "not describe them as the whole result. approx_distinct is an estimate. The full "
            "result is displayed to the user and available to download. To go further, run a "
            "narrower or aggregated query."
        )

        def build(rows: list[object], described: dict[str, dict[str, object]]) -> dict[str, object]:
            digest: dict[str, object] = {
                "returned": "digest",
                "row_count": result.total_rows,
                "columns": described,
                "sample_rows": rows,
                "note": note,
            }
            if len(described) < len(columns):
                digest["columns_described"] = f"{len(described)} of {len(columns)}"
            return digest

        def too_big(digest: dict[str, object]) -> bool:
            return len(json.dumps(digest, default=str)) > MAX_LLM_PAYLOAD_CHARS

        # The digest has to fit the budget too. Sample rows go first, since the
        # statistics are the part worth keeping; on a result wide enough that the
        # statistics alone overflow, describe fewer columns rather than overrun.
        described = columns
        digest = build(sample, described)
        while sample and too_big(digest):
            sample = sample[: len(sample) // 2]
            digest = build(sample, described)
        while len(described) > 1 and too_big(digest):
            described = dict(list(described.items())[: len(described) // 2])
            digest = build(sample, described)
        return digest

    def query(self, sql: str, row_limit: int = MAX_DISPLAY_ROWS) -> QueryResult:
        if self.queries_run >= MAX_SESSION_QUERIES:
            raise RuntimeError(
                f"This session has run its {MAX_SESSION_QUERIES:,} queries. "
                "Reload the data to start a fresh workspace."
            )
        self.queries_run += 1
        clean_sql = validate_select(sql, set(self.tables))
        # COUNT(*) OVER () is evaluated across the whole result before LIMIT applies,
        # so a single execution yields both the page of rows and the true total.
        counted_sql = (
            f"SELECT *, COUNT(*) OVER () AS {TOTAL_ROWS_COLUMN} "
            f"FROM ({clean_sql}) AS result_rows LIMIT {int(row_limit)}"
        )
        with logs.timed("query", sql=clean_sql) as fields, self._deadline():
            frame = self.connection.execute(counted_sql).fetchdf()
            fields["returned"] = len(frame)
        # Read the count positionally. A result of its own carrying this column name
        # would otherwise shadow ours, and we would report the user's data as the
        # row count and drop their column instead of the one we added.
        total_rows = int(frame.iloc[0, -1]) if len(frame) else 0
        return QueryResult(
            sql=clean_sql,
            frame=frame.iloc[:, :-1],
            total_rows=total_rows,
        )

    def columns_mentioned_in(self, text: str) -> list[str]:
        """Loaded column names that appear in free text.

        Used to confirm a metric definition refers to real columns: a name absent
        from the result is the typo, and the user sees it before the model does.
        """
        words = {word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)}
        found = {
            name for table in self.tables for name, _ in self.schema(table) if name.lower() in words
        }
        return sorted(found)

    @contextmanager
    def _deadline(self):
        """Interrupt a query that outstays its welcome.

        DuckDB has no statement timeout, but a connection can be interrupted from
        another thread, which is enough to stop a runaway cross join without
        losing the session.
        """
        timer = threading.Timer(QUERY_TIMEOUT_SECONDS, self.connection.interrupt)
        timer.daemon = True
        timer.start()
        try:
            yield
        except duckdb.InterruptException as error:
            raise TimeoutError(
                f"The query ran past {QUERY_TIMEOUT_SECONDS}s and was cancelled. "
                "Narrow it — filter, aggregate, or add a LIMIT."
            ) from error
        finally:
            timer.cancel()

    def sample_text(self, limit: int = SAMPLE_ROWS) -> str:
        """First rows of each table, so the model sees real values and not only types."""
        sections = []
        for table in self.tables:
            visible = [name for name, _ in self.schema(table) if not is_sensitive(name)]
            projection = ", ".join(quote_identifier(name) for name in visible) or "*"
            frame = self.connection.execute(
                f"SELECT {projection} FROM {quote_identifier(table)} LIMIT {int(limit)}"
            ).fetchdf()
            sections.append(
                f"Sample rows from {table}:\n{frame.to_string(index=False, max_cols=30)}"
            )
        return "\n\n".join(sections)

    def close(self) -> None:
        self.connection.close()

    def _require_table(self, table_name: str) -> None:
        if table_name not in self.tables:
            raise ValueError(f"Unknown table: {table_name}")
