"""CSV ingestion and the locked-down DuckDB workspace."""

from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from smart_data_studio import logs
from smart_data_studio.config import (
    ALLOW_LOCAL_PATHS,
    CODE_COLUMN_WORDS,
    DIGEST_SAMPLE_ROWS,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    IDENTIFIER_WORDS,
    MAX_CELL_CHARS_TO_MODEL,
    MAX_DISPLAY_ROWS,
    MAX_HEADER_LENGTH,
    MAX_INGEST_CELLS,
    MAX_INGEST_COLUMNS,
    MAX_INGEST_ROWS,
    MAX_LLM_PAYLOAD_CHARS,
    MAX_LLM_ROWS,
    MAX_LOCAL_FILE_BYTES,
    MAX_SESSION_QUERIES,
    MAX_UPLOAD_BYTES,
    MISSING_VALUE_MARKERS,
    QUERY_TIMEOUT_SECONDS,
    SAMPLE_ROWS,
    SENSITIVE_COLUMNS,
    temp_directory,
)
from smart_data_studio.sql_guard import validate_select

TOTAL_ROWS_COLUMN = "__total_rows"
# Enough to settle a heuristic without scanning a large file for advice.
WARNING_SAMPLE_ROWS = 200_000


def safe_name(value: str) -> str:
    """A filename fit for a log line or a download header.

    Upload names arrive from a browser and can carry newlines, quotes or path
    separators; none of those belong in a log or a Content-Disposition header.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(value).name).strip("._-")
    return (cleaned or "upload")[:100]


# Separator or camelCase boundary — playerId, player_id and player-id all split.
WORD_BREAK = r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])"


def words_in(name: str) -> list[str]:
    return [part for part in re.split(WORD_BREAK, name) if part]


def _looks_like_code(name: str) -> bool:
    """Is this column an identifier rather than a quantity?

    A zip code is entirely digits and entirely not a number: casting it drops the
    leading zero that makes it correct.
    """
    return any(part.lower() in CODE_COLUMN_WORDS for part in words_in(name))


def looks_like_identifier(name: str) -> bool:
    """Whether the name says this column identifies a row rather than measures one.

    Matched on the last word, so playerId, player_id and order_no qualify while
    PAID, VOID, PYRAMID and casino — which merely end in those letters — do not.
    Calling a measure an identifier is the failure that matters: it offers the
    measure as a key and withholds it from the totals.

    A long word also matches unanchored, because an all-lowercase compound like
    barcode or accountnumber has no boundary to split on. Four characters up
    only: the short words are exactly the ones that produce paid and casino.
    """
    parts = words_in(name)
    if parts and parts[-1].lower() in IDENTIFIER_WORDS:
        return True
    lowered = name.lower()
    return any(lowered.endswith(word) for word in IDENTIFIER_WORDS if len(word) >= 4)


class OutOfQueries(RuntimeError):
    """Raised when a workspace has spent its query budget.

    Its own type because a caller that retries per column — the profile falling
    back when SUMMARIZE refuses a table — would otherwise catch this alongside
    the failure it is handling and try the next column, and the next, turning one
    exhausted budget into a table with no statistics and no explanation.
    """


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
    encoding: str = "utf-8"

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
        size = resolved.stat().st_size
        if size > MAX_LOCAL_FILE_BYTES:
            raise ValueError(
                f"{resolved.name} is {size / 1e9:.1f}GB; the limit is "
                f"{MAX_LOCAL_FILE_BYTES / 1e9:.1f}GB."
            )
        # The check decode_csv makes on an upload: a binary file decodes to garbage
        # rather than failing, so it is named here rather than as a parse error.
        with resolved.open("rb") as handle:
            if b"\x00" in handle.read(65536):
                raise ValueError(
                    f"{resolved.name} looks binary, not CSV. Export it as text and try again."
                )
        return cls(name=resolved.name, path=resolved)

    def header_names(self) -> list[str]:
        """The header row as written.

        Checked before loading because DuckDB silently renames a duplicate to
        `a_1`, so by the time the table exists the collision has been papered over
        and the model is reading a column nobody named.
        """
        if self.content is not None:
            raw = self.content.split(b"\n", 1)[0]
        else:
            with self.path.open("rb") as handle:
                raw = handle.readline()
        # The same ladder decode_csv applies to an upload, so an accented header is
        # not replaced into a false duplicate. A line break cannot fall inside a
        # multi-byte sequence, so one line decodes on its own.
        first, _ = decode_csv(self.name, raw)
        first = first.rstrip("\r")
        # Split on whichever separator actually divides this line: assuming a comma
        # reads a semicolon or tab file as one enormous field, which disables the
        # duplicate check below.
        best: list[str] = []
        for delimiter in (",", ";", "\t", "|"):
            fields = next(csv.reader([first], delimiter=delimiter), [])
            if len(fields) > len(best):
                best = fields
        return best

    @classmethod
    def from_upload(cls, name: str, content: bytes) -> CsvSource:
        if not content:
            raise ValueError(f"{name} is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"{name} is {len(content) / 1e6:.0f}MB; the limit is "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f}MB."
            )
        text, encoding = decode_csv(name, content)
        # Normalise to UTF-8 once, here, so nothing downstream has to care.
        return cls(
            name=safe_name(name),
            content=text.encode("utf-8"),
            encoding=encoding,
        )


def decode_csv(name: str, content: bytes) -> tuple[str, str]:
    """Decode a CSV that was not necessarily written in UTF-8.

    Order matters: UTF-8 first because it fails loudly on non-UTF-8 input, cp1252
    last because it decodes almost every byte and would silently mangle UTF-8.
    """
    head = content[:65536]
    if b"\x00" in head:
        raise ValueError(f"{name} looks binary, not CSV. Export it as text and try again.")
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"{name} is not readable as UTF-8 or Windows-1252. Re-save it as UTF-8 — "
        "most spreadsheets offer 'CSV UTF-8'."
    )


def _markers() -> str:
    """The missing-value markers as a SQL list, each literal escaped.

    Backslash-N is what MySQL and Postgres write on export, and DuckDB reads a
    backslash in a plain string literally, so it needs no unescaping — only the
    quote does.
    """
    return ", ".join("'" + marker.replace("'", "''") + "'" for marker in MISSING_VALUE_MARKERS)


def transcode_to_utf8(source: Path) -> Path:
    """Rewrite a CSV as UTF-8 in a temporary file, returning where it went.

    Streamed rather than decoded whole: a local path exists so that a file larger
    than memory need not be held in it.

    Two encodings, not decode_csv's three, since this only runs on a file DuckDB
    has already refused as non-UTF-8 and utf-8-sig differs by a BOM alone.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        destination = Path(handle.name)
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            # newline="" so the file's own line endings survive the round trip.
            with (
                source.open("r", encoding=encoding, newline="") as reader,
                destination.open("w", encoding="utf-8", newline="") as writer,
            ):
                shutil.copyfileobj(reader, writer)
            return destination
        except UnicodeDecodeError:
            continue
    destination.unlink(missing_ok=True)
    raise ValueError(
        f"{source.name} is not readable as UTF-8 or Windows-1252. Re-save it as UTF-8 — "
        "most spreadsheets offer 'CSV UTF-8'."
    )


@dataclass
class TableLineage:
    """Where a table came from and what it turned into."""

    table: str
    source: str
    loaded_at: str
    rows: int
    columns: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    sql: str
    frame: pd.DataFrame
    total_rows: int
    # Sensitive columns a SELECT * picked up and query() then removed. Named so
    # the absence reads as withheld rather than as data the file does not hold.
    withheld: tuple[str, ...] = ()

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
    """Shorten long free text before it reaches the prompt."""
    trimmed = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > MAX_CELL_CHARS_TO_MODEL:
            trimmed[key] = f"{value[:MAX_CELL_CHARS_TO_MODEL]}… (truncated)"
        else:
            trimmed[key] = value
    return trimmed


class Dataset:
    """An in-memory database that becomes read-only after all CSVs are loaded."""

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        tables: tuple[str, ...],
        lineage: tuple[TableLineage, ...] = (),
        rejected: tuple[str, ...] = (),
    ):
        self.connection = connection
        self.tables = tables
        self.lineage = lineage
        # Files that could not be read, kept so the panel can name them.
        self.rejected = rejected
        self.queries_run = 0

    @classmethod
    def load(cls, sources: Iterable[CsvSource]) -> Dataset:
        source_list = list(sources)
        if not source_list:
            raise ValueError("Choose at least one CSV file")

        connection = duckdb.connect(database=":memory:")
        table_names: list[str] = []
        lineage: list[TableLineage] = []
        rejected: list[str] = []
        try:
            cls._apply_budget(connection)
            for source in source_list:
                table_name = cls._unique_table_name(source.name, table_names)
                try:
                    cls._check_header(source)
                    with logs.timed("ingest", table=table_name) as fields:
                        cls._load_source(connection, table_name, source)
                        shape = cls._check_size(connection, table_name)
                        fields.update(shape)
                except Exception as error:
                    # One unreadable file should not cost the others. A half-built
                    # table is dropped so it cannot be queried as though it loaded.
                    logs.failure("ingest.failed")
                    connection.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")
                    # Most of these errors name the file themselves.
                    name = safe_name(source.name)
                    reason = str(error)
                    rejected.append(reason if name in reason else f"{name} — {reason}")
                    continue
                table_names.append(table_name)
                lineage.append(
                    TableLineage(
                        table=table_name,
                        source=safe_name(source.name),
                        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        warnings=cls._column_warnings(connection, table_name),
                        **shape,
                    )
                )
            if not table_names:
                raise ValueError("; ".join(rejected))
            connection.execute("SET enable_external_access = false")
            connection.execute("SET lock_configuration = true")
        except Exception:
            connection.close()
            raise
        return cls(connection, tuple(table_names), tuple(lineage), tuple(rejected))

    @staticmethod
    def _apply_budget(connection: duckdb.DuckDBPyConnection) -> None:
        """Bound memory, threads and spill before the connection is locked shut.

        After the lock these settings can no longer be changed, which is the point.
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
    def _column_warnings(connection: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
        """Everything worth saying about how the columns parsed, in one pass.

        Heuristics rather than facts, so they are measured on a sample and reported
        as shares: a regex per text column over a whole large file costs tens of
        seconds for advice a sample settles just as well.
        """
        quoted = quote_identifier(table_name)
        described = [
            (row[0], str(row[1]).upper())
            for row in connection.execute(f"DESCRIBE {quoted}").fetchall()
        ]
        if not described:
            return []

        projections = ["count(*) AS total"]
        for index, (name, kind) in enumerate(described):
            column = quote_identifier(name)
            if "VARCHAR" in kind:
                projections += [
                    f"count({column}) AS present_{index}",
                    f"count(TRY_CAST({column} AS DOUBLE)) AS numeric_{index}",
                    # Decoration only: a bare digit string must not match, or a zip
                    # code is advised into losing its leading zero. Three shapes
                    # count — a currency or percent sign, a grouped number like
                    # 1.234,56, and a bare comma decimal like 2,6.
                    # \p{Sc} is every currency symbol Unicode knows. A list of the
                    # familiar few reads as complete and is not: a column of ₹, ₩ or
                    # ₽ amounts loaded as text and said nothing about it.
                    f"count_if(regexp_matches({column}, '[\\p{{Sc}}%]') OR "
                    f"regexp_matches({column}, "
                    f"'^[-+]?[0-9]{{1,3}}([.,][0-9]{{3}})+([.,][0-9]+)?$') OR "
                    f"regexp_matches({column}, '^[-+]?[0-9]+,[0-9]+$')"
                    f") AS decorated_{index}",
                    # Markers, not values: a column really holding "-" as a category
                    # is rarer than one where "-" means nobody filled it in.
                    f"count_if(lower(trim({column})) IN ({_markers()})) AS missing_{index}",
                    # A leading zero marks a code, not a quantity. Casting it away
                    # is the bug, not the fix.
                    f"count_if(regexp_matches({column}, '^0[0-9]+$')) AS coded_{index}",
                ]
            elif "DATE" in kind or "TIMESTAMP" in kind:
                projections += [
                    f"max(day({column})) AS maxday_{index}",
                    f"count({column}) AS present_{index}",
                ]
        blank = " AND ".join(f"{quote_identifier(name)} IS NULL" for name, _ in described)
        projections.append(f"count_if({blank}) AS blank_rows")
        # Does each column's own name occur among its values? A real header almost
        # never repeats as data in its own column; a data row promoted to a header
        # does, which catches a headerless file whose first row is words not numbers.
        for index, (name, _) in enumerate(described):
            literal = name.replace("'", "''")
            # Trimmed both sides: a file separated by ", " leaves a leading space on
            # every value while DuckDB strips it from the header.
            projections.append(
                f"count_if(trim(CAST({quote_identifier(name)} AS VARCHAR)) = trim('{literal}')) "
                f"AS selfnamed_{index}"
            )

        # LIMIT, not a random sample: reservoir sampling still reads every row, and
        # a *format* does not vary down the file the way a statistic does. The
        # statistical tools sample randomly for that reason; this deliberately does not.
        row = connection.execute(
            f"SELECT {', '.join(projections)} FROM "
            f"(SELECT * FROM {quoted} LIMIT {WARNING_SAMPLE_ROWS})"
        ).fetchdf()
        values = row.iloc[0].to_dict()
        total = int(values["total"])

        warnings: list[str] = []
        numeric_names = sum(1 for name, _ in described if re.fullmatch(r"[-+]?[0-9.,]+", name))
        # Compared rather than cast: these arrive as floats and an empty table makes
        # them NaN, which int() refuses. This runs before the no-rows guard below.
        self_named = sum(
            1 for index in range(len(described)) if (values.get(f"selfnamed_{index}") or 0) > 0
        )
        if numeric_names / len(described) > 0.5:
            evidence = f"{numeric_names} of {len(described)} column names are numbers"
        elif self_named / len(described) > 0.5:
            evidence = (
                f"{self_named} of {len(described)} column names also appear as values in "
                "their own column"
            )
        else:
            evidence = ""
        if evidence:
            warnings.append(
                f"The first row looks like data rather than headers ({evidence}). It has been "
                "used as the header, so that row is missing from the table."
            )
        if not total:
            warnings.append(
                "This table loaded with no rows at all. The file probably has rows with a "
                "different number of columns than the header, or only a header."
            )
            return warnings
        if int(values["blank_rows"]):
            warnings.append(
                f"{int(values['blank_rows']):,} row(s) are entirely empty, which usually means "
                "the file has rows with a different number of columns than the header."
            )

        for index, (name, kind) in enumerate(described):
            # These warnings are prose that names the column and quotes its shape,
            # and they travel into the prompt with the rest of the lineage.
            if is_sensitive(name):
                continue
            if "VARCHAR" in kind:
                present = int(values[f"present_{index}"] or 0)
                if not present:
                    continue
                numeric = int(values[f"numeric_{index}"] or 0)
                decorated = int(values[f"decorated_{index}"] or 0)
                missing = int(values[f"missing_{index}"] or 0)
                coded = int(values[f"coded_{index}"] or 0)
                if coded / present >= 0.05 or _looks_like_code(name):
                    # An identifier, not a quantity: text is the right type, and a
                    # cast would strip the leading zero.
                    continue
                # Decorated and plainly-numeric together: a column of comma decimals
                # still holds bare integers like "2", which leaves each share alone
                # under its own threshold and neither branch firing.
                convertible = (decorated + numeric) / present
                if decorated / present >= 0.2 and convertible >= 0.9:
                    warnings.append(
                        f"{name} was read as text because its values carry a currency symbol, "
                        f"thousands separator, percent sign or comma decimal "
                        f"({decorated / present:.0%} of a sample). Strip those in SQL before "
                        "summing it."
                    )
                elif numeric and 0.5 <= numeric / present < 1.0:
                    warnings.append(
                        f"{name} was read as text although {numeric / present:.0%} of a sample "
                        "is numeric — a few stray values changed the type. Sums and averages "
                        "on it will need a cast."
                    )
                if missing / present >= 0.2:
                    warnings.append(
                        f"{name} holds values that mean 'missing' ({missing / present:.0%} of a "
                        "sample: NA, null, -, ?). They are text here, not NULL, so counts and "
                        "averages will include them unless you convert them."
                    )
            elif ("DATE" in kind or "TIMESTAMP" in kind) and values.get(f"maxday_{index}"):
                # Every day at or below the twelfth means day-first and month-first
                # both parse, and the file cannot say which was meant.
                if int(values[f"maxday_{index}"]) <= 12 and int(values[f"present_{index}"] or 0):
                    warnings.append(
                        f"Every date in {name} falls on or before the 12th, so day-first and "
                        "month-first readings both fit. Check which one this file meant."
                    )
        return warnings

    @staticmethod
    def _check_header(source: CsvSource) -> None:
        names = source.header_names()
        overlong = [name for name in names if len(name) > MAX_HEADER_LENGTH]
        if overlong:
            raise ValueError(
                f"{source.name} has a column name longer than {MAX_HEADER_LENGTH} characters: "
                f"{overlong[0][:60]}… — the first row is probably data, not a header."
            )
        # Compared case-insensitively because DuckDB is: a file with both `a` and
        # `A` passed this check and then arrived as `a` and `A_1`, which is the
        # silent rename the check exists to prevent.
        folded = [name.casefold() for name in names]
        duplicates = sorted(
            {name for name, key in zip(names, folded, strict=True) if key and folded.count(key) > 1}
        )
        if duplicates:
            raise ValueError(
                f"{source.name} repeats column name(s): {', '.join(duplicates[:5])}. "
                "DuckDB matches names without regard to case and would rename the second "
                "to name_1 without saying so — rename them yourself so the right one is read."
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
            try:
                Dataset._read_csv(connection, table_name, path)
            except duckdb.InvalidInputException as error:
                # An upload arrives decoded; a path is handed to DuckDB as it sits
                # on disk. DuckDB reads UTF-8 only — its latin-1 mode refuses the
                # 0x80-0x9F range Windows uses for quotes and dashes, and
                # windows-1252 needs an extension — so the file is rewritten rather
                # than refused. A path only: content is already UTF-8.
                if temporary_path is not None or "utf-8 encoded" not in str(error):
                    raise
                temporary_path = transcode_to_utf8(path)
                Dataset._read_csv(connection, table_name, temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _read_csv(connection: duckdb.DuckDBPyConnection, table_name: str, path: Path) -> None:
        connection.execute(
            f"CREATE TABLE {quote_identifier(table_name)} AS "
            "SELECT * FROM read_csv_auto(?, header = true, sample_size = -1)",
            [str(path)],
        )

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
        payload = (
            rows
            if len(json.dumps(rows, default=str)) <= MAX_LLM_PAYLOAD_CHARS
            else self._digest(result)
        )
        if result.withheld:
            payload["withheld_columns"] = (
                f"{', '.join(result.withheld)} — withheld as sensitive on this deployment and "
                "removed from this result. Do not ask for them again and do not describe the "
                "result as every column of the table."
            )
        return payload

    def _digest(self, result: QueryResult) -> dict[str, object]:
        """A compact stand-in for a result too large to put in the prompt.

        SUMMARIZE runs over the whole result rather than the rows on screen, so
        anything quoted from here holds for every row.
        """
        stats = self.run(f"SUMMARIZE ({result.sql})").fetchdf()
        columns: dict[str, dict[str, object]] = {}
        for row in stats.to_dict(orient="records"):
            # SUMMARIZE runs over the SQL, not over the frame the withheld columns
            # were cut from, so a SELECT * would otherwise describe them here.
            if is_sensitive(str(row["column_name"])):
                continue
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

        # The digest has to fit the budget too. Sample rows go first, the statistics
        # being the part worth keeping; only then are fewer columns described.
        described = columns
        digest = build(sample, described)
        while sample and too_big(digest):
            sample = sample[: len(sample) // 2]
            digest = build(sample, described)
        while len(described) > 1 and too_big(digest):
            described = dict(list(described.items())[: len(described) // 2])
            digest = build(sample, described)
        return digest

    def run(self, sql: str, parameters: list[object] | None = None):
        """Execute one query against the workspace, under the budget and the deadline.

        Every path that scans data comes through here — the tools, the profile, the
        cohort grid, the digest — because a runaway is a runaway wherever it was
        issued from. A helper query that went straight to the connection could not
        be cancelled at all: DuckDB has no statement timeout, only the interrupt
        this arranges, so skipping it does not mean a longer limit, it means none.

        DESCRIBE and a table's row count stay off this path. They read catalog
        metadata rather than rows, they cannot run long, and they are asked often
        enough that charging them would spend the session's budget on bookkeeping.
        """
        if self.queries_run >= MAX_SESSION_QUERIES:
            raise OutOfQueries(
                f"This session has run its {MAX_SESSION_QUERIES:,} queries. "
                "Reload the data to start a fresh workspace."
            )
        self.queries_run += 1
        with self._deadline():
            return self.connection.execute(sql, parameters)

    def query(self, sql: str, row_limit: int = MAX_DISPLAY_ROWS) -> QueryResult:
        clean_sql = validate_select(sql, set(self.tables), self._withheld_columns())
        # COUNT(*) OVER () is evaluated across the whole result before LIMIT applies,
        # so a single execution yields both the page of rows and the true total.
        counted_sql = (
            f"SELECT *, COUNT(*) OVER () AS {TOTAL_ROWS_COLUMN} "
            f"FROM ({clean_sql}) AS result_rows LIMIT {int(row_limit)}"
        )
        with logs.timed("query", sql=clean_sql) as fields:
            frame = self.run(counted_sql).fetchdf()
            fields["returned"] = len(frame)
        # Positional, so a result carrying this column name of its own does not
        # shadow ours and get dropped in place of the column we added.
        total_rows = int(frame.iloc[0, -1]) if len(frame) else 0
        frame = frame.iloc[:, :-1]

        # The guard above refuses a query that names a sensitive column, which
        # leaves SELECT *: it names nothing and returns everything. So the result
        # is cut too. Both halves are needed — one cannot see a star, the other
        # cannot see through an alias — and this is the last point before the rows
        # reach the model, the screen and the export alike.
        withheld = tuple(name for name in frame.columns if is_sensitive(str(name)))
        if withheld:
            frame = frame.drop(columns=list(withheld))
            logs.event("query.withheld", columns=len(withheld))
        return QueryResult(
            sql=clean_sql,
            frame=frame,
            total_rows=total_rows,
            withheld=withheld,
        )

    def _withheld_columns(self) -> set[str]:
        """Loaded column names the operator marked sensitive, lowercased to match."""
        if not SENSITIVE_COLUMNS:
            return set()
        return {
            name.lower()
            for table in self.tables
            for name, _ in self.schema(table)
            if is_sensitive(name)
        }

    def convert_to_number(self, table: str, column: str) -> str:
        """Rewrite a text column as a number, stripping whatever kept it text.

        Which separator convention applies is decided from the data rather than
        assumed: getting it backwards turns 1.234,56 into 1.23456.
        """
        self._require_table(table)
        kinds = dict(self.schema(table))
        if column not in kinds:
            raise ValueError(f"{table} has no column {column}.")
        if "VARCHAR" not in kinds[column].upper():
            raise ValueError(
                f"{column} is already {kinds[column]}, so there is nothing to convert."
            )
        quoted, source = quote_identifier(table), quote_identifier(column)
        # Character classes rather than backslash escapes: a backslash does not
        # survive a SQL string literal on its way into DuckDB's regex.
        european = self.run(
            f"SELECT count_if(regexp_matches({source}, ',[0-9]{{1,2}}$')) > "
            f"count_if(regexp_matches({source}, '[.][0-9]{{1,2}}$')) FROM {quoted}"
        ).fetchone()[0]
        if european:
            stripped = f"regexp_replace({source}, '[^0-9.,-]', '', 'g')"
            cleaned = f"replace(replace({stripped}, '.', ''), ',', '.')"
            convention = "European (dot thousands, comma decimal)"
        else:
            cleaned = f"regexp_replace({source}, '[^0-9.-]', '', 'g')"
            convention = "plain (comma thousands, dot decimal)"

        would_fail, total = self.run(
            f"SELECT count(*) FILTER (WHERE {source} IS NOT NULL AND "
            f"TRY_CAST({cleaned} AS DOUBLE) IS NULL), count({source}) FROM {quoted}"
        ).fetchone()
        if total and would_fail / total > 0.5:
            # Converting a genuinely textual column empties it, so refuse instead.
            raise ValueError(
                f"{column} is not a number in disguise: {would_fail:,} of {total:,} values "
                "would be emptied by the conversion, so it was left as text."
            )

        # In schema order, with the conversion substituted in place: appending it
        # would move the repaired column to the end and reorder every SELECT *.
        projection = ", ".join(
            f"TRY_CAST({cleaned} AS DOUBLE) AS {source}"
            if name == column
            else quote_identifier(name)
            for name, _ in self.schema(table)
        )
        with logs.timed("column.converted", table=table, column=column):
            self.run(f"CREATE OR REPLACE TABLE {quoted} AS SELECT {projection} FROM {quoted}")
        failed = self.run(f"SELECT count(*) FROM {quoted} WHERE {source} IS NULL").fetchone()[0]
        note = f"{column} converted to a number, reading it as {convention}" + (
            f"; {failed:,} value(s) would not convert and are now empty." if failed else "."
        )
        self.lineage = tuple(
            TableLineage(
                table=item.table,
                source=item.source,
                loaded_at=item.loaded_at,
                rows=item.rows,
                columns=item.columns,
                warnings=(
                    [w for w in item.warnings if not w.startswith(f"{column} ")] + [note]
                    if item.table == table
                    else item.warnings
                ),
            )
            for item in self.lineage
        )
        return note

    def text_columns(self, table: str) -> list[str]:
        return [name for name, kind in self.schema(table) if "VARCHAR" in kind.upper()]

    def columns_mentioned_in(self, text: str) -> list[str]:
        """Loaded column names that appear in free text.

        Confirms a metric definition refers to real columns: a name absent from the
        result is the typo.
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
        another thread, which stops a runaway without losing the session.
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
            frame = self.run(
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
