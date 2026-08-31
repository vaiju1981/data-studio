"""CSV ingestion and the locked-down DuckDB workspace."""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import openpyxl
import pandas as pd

from smart_data_studio import logs
from smart_data_studio.config import (
    ALLOW_LOCAL_PATHS,
    CODE_COLUMN_WORDS,
    DIGEST_SAMPLE_ROWS,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_LIMIT,
    DUCKDB_THREADS,
    IDENTIFIER_WORDS,
    MAX_CELL_CHARS_TO_MODEL,
    MAX_DISPLAY_ROWS,
    MAX_EXPORT_ROWS,
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
    PERSONAL_DATA_SHARE,
    QUERY_TIMEOUT_SECONDS,
    SAMPLE_ROWS,
    SENSITIVE_COLUMNS,
    TIME_ZONE,
    temp_directory,
)
from smart_data_studio.sql_guard import redact_literals, validate_select

TOTAL_ROWS_COLUMN = "__total_rows"
# Shapes that say a value is about a person rather than about the business. Both
# are deliberately narrow: this warning's whole worth is that it is worth reading,
# and a rule that fires on every long number would be ignored within a day.
EMAIL_SHAPE = r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$"
# Thirteen to nineteen digits behind a major issuer's prefix, spaces and dashes
# allowed. An account number of the same length does not begin 4, 51-55, 34, 37
# or 6011, which is what keeps ordinary identifiers out of this.
CARD_SHAPE = r"^(4|5[1-5]|3[47]|6011)[0-9 -]{11,17}$"
# Enough of the file to hold any header a load would accept: MAX_INGEST_COLUMNS
# names of MAX_HEADER_LENGTH characters do not reach a tenth of it. A header that
# does not end inside this much of the file is the "first row is data" case, and
# the checks below are what say so.
HEADER_SCAN_BYTES = 1024 * 1024
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


def is_text(kind: str) -> bool:
    """Whether this column holds text, rather than something built out of it.

    `"VARCHAR" in kind` reads a nested column as text: DuckDB describes a JSON
    object as `STRUCT("name" VARCHAR)` and a list of strings as `VARCHAR[]`, so
    the regex meant for free text was handed a struct and ingest failed on an
    ordinary nested JSON file.
    """
    return kind.strip().upper() == "VARCHAR"


def quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


class _CsvByteCounter:
    """Count UTF-8 output from pandas without retaining the rendered CSV."""

    def __init__(self) -> None:
        self.size = 0

    def write(self, text: str) -> int:
        self.size += len(text.encode("utf-8"))
        return len(text)


# A cell starting with one of these is a formula to Excel, Sheets and LibreOffice,
# whatever the CSV quoting says. Tab and carriage return are here because both are
# stripped before the cell is read, exposing whatever follows.
_FORMULA_STARTERS = ("=", "+", "-", "@", "\t", "\r")
# A negative number is not a formula, and prefixing it would corrupt an ordinary
# column of them that happened to be read as text.
_PLAIN_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def _defuse(value: object) -> object:
    if not isinstance(value, str) or not value.startswith(_FORMULA_STARTERS):
        return value
    if _PLAIN_NUMBER.fullmatch(value):
        return value
    return "'" + value


def defuse_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Stop a downloaded cell from running when the file is opened.

    A cell reading `=cmd|' /C calc'!A1` in the source CSV comes back out of the
    export unchanged, and a spreadsheet evaluates it. CSV quoting does not help:
    the value is a formula once it is a cell. The convention for "this is text" is
    a leading apostrophe, which spreadsheets strip on display.

    Applied to the export and to the size counted for it, so the ceiling is
    measured on the bytes that are actually written.
    """
    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    if text_columns.empty:
        return frame
    return frame.assign(**{str(name): frame[name].map(_defuse) for name in text_columns})


def csv_size(frame: pd.DataFrame, header: bool = True) -> int:
    counter = _CsvByteCounter()
    frame.to_csv(counter, index=False, header=header)
    return counter.size


def check_column_names(source_name: str, names: list[str]) -> None:
    """Refuse a header that would arrive as something other than it says.

    Applies to any format that names its own columns. DuckDB matches names
    without regard to case and silently renames the second `a` to `a_1`, so by the
    time the table exists the collision has been papered over and the model is
    reading a column nobody named.
    """
    overlong = [name for name in names if len(name) > MAX_HEADER_LENGTH]
    if overlong:
        raise ValueError(
            f"{source_name} has a column name longer than {MAX_HEADER_LENGTH} characters: "
            f"{overlong[0][:60]}… — the first row is probably data, not a header."
        )
    folded = [name.casefold() for name in names]
    duplicates = sorted(
        {name for name, key in zip(names, folded, strict=True) if key and folded.count(key) > 1}
    )
    if duplicates:
        raise ValueError(
            f"{source_name} repeats column name(s): {', '.join(duplicates[:5])}. "
            "DuckDB matches names without regard to case and would rename the second "
            "to name_1 without saying so — rename them yourself so the right one is read."
        )


def _read_csv(connection: duckdb.DuckDBPyConnection, table_name: str, path: Path) -> None:
    connection.execute(
        f"CREATE TABLE {quote_identifier(table_name)} AS "
        "SELECT * FROM read_csv_auto(?, header = true, sample_size = -1)",
        [str(path)],
    )


# What each suffix is read as. The CSV family goes through CsvSource, which
# carries the encoding, delimiter and header machinery that only text files need;
# the rest name their own columns and types, so none of that applies to them.
CSV_SUFFIXES = (".csv", ".tsv", ".txt")
PARQUET_SUFFIXES = (".parquet", ".pq")
JSON_SUFFIXES = (".json", ".ndjson", ".jsonl")
EXCEL_SUFFIXES = (".xlsx", ".xlsm")
SUPPORTED_SUFFIXES = CSV_SUFFIXES + PARQUET_SUFFIXES + JSON_SUFFIXES + EXCEL_SUFFIXES
# Unwrapped before the suffix underneath decides the reader.
COMPRESSED_SUFFIXES = (".gz", ".zip")


@dataclass
class DataFileSource:
    """A Parquet, JSON or Excel file, as an upload or a local path.

    Apart from CsvSource because none of the CSV hazards exist here: there is no
    delimiter to sniff, no encoding to guess, and no first row that might be data
    wearing a header's clothes. What these have instead is Excel's several sheets,
    which is why a source yields parts rather than one table.
    """

    name: str
    kind: str  # one of "parquet", "json", "excel"
    path: Path | None = None
    content: bytes | None = None
    # A sheet parsed once and kept, because check() and create() both read it.
    _sheets: dict[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if (self.path is None) == (self.content is None):
            raise ValueError("A file source needs exactly one of path or content")

    def parts(self) -> list[str]:
        """The sheet names of a workbook, or one unnamed part for a single table.

        Read with openpyxl rather than pandas because pandas parses every sheet to
        answer this, and naming the sheets should not cost reading them.
        """
        if self.kind != "excel":
            return [""]
        book = openpyxl.load_workbook(self._excel_source(), read_only=True)
        try:
            return list(book.sheetnames)
        finally:
            book.close()

    def check(self, connection: duckdb.DuckDBPyConnection, part: str = "") -> None:
        if self.kind == "excel":
            frame = self._sheet(part)
            check_column_names(f"{self.name} ({part})", [str(name) for name in frame.columns])

    def create(
        self, connection: duckdb.DuckDBPyConnection, table_name: str, part: str = ""
    ) -> None:
        quoted = quote_identifier(table_name)
        if self.kind == "excel":
            frame = self._sheet(part)
            if not len(frame.columns):
                raise ValueError(f"{self.name} sheet {part} is empty")
            # Registered under a name of our own rather than relied on to be found
            # as a local variable, which is what DuckDB's replacement scan would do.
            connection.register("_incoming", frame)
            try:
                connection.execute(f"CREATE TABLE {quoted} AS SELECT * FROM _incoming")
            finally:
                connection.unregister("_incoming")
            return
        reader = "read_parquet" if self.kind == "parquet" else "read_json_auto"
        with self._on_disk() as path:
            connection.execute(f"CREATE TABLE {quoted} AS SELECT * FROM {reader}(?)", [str(path)])

    def _sheet(self, part: str) -> pd.DataFrame:
        """One sheet as a frame. Parsed on first use, then kept, because the name
        check and the load both read it."""
        if part not in self._sheets:
            # The engine is named rather than guessed, so an .xlsm is read by the
            # one that handles it rather than refused.
            self._sheets[part] = pd.read_excel(
                self._excel_source(), sheet_name=part, engine="openpyxl"
            )
        return self._sheets[part]

    def _excel_source(self):
        """A fresh reader over the workbook, since each parse consumes one."""
        return io.BytesIO(self.content) if self.content is not None else self.path

    @contextmanager
    def _on_disk(self):
        """The file as a path, which is what DuckDB's readers take."""
        if self.path is not None:
            yield self.path
            return
        suffix = Path(self.name).suffix or ".dat"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(self.content)
            temporary = Path(handle.name)
        try:
            yield temporary
        finally:
            temporary.unlink(missing_ok=True)


def _kind_of(name: str) -> str:
    """Which reader this filename asks for, refusing anything unsupported."""
    suffix = Path(name).suffix.lower()
    if suffix in CSV_SUFFIXES:
        return "csv"
    if suffix in PARQUET_SUFFIXES:
        return "parquet"
    if suffix in JSON_SUFFIXES:
        return "json"
    if suffix in EXCEL_SUFFIXES:
        return "excel"
    raise ValueError(
        f"{safe_name(name)} is a {suffix or 'file with no'} extension, which is not one this "
        f"reads: {', '.join(SUPPORTED_SUFFIXES)}, optionally .gz or .zip."
    )


def _decompressed(name: str, content: bytes) -> tuple[str, bytes]:
    """The file inside a .gz or .zip, and the name it carries.

    Bounded by the same ceiling a local file has, because the ratio between a zip
    and what it holds is chosen by whoever made it and can be enormous.
    """
    suffix = Path(name).suffix.lower()
    if suffix == ".gz":
        inner = Path(name).stem or "data.csv"
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as handle:
            unpacked = handle.read(MAX_LOCAL_FILE_BYTES + 1)
    else:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1:
                listed = ", ".join(item.filename for item in members[:5]) or "nothing"
                raise ValueError(
                    f"{safe_name(name)} holds {len(members)} files ({listed}); this reads a zip "
                    "of exactly one. Extract it and load the files you want."
                )
            inner = Path(members[0].filename).name
            with archive.open(members[0]) as handle:
                unpacked = handle.read(MAX_LOCAL_FILE_BYTES + 1)
    if len(unpacked) > MAX_LOCAL_FILE_BYTES:
        raise ValueError(
            f"{safe_name(name)} expands to more than "
            f"{MAX_LOCAL_FILE_BYTES / 1e9:.1f}GB; the limit is on what it holds, not its own size."
        )
    if not unpacked:
        raise ValueError(f"{safe_name(name)} is empty inside")
    return inner, unpacked


def source_from_upload(name: str, content: bytes):
    """The right source for an uploaded file, unwrapping .gz and .zip first."""
    if Path(name).suffix.lower() in COMPRESSED_SUFFIXES:
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"{safe_name(name)} is {len(content) / 1e6:.0f}MB; the limit is "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f}MB."
            )
        name, content = _decompressed(name, content)
    kind = _kind_of(name)
    if kind == "csv":
        return CsvSource.from_upload(name, content)
    if not content:
        raise ValueError(f"{safe_name(name)} is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"{safe_name(name)} is {len(content) / 1e6:.0f}MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f}MB."
        )
    return DataFileSource(name=safe_name(name), kind=kind, content=content)


def source_from_path(path: str | Path):
    """The right source for a local path, unwrapping .gz and .zip first."""
    if not ALLOW_LOCAL_PATHS:
        raise PermissionError(
            "Loading from a server path is disabled on this deployment. Upload the file."
        )
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() in COMPRESSED_SUFFIXES:
        # Read as bytes because the ceiling below is on what it expands to, and a
        # compressed file small enough to sit on disk is small enough to hold.
        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        return source_from_upload(resolved.name, resolved.read_bytes())
    kind = _kind_of(resolved.name)
    if kind == "csv":
        return CsvSource.from_path(resolved)
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    size = resolved.stat().st_size
    if size > MAX_LOCAL_FILE_BYTES:
        raise ValueError(
            f"{resolved.name} is {size / 1e9:.1f}GB; the limit is "
            f"{MAX_LOCAL_FILE_BYTES / 1e9:.1f}GB."
        )
    return DataFileSource(name=resolved.name, kind=kind, path=resolved)


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

    def _prefix(self) -> tuple[str, str, bool]:
        """The leading chunk decoded, the encoding it was written in, and whether
        the read stopped short of the end of the file."""
        if self.content is not None:
            raw = self.content[:HEADER_SCAN_BYTES]
            truncated = len(self.content) > HEADER_SCAN_BYTES
        else:
            with self.path.open("rb") as handle:
                raw = handle.read(HEADER_SCAN_BYTES)
            truncated = len(raw) == HEADER_SCAN_BYTES
        text, encoding = decode_csv(self.name, _whole_characters(raw))
        return text, encoding, truncated

    @contextmanager
    def _sniffable(self):
        """The file as DuckDB will read it, for the sniffer to look at.

        A path already in UTF-8 is handed over as it stands, so the sniffer sees
        the whole file exactly as the loader will. Anything else is rewritten:
        DuckDB reads UTF-8 only, so sniffing Windows-1252 bytes as they sit
        returns nothing, the delimiter falls back to a guess, and the load then
        transcodes and sniffs the *converted* file — the two disagreeing by a
        different route. A cp1252 file headed `a;b;c;d,x,x` was guessed as
        semicolons here, read as commas there, and arrived as x and x_1.

        A rewritten copy is only the leading chunk, and it is cut back to the last
        line break. Handed a file ending mid-row the sniffer read a 71-column
        comma file as one pipe-separated field, and its header came back as a
        single 1,400-character column name.
        """
        text, encoding, truncated = self._prefix()
        if self.path is not None and encoding.startswith("utf-8"):
            yield self.path
            return
        if truncated:
            text = text[: text.rfind("\n") + 1] or text
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", encoding="utf-8", newline="", delete=False
        ) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        try:
            yield temporary
        finally:
            temporary.unlink(missing_ok=True)

    def dialect(self, connection: duckdb.DuckDBPyConnection) -> tuple[str, str] | None:
        """The delimiter and quote DuckDB will actually read this file with.

        Deciding separately is how the check and the reader disagree.
        `a;b;c;d,x,x` is four fields read as semicolons and three read as commas:
        the header check picked semicolons and saw no duplicate, DuckDB picked
        commas, and the second `x` arrived as `x_1` without a word — the silent
        rename the check exists to prevent, reached by a different road.

        None when the file cannot be sniffed at all, which read_csv_auto then
        reports in its own words.
        """
        try:
            with self._sniffable() as path:
                row = connection.execute(
                    "SELECT Delimiter, Quote FROM sniff_csv(?)", [str(path)]
                ).fetchone()
        except duckdb.Error:
            return None
        if not row or not row[0]:
            return None
        delimiter, quote = row[0], row[1] or ""
        return delimiter, (quote if len(quote) == 1 else '"')

    def header_names(self, dialect: tuple[str, str] | None = None) -> list[str]:
        """The header row as written.

        Checked before loading because DuckDB silently renames a duplicate to
        `a_1`, so by the time the table exists the collision has been papered over
        and the model is reading a column nobody named.

        One logical record, not one line. CSV allows a newline inside a quoted
        field, and a header cut at the first physical newline is a different,
        shorter header: `a,"note\nwrapped",a` came back as ['a', 'note'], the two
        `a` columns never met each other, and DuckDB performed exactly the silent
        rename this check exists to prevent.
        """
        # Decoded with the same ladder decode_csv applies to an upload, so an
        # accented header is not replaced into a false duplicate.
        text, _, _ = self._prefix()
        # The reader's own dialect where it could be sniffed. Otherwise whichever
        # separator divides the record into the most fields — assuming a comma
        # reads a semicolon or tab file as one enormous field, which disables the
        # duplicate check below entirely.
        candidates = [dialect] if dialect else [(mark, '"') for mark in (",", ";", "\t", "|")]
        best: list[str] = []
        for delimiter, quote in candidates:
            try:
                reader = csv.reader(io.StringIO(text), delimiter=delimiter, quotechar=quote)
                fields = next(reader, [])
            except csv.Error:
                continue
            if len(fields) > len(best):
                best = fields
        return [name.rstrip("\r") for name in best]

    def parts(self) -> list[str]:
        """One table per CSV. Excel is the format that splits; this is the answer
        every other format gives."""
        return [""]

    def check(self, connection: duckdb.DuckDBPyConnection, part: str = "") -> None:
        check_column_names(self.name, self.header_names(self.dialect(connection)))

    def create(
        self, connection: duckdb.DuckDBPyConnection, table_name: str, part: str = ""
    ) -> None:
        temporary_path: Path | None = None
        path = self.path
        if self.content is not None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary_file:
                temporary_file.write(self.content)
                temporary_path = Path(temporary_file.name)
            path = temporary_path

        try:
            try:
                _read_csv(connection, table_name, path)
            except duckdb.InvalidInputException as error:
                # An upload arrives decoded; a path is handed to DuckDB as it sits
                # on disk. DuckDB reads UTF-8 only — its latin-1 mode refuses the
                # 0x80-0x9F range Windows uses for quotes and dashes, and
                # windows-1252 needs an extension — so the file is rewritten rather
                # than refused. A path only: content is already UTF-8.
                if temporary_path is not None or "utf-8 encoded" not in str(error):
                    raise
                temporary_path = transcode_to_utf8(path)
                _read_csv(connection, table_name, temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

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


def _whole_characters(raw: bytes) -> bytes:
    """The prefix of a chunk that does not end mid-character.

    decode_csv falls back to cp1252, which decodes every byte and never raises, so
    a chunk cut through a multi-byte sequence would be read as Windows-1252 in its
    entirety — turning every accented header in a UTF-8 file into mojibake, and a
    pair of identical names into two different ones. A genuinely cp1252 file fails
    all four trims and is handed over whole, which is correct for it.
    """
    for trim in range(4):
        candidate = raw[: len(raw) - trim]
        try:
            candidate.decode("utf-8")
        except UnicodeDecodeError:
            continue
        return candidate
    return raw


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


def _personal_data_note(name: str, present: int, emails: int, cards: int) -> str | None:
    """Say when a column's *values* are personal, whatever the column is called.

    `SDS_SENSITIVE_COLUMNS` matches names, so it protects `email` and misses
    `notes` — and a hosted model endpoint receives the profile, the samples and
    every query result. Naming the column is the operator's decision; noticing it
    is this code's job, and it cannot be made by reading the header alone.
    """
    if is_sensitive(name):
        return None  # already withheld; saying so again helps nobody
    for kind, count in (("email addresses", emails), ("payment card numbers", cards)):
        share = count / present
        if share >= PERSONAL_DATA_SHARE:
            return (
                f"{name} holds what look like {kind} ({share:.0%} of a sample). It is not "
                "withheld — the schema, samples and query results from it are sent to the "
                "model. Add it to SDS_SENSITIVE_COLUMNS to keep it out of everything the "
                "model sees."
            )
    return None


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
    # Columns the operator marked sensitive. They were dropped as the table was
    # built, so this is the only record that the file had them at all.
    withheld: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    sql: str
    frame: pd.DataFrame
    total_rows: int
    # What the guards said about this query. Carried on the result rather than only
    # returned to the model, because a warning the model chose not to repeat is a
    # warning nobody saw — and the same reasoning already puts the SQL and the row
    # count on screen whether the answer mentions them or not.
    warnings: dict[str, str] = field(default_factory=dict)

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
    def load(cls, sources: Iterable[CsvSource | DataFileSource]) -> Dataset:
        source_list = list(sources)
        if not source_list:
            raise ValueError("Choose at least one file")

        connection = duckdb.connect(database=":memory:")
        table_names: list[str] = []
        lineage: list[TableLineage] = []
        rejected: list[str] = []
        try:
            cls._apply_budget(connection)
            for source in source_list:
                try:
                    parts = source.parts()
                except Exception as error:
                    # A workbook that cannot even be opened names no sheets, and
                    # that is one rejection rather than none.
                    logs.failure("ingest.failed")
                    rejected.append(cls._rejection(source.name, error))
                    continue
                for part in parts:
                    # A sheet becomes its own table, so its name has to reach the
                    # table name — two sheets of one workbook are not one table.
                    # Two names for one thing. The table name is built from the
                    # stem, because Path reads "book.xlsx Orders" as a suffix of
                    # ".xlsx Orders" and the sheet would never survive into it. The
                    # origin is what a person reads in the lineage panel, where the
                    # file and the sheet are both worth seeing.
                    stem = f"{Path(source.name).stem} {part}" if part else source.name
                    origin = (
                        f"{safe_name(source.name)} ({safe_name(part)})"
                        if part
                        else safe_name(source.name)
                    )
                    table_name = cls._unique_table_name(stem, table_names)
                    try:
                        source.check(connection, part)
                        with logs.timed("ingest", table=table_name) as fields:
                            source.create(connection, table_name, part)
                            withheld = cls._withhold_sensitive(connection, table_name)
                            shape = cls._check_size(connection, table_name)
                            fields.update(shape)
                    except Exception as error:
                        # One unreadable file — or one unreadable sheet — should not
                        # cost the others. A half-built table is dropped so it cannot
                        # be queried as though it loaded.
                        logs.failure("ingest.failed")
                        connection.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")
                        rejected.append(cls._rejection(origin, error))
                        continue
                    table_names.append(table_name)
                    lineage.append(
                        TableLineage(
                            table=table_name,
                            source=origin,
                            loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            warnings=cls._column_warnings(connection, table_name),
                            withheld=withheld,
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
        # Bounded because memory_limit alone does not bound this. An in-memory
        # database offloads table data here once the budget is reached, so what
        # memory_limit refuses to hold, the disk holds instead — up to 90% of the
        # volume, which is DuckDB's default and no bound worth the name.
        connection.execute(f"SET max_temp_directory_size='{DUCKDB_TEMP_LIMIT}'")
        # Pinned, or a timestamp carrying an offset is bucketed by the host's own
        # zone and the same question answers differently on two machines.
        connection.execute(f"SET TimeZone='{TIME_ZONE}'")

    @staticmethod
    def _withhold_sensitive(connection: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
        """Drop the operator's sensitive columns out of the table as it is built.

        Filtering them on the way out cannot be made to work. Hiding them from the
        schema left SELECT * returning them; cutting them from the result left a
        bare table name, which DuckDB reads as a struct of the whole row; refusing
        that left `FROM people AS t(w, x, y, z)`, which renames every column
        positionally without naming one, and `[COLUMNS(*)]`, which packs them into
        a list. Each is a different DuckDB feature and the next release will add
        another, because none of them is doing anything wrong — they are only
        reshaping a table that should never have held the column.

        So it does not. DROP COLUMN is a catalog edit, instant on any size of
        table, and after it there is nothing left to reshape.
        """
        if not SENSITIVE_COLUMNS:
            return []
        quoted = quote_identifier(table_name)
        names = [row[0] for row in connection.execute(f"DESCRIBE {quoted}").fetchall()]
        sensitive = [name for name in names if is_sensitive(name)]
        if len(sensitive) == len(names):
            raise ValueError(
                f"Every column in {table_name} is withheld as sensitive on this deployment "
                f"({', '.join(sensitive[:5])}), so there would be nothing left to analyse."
            )
        for name in sensitive:
            connection.execute(f"ALTER TABLE {quoted} DROP COLUMN {quote_identifier(name)}")
        return sensitive

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
            if is_text(kind):
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
                    # Personal data by shape rather than by column name, because
                    # the column that carries it is usually called notes.
                    f"count_if(regexp_matches(trim({column}), '{EMAIL_SHAPE}')) AS emails_{index}",
                    f"count_if(regexp_matches(replace(trim({column}), '-', ''), "
                    f"'{CARD_SHAPE}')) AS cards_{index}",
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
            if is_text(kind):
                present = int(values[f"present_{index}"] or 0)
                if not present:
                    continue
                numeric = int(values[f"numeric_{index}"] or 0)
                decorated = int(values[f"decorated_{index}"] or 0)
                missing = int(values[f"missing_{index}"] or 0)
                coded = int(values[f"coded_{index}"] or 0)
                personal = _personal_data_note(
                    name,
                    present,
                    int(values[f"emails_{index}"] or 0),
                    int(values[f"cards_{index}"] or 0),
                )
                if personal:
                    warnings.append(personal)
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
    def _rejection(label: str, error: Exception) -> str:
        """What to show for a file that could not be read. Most of these errors
        name the file themselves, and repeating it reads as two problems."""
        name = safe_name(label)
        reason = str(error)
        return reason if name in reason else f"{name} — {reason}"

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
        # The table holds only what was loaded, so this needs no filter. The count
        # comes from the lineage because that is now the only record the file ever
        # had those columns, and an unexplained gap invites the model to guess.
        withheld = {item.table: len(item.withheld) for item in self.lineage}
        sections = []
        for table in self.tables:
            columns = ", ".join(
                f"{quote_identifier(name)} {kind}" for name, kind in self.schema(table)
            )
            hidden = withheld.get(table, 0)
            note = f" — {hidden} column(s) withheld as sensitive, not loaded" if hidden else ""
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

        SUMMARIZE runs over the whole result rather than the rows on screen, so
        anything quoted from here holds for every row.
        """
        stats = self.run(f"SUMMARIZE ({result.sql})").fetchdf()
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
        with logs.timed("query", sql=redact_literals(clean_sql)) as fields:
            frame = self.run(counted_sql).fetchdf()
            fields["returned"] = len(frame)
        # Positional, so a result carrying this column name of its own does not
        # shadow ours and get dropped in place of the column we added.
        total_rows = int(frame.iloc[0, -1]) if len(frame) else 0
        return QueryResult(sql=clean_sql, frame=frame.iloc[:, :-1], total_rows=total_rows)

    def export_size(self, sql: str, row_limit: int = MAX_EXPORT_ROWS) -> int:
        """Exact CSV bytes for the capped result, counted without retaining them.

        Priced from the page on screen this was a guess, and a bad one: the first
        five thousand rows of a result are not its widest. A note column holding
        "x" at the top and four hundred characters lower down under-priced a
        50,000-row export by thirty-nine times, which is no bound at all.

        The LIMIT belongs inside the query: the UI exports at most that many rows,
        and counting the rest falsely refuses a result only because it has a long
        tail. DuckDB yields bounded chunks and pandas writes them into a counter,
        so quoting, nested values, blobs and Unicode are measured exactly without
        first allocating the full DataFrame or CSV.
        """
        # Validated here too. Callers pass a result's own SQL, which was validated
        # on the way in, but a method that interpolates whatever it is handed is one
        # refactor away from being the way round the guard.
        capped = (
            f"SELECT * FROM ({validate_select(sql, set(self.tables), self._withheld_columns())}) "
            f"AS export_rows LIMIT {int(row_limit)}"
        )
        # The deadline is opened around the fetch as well. run()'s own timer is
        # cancelled the moment it returns, and the chunks are pulled afterwards —
        # so without this the one path that streams is the one path with no
        # deadline on it.
        with self._deadline():
            cursor = self.run(capped)
            total = 0
            first = True
            while True:
                frame = cursor.fetch_df_chunk()
                if frame.empty:
                    if first:
                        total += csv_size(defuse_formulas(frame))
                    break
                total += csv_size(defuse_formulas(frame), header=first)
                first = False
        return total

    def _withheld_columns(self) -> set[str]:
        """Names dropped as sensitive, so asking for one gets a reason.

        Not a guard — the columns are not in the workspace and no query could
        reach them. DuckDB would say the column does not exist, which reads like
        a typo and invites the model to try three spellings of it.
        """
        return {name.lower() for item in self.lineage for name in item.withheld}

    def convert_to_number(self, table: str, column: str) -> str:
        """Rewrite a text column as a number, stripping whatever kept it text.

        Which separator convention applies is decided from the data rather than
        assumed: getting it backwards turns 1.234,56 into 1.23456.
        """
        self._require_table(table)
        kinds = dict(self.schema(table))
        if column not in kinds:
            raise ValueError(f"{table} has no column {column}.")
        if not is_text(kinds[column]):
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
        # Only the values the conversion could not read. Counting every NULL after
        # the fact charges it for the blanks that were already there — a column
        # with one empty cell and nothing wrong with it reported one failure.
        failed = int(total) - int(self.run(f"SELECT count({source}) FROM {quoted}").fetchone()[0])
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
                # Carried through: dropped here, one conversion stopped schema_text
                # disclosing the withheld columns and left the guard's refusal to
                # decay into DuckDB's "column not found", which reads like a typo.
                withheld=item.withheld,
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
        return [name for name, kind in self.schema(table) if is_text(kind)]

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
            frame = self.run(
                f"SELECT * FROM {quote_identifier(table)} LIMIT {int(limit)}"
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
