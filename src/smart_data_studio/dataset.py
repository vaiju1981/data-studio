"""CSV ingestion and the locked-down DuckDB workspace."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from smart_data_studio.config import (
    DIGEST_SAMPLE_ROWS,
    MAX_DISPLAY_ROWS,
    MAX_LLM_PAYLOAD_CHARS,
    MAX_LLM_ROWS,
    SAMPLE_ROWS,
)
from smart_data_studio.sql_guard import validate_select

TOTAL_ROWS_COLUMN = "__total_rows"


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
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"CSV file not found: {resolved}")
        return cls(name=resolved.name, path=resolved)

    @classmethod
    def from_upload(cls, name: str, content: bytes) -> CsvSource:
        if not content:
            raise ValueError(f"{name} is empty")
        return cls(name=name, content=content)


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
        return {
            "columns": head.columns.tolist(),
            "rows": json.loads(head.to_json(orient="records", date_format="iso")),
            "row_count": self.total_rows,
            "truncated": self.total_rows > len(head),
        }


class Dataset:
    """An in-memory database that becomes read-only after all CSVs are loaded."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, tables: tuple[str, ...]):
        self.connection = connection
        self.tables = tables

    @classmethod
    def load(cls, sources: Iterable[CsvSource]) -> Dataset:
        source_list = list(sources)
        if not source_list:
            raise ValueError("Choose at least one CSV file")

        connection = duckdb.connect(database=":memory:")
        table_names: list[str] = []
        try:
            for source in source_list:
                table_name = cls._unique_table_name(source.name, table_names)
                cls._load_source(connection, table_name, source)
                table_names.append(table_name)
            connection.execute("SET enable_external_access = false")
            connection.execute("SET lock_configuration = true")
        except Exception:
            connection.close()
            raise
        return cls(connection, tuple(table_names))

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
            columns = ", ".join(
                f"{quote_identifier(name)} {data_type}" for name, data_type in self.schema(table)
            )
            sections.append(f"Table {quote_identifier(table)} ({columns})")
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
        clean_sql = validate_select(sql, set(self.tables))
        # COUNT(*) OVER () is evaluated across the whole result before LIMIT applies,
        # so a single execution yields both the page of rows and the true total.
        counted_sql = (
            f"SELECT *, COUNT(*) OVER () AS {TOTAL_ROWS_COLUMN} "
            f"FROM ({clean_sql}) AS result_rows LIMIT {int(row_limit)}"
        )
        frame = self.connection.execute(counted_sql).fetchdf()
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

    def sample_text(self, limit: int = SAMPLE_ROWS) -> str:
        """First rows of each table, so the model sees real values and not only types."""
        sections = []
        for table in self.tables:
            frame = self.connection.execute(
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
