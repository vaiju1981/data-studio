import pytest

from smart_data_studio.sql_guard import UnsafeQuery, validate_select


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "INSERT INTO sales VALUES (1)",
        "SELECT * FROM sales; DROP TABLE sales",
        "ATTACH 'other.db' AS other",
    ],
)
def test_rejects_non_select_and_multiple_statements(sql: str) -> None:
    with pytest.raises(UnsafeQuery):
        validate_select(sql, {"sales"})


def test_rejects_unknown_tables() -> None:
    with pytest.raises(UnsafeQuery, match="missing"):
        validate_select("SELECT * FROM missing", {"sales"})


def test_allows_registered_tables_and_ctes() -> None:
    sql = validate_select(
        "WITH totals AS (SELECT region, SUM(amount) total FROM sales GROUP BY region) "
        "SELECT * FROM totals",
        {"sales"},
    )
    assert "totals" in sql
    assert "sales" in sql


def test_filesystem_escape_passes_shape_guard_for_duckdb_to_block() -> None:
    sql = validate_select("SELECT * FROM read_csv_auto('/etc/passwd')", {"sales"})
    assert "READ_CSV_AUTO" in sql.upper()
