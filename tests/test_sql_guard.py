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


@pytest.mark.parametrize("operator", ["UNION", "UNION ALL", "EXCEPT", "INTERSECT"])
def test_allows_set_operations_over_registered_tables(operator: str) -> None:
    """Stacking period files and set-differencing two of them are the two most
    natural multi-CSV questions. Requiring a Select root refused both, while the
    same query wrapped in a subquery passed — arbitrary as well as wrong."""
    sql = validate_select(
        f"SELECT id FROM sales {operator} SELECT id FROM regions", {"sales", "regions"}
    )
    assert operator.split()[0] in sql.upper()


def test_a_set_operation_is_still_checked_for_unknown_tables() -> None:
    with pytest.raises(UnsafeQuery, match="missing"):
        validate_select("SELECT id FROM sales UNION SELECT id FROM missing", {"sales"})


def test_the_refusal_says_which_mistake_was_made() -> None:
    """Told it had written two statements when it had written one set operation,
    the model went looking for a semicolon it had never typed."""
    with pytest.raises(UnsafeQuery, match="Only one statement"):
        validate_select("SELECT 1; SELECT 2", {"sales"})
    with pytest.raises(UnsafeQuery, match="Only a SELECT"):
        validate_select("INSERT INTO sales SELECT * FROM regions", {"sales", "regions"})


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


def test_flat_predicates_are_not_mistaken_for_nesting() -> None:
    """A WHERE clause of ten ANDs parses as a leaning binary tree, and counting
    every AST child read that as thirteen levels deep and refused it — an ordinary
    filter, turned away by the guard meant for a runaway generated query."""
    flat = "SELECT * FROM sales WHERE " + " AND ".join(f"c{index} = {index}" for index in range(10))
    assert "c9" in validate_select(flat, {"sales"})


def test_genuinely_nested_queries_are_still_counted() -> None:
    """The other half: the old measure called three subqueries twelve levels and
    let them through, so it was refusing the wrong thing in both directions."""
    sql = "SELECT * FROM sales"
    for _ in range(20):
        sql = f"SELECT * FROM ({sql})"
    with pytest.raises(UnsafeQuery, match="queries deep"):
        validate_select(sql, {"sales"})


def test_a_column_withheld_as_sensitive_cannot_be_named() -> None:
    """Hiding a column from the schema is not the same as refusing it in SQL, and
    an alias would carry it past any check made on the result."""
    for sql in (
        "SELECT ssn FROM people",
        "SELECT ssn AS taxid FROM people",
        "SELECT name FROM people WHERE ssn IS NOT NULL",
        "SELECT * FROM (SELECT ssn AS x FROM people)",
    ):
        with pytest.raises(UnsafeQuery, match="withheld as sensitive"):
            validate_select(sql, {"people"}, {"ssn"})
    assert "name" in validate_select("SELECT name FROM people", {"people"}, {"ssn"})
