"""What a column means, rather than what shape it is.

The profile is strong on shape — types, cardinality, nulls, sentinels — and the
failures here are of a different kind: every value parses, the arithmetic is
valid, and the number means something other than what it is read as. Two of them
are detectable from the data itself, and those are the two tested here.

A third, the start of the business week, is not: `date_trunc('week', …)` is
Monday-based and whether that is right is a convention nobody's file records. It
is deliberately not warned about, because a note that fires on every weekly query
without knowing whether anything is wrong spends attention the notes that do know
have to share.
"""

from __future__ import annotations

import json

import pytest

from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.tools import AnalysisTools

MIXED = (
    b"region,currency,amount,booked\n"
    b"north,GBP,100,2026-01-01T23:30:00Z\n"
    b"north,JPY,15000,2026-01-02T00:30:00Z\n"
    b"south,GBP,200,2026-01-02T11:00:00Z\n"
    b"south,JPY,30000,2026-01-03T02:15:00Z\n"
)
SINGLE = b"region,currency,amount,booked\nnorth,GBP,100,2026-01-01\nsouth,GBP,200,2026-01-02\n"


@pytest.fixture
def mixed():
    dataset = Dataset.load([CsvSource.from_upload("sales.csv", MIXED)])
    try:
        yield AnalysisTools(dataset)
    finally:
        dataset.close()


@pytest.fixture
def single():
    dataset = Dataset.load([CsvSource.from_upload("sales.csv", SINGLE)])
    try:
        yield AnalysisTools(dataset)
    finally:
        dataset.close()


def warnings_for(tools: AnalysisTools, sql: str) -> dict:
    payload = json.loads(tools.run_sql(sql))
    assert "error" not in payload, payload.get("error")
    return {key: value for key, value in payload.items() if key.endswith("_warning")}


# --- adding pounds to yen -----------------------------------------------------


def test_a_total_across_two_currencies_says_so(mixed) -> None:
    """£300 plus ¥45,000 is 45,300 of nothing. The amounts parse, the column is
    numeric, and the currency sits in a column the total never reads."""
    found = warnings_for(mixed, "SELECT region, sum(amount) AS total FROM sales GROUP BY region")
    assert "currency" in found.get("currency_warning", "")
    assert "amount" in found["currency_warning"]


def test_grouping_by_the_currency_is_not_warned_about(mixed) -> None:
    """The note exists to be acted on, and this is the action."""
    found = warnings_for(
        mixed, "SELECT currency, sum(amount) AS total FROM sales GROUP BY currency"
    )
    assert "currency_warning" not in found


def test_filtering_to_one_currency_is_not_warned_about(mixed) -> None:
    """The other action. A total of the GBP rows has a unit."""
    found = warnings_for(
        mixed,
        "SELECT region, sum(amount) AS total FROM sales WHERE currency = 'GBP' GROUP BY region",
    )
    assert "currency_warning" not in found


def test_one_currency_in_the_file_is_not_a_hazard(single) -> None:
    """A file entirely in euros totals correctly, and a note on it is noise that
    costs attention the real ones need."""
    found = warnings_for(single, "SELECT region, sum(amount) AS total FROM sales GROUP BY region")
    assert "currency_warning" not in found


# --- a day that is not the day anybody means ----------------------------------


def test_a_day_taken_from_a_utc_instant_says_which_day_it_is(mixed) -> None:
    """23:30Z on the 1st and 00:30Z on the 2nd are two UTC days and one evening
    almost anywhere else. Nothing about the result looks wrong."""
    found = warnings_for(
        mixed,
        "SELECT date_trunc('day', CAST(booked AS TIMESTAMP)) AS day, count(*) AS n "
        "FROM sales GROUP BY day ORDER BY day",
    )
    assert "UTC" in found.get("timezone_warning", "")


def test_the_timezone_note_does_not_invent_a_zone(mixed) -> None:
    """The file records an offset, never a place. Naming one would be the tool
    guessing the answer to the question it is raising."""
    found = warnings_for(
        mixed,
        "SELECT date_trunc('day', CAST(booked AS TIMESTAMP)) AS day, count(*) AS n "
        "FROM sales GROUP BY day",
    )
    note = found["timezone_warning"]
    assert "has to come from whoever asked" in note
    assert "/" not in note.replace("UTC", ""), "a zone name would be a guess"


def test_a_plain_date_column_is_not_warned_about(single) -> None:
    """No offset, no instant, nothing to be wrong about."""
    found = warnings_for(
        single,
        "SELECT date_trunc('day', CAST(booked AS TIMESTAMP)) AS day, count(*) AS n "
        "FROM sales GROUP BY day",
    )
    assert "timezone_warning" not in found


def test_a_column_is_measured_once_per_workspace(mixed) -> None:
    """These notes each cost a probe query, and the session has a budget of them.
    Asked twice, the second question re-reads what the first measured."""
    warnings_for(mixed, "SELECT region, sum(amount) AS total FROM sales GROUP BY region")
    after_first = mixed.dataset.queries_run
    warnings_for(mixed, "SELECT region, sum(amount) AS t2 FROM sales GROUP BY region")
    # One query for the question itself, and none for the probe that already ran.
    assert mixed.dataset.queries_run == after_first + 1


def test_the_workspace_pins_the_zone_it_reads_instants_in(mixed) -> None:
    """Otherwise the answer depends on the machine. Measured: two events 23:30Z and
    00:30Z bucket as two days on a UTC host, one day in Los Angeles and one in
    Tokyo — the same file and the same question, three answers.

    Locked with the rest of the budget, so model-written SQL cannot move the day
    boundary either.
    """
    import duckdb

    from smart_data_studio import config

    setting = mixed.dataset.connection.execute(
        "SELECT value FROM duckdb_settings() WHERE name = 'TimeZone'"
    ).fetchone()
    assert setting[0] == config.TIME_ZONE
    with pytest.raises(duckdb.Error):
        mixed.dataset.connection.execute("SET TimeZone='Asia/Tokyo'")

    days = mixed.dataset.query(
        "SELECT date_trunc('day', booked) AS day, count(*) AS n FROM sales GROUP BY day"
    ).frame
    assert len(days) == 3, "the UTC days these four rows fall in"


# --- shapes the profile did not model -----------------------------------------


def test_rows_this_table_holds_twice_are_reported() -> None:
    """The classic bad export: the same row twice, every total over it inflated,
    and nothing about the file looking wrong."""
    from smart_data_studio.profile import profile_table

    doubled = b"region,amount\nnorth,10\nsouth,20\nnorth,10\nnorth,10\n"
    dataset = Dataset.load([CsvSource.from_upload("sales.csv", doubled)])
    try:
        profile = profile_table(dataset, "sales")
        note = next((line for line in profile.findings if "repeat another row" in line), "")
        assert note, profile.findings
        assert "2 of 4 rows" in note
    finally:
        dataset.close()


def test_a_table_with_no_repeats_says_nothing_about_them(single) -> None:
    """Most files are fine, and a finding on every one of them is noise."""
    from smart_data_studio.profile import profile_table

    profile = profile_table(single.dataset, "sales")
    assert not any("repeat another row" in line for line in profile.findings)


def test_nested_columns_are_named_with_the_way_into_them() -> None:
    """A struct listed only by name reads as a column with nothing in it — the same
    failure the value dictionary was built for, in a different shape."""
    from smart_data_studio.dataset import source_from_upload
    from smart_data_studio.profile import profile_table

    content = b'[{"id": 1, "customer": {"name": "ada"}, "tags": ["x"]}]'
    dataset = Dataset.load([source_from_upload("events.json", content)])
    try:
        profile = profile_table(dataset, "events")
        note = next((line for line in profile.findings if "Nested columns" in line), "")
        assert "customer" in note and "tags" in note
        assert "customer.name" in note, "the note has to say how to reach a field"
    finally:
        dataset.close()
