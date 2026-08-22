from __future__ import annotations

import re

import pytest

from smart_data_studio.config import MAX_VARYING_COLUMNS
from smart_data_studio.dataset import CsvSource, Dataset, looks_like_identifier
from smart_data_studio.profile import profile_table

ROW_COUNT = 5000


def build_dataset() -> Dataset:
    """A table large enough that SUMMARIZE's sketch drifts away from the true count."""
    rows = ["id,paired,constant"]
    rows.extend(f"{index},{index // 2},same" for index in range(ROW_COUNT))
    return Dataset.load([CsvSource.from_upload("wide.csv", ("\n".join(rows) + "\n").encode())])


def load_column(name: str, values: list[float]) -> Dataset:
    body = f"{name}\n" + "\n".join(str(value) for value in values) + "\n"
    return Dataset.load([CsvSource.from_upload("t.csv", body.encode())])


def test_key_detection_uses_exact_counts_not_the_sketch() -> None:
    dataset = build_dataset()
    try:
        profile = profile_table(dataset, "wide")
        findings = "\n".join(profile.findings)

        # id is genuinely unique; paired repeats every value twice and is not a key.
        assert "id is unique across all" in findings
        assert "paired is unique across all" not in findings
        assert "constant is constant" in findings

        # approx_unique overestimates here, so no finding may quote a distinct count
        # larger than the number of rows -- that was the bug this guards.
        for number in re.findall(r"([\d,]+) distinct values", findings):
            assert int(number.replace(",", "")) <= ROW_COUNT
    finally:
        dataset.close()


def test_profile_prompt_flags_the_estimate_as_an_estimate() -> None:
    dataset = build_dataset()
    try:
        assert "approx_unique is an estimate" in profile_table(dataset, "wide").prompt_text()
    finally:
        dataset.close()


VISITS = (
    b"player_id,tier,last_seen,amount\n"
    b"1,GOLD,2024-01-05,10\n"
    b"1,GOLD,2024-02-05,20\n"
    b"2,SILVER,2024-01-06,5\n"
    b"2,SILVER,2024-03-06,7\n"
)


def test_grain_finding_names_what_is_safe_to_group_by() -> None:
    """The win-back bug: GROUP BY player_id, last_seen splits one player into two rows."""
    dataset = Dataset.load([CsvSource.from_upload("visits.csv", VISITS)])
    try:
        grain = profile_table(dataset, "visits").findings[0]
        assert "player_id repeats: 2 values across 4 rows" in grain
        # tier holds steady per player; last_seen and amount do not.
        assert "Constant within it: tier." in grain
        assert "GROUP BY player_id" in grain
    finally:
        dataset.close()


def test_no_grain_finding_when_the_key_is_the_grain() -> None:
    rows = b"order_id,amount\n1,10\n2,20\n3,30\n"
    dataset = Dataset.load([CsvSource.from_upload("orders.csv", rows)])
    try:
        findings = " ".join(profile_table(dataset, "orders").findings)
        assert "repeats" not in findings
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("playerId", True),
        ("player_id", True),
        ("id", True),
        ("visitId", True),
        ("playerID", True),
        ("paid", False),
        ("valid", False),
        ("void", False),
        ("grid", False),
        # Breadth the narrow version lacked: a file need not spell its key "id".
        ("customer_ref", True),
        ("member_key", True),
        ("order_no", True),
        ("sku", True),
        # ...without the naive suffix match that called these identifiers.
        ("casino", False),
        ("pyramid", False),
        ("turkey", False),
        ("turnover", False),
        # An all-lowercase compound has no boundary, so a long word matches loose.
        ("barcode", True),
        ("accountnumber", True),
    ],
)
def test_only_real_key_names_are_treated_as_entity_keys(name: str, expected: bool) -> None:
    assert looks_like_identifier(name) is expected


def test_a_column_merely_ending_in_id_gets_no_grain_finding() -> None:
    rows = b"paid,amount\ntrue,10\ntrue,20\nfalse,5\nfalse,7\n"
    dataset = Dataset.load([CsvSource.from_upload("payments.csv", rows)])
    try:
        assert "paid repeats" not in " ".join(profile_table(dataset, "payments").findings)
    finally:
        dataset.close()


def test_a_repeated_extreme_is_reported_as_a_missing_value_code() -> None:
    """UCI Air Quality stores "missing" as -200 inside its numeric columns. The
    value parses, the column is numeric, the average is arithmetic — and mean CO
    comes out at -34.2 against a true 2.15. Nothing about the load is wrong, which
    is exactly why it has to be said out loud.
    """
    real = [1.2, 2.6, 3.1, 2.2, 4.0, 1.6, 11.9, 0.6] * 25
    dataset = load_column("co", real + [-200.0] * 40)
    try:
        findings = " ".join(profile_table(dataset, dataset.tables[0]).findings)
        assert "-200" in findings and "missing-value code" in findings
    finally:
        dataset.close()


def test_a_sentinel_is_caught_in_a_wide_column_too() -> None:
    """The first version of this rule compared the gap to the column's own range,
    which works only when the range is narrow. Air Quality's NOx spans 2 to 1,479,
    so -200 sat well inside a range that wide and went unflagged — while its mean
    read 168.6 against a true 246.9. What marks the value is that it stands far
    from the *next* value, not from the far end of the data.
    """
    real = [float(2 + (n * 7) % 1477) for n in range(400)]
    dataset = load_column("nox", real + [-200.0] * 80)
    try:
        findings = " ".join(profile_table(dataset, dataset.tables[0]).findings)
        assert "-200" in findings and "missing-value code" in findings
    finally:
        dataset.close()


def test_a_binned_column_is_not_called_a_sentinel() -> None:
    """Values that all sit 100 apart make the lowest bin look isolated by any
    measure that ignores how far apart the others are."""
    dataset = load_column("bucket", [0.0] * 60 + [float((n % 9 + 1) * 100) for n in range(300)])
    try:
        findings = " ".join(profile_table(dataset, dataset.tables[0]).findings)
        assert "missing-value code" not in findings
    finally:
        dataset.close()


def test_a_genuine_spread_is_not_called_a_sentinel() -> None:
    """A low value that repeats is only suspicious when it sits further out than
    the whole rest of the data. An ordinary distribution with a common minimum
    must pass silently, or the warning is noise on every table."""
    dataset = load_column("amount", [0.0] * 40 + [float(n % 50) for n in range(200)])
    try:
        findings = " ".join(profile_table(dataset, dataset.tables[0]).findings)
        assert "missing-value code" not in findings
    finally:
        dataset.close()


def test_dimension_values_are_listed_so_a_column_need_not_be_queried_to_be_seen() -> None:
    """The schema says ageGroup exists; it does not say what the column holds. A
    summary of a 57-column file covered geoType and clubLevel and never mentioned
    ageGroup, city or state — all three were listed all along, but only as names,
    so the only ones described were the ones exploration happened to query.
    """
    rows = ["region,tier,note"]
    rows += [
        f"{'North' if n % 2 else 'South'},{'gold' if n % 3 else 'silver'},x{n}" for n in range(60)
    ]
    dataset = Dataset.load([CsvSource.from_upload("t.csv", ("\n".join(rows) + "\n").encode())])
    try:
        profile = profile_table(dataset, "t")
        listed = "\n".join(profile.dictionary)
        assert "region (2 values): North, South" in listed
        assert "tier (2 values): gold, silver" in listed
        # note is near-unique — an identifier, whose commonest values say nothing.
        assert "note" not in listed
        assert "Values held by the dimension columns" in profile.prompt_text()
    finally:
        dataset.close()


def test_a_wide_dimension_is_summarised_rather_than_listed_in_full() -> None:
    rows = ["city"] + [f"city{n % 200}" for n in range(2000)]
    dataset = Dataset.load([CsvSource.from_upload("t.csv", ("\n".join(rows) + "\n").encode())])
    try:
        listed = "\n".join(profile_table(dataset, "t").dictionary)
        assert "most common:" in listed
        assert listed.count(",") <= 12, "a wide column was enumerated in full"
    finally:
        dataset.close()


def test_a_sensitive_column_is_not_given_a_dictionary() -> None:
    """Sensitive columns are withheld from everything the model sees, and a list of
    their commonest values would be the most revealing form of all."""
    import smart_data_studio.dataset as dataset_module

    original = dataset_module.SENSITIVE_COLUMNS
    dataset_module.SENSITIVE_COLUMNS = ("email",)
    rows = ["email,region"] + [
        f"user{n % 3}@x.com,{'North' if n % 2 else 'South'}" for n in range(60)
    ]
    dataset = Dataset.load([CsvSource.from_upload("t.csv", ("\n".join(rows) + "\n").encode())])
    try:
        listed = "\n".join(profile_table(dataset, "t").dictionary)
        assert "@x.com" not in listed and "email" not in listed
        assert "region" in listed
    finally:
        dataset_module.SENSITIVE_COLUMNS = original
        dataset.close()


def test_the_grain_finding_says_which_dimensions_move_within_an_entity() -> None:
    """ "How do we upsell tiers" is answerable from history only if the data says a
    tier moves at all. Restricted to dimensions, because a visit id and a timestamp
    differ per row by definition and listing those buries the one that matters.
    """
    rows = ["playerId,tier,visitId"]
    rows += [f"1,GOLD,{n}" for n in range(5)]
    rows += [f"2,GOLD,{n + 5}" for n in range(5)]
    rows += [f"3,{'GOLD' if n < 3 else 'PLATINUM'},{n + 10}" for n in range(6)]
    dataset = Dataset.load([CsvSource.from_upload("v.csv", ("\n".join(rows) + "\n").encode())])
    try:
        profile = profile_table(dataset, "v")
        assert profile.entity_key == "playerId"
        grain = profile.findings[0]
        assert "tier for 1" in grain, grain
        assert "visitId" not in grain.split("history can be followed")[-1]
    finally:
        dataset.close()


def test_a_sensitive_column_never_reaches_the_prompt_through_its_statistics() -> None:
    """SUMMARIZE covers every column, so its min and max carry real cell values —
    the first and last email, the smallest and largest account number. The schema,
    the samples and the dictionary all excluded sensitive columns; this was the way
    round them.
    """
    import smart_data_studio.dataset as dataset_module
    import smart_data_studio.profile as profile_module

    original = dataset_module.SENSITIVE_COLUMNS, profile_module.SENSITIVE_COLUMNS
    dataset_module.SENSITIVE_COLUMNS = profile_module.SENSITIVE_COLUMNS = ("email",)
    rows = ["email,region"] + [
        f"user{n}@example.com,{'North' if n % 2 else 'South'}" for n in range(40)
    ]
    dataset = Dataset.load([CsvSource.from_upload("t.csv", ("\n".join(rows) + "\n").encode())])
    try:
        profile = profile_table(dataset, "t")
        prompt = profile.prompt_text()
        assert "@example.com" not in prompt, "a real address reached the model"
        assert "email" not in prompt
        assert "region" in prompt, "the rest of the profile still has to be there"
        # The owner's own panel is a different audience and keeps the full table.
        assert "email" in profile.stats["column_name"].tolist()
    finally:
        dataset_module.SENSITIVE_COLUMNS, profile_module.SENSITIVE_COLUMNS = original
        dataset.close()


def test_one_loaded_table_gains_no_key_search_and_no_extra_lines() -> None:
    """§10 of the multi-CSV plan: one CSV adds no profile query and no output. The
    key search and the shared-column report are multi-table features."""
    dataset = Dataset.load([CsvSource.from_upload("t.csv", b"a,b\n1,x\n2,y\n")])
    try:
        profile = profile_table(dataset, "t")
        assert profile.keys == [] and profile.shared == []
        text = profile.prompt_text()
        assert "How a row is identified" not in text
        assert "another table also has" not in text
    finally:
        dataset.close()


def test_two_tables_state_their_keys_and_shared_columns() -> None:
    """Said before anything joins, so the right key is used first time rather than
    a wrong one being refused and rewritten."""
    dataset = Dataset.load(
        [
            CsvSource.from_upload("sessions.csv", b"assetId,day,coinIn\n1,x,10\n1,y,20\n2,x,5\n"),
            CsvSource.from_upload(
                "assets.csv", b"assetId,day,maker\n1,x,IGT\n1,y,IGT\n2,x,BALLY\n"
            ),
        ]
    )
    try:
        text = profile_table(dataset, "assets").prompt_text()
        assert "How a row is identified" in text
        assert "another table also has" in text
        assert "assetId repeats" in text, text
    finally:
        dataset.close()


def test_a_column_that_cannot_be_summarised_costs_only_its_own_statistics() -> None:
    """A real 962MB asset file carries NaN in a DOUBLE column, and stddev over NaN
    raises OutOfRange in DuckDB. That took the whole table's profile down with it,
    so a file that loaded and queried perfectly well could not be profiled — and a
    file that cannot be profiled cannot be used.
    """
    body = b"id,ok,broken\n1,10.5,1.0\n2,20.5,nan\n3,30.5,3.0\n"
    dataset = Dataset.load([CsvSource.from_upload("t.csv", body)])
    try:
        # The column really does defeat SUMMARIZE, or this test proves nothing.
        import duckdb

        with pytest.raises(duckdb.OutOfRangeException):
            dataset.connection.execute("SUMMARIZE t").fetchdf()

        profile = profile_table(dataset, "t")
        assert "ok" in profile.stats["column_name"].tolist(), "a good column lost its statistics"
        findings = " ".join(profile.findings)
        assert "broken" in findings and "NaN" in findings
        assert dataset.query("SELECT count(*) AS n FROM t").frame.iloc[0, 0] == 3
    finally:
        dataset.close()


def test_a_row_id_whose_sketch_undercounts_is_not_taken_for_the_entity() -> None:
    """approx_unique comes from a sketch that drifts. A row id reading 172 of 180
    is a candidate key ahead of the real one, and once it is chosen the table looks
    like it is already at entity grain, so nothing about grain is reported at all —
    which is any table carrying both a row id and an entity id.
    """
    rows = ["customer_id,ticket_id,plan"]
    for customer in range(60):
        for visit in range(3):
            rows.append(f"c{customer},t{customer}_{visit},{'pro' if visit else 'basic'}")
    csv = ("\n".join(rows) + "\n").encode()

    dataset = Dataset.load([CsvSource.from_upload("tickets.csv", csv)])
    try:
        grain = next(
            (item for item in profile_table(dataset, "tickets").findings if "repeats" in item), ""
        )
        assert grain.startswith("customer_id repeats: 60 values across 180 rows"), grain
    finally:
        dataset.close()


def test_a_column_that_changes_for_a_handful_of_entities_is_still_named() -> None:
    """Ranked by count alone, the rarest-changing column is always the first thing
    a cap discards — and it is the one worth naming, because everywhere else it
    reads as a property of the entity. Grouping or filtering on it then silently
    splits or drops entities.

    Nine varying columns so the cap bites; the interesting one changes for one
    customer in sixty and would sit last in a top-N.
    """
    varying = {
        f"attr_{index}": count for index, count in enumerate([50, 45, 40, 35, 30, 25, 20, 15])
    }
    varying["signup_source"] = 1  # the rare one, last by volume

    header = "customer_id,ticket_id," + ",".join(varying) + "\n"
    rows = []
    for customer in range(60):
        for visit in range(3):
            cells = [
                f"v{visit}" if customer < count and visit else "v0" for count in varying.values()
            ]
            rows.append(f"c{customer},t{customer}_{visit}," + ",".join(cells))
    csv = (header + "\n".join(rows) + "\n").encode()

    dataset = Dataset.load([CsvSource.from_upload("tickets.csv", csv)])
    try:
        finding = next(
            item for item in profile_table(dataset, "tickets").findings if "repeats" in item
        )
        assert "signup_source for 1" in finding, f"the rare column was dropped: {finding}"
        assert "attr_0 for 50" in finding, "the most-varying column should still lead"
        # The cap still applies, and it is the middle that goes: nine varying
        # columns, eight named, and the one dropped is the least informative.
        named = [column for column in varying if column in finding]
        assert len(named) == MAX_VARYING_COLUMNS, named
        assert "attr_4" not in finding, finding
    finally:
        dataset.close()


def test_a_numeric_attribute_is_named_and_per_row_values_are_not() -> None:
    """Reading only text columns hid a tier or a store number stored as an integer.
    Reading every column instead let per-row values and measures drown the list, so
    both are excluded: one by how widely it changes, the other by name.
    """
    rows = ["customer_id,tier_code,status,seen_at,amount"]
    for customer in range(50):
        for visit in range(3):
            rows.append(
                f"c{customer},{2 if customer < 3 and visit else 1},"
                f"{'open' if visit else 'new'},2024-01-0{visit + 1},{100 + customer * visit}"
            )
    csv = ("\n".join(rows) + "\n").encode()

    dataset = Dataset.load([CsvSource.from_upload("accounts.csv", csv)])
    try:
        assert dict(dataset.schema("accounts"))["tier_code"] == "BIGINT"
        finding = next(
            item for item in profile_table(dataset, "accounts").findings if "repeats" in item
        )
        assert "tier_code for 3" in finding, f"the numeric attribute was hidden: {finding}"
        # status and seen_at change for every customer that has more than one row,
        # so they are per-row values; amount is a measure by name.
        for per_row in ("status", "seen_at", "amount"):
            assert f"{per_row} for" not in finding, f"{per_row} should not be named: {finding}"
    finally:
        dataset.close()
