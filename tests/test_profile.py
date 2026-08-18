from __future__ import annotations

import re

import pytest

from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import _looks_like_key, profile_table

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
    ],
)
def test_only_real_key_names_are_treated_as_entity_keys(name: str, expected: bool) -> None:
    assert _looks_like_key(name) is expected


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
