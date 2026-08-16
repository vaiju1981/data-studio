from __future__ import annotations

import re

from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_table

ROW_COUNT = 5000


def build_dataset() -> Dataset:
    """A table large enough that SUMMARIZE's sketch drifts away from the true count."""
    rows = ["id,paired,constant"]
    rows.extend(f"{index},{index // 2},same" for index in range(ROW_COUNT))
    return Dataset.load([CsvSource.from_upload("wide.csv", ("\n".join(rows) + "\n").encode())])


def test_key_detection_uses_exact_counts_not_the_sketch() -> None:
    dataset = build_dataset()
    try:
        profile = profile_table(dataset, "wide")
        findings = "\n".join(profile.findings)

        # id is genuinely unique; paired repeats every value twice and is not a key.
        assert "id is a unique key" in findings
        assert "paired is a unique key" not in findings
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
