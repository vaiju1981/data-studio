"""Following a cohort forward, and dividing by the right number.

The bug this exists for produced a retention curve in which every figure was
arithmetically correct: each month was divided by the entities active in the
first month rather than by the cohort, so entities that joined the cohort and
first appeared later counted in the numerators and not in the base.
"""

from __future__ import annotations

import pytest

from smart_data_studio import cohorts
from smart_data_studio.dataset import CsvSource, Dataset


def signups(late_starters: int = 20) -> bytes:
    """A January cohort of 100, of which some are not seen until February.

    That gap is the whole point: 80 are active in January, 100 are in the cohort,
    and the two denominators give different answers to the same question.
    """
    rows = ["customer_id,signed_up,ordered_on"]
    for index in range(100 - late_starters):
        rows.append(f"c{index},2026-01-05,2026-01-20")
    for index in range(100 - late_starters, 100):
        rows.append(f"c{index},2026-01-05,2026-02-11")
    # Half of the cohort comes back in February.
    for index in range(50):
        rows.append(f"c{index},2026-01-05,2026-02-14")
    # A second cohort, so the tool has more than one to line up.
    for index in range(200, 240):
        rows.append(f"c{index},2026-02-03,2026-02-20")
    return ("\n".join(rows) + "\n").encode()


def loaded(body: bytes) -> Dataset:
    return Dataset.load([CsvSource.from_upload("orders.csv", body)])


def test_the_base_is_the_cohort_not_the_first_period() -> None:
    dataset = loaded(signups())
    try:
        found = cohorts.cohort_window(
            dataset, "orders", "customer_id", "signed_up", "ordered_on", "month", 3
        )
        january = next(c for c in found["cohorts"] if c["cohort"] == "2026-01-01")
        assert january["size"] == 100, "the cohort is everyone who signed up in January"

        by_offset = {item["offset"]: item for item in january["retention"]}
        assert by_offset[0]["active"] == 80
        assert by_offset[0]["rate"] == 0.8, "80 of the cohort of 100"

        # 20 late starters plus 50 returning. Against the cohort that is 70%;
        # against January's actives it would read 87.5%, which is the bug.
        assert by_offset[1]["active"] == 70
        assert by_offset[1]["rate"] == 0.7
        assert by_offset[1]["active"] / by_offset[0]["active"] == 0.875
    finally:
        dataset.close()


def test_the_reading_names_the_base_so_it_cannot_be_quoted_as_the_other_one() -> None:
    dataset = loaded(signups())
    try:
        found = cohorts.cohort_window(
            dataset, "orders", "customer_id", "signed_up", "ordered_on", "month", 3
        )
        assert "cohort's own size" in found["reading"]
        assert "not a share of the entities active at offset 0" in found["reading"]
    finally:
        dataset.close()


def test_activity_before_the_cohort_is_reported_not_explained() -> None:
    """Two players had visits dated days before they registered. Left unreported
    the answer reached for a reason — a data entry error, a system offset — in
    every run. The count is a fact; the reason is not in the data."""
    body = signups().decode() + "c900,2026-03-01,2026-02-01\n"
    dataset = loaded(body.encode())
    try:
        found = cohorts.cohort_window(
            dataset, "orders", "customer_id", "signed_up", "ordered_on", "month", 3
        )
        early = found["activity_before_the_cohort_started"]
        assert early["entities"] == 1
        assert "do not offer a reason" in early["note"]
    finally:
        dataset.close()


def test_the_cohorts_shown_are_the_recent_ones() -> None:
    """Taking from the front hid the cohort being asked about behind two years of
    finished ones."""
    rows = ["customer_id,signed_up,ordered_on"]
    for month in range(1, 13):
        for index in range(5):
            day = f"2025-{month:02d}-05"
            rows.append(f"m{month}_{index},{day},{day}")
    dataset = loaded(("\n".join(rows) + "\n").encode())
    try:
        found = cohorts.cohort_window(
            dataset, "orders", "customer_id", "signed_up", "ordered_on", "month", 2
        )
        shown = [item["cohort"] for item in found["cohorts"]]
        assert found["cohorts_found"] == 12
        assert shown[-1] == "2025-12-01", shown
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("arguments", "because"),
    [
        (("nosuch", "customer_id", "signed_up", "ordered_on", "month"), "Unknown table"),
        (("orders", "nope", "signed_up", "ordered_on", "month"), "not found"),
        (("orders", "customer_id", "signed_up", "ordered_on", "fortnight"), "period must be"),
        (("orders", "customer_id", "customer_id", "ordered_on", "month"), "did not parse"),
    ],
)
def test_cohort_window_refuses_what_it_cannot_answer(arguments, because) -> None:
    dataset = loaded(signups())
    try:
        with pytest.raises(cohorts.NotCohortable, match=because):
            cohorts.cohort_window(dataset, *arguments)
    finally:
        dataset.close()


def test_an_entity_with_two_start_dates_belongs_to_one_cohort() -> None:
    """A start date on every activity row can disagree with itself.

    Read per row, one customer sat in January and again in February: counted in
    both cohort sizes and both sets of numerators, which is the one thing a
    cohort cannot be. Each entity is placed in its earliest start, and the
    disagreement is reported rather than quietly resolved.
    """
    rows = ["customer_id,signed_up,ordered_on"]
    rows.append("c1,2026-01-05,2026-01-20")
    rows.append("c1,2026-02-05,2026-02-20")  # the same customer, a later start
    for index in range(2, 30):
        rows.append(f"c{index},2026-01-06,2026-01-21")
    dataset = loaded(("\n".join(rows) + "\n").encode())
    try:
        found = cohorts.cohort_window(
            dataset, "orders", "customer_id", "signed_up", "ordered_on", "month", 3
        )
    finally:
        dataset.close()

    starts = {cohort["cohort"]: cohort["size"] for cohort in found["cohorts"]}
    assert starts == {"2026-01-01": 29}, "c1 was counted in a February cohort as well"
    assert found["entities_with_more_than_one_start"]["entities"] == 1


def test_activity_rows_need_not_repeat_the_start_date() -> None:
    """An event table commonly writes the signup on the first row and leaves it
    blank after.

    Requiring it on every row dropped all the later ones, so two users who signed
    up in January and came back in February came back as offset 0 alone — a
    retention curve with the retention missing from it.
    """
    rows = ["customer_id,signed_up,ordered_on"]
    for index in range(30):
        rows.append(f"c{index},2026-01-05,2026-01-20")
        rows.append(f"c{index},,2026-02-11")  # the signup is not repeated
    dataset = loaded(("\n".join(rows) + "\n").encode())
    try:
        found = cohorts.cohort_window(
            dataset, "orders", "customer_id", "signed_up", "ordered_on", "month", 3
        )
    finally:
        dataset.close()

    cohort = found["cohorts"][0]
    assert cohort["size"] == 30
    assert [(step["offset"], step["active"]) for step in cohort["retention"]] == [(0, 30), (1, 30)]
