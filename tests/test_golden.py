"""Fixtures with answers worked out by hand, and properties that must always hold.

The question bank proves the agent behaves on real data; these prove the machinery
underneath is right on data whose correct answer is not in dispute. They run in the
fast suite because none of them needs a model.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smart_data_studio import analysis, timeseries
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.tools import AnalysisTools


def load(name: str, rows: str) -> Dataset:
    return Dataset.load([CsvSource.from_upload(name, rows.encode())])


# --- fixtures whose answers are arithmetic, not opinion -------------------------


def test_a_join_does_not_multiply_rows_or_totals() -> None:
    """The classic fan-out: joining a dimension must not inflate a measure."""
    facts = load("facts.csv", "id,region_id,amount\n1,10,100\n2,10,200\n3,20,300\n")
    try:
        dimension = "region_id,name\n10,North\n20,South\n"
        both = Dataset.load(
            [
                CsvSource.from_upload(
                    "facts.csv",
                    facts.query("SELECT * FROM facts").frame.to_csv(index=False).encode(),
                ),
                CsvSource.from_upload("regions.csv", dimension.encode()),
            ]
        )
        try:
            joined = both.query(
                "SELECT r.name, SUM(f.amount) AS total FROM facts f "
                "JOIN regions r ON f.region_id = r.region_id GROUP BY r.name ORDER BY r.name"
            )
            assert joined.frame.to_dict("records") == [
                {"name": "North", "total": 300},
                {"name": "South", "total": 300},
            ]
            assert both.query("SELECT SUM(amount) AS t FROM facts").frame.iloc[0, 0] == 600
        finally:
            both.close()
    finally:
        facts.close()


def test_nulls_are_excluded_from_averages_but_counted_in_rows() -> None:
    dataset = load("n.csv", "a\n10\n\n20\n")
    try:
        result = dataset.query(
            "SELECT count(*) AS rows, count(a) AS present, avg(a) AS mean FROM n"
        )
        assert result.frame.to_dict("records") == [{"rows": 3, "present": 2, "mean": 15.0}]
    finally:
        dataset.close()


def test_simpsons_paradox_survives_the_tools() -> None:
    """Each group favours B while the pooled total favours A.

    A driver sweep must show the reversal rather than smooth it away, which is the
    whole reason to sweep dimensions instead of trusting the aggregate.
    """
    rows = [
        "arm,segment,outcome",
        *["A,small,1"] * 63,
        *["A,small,0"] * 24,
        *["A,large,1"] * 6,
        *["A,large,0"] * 7,
        *["B,small,1"] * 8,
        *["B,small,0"] * 2,
        *["B,large,1"] * 57,
        *["B,large,0"] * 33,
    ]
    frame = pd.DataFrame([line.split(",") for line in rows[1:]], columns=rows[0].split(","))
    frame["outcome"] = frame["outcome"].astype(int)

    pooled = frame.groupby("arm")["outcome"].mean()
    within = frame.groupby(["segment", "arm"])["outcome"].mean()
    assert pooled["A"] > pooled["B"]  # pooled prefers A
    assert within[("small", "B")] > within[("small", "A")]  # every segment prefers B
    assert within[("large", "B")] > within[("large", "A")]

    # The sweep reports the segment split, so the reversal is visible.
    drivers = analysis.rank_drivers(
        frame.assign(one=1).groupby(["arm", "segment"], as_index=False)["outcome"].sum(),
        "outcome",
        "arm",
    )
    assert drivers["drivers"][0]["dimension"] == "segment"


def test_an_imbalanced_comparison_reports_both_group_sizes() -> None:
    """Twenty-to-one, and the small group still gets its own row count."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "g": ["A"] * 500 + ["B"] * 25,
            "v": np.concatenate([rng.normal(1, 0.5, 500), rng.normal(9, 0.5, 25)]),
        }
    )
    result = analysis.compare_groups(frame, "g", "v")
    assert [group["rows"] for group in result["groups"]] == [500, 25]
    assert result["test"]["effect"] == "large"


def test_negative_values_are_summed_not_dropped() -> None:
    dataset = load("r.csv", "status,amount\nsale,100\nrefund,-30\nsale,50\n")
    try:
        assert dataset.query("SELECT SUM(amount) AS t FROM r").frame.iloc[0, 0] == 120
    finally:
        dataset.close()


def test_a_partial_period_does_not_become_a_trend() -> None:
    """Twelve steady months plus a stub must not forecast a decline."""
    values = [100.0] * 24 + [18.0]
    series = timeseries.prepare(
        pd.DataFrame(
            {"month": pd.date_range("2024-01-01", periods=25, freq="MS"), "amount": values}
        ),
        "month",
        "amount",
    )
    ahead = timeseries.forecast(series, 6)["forecast"]
    assert all(95 <= point["value"] <= 105 for point in ahead)


# --- properties that must hold for any input ------------------------------------


@pytest.mark.parametrize("width", [1, 3, 12])
@pytest.mark.parametrize("height", [1, 50, 999])
def test_a_query_reports_the_row_count_it_actually_has(width: int, height: int) -> None:
    header = ",".join(f"c{index}" for index in range(width))
    row = ",".join("1" for _ in range(width))
    dataset = load("p.csv", header + "\n" + "\n".join([row] * height) + "\n")
    try:
        result = dataset.query("SELECT * FROM p")
        assert result.total_rows == height
        assert len(result.frame.columns) == width
        assert len(result.frame) == min(height, 5000)
    finally:
        dataset.close()


def test_sampling_is_reproducible() -> None:
    """Same question, same answer — the seed is fixed for exactly this reason."""
    rows = "g,v\n" + "\n".join(f"{'A' if i % 2 else 'B'},{i}" for i in range(20_000)) + "\n"
    dataset = load("s.csv", rows)
    try:
        first = AnalysisTools(dataset)
        first.run_sql("SELECT g, v FROM s")
        second = AnalysisTools(dataset)
        second.run_sql("SELECT g, v FROM s")
        assert first.compare_groups("g", "v") == second.compare_groups("g", "v")
    finally:
        dataset.close()


@pytest.mark.parametrize(
    "rows",
    [
        "a\n1\n2\n",
        "a,b\n1,\n2,x\n",
        "a\n-1.5\n0\n",
        "a\n2024-01-01\n2024-02-01\n",
    ],
)
def test_every_tool_payload_is_valid_json(rows: str) -> None:
    dataset = load("j.csv", rows)
    try:
        tools = AnalysisTools(dataset)
        payload = tools.run_sql("SELECT * FROM j")
        parsed = json.loads(payload)
        assert "NaN" not in payload and "Infinity" not in payload
        assert isinstance(parsed, dict)
    finally:
        dataset.close()
