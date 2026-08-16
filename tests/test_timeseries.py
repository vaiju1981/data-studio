from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smart_data_studio import timeseries
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.tools import AnalysisTools


def monthly(values: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"month": index, "amount": values})


def test_partial_trailing_period_is_dropped_before_modelling() -> None:
    """A part-covered final month otherwise turns a flat series into a decline."""
    frame = monthly([100.0] * 24 + [30.0])
    series = timeseries.prepare(frame, "month", "amount")

    assert len(series.values) == 24
    assert series.notes and "Dropped the last period" in series.notes[0]
    # The forecast tracks the real level rather than the stub.
    ahead = timeseries.forecast(series, 3)["forecast"]
    assert all(95 < point["value"] < 105 for point in ahead)


def test_a_genuine_dip_at_the_boundary_is_kept() -> None:
    frame = monthly([100.0] * 24 + [95.0])
    assert len(timeseries.prepare(frame, "month", "amount").values) == 25


def test_forecast_reports_its_accuracy_against_do_nothing_baselines() -> None:
    rng = np.random.default_rng(0)
    flat = list(100 + rng.normal(0, 1, 30))
    result = timeseries.forecast(timeseries.prepare(monthly(flat), "month", "amount"), 12)

    assert len(result["forecast"]) == 12
    assert all(point["low_80"] < point["value"] < point["high_80"] for point in result["forecast"])
    # The comparison always travels with the forecast, whichever way it lands.
    accuracy = result["accuracy"]
    assert {"model_mape_pct", "repeat_last_value_mape_pct", "history_mean_mape_pct"} <= set(
        accuracy
    )
    assert "beats" in accuracy["verdict"] or "does not beat" in accuracy["verdict"]


def test_trend_is_measured_and_seasonality_reported_only_when_assessable() -> None:
    rising = timeseries.decompose(
        timeseries.prepare(monthly([100 + 5 * i for i in range(30)]), "month", "amount")
    )
    assert rising["trend_direction"] == "rising"
    assert rising["total_change_pct"] > 100

    short = timeseries.decompose(
        timeseries.prepare(monthly([100.0 + i for i in range(10)]), "month", "amount")
    )
    assert "Not assessed" in short["seasonality"]


def seasonal(count: int, seed: int, amplitude: float = 20.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(
        100 + amplitude * np.sin(np.arange(count) * 2 * np.pi / 12) + rng.normal(0, 1, count)
    )


def spiked(values: list[float], position: int, delta: float) -> list[float]:
    values = list(values)
    values[position] += delta
    return values


@pytest.mark.parametrize(
    ("label", "values", "expected"),
    [
        (
            "flat with one spike",
            spiked(list(100 + np.random.default_rng(1).normal(0, 1, 30)), 17, 300.0),
            ["2025-06-01"],
        ),
        ("flat and clean", list(100 + np.random.default_rng(3).normal(0, 1, 30)), []),
        (
            "two spikes far apart",
            spiked(
                spiked(list(100 + np.random.default_rng(7).normal(0, 1, 36)), 10, 40.0), 25, -45.0
            ),
            ["2024-11-01", "2026-02-01"],
        ),
        ("seasonal with one spike", spiked(seasonal(48, 5), 30, 90.0), ["2026-07-01"]),
        ("seasonal and clean", seasonal(60, 9), []),
        ("seasonal with one dip", spiked(seasonal(72, 11), 50, -70.0), ["2028-03-01"]),
    ],
)
def test_anomalies_find_real_breaks_without_inventing_them(
    label: str, values: list[float], expected: list[str]
) -> None:
    """Seasonal peaks must not read as anomalies, and clean noise must flag nothing."""
    series = timeseries.prepare(monthly(values), "month", "amount")
    found = timeseries.anomalies(series)["anomalies"]
    assert sorted(item["period"] for item in found) == sorted(expected), label


def test_unusable_input_is_reported_not_raised() -> None:
    with pytest.raises(timeseries.NotEnoughData, match="Column not found"):
        timeseries.prepare(monthly([1.0] * 10), "nope", "amount")
    with pytest.raises(timeseries.NotEnoughData, match="at least"):
        timeseries.prepare(monthly([1.0, 2.0]), "month", "amount")

    uneven = pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-05-01", "2024-06-01", "2024-09-01", "2024-12-01"]
            ),
            "amount": [1.0] * 6,
        }
    )
    with pytest.raises(timeseries.NotEnoughData, match="evenly spaced"):
        timeseries.prepare(uneven, "month", "amount")


def test_series_tools_reach_the_model_as_readable_json() -> None:
    rows = ["month,amount"]
    rows += [f"2024-{m:02d}-01,{100 + m}" for m in range(1, 13)]
    rows += [f"2025-{m:02d}-01,{110 + m}" for m in range(1, 13)]
    dataset = Dataset.load([CsvSource.from_upload("t.csv", ("\n".join(rows) + "\n").encode())])
    try:
        tools = AnalysisTools(dataset)
        assert json.loads(tools.forecast("month", "amount", 6))["error"]  # no query yet

        tools.run_sql("SELECT month, amount FROM t ORDER BY month")
        assert len(json.loads(tools.forecast("month", "amount", 6))["forecast"]) == 6
        assert json.loads(tools.analyze_trend("month", "amount"))["trend_direction"] == "rising"
        assert "anomalies" in json.loads(tools.detect_anomalies("month", "amount"))
        assert "Column not found" in json.loads(tools.forecast("month", "nope", 3))["error"]
    finally:
        dataset.close()


def test_a_steady_decline_at_the_boundary_survives() -> None:
    """Only a step out of line with the series marks a partial period."""
    falling = [100.0 * (0.93**index) for index in range(24)]  # ends at 82% of the median
    series = timeseries.prepare(monthly(falling), "month", "amount")
    assert len(series.values) == 24, "a genuine decline was removed"
    assert series.notes and "kept as a real change" in series.notes[0]

    # A stub that breaks the step is still removed.
    stubbed = timeseries.prepare(monthly([100.0] * 24 + [30.0]), "month", "amount")
    assert len(stubbed.values) == 24
    assert "Dropped the last period" in stubbed.notes[0]


def test_backtest_survives_a_zero_in_the_held_out_window() -> None:
    values = [100.0] * 20 + [0.0, 100.0, 100.0, 100.0]
    result = timeseries.forecast(timeseries.prepare(monthly(values), "month", "amount"), 3)
    accuracy = result["accuracy"]

    assert accuracy["zero_periods_skipped"] == 1
    for key in ("model_mape_pct", "repeat_last_value_mape_pct", "history_mean_mape_pct"):
        assert np.isfinite(accuracy[key]), f"{key} is not a finite number"
    # Infinity is not valid JSON, whatever json.dumps permits.
    assert "Infinity" not in json.dumps(result)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("MS", 12),
        ("ME", 12),
        ("M", 12),
        ("W-SUN", 52),
        ("W", 52),
        ("QS-OCT", 4),
        ("D", 7),
        ("h", 24),
    ],
)
def test_calendar_aliases_keep_their_season(alias: str, expected: int) -> None:
    """pandas reports W-SUN and QS-OCT, not W and QS."""
    assert timeseries.seasonal_period(alias) == expected


def test_weekly_data_is_recognised_as_seasonal() -> None:
    index = pd.date_range("2022-01-02", periods=120, freq="W")
    frame = pd.DataFrame({"week": index, "amount": [100.0 + i % 52 for i in range(120)]})
    assert timeseries.prepare(frame, "week", "amount").season == 52
