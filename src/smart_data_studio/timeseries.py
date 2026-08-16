"""Trend, forecast and anomaly analysis over a query result, using statsmodels."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL

from smart_data_studio.config import MAX_FORECAST_PERIODS

# A first or last period this far below the interior median is nearly always a
# partial period rather than a real collapse. Left in, a 23-day trailing month
# turned a flat series into a fabricated 15% decline.
BOUNDARY_RATIO = 0.85
# ...and the step into it must be this many times the usual one.
BOUNDARY_STEP_MULTIPLE = 3.0
MIN_POINTS = 6
# Deliberately strict. Looser values flagged ordinary noise as anomalies on clean
# series, and inventing an anomaly is worse here than missing a marginal one.
ANOMALY_ALPHA = 0.001
MIN_CYCLES_FOR_ANOMALY = 4
# pandas reports aliases like "W-SUN", "QS-OCT" and "ME", so match on the base
# rather than the whole string or weekly and quarterly data lose their season.
SEASONAL_PERIODS = {
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "W": 52,
    "D": 7,
    "B": 5,
    "H": 24,
}


def seasonal_period(freq: str) -> int | None:
    return SEASONAL_PERIODS.get(freq.split("-")[0].upper())


class NotEnoughData(ValueError):
    """Raised when a series is too short or too irregular to model."""


@dataclass
class Series:
    values: pd.Series
    freq: str
    notes: list[str] = field(default_factory=list)

    @property
    def season(self) -> int | None:
        """Points per cycle, or None when the series cannot cover two cycles."""
        period = seasonal_period(self.freq)
        return period if period and len(self.values) >= 2 * period else None


def prepare(frame: pd.DataFrame, date_column: str, value_column: str) -> Series:
    """Turn two columns of a result into a regular series fit to model."""
    for column in (date_column, value_column):
        if column not in frame.columns:
            raise NotEnoughData(f"Column not found: {column}. Available: {list(frame.columns)}")

    working = frame[[date_column, value_column]].dropna()
    dates = pd.to_datetime(working[date_column], errors="coerce")
    if dates.isna().any():
        raise NotEnoughData(f"{date_column} does not parse as dates")

    values = pd.to_numeric(working[value_column], errors="coerce")
    if values.isna().any():
        raise NotEnoughData(f"{value_column} is not numeric")

    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates)).sort_index()
    series = series.groupby(level=0).sum()
    if len(series) < MIN_POINTS:
        raise NotEnoughData(f"Need at least {MIN_POINTS} periods, got {len(series)}")

    freq = pd.infer_freq(series.index)
    if freq is None:
        raise NotEnoughData(
            "The dates are not evenly spaced. Aggregate to whole periods in SQL "
            "(for example date_trunc('month', ...)) before analysing."
        )
    series = series.asfreq(freq)

    notes: list[str] = []
    series = _drop_partial_boundaries(series, notes)
    if len(series) < MIN_POINTS:
        raise NotEnoughData(f"Only {len(series)} complete periods remain")
    return Series(values=series, freq=freq, notes=notes)


def _drop_partial_boundaries(series: pd.Series, notes: list[str]) -> pd.Series:
    """Drop a leading or trailing period that is clearly only partly covered.

    The tool cannot see day coverage, only the aggregate, so this is a safety net.
    Filtering incomplete periods in SQL is the reliable fix, and the note says so.
    """
    for edge in ("first", "last"):
        if len(series) < MIN_POINTS + 1:
            break
        interior = series.iloc[1:] if edge == "first" else series.iloc[:-1]
        point = float(series.iloc[0] if edge == "first" else series.iloc[-1])
        neighbour = float(series.iloc[1] if edge == "first" else series.iloc[-2])
        stamp = series.index[0] if edge == "first" else series.index[-1]
        median = float(interior.median())
        if median <= 0 or point >= BOUNDARY_RATIO * median:
            continue

        # A steady decline also leaves the last point below the median. What marks
        # a partial period is that the step into it dwarfs every other step, so
        # require both before removing a point that may be a real fall.
        typical_step = float(interior.diff().abs().median())
        if typical_step > 0 and abs(point - neighbour) < BOUNDARY_STEP_MULTIPLE * typical_step:
            notes.append(
                f"The {edge} period {stamp.date()} is {point / median:.0%} of the median "
                "but moves no more than the series usually does, so it was kept as a "
                "real change. Exclude it in SQL if the period is incomplete."
            )
            continue

        notes.append(
            f"Dropped the {edge} period {stamp.date()} ({point:,.0f} is "
            f"{point / median:.0%} of the median {median:,.0f} and breaks the usual "
            "step) — it looks partly covered. Filter incomplete periods in SQL if "
            "that is wrong."
        )
        series = series.iloc[1:] if edge == "first" else series.iloc[:-1]
    return series


def forecast(series: Series, periods: int) -> dict[str, object]:
    """Forecast ahead, and say plainly whether the model beats doing nothing."""
    values = series.values
    season = series.season
    # Bound the horizon before statsmodels does it for us: zero and negative raise
    # an opaque ValueError, and a huge one overflows the timestamp range.
    limit = min(MAX_FORECAST_PERIODS, len(values))
    if periods < 1 or periods > limit:
        raise NotEnoughData(
            f"Forecast between 1 and {limit} periods. {len(values)} periods of history "
            "cannot support a longer horizon than itself."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = _fit(values, season)
        prediction = fitted.get_prediction(start=len(values), end=len(values) + periods - 1)
        frame = prediction.summary_frame(alpha=0.20)

    # A quantity that has never been negative should not be forecast negative; the
    # additive model happily projects a declining series straight through zero.
    floor_at_zero = bool((values >= 0).all())
    ahead = [
        {
            "period": str(stamp.date()),
            "value": _bounded(row["mean"], floor_at_zero),
            "low_80": _bounded(row["pi_lower"], floor_at_zero),
            "high_80": _bounded(row["pi_upper"], floor_at_zero),
        }
        for stamp, row in frame.iterrows()
    ]
    notes = list(series.notes)
    if floor_at_zero and (frame[["mean", "pi_lower"]].to_numpy() < 0).any():
        notes.append(
            "The history is never negative, so the forecast and its range were held at "
            "zero where the model projected below it. Read those periods as 'at or near "
            "zero', not as a precise figure."
        )
    return {
        "model": f"ETS(add trend, damped{f', seasonal {season}' if season else ', no seasonal'})",
        "periods_used": len(values),
        "history_mean": round(float(values.mean()), 2),
        "forecast": ahead,
        "accuracy": _backtest(values, season),
        "notes": notes,
    }


def _bounded(value: float, floor_at_zero: bool) -> float:
    number = float(value)
    return round(max(number, 0.0) if floor_at_zero else number, 2)


def _fit(values: pd.Series, season: int | None):
    return ETSModel(
        values,
        error="add",
        trend="add",
        damped_trend=True,
        seasonal="add" if season else None,
        seasonal_periods=season,
    ).fit(disp=False)


def _backtest(values: pd.Series, season: int | None) -> dict[str, object]:
    """Hold out the tail and compare the model against two do-nothing baselines.

    On flat series a seasonal model can look impressive while adding nothing over
    the mean, so the comparison travels with every forecast.
    """
    horizon = min(4, len(values) // 5)
    if horizon < 1:
        return {"note": "Too few periods to backtest"}
    train, test = values.iloc[:-horizon], values.iloc[-horizon:]

    actual = test.to_numpy()
    usable = actual != 0
    if not usable.any():
        return {"note": "Every held-out period is zero, so percentage error is undefined"}

    def mape(prediction) -> float | None:
        """Percentage error over the non-zero periods; dividing by zero gives Infinity,
        which is not valid JSON and means nothing to a reader."""
        errors = np.abs((actual[usable] - np.asarray(prediction)[usable]) / actual[usable])
        return round(float(np.mean(errors) * 100), 2)

    # Holding out the tail can leave too few points for a seasonal fit even when
    # the full series had enough. Fall back rather than lose the comparison, which
    # is the part that keeps a flat forecast honest.
    model_error = None
    for attempt in (season, None) if season else (None,):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_error = mape(_fit(train, attempt).forecast(horizon).to_numpy())
            break
        except Exception:
            continue
    if model_error is None:
        return {"note": "Backtest could not be fitted"}

    naive = mape(np.repeat(train.iloc[-1], horizon))
    mean = mape(np.repeat(train.mean(), horizon))
    best_baseline = min(naive, mean)
    skipped = int((~usable).sum())
    return {
        "held_out_periods": horizon,
        **({"zero_periods_skipped": skipped} if skipped else {}),
        "model_mape_pct": model_error,
        "repeat_last_value_mape_pct": naive,
        "history_mean_mape_pct": mean,
        "verdict": (
            "The model beats both do-nothing baselines."
            if model_error < best_baseline
            else "The model does not beat simply repeating the last value or the average. "
            "Treat the forecast as a level with a range, not as a trend."
        ),
    }


def decompose(series: Series) -> dict[str, object]:
    """Split the series into trend, season and remainder, and size each part."""
    values, season = series.values, series.season
    first, last = float(values.iloc[0]), float(values.iloc[-1])
    result: dict[str, object] = {
        "periods_used": len(values),
        "first": round(first, 2),
        "last": round(last, 2),
        "total_change_pct": round((last - first) / abs(first) * 100, 2) if first else None,
        "notes": series.notes,
    }
    if season is None:
        result["seasonality"] = (
            f"Not assessed — needs two full cycles ({seasonal_period(series.freq) or '?'} "
            f"periods each) and there are {len(values)}."
        )
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = STL(values, period=season, robust=True).fit()
    trend, seasonal, resid = fitted.trend, fitted.seasonal, fitted.resid
    result["trend_direction"] = "rising" if trend.iloc[-1] > trend.iloc[0] else "falling"
    result["trend_change_pct"] = round(
        (trend.iloc[-1] - trend.iloc[0]) / abs(trend.iloc[0]) * 100, 2
    )
    # Strengths follow Hyndman: how much of the variation each component explains.
    result["seasonal_strength"] = round(max(0.0, 1 - resid.var() / (seasonal + resid).var()), 3)
    result["trend_strength"] = round(max(0.0, 1 - resid.var() / (trend + resid).var()), 3)
    result["reading"] = "0 means the component explains nothing, 1 means it explains everything."
    return result


def _generalized_esd(values: np.ndarray, max_outliers: int) -> list[int]:
    """Rosner's test: remove the most extreme point, then retest.

    Peeling one at a time is what stops two outliers from hiding each other — a
    single spike inflates the standard deviation enough to mask its neighbour, and
    a plain z-score never recovers from that.
    """
    count = len(values)
    remaining = list(range(count))
    found: list[int] = []
    for step in range(1, max_outliers + 1):
        current = values[remaining]
        spread = current.std(ddof=1)
        if spread == 0:
            break
        worst = int(np.argmax(np.abs(current - current.mean())))
        statistic = abs(current[worst] - current.mean()) / spread
        degrees = count - step - 1
        if degrees <= 0:
            break
        critical = stats.t.ppf(1 - ANOMALY_ALPHA / (2 * (count - step + 1)), degrees)
        threshold = (
            (count - step) * critical / np.sqrt((degrees + critical**2) * (count - step + 1))
        )
        if statistic > threshold:
            found.append(remaining[worst])
        remaining.pop(worst)
    return sorted(found)


def anomalies(series: Series) -> dict[str, object]:
    """Flag periods that break the pattern, after removing seasonality."""
    values = series.values
    # A seasonal estimate from only two or three cycles is itself shaky, and
    # subtracting it invents outliers. Below that, test the raw level instead.
    period = seasonal_period(series.freq)
    season = period if period and len(values) >= MIN_CYCLES_FOR_ANOMALY * period else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = values - STL(values, period=season, robust=True).fit().seasonal if season else values
    # Centre on the median, not a fitted trend: a trend line bends toward a spike
    # and hides it.
    resid = (base - float(np.median(base))).to_numpy()

    found = _generalized_esd(resid, max_outliers=max(1, len(values) // 10))
    spread = float(np.std(resid, ddof=1)) or 1.0
    flagged = [
        {
            "period": str(values.index[position].date()),
            "value": round(float(values.iloc[position]), 2),
            "deviations": round(float(resid[position] / spread), 2),
            "direction": "high" if resid[position] > 0 else "low",
        }
        for position in found
    ]
    return {
        "method": (
            f"{'seasonally adjusted, ' if season else ''}generalized ESD at alpha {ANOMALY_ALPHA}"
        ),
        "periods_checked": len(values),
        "anomalies": sorted(flagged, key=lambda item: abs(item["deviations"]), reverse=True),
        "notes": series.notes,
    }
