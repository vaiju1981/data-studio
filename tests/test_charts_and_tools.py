import json

import pandas as pd
import pytest

from smart_data_studio.charts import ChartSpec, label, make_figure
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.tools import AnalysisTools


def test_chart_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"month": ["Jan"], "revenue": [10]})
    with pytest.raises(ValueError, match="profit"):
        make_figure(frame, ChartSpec(kind="bar", x="month", y="profit"))


def test_analysis_tools_return_data_and_chart_errors_without_tracebacks() -> None:
    dataset = Dataset.load([CsvSource.from_upload("sales.csv", b"month,revenue\nJan,10\nFeb,20\n")])
    try:
        tools = AnalysisTools(dataset)
        payload = json.loads(
            tools.run_sql("SELECT month, SUM(revenue) AS revenue FROM sales GROUP BY month")
        )
        assert payload["row_count"] == 2
        assert json.loads(tools.make_chart("bar", "month", "profit"))["error"]
        assert json.loads(tools.make_chart("bar", "month", "revenue"))["status"] == (
            "chart_created"
        )
        assert tools.chart is not None
    finally:
        dataset.close()


def test_chart_plots_every_row_even_when_the_model_saw_a_truncated_result() -> None:
    rows = "n,value\n" + "\n".join(f"{index},{index * 2}" for index in range(1000)) + "\n"
    dataset = Dataset.load([CsvSource.from_upload("series.csv", rows.encode())])
    try:
        tools = AnalysisTools(dataset)
        payload = json.loads(tools.run_sql("SELECT n, value FROM series ORDER BY n"))
        assert payload["truncated"] is True and payload["row_count"] == 1000

        status = json.loads(tools.make_chart("line", "n", "value"))
        assert status["rows_plotted"] == 1000
        assert len(tools.chart.data[0].x) == 1000
    finally:
        dataset.close()


def test_chart_refuses_a_result_too_large_to_draw() -> None:
    rows = "n\n" + "\n".join(str(index) for index in range(6000)) + "\n"
    dataset = Dataset.load([CsvSource.from_upload("many.csv", rows.encode())])
    try:
        tools = AnalysisTools(dataset)
        tools.run_sql("SELECT n FROM many")
        error = json.loads(tools.make_chart("histogram", "n"))["error"]
        assert "too many to chart" in error
        assert tools.chart is None
    finally:
        dataset.close()


def test_column_names_become_readable_axis_labels() -> None:
    assert label("total_theo_win") == "Total Theo Win"
    assert label("avgBet") == "Avg Bet"
    assert label("playerId") == "Player Id"
    assert label("region") == "Region"


def test_bars_are_ranked_but_dated_axes_keep_their_order() -> None:
    unsorted = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [10, 30, 20]})
    figure = make_figure(unsorted, ChartSpec(kind="bar", x="region", y="revenue"))
    assert list(figure.data[0].x) == ["B", "C", "A"]

    over_time = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "revenue": [30, 10, 20],
        }
    )
    dated = make_figure(over_time, ChartSpec(kind="bar", x="month", y="revenue"))
    assert list(dated.data[0].y) == [30, 10, 20]


def test_large_values_get_short_axis_ticks_and_sparse_lines_get_markers() -> None:
    big = pd.DataFrame({"month": range(1, 7), "revenue": [22_000_000 + i for i in range(6)]})
    figure = make_figure(big, ChartSpec(kind="line", x="month", y="revenue"))
    assert figure.layout.yaxis.tickformat == "~s"
    assert figure.data[0].mode == "lines+markers"

    small = pd.DataFrame({"month": range(1, 7), "revenue": [10, 12, 11, 13, 12, 14]})
    assert (
        make_figure(small, ChartSpec(kind="line", x="month", y="revenue")).layout.yaxis.tickformat
        is None
    )


def test_single_series_charts_use_the_brand_accent_not_plotly_default() -> None:
    """A layout colorway alone leaves Express traces blue; the sequence must be passed in."""
    from smart_data_studio.charts import PALETTE

    frame = pd.DataFrame({"region": ["A", "B"], "revenue": [10, 20]})
    figure = make_figure(frame, ChartSpec(kind="bar", x="region", y="revenue"))
    assert figure.data[0].marker.color == PALETTE[0]
