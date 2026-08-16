import json

import pandas as pd
import pytest

from smart_data_studio.charts import ChartSpec, make_figure
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
