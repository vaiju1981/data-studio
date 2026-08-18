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


# --- resolving a name to the values a column actually holds ---------------------

CITIES = b"city,amount\n" + b"".join(
    [b"NORTH LAS VEGAS,1\n"] * 40
    + [b"N LAS VEGAS,1\n"] * 12
    + [b"N. LAS VEGAS,1\n"] * 2
    + [b"LAS VEGAS,1\n"] * 60
    + [b"DALLAS,1\n"] * 8
    + [b"HENDERSON,1\n"] * 5
)


def city_tools() -> tuple[AnalysisTools, Dataset]:
    dataset = Dataset.load([CsvSource.from_upload("visits.csv", CITIES)])
    return AnalysisTools(dataset), dataset


def test_find_values_finds_the_abbreviated_spellings_of_a_name() -> None:
    """The failure this exists for: filtering city = 'NORTH LAS VEGAS' on the real
    file returned 293,836 visits and silently missed 69,910 more written another
    way — a 19% undercount that looks entirely correct.

    Matching the whole phrase cannot find them, because "N LAS VEGAS" does not
    contain "north las vegas". Each word is scored separately for that reason.
    """
    tools, dataset = city_tools()
    try:
        found = json.loads(tools.find_values("visits", "city", "North Las Vegas"))
        values = [item["value"] for item in found["matches"]]
        assert "NORTH LAS VEGAS" in values
        assert "N LAS VEGAS" in values, "the abbreviated spelling was missed"
        assert "N. LAS VEGAS" in values
        assert found["matches"][0]["rows"] == 40  # commonest first
    finally:
        dataset.close()


def test_find_values_does_not_return_a_word_it_merely_shares_a_fragment_with() -> None:
    """DALLAS contains "las". Burying the real spellings under matches like that
    defeats the purpose, so only values scoring near the best are kept."""
    tools, dataset = city_tools()
    try:
        found = json.loads(tools.find_values("visits", "city", "Las Vegas"))
        assert "DALLAS" not in [item["value"] for item in found["matches"]]
    finally:
        dataset.close()


def test_find_values_says_so_when_nothing_matches() -> None:
    """A state column of two-letter codes holds no value containing "Nevada". Being
    told that is what sends the model to the code instead of filtering on nothing."""
    tools, dataset = city_tools()
    try:
        found = json.loads(tools.find_values("visits", "city", "Nevada"))
        assert found["matches"] == []
        assert "No value" in found["note"]
    finally:
        dataset.close()


def test_find_values_lists_the_commonest_when_given_nothing_to_match() -> None:
    tools, dataset = city_tools()
    try:
        found = json.loads(tools.find_values("visits", "city", ""))
        assert found["matches"][0] == {"value": "LAS VEGAS", "rows": 60}
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("table", "column"), [("visits", "nope"), ("nope", "city"), ("visits", "amount ; DROP")]
)
def test_find_values_refuses_what_is_not_in_the_schema(table: str, column: str) -> None:
    tools, dataset = city_tools()
    try:
        assert "error" in json.loads(tools.find_values(table, column, "x"))
    finally:
        dataset.close()


def test_find_values_treats_the_search_text_as_data_not_sql() -> None:
    """The words are bound parameters, never concatenated into the statement."""
    tools, dataset = city_tools()
    try:
        found = json.loads(tools.find_values("visits", "city", "'; DROP TABLE visits; --"))
        assert "error" not in found
        # The table is still there, which is the only thing this is really asking.
        assert json.loads(tools.run_sql("SELECT count(*) AS n FROM visits"))["rows"][0]["n"] == 127
    finally:
        dataset.close()


def test_find_values_will_not_expose_a_sensitive_column() -> None:
    """Sensitive columns are withheld from everything the model sees, and listing
    their values on request would be the most direct way around that."""
    import smart_data_studio.dataset as dataset_module

    original = dataset_module.SENSITIVE_COLUMNS
    dataset_module.SENSITIVE_COLUMNS = ("city",)
    tools, dataset = city_tools()
    try:
        found = json.loads(tools.find_values("visits", "city", "Las Vegas"))
        assert "error" in found and "LAS VEGAS" not in json.dumps(found)
    finally:
        dataset_module.SENSITIVE_COLUMNS = original
        dataset.close()


def test_a_name_the_data_has_no_value_for_is_recorded_as_an_assumption() -> None:
    """Answering "how many visits from Summerlin" is useful even though no city is
    called that — the model bridges it to postcodes. The number still comes from
    SQL; the mapping came from the model, and only recording that makes it
    correctable if it is ever wrong.
    """
    tools, dataset = city_tools()
    try:
        tools.find_values("visits", "city", "Summerlin")
        assert tools.unresolved == ["Summerlin (looked for in city)"]
    finally:
        dataset.close()


def test_a_name_the_data_does_hold_records_nothing() -> None:
    tools, dataset = city_tools()
    try:
        tools.find_values("visits", "city", "Henderson")
        tools.find_values("visits", "city", "")
        assert tools.unresolved == []
    finally:
        dataset.close()
