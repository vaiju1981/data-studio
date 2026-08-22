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


# --- counting rows when the question asked about entities -----------------------

VISITS = b"playerId,tier,spend\n" + b"".join(
    [b"1,GOLD,10\n"] * 6 + [b"2,GOLD,90\n"] + [b"3,PLATINUM,5\n"] * 3
)


def visit_tools() -> tuple[AnalysisTools, Dataset]:
    dataset = Dataset.load([CsvSource.from_upload("visits.csv", VISITS)])
    tools = AnalysisTools(dataset)
    tools.entity_keys = {"visits": "playerId"}
    return tools, dataset


def test_a_question_about_entities_answered_by_counting_rows_is_flagged() -> None:
    """The failure this exists for: asked what share of players beat the tier above,
    a query counted visit rows and answered 4.96%. Per player it is 30.54% — six
    times out, and reported as a percentage of players.
    """
    tools, dataset = visit_tools()
    try:
        tools.question = "What percentage of players spend more than 50?"
        payload = json.loads(tools.run_sql("SELECT count(*) AS n FROM visits WHERE spend > 50"))
        assert "grain_warning" in payload
        assert "one visit rather than one player" in payload["grain_warning"]
    finally:
        dataset.close()


def test_no_warning_once_the_query_aggregates_to_the_entity() -> None:
    tools, dataset = visit_tools()
    try:
        tools.question = "What percentage of players spend more than 50?"
        for sql in (
            "SELECT count(DISTINCT playerId) AS n FROM visits WHERE spend > 50",
            "SELECT playerId, sum(spend) AS s FROM visits GROUP BY playerId",
            "SELECT count(*) FROM (SELECT DISTINCT playerId FROM visits)",
        ):
            assert "grain_warning" not in json.loads(tools.run_sql(sql)), sql
    finally:
        dataset.close()


def test_a_question_about_rows_is_left_alone() -> None:
    """Counting visits is exactly right when visits are what was asked about."""
    tools, dataset = visit_tools()
    try:
        tools.question = "How many visits spent more than 50?"
        assert "grain_warning" not in json.loads(
            tools.run_sql("SELECT count(*) AS n FROM visits WHERE spend > 50")
        )
    finally:
        dataset.close()


def test_no_warning_when_the_table_has_no_repeating_entity() -> None:
    tools, dataset = visit_tools()
    try:
        tools.entity_keys = {}
        tools.question = "What percentage of players spend more than 50?"
        assert "grain_warning" not in json.loads(tools.run_sql("SELECT count(*) AS n FROM visits"))
    finally:
        dataset.close()


# --- answering about a value nobody asked about ---------------------------------

GEO = b"geoType,amount\n" + b"".join(
    [b"LOCAL,10\n"] * 30 + [b"NATIONAL,20\n"] * 20 + [b"REGIONAL,30\n"] * 15
)


def geo_tools() -> tuple[AnalysisTools, Dataset]:
    dataset = Dataset.load([CsvSource.from_upload("visits.csv", GEO)])
    tools = AnalysisTools(dataset)
    tools.dimension_values = {"visits": {"geoType": ["LOCAL", "NATIONAL", "REGIONAL"]}}
    return tools, dataset


def test_filtering_on_a_value_the_question_did_not_name_is_flagged() -> None:
    """Asked about NATIONAL players, an answer once described REGIONAL ones. Both
    values are real and the SQL is valid, so nothing fails — the answer is simply
    about somebody else, and reads exactly as though it were not.
    """
    tools, dataset = geo_tools()
    try:
        tools.question = "How much do NATIONAL players spend?"
        payload = json.loads(
            tools.run_sql("SELECT sum(amount) AS s FROM visits WHERE geoType = 'REGIONAL'")
        )
        assert "NATIONAL" in payload["filter_warning"]
        assert "REGIONAL" in payload["filter_warning"], "the substituted value is not named"
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("question", "sql"),
    [
        # Asked for and filtered on the same value.
        (
            "How much do NATIONAL players spend?",
            "SELECT sum(amount) FROM visits WHERE geoType = 'NATIONAL'",
        ),
        # A comparison names two values and uses both.
        (
            "Compare NATIONAL and REGIONAL spend",
            "SELECT geoType, sum(amount) FROM visits "
            "WHERE geoType IN ('NATIONAL', 'REGIONAL') GROUP BY 1",
        ),
        # Naming a value while grouping is asking for a comparison, not a filter,
        # so there is nothing for it to be wrong about.
        ("Is NATIONAL the biggest segment?", "SELECT geoType, sum(amount) FROM visits GROUP BY 1"),
        # A filter the question never spoke to is the model's own scoping.
        ("What is total spend?", "SELECT sum(amount) FROM visits WHERE geoType = 'LOCAL'"),
    ],
)
def test_a_filter_that_matches_the_question_is_left_alone(question: str, sql: str) -> None:
    tools, dataset = geo_tools()
    try:
        tools.question = question
        assert "filter_warning" not in json.loads(tools.run_sql(sql)), question
    finally:
        dataset.close()


def test_a_value_inside_a_longer_word_does_not_count_as_asking_for_it() -> None:
    """ "LOCAL" sits inside "locality", and treating that as a request for the value
    would warn on questions that never named one."""
    tools, dataset = geo_tools()
    try:
        tools.question = "How do locality tiers compare?"
        assert "filter_warning" not in json.loads(
            tools.run_sql("SELECT sum(amount) AS s FROM visits WHERE geoType = 'NATIONAL'")
        )
    finally:
        dataset.close()


def test_two_tables_sharing_a_column_name_keep_separate_vocabularies() -> None:
    """Merging them checks a filter against the wrong table's values: a status of
    OPEN in one file and PAID in another are not alternatives to each other."""
    orders = Dataset.load(
        [
            CsvSource.from_upload("orders.csv", b"status,amount\nOPEN,1\nSHIPPED,2\n"),
            CsvSource.from_upload("invoices.csv", b"status,amount\nPAID,3\nDUE,4\n"),
        ]
    )
    try:
        tools = AnalysisTools(orders)
        tools.dimension_values = {
            "orders": {"status": ["OPEN", "SHIPPED"]},
            "invoices": {"status": ["PAID", "DUE"]},
        }
        tools.question = "How much is OPEN?"
        # Filtering invoices, whose vocabulary has no OPEN, must not be measured
        # against the orders vocabulary that does.
        assert "filter_warning" not in json.loads(
            tools.run_sql("SELECT sum(amount) AS s FROM invoices WHERE status = 'PAID'")
        )
        # The same question against orders is a genuine substitution.
        payload = json.loads(
            tools.run_sql("SELECT sum(amount) AS s FROM orders WHERE status = 'SHIPPED'")
        )
        assert "OPEN" in payload["filter_warning"]
    finally:
        orders.close()


def test_an_analysis_cannot_reach_a_result_the_turn_forgot() -> None:
    """Single-turn rebuilds the chat but the tools live for the session, so the
    newest result may belong to a turn this one is meant to know nothing about."""
    dataset = Dataset.load([CsvSource.from_upload("t.csv", b"g,v\nA,1\nB,2\nA,3\nB,4\n")])
    try:
        tools = AnalysisTools(dataset)
        tools.run_sql("SELECT g, v FROM t")
        tools.reset_chart(keep_history=False)
        assert "Run a SQL query" in json.loads(tools.compare_groups("g", "v"))["error"]
    finally:
        dataset.close()


MONTHLY = b"d,v\n" + b"".join(
    f"2024-{month:02d}-01,{100 + month}\n".encode() for month in range(1, 13)
)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [("forecast", ("d", "v", 3)), ("analyze_trend", ("d", "v")), ("detect_anomalies", ("d", "v"))],
)
def test_a_series_tool_cannot_reach_a_result_the_turn_forgot(tool: str, arguments: tuple) -> None:
    """The same boundary the frame analyses respect. Reading self.results[-1] here
    forecast a hidden turn's result and reported it as an answer to this one."""
    dataset = Dataset.load([CsvSource.from_upload("s.csv", MONTHLY)])
    try:
        tools = AnalysisTools(dataset)
        tools.run_sql("SELECT d, v FROM s ORDER BY d")
        tools.reset_chart(keep_history=False)
        assert "Run a SQL query" in json.loads(getattr(tools, tool)(*arguments))["error"]

        # ...and a result this turn can see is analysed as usual.
        tools.run_sql("SELECT d, v FROM s ORDER BY d")
        assert "error" not in json.loads(getattr(tools, tool)(*arguments))
    finally:
        dataset.close()


# --- a figure per group that rests on very different amounts of data -------------

UNEVEN = (
    b"order_id,region,revenue,delivery_days\n"
    + b"".join(
        # North is fastest and almost entirely unmeasured; the rest are well covered.
        f"n{i},North,100,{'1.4' if i < 4 else ''}\n".encode()
        for i in range(200)
    )
    + b"".join(
        f"{r}{i},{r},100,5.{i % 9}\n".encode() for r in ("South", "East") for i in range(200)
    )
)


def uneven_tools() -> tuple[AnalysisTools, Dataset]:
    from smart_data_studio.profile import profile_dataset

    dataset = Dataset.load([CsvSource.from_upload("orders.csv", UNEVEN)])
    tools = AnalysisTools(dataset)
    tools.null_shares = {
        profile.table_name: {
            str(row["column_name"]): float(row["null_percentage"] or 0)
            for row in profile.stats.to_dict(orient="records")
        }
        for profile in profile_dataset(dataset)
    }
    return tools, dataset


def test_a_group_average_over_almost_no_rows_is_flagged() -> None:
    """The failure this exists for: delivery_days was 8% null overall, which reads
    as unremarkable, and every one of those nulls sat in one region. That region's
    average covered 2% of it and led five answers out of five as the fastest."""
    tools, dataset = uneven_tools()
    try:
        payload = json.loads(
            tools.run_sql("SELECT region, avg(delivery_days) FROM orders GROUP BY region")
        )
        warning = payload.get("coverage_warning", "")
        assert "delivery_days" in warning and "region" in warning, payload
        assert "2%" in warning and "100%" in warning, warning
    finally:
        dataset.close()


def test_the_coverage_warning_reads_group_by_ordinals_too() -> None:
    """GROUP BY 1 is as common as GROUP BY region and names nothing."""
    tools, dataset = uneven_tools()
    try:
        payload = json.loads(
            tools.run_sql("SELECT region, avg(delivery_days) AS d FROM orders GROUP BY 1")
        )
        assert "coverage_warning" in payload, payload
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("label", "sql"),
    [
        (
            "a column filled in everywhere",
            "SELECT region, avg(revenue) FROM orders GROUP BY region",
        ),
        ("nothing grouped, so nothing to compare", "SELECT avg(delivery_days) FROM orders"),
        (
            "counting, which is what you do about nulls",
            "SELECT region, count(delivery_days) FROM orders GROUP BY region",
        ),
    ],
)
def test_the_coverage_warning_stays_quiet_when_it_has_nothing_to_say(label: str, sql: str) -> None:
    """A warning on every grouped average would teach the model to skip them all."""
    tools, dataset = uneven_tools()
    try:
        assert "coverage_warning" not in json.loads(tools.run_sql(sql)), label
    finally:
        dataset.close()


# --- a cohort assembled by hand, which gets the base wrong -----------------------

COHORT_ROWS = b"customer_id,signed_up,ordered_on,region\n" + b"".join(
    f"c{c},2026-{1 if c < 30 else 2:02d}-05,2026-{1 + k:02d}-15,{'N' if c % 3 else 'S'}\n".encode()
    for c in range(60)
    for k in range(3)
)


def cohort_tools() -> tuple[AnalysisTools, Dataset]:
    from smart_data_studio.profile import profile_dataset

    dataset = Dataset.load([CsvSource.from_upload("orders.csv", COHORT_ROWS)])
    tools = AnalysisTools(dataset)
    tools.entity_keys = {
        profile.table_name: profile.entity_key
        for profile in profile_dataset(dataset)
        if profile.entity_key
    }
    return tools, dataset


GROUPED = "date_trunc('month', CAST(ordered_on AS DATE))"


def test_a_cohort_written_by_hand_is_told_what_its_base_became() -> None:
    """Six runs in six the model wrote this instead of calling the tool, and the
    number it led with was the entities active in the first period rather than the
    cohort. The prompt names the tool; naming it there moved three runs of six.
    The correction has to arrive with the result to move the rest."""
    tools, dataset = cohort_tools()
    try:
        payload = json.loads(
            tools.run_sql(
                f"SELECT {GROUPED} AS m, count(DISTINCT customer_id) AS n FROM orders "
                "WHERE CAST(signed_up AS DATE) < '2026-02-01' GROUP BY 1"
            )
        )
        warning = payload.get("cohort_warning", "")
        assert "signed_up" in warning and "cohort" in warning, payload
        assert "cohort_window" in warning, "the warning should name what to call instead"
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("label", "sql"),
    [
        (
            "a segment counted over time is not a cohort",
            f"SELECT {GROUPED} AS m, count(DISTINCT customer_id) AS n FROM orders "
            "WHERE region = 'N' GROUP BY 1",
        ),
        (
            "counting rows rather than entities",
            f"SELECT {GROUPED} AS m, count(*) AS n FROM orders "
            "WHERE CAST(signed_up AS DATE) < '2026-02-01' GROUP BY 1",
        ),
        (
            "nothing filtered, so no cohort was chosen",
            f"SELECT {GROUPED} AS m, count(DISTINCT customer_id) FROM orders GROUP BY 1",
        ),
    ],
)
def test_the_cohort_warning_stays_quiet_on_what_is_not_a_cohort(label: str, sql: str) -> None:
    tools, dataset = cohort_tools()
    try:
        payload = json.loads(tools.run_sql(sql))
        assert "error" not in payload, payload
        assert "cohort_warning" not in payload, label
    finally:
        dataset.close()
