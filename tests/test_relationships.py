"""M1: what the model proposes about how tables relate is a hypothesis.

Everything here tests the validator rather than the proposal, because the proposal
comes from a model and the guarantee is deterministic: a reference that does not
resolve against the loaded schema never reaches a verification query.
"""

from __future__ import annotations

import pytest

from smart_data_studio import relationships
from smart_data_studio.config import MAX_JOIN_CANDIDATES, MAX_KEY_CANDIDATES
from smart_data_studio.dataset import CsvSource, Dataset

SESSIONS = b"assetId,day,coinIn\n1,2024-01-01,10\n1,2024-01-02,20\n2,2024-01-01,30\n"
ASSETS = b"assetId,day,manufacturer\n1,2024-01-01,IGT\n1,2024-01-02,IGT\n2,2024-01-01,BALLY\n"


@pytest.fixture
def data():
    dataset = Dataset.load(
        [
            CsvSource.from_upload("sessions.csv", SESSIONS),
            CsvSource.from_upload("assets.csv", ASSETS),
        ]
    )
    try:
        yield dataset
    finally:
        dataset.close()


def join(left_cols, right_cols, left="sessions", right="assets", **extra):
    return {
        "kind": "join",
        "left": {"table": left, "columns": left_cols},
        "right": {"table": right, "columns": right_cols},
        **extra,
    }


def test_a_composite_join_resolves(data) -> None:
    found = relationships.validate(data, [join(["assetId", "day"], ["assetId", "day"])])
    assert not found.rejected
    assert str(found.joins[0]) == "sessions(assetId, day) = assets(assetId, day)"


def test_identifiers_resolve_without_case_but_keep_the_file_s_spelling(data) -> None:
    """A proposal saying assetid must resolve, and what is stored must be what the
    file calls the column — anything else produces SQL that does not run."""
    found = relationships.validate(data, [join(["ASSETID"], ["assetid"], left="SESSIONS")])
    assert not found.rejected
    assert found.joins[0].left == relationships.Ref("sessions", ("assetId",))


@pytest.mark.parametrize(
    ("proposal", "because"),
    [
        (join(["assetId"], ["assetId"], left="nosuch"), "unknown table"),
        (join(["nosuch"], ["assetId"]), "has no column"),
        (join(["assetId", "day"], ["assetId"]), "different numbers of columns"),
        (join([], ["assetId"]), "no columns given"),
        (join(["assetId", "assetId"], ["assetId", "day"]), "repeated"),
        (join(["assetId"], ["assetId"], right="sessions"), "joined to itself"),
        ({"kind": "nonsense"}, "unknown proposal kind"),
        ({"kind": "key", "table": "sessions", "columns": ["a", "b", "c", "d", "e"]}, "more than"),
    ],
)
def test_a_proposal_that_does_not_resolve_is_rejected_with_a_reason(
    data, proposal, because
) -> None:
    found = relationships.validate(data, [proposal])
    assert not found.joins and not found.keys
    assert any(because in reason for reason in found.rejected), found.rejected


def test_a_withheld_column_cannot_be_proposed_or_named_in_the_refusal(data) -> None:
    """Naming it in the rejection would defeat withholding it."""
    import smart_data_studio.dataset as dataset_module

    original = dataset_module.SENSITIVE_COLUMNS
    dataset_module.SENSITIVE_COLUMNS = ("coinin",)
    try:
        found = relationships.validate(
            data, [{"kind": "key", "table": "sessions", "columns": ["coinIn"]}]
        )
        assert not found.keys
        assert found.rejected == ["sessions: a withheld column was proposed"]
        assert "coinIn" not in found.rejected[0]
    finally:
        dataset_module.SENSITIVE_COLUMNS = original


def test_proposals_are_bounded_per_table_and_per_pair(data) -> None:
    """Forty joins between two tables cost forty verification queries for no more
    insight than four."""
    # The same join worded differently is one candidate, not several.
    same = [
        join(["assetId"], ["assetId"], reason=f"wording {n}")
        for n in range(MAX_JOIN_CANDIDATES + 3)
    ]
    found = relationships.validate(data, same)
    assert len(found.joins) == 1, "the same join counted more than once"
    assert found.joins[0].reason == "wording 0", "the first reason should survive"

    # Genuinely different column pairs do consume the budget.
    pairs = [
        (["assetId"], ["assetId"]),
        (["day"], ["day"]),
        (["assetId", "day"], ["assetId", "day"]),
        (["day", "assetId"], ["day", "assetId"]),
        (["assetId"], ["day"]),
        (["day"], ["assetId"]),
    ]
    found = relationships.validate(data, [join(left, right) for left, right in pairs])
    assert len(found.joins) == MAX_JOIN_CANDIDATES
    assert (
        sum("more than" in reason for reason in found.rejected) == len(pairs) - MAX_JOIN_CANDIDATES
    )

    distinct_keys = [["assetId"], ["day"], ["coinIn"], ["assetId", "day"], ["day", "coinIn"]]
    keys = [{"kind": "key", "table": "sessions", "columns": cols} for cols in distinct_keys]
    assert len(relationships.validate(data, keys).keys) == MAX_KEY_CANDIDATES


def test_a_malformed_proposal_does_not_stop_the_good_ones(data) -> None:
    found = relationships.validate(
        data,
        [
            {"kind": "join", "left": "not a dict", "right": None},
            join(["assetId", "day"], ["assetId", "day"]),
            "not a dict at all",
        ],
    )
    assert len(found.joins) == 1
    assert len(found.rejected) == 2


def test_nothing_proposed_is_not_an_error(data) -> None:
    found = relationships.validate(data, [])
    assert (found.keys, found.joins, found.rejected) == ([], [], [])


# --- M2: measuring a join without building it -----------------------------------


def verify_join(dataset, left_cols, right_cols):
    found = relationships.validate(dataset, [join(left_cols, right_cols)])
    assert not found.rejected, found.rejected
    return relationships.verify(dataset, found.joins[0])


def test_an_incomplete_key_is_measured_as_the_explosion_it_is(data) -> None:
    """The real case: joining sessions to assets on assetId alone multiplies every
    session by every day that machine existed. Measured by counting, so the 439x
    explosion is priced without ever being built."""
    result = verify_join(data, ["assetId"], ["assetId"])
    # sessions: rows 1+2 share assetId 1, asset rows 1+2 likewise -> 2*2 + 1*1.
    assert result.joined_rows == 5
    assert result.cardinality == "N:N"
    assert result.multiplies("sessions") and result.multiplies("assets")


def test_the_complete_key_does_not_multiply(data) -> None:
    result = verify_join(data, ["assetId", "day"], ["assetId", "day"])
    assert result.joined_rows == 3, "one row out per session"
    assert not result.multiplies("sessions")
    assert result.cardinality == "1:1"


def test_a_predicted_row_count_matches_the_join_it_declined_to_run(data) -> None:
    for left, right in ([["assetId"], ["assetId"]], [["assetId", "day"], ["assetId", "day"]]):
        predicted = verify_join(data, left, right).joined_rows
        on = " AND ".join(f"s.{c} = a.{c}" for c in left)
        actual = data.query(
            f"SELECT count(*) AS n FROM sessions s JOIN assets a ON {on}"
        ).frame.iloc[0, 0]
        assert predicted == actual, f"{left}: predicted {predicted}, actual {actual}"


def test_null_keys_are_counted_apart_from_matches() -> None:
    """DuckDB counts NULL-bearing tuples in count(DISTINCT (a, b)) while an equality
    join matches none of them, so folding them together overstates containment."""
    dataset = Dataset.load(
        [
            CsvSource.from_upload("sessions.csv", b"assetId,day,coinIn\n1,x,10\n,x,20\n"),
            CsvSource.from_upload("assets.csv", b"assetId,day,manufacturer\n1,x,IGT\n"),
        ]
    )
    try:
        result = verify_join(dataset, ["assetId"], ["assetId"])
        assert result.left.rows == 2
        assert result.left.joinable == 1, "the null-key row cannot join"
        assert result.left.null_keys == 1
        assert result.joined_rows == 1
    finally:
        dataset.close()


def test_a_join_that_drops_rows_is_partial_even_when_nothing_multiplies() -> None:
    """A multiplier of exactly 1.0 can still hide a join that keeps some rows and
    silently drops others."""
    dataset = Dataset.load(
        [
            CsvSource.from_upload("sessions.csv", b"assetId,day,coinIn\n1,x,10\n2,x,20\n"),
            CsvSource.from_upload("assets.csv", b"assetId,day,manufacturer\n1,x,IGT\n3,x,BALLY\n"),
        ]
    )
    try:
        result = verify_join(dataset, ["assetId"], ["assetId"])
        assert result.joined_rows == 1
        assert result.partial, "half of each side matched nothing"
        assert result.left.unmatched == 1 and result.right.unmatched == 1
    finally:
        dataset.close()


# --- M3: refusing before execution ----------------------------------------------


def guard(dataset, sql):
    """The refusal alone; the second element is the non-blocking note."""
    return relationships.preflight(dataset, sql, {})[0]


def test_the_incomplete_join_is_refused_before_it_runs(data) -> None:
    """The motivating failure: joining on assetId alone multiplies every session by
    every day that machine existed, and the total came out 439 times too large."""
    refusal = guard(
        data, "SELECT SUM(s.coinIn) FROM sessions s JOIN assets a ON s.assetId=a.assetId"
    )
    assert refusal and "repeats rows of s (sessions)" in refusal
    assert "add the rest of its key" in refusal


def test_the_complete_join_runs(data) -> None:
    assert (
        guard(
            data,
            "SELECT SUM(s.coinIn) FROM sessions s JOIN assets a "
            "ON s.assetId=a.assetId AND s.day=a.day",
        )
        is None
    )


def test_a_query_that_totals_nothing_is_left_alone(data) -> None:
    """Fan-out changes the row count, but with no aggregate no figure is wrong, and
    the rows are shown with their count beside them."""
    assert (
        guard(data, "SELECT s.coinIn FROM sessions s JOIN assets a ON s.assetId=a.assetId") is None
    )


def test_a_single_table_query_never_reaches_the_guard(data) -> None:
    assert guard(data, "SELECT SUM(coinIn) FROM sessions") is None


def test_aggregating_the_side_that_does_not_repeat_is_allowed(data) -> None:
    """N:1 from sessions to a unique asset key: session rows are not multiplied, so
    totalling a session measure is safe."""
    assert (
        guard(
            data,
            "SELECT a.manufacturer, SUM(s.coinIn) FROM sessions s "
            "JOIN assets a ON s.assetId=a.assetId AND s.day=a.day GROUP BY 1",
        )
        is None
    )


def test_a_derived_relation_that_fixed_its_own_grain_is_allowed(data) -> None:
    """The model writes this unprompted, and warning about it would refuse correct
    SQL: the subquery is one row per assetId whatever the base table does."""
    for inner in (
        "SELECT DISTINCT assetId FROM assets",
        "SELECT assetId, MAX(manufacturer) AS manufacturer FROM assets GROUP BY assetId",
    ):
        sql = f"SELECT SUM(s.coinIn) FROM sessions s JOIN ({inner}) a ON s.assetId=a.assetId"
        assert guard(data, sql) is None, inner


@pytest.mark.parametrize(
    ("shape", "sql"),
    [
        ("range", "SELECT SUM(s.coinIn) FROM sessions s JOIN assets a ON s.assetId>=a.assetId"),
        (
            "expression",
            "SELECT SUM(s.coinIn) FROM sessions s JOIN assets a ON lower(s.day)=lower(a.day)",
        ),
        (
            "or",
            "SELECT SUM(s.coinIn) FROM sessions s JOIN assets a "
            "ON s.assetId=a.assetId OR s.day=a.day",
        ),
        ("cross", "SELECT SUM(s.coinIn) FROM sessions s, assets a"),
    ],
)
def test_a_shape_that_cannot_be_checked_cannot_bypass_the_guard(data, shape, sql) -> None:
    """A predicate we cannot measure used to fall through as "nothing found to
    multiply", which let an unmeasurable join run precisely because it was
    unmeasurable."""
    refusal = guard(data, sql)
    assert refusal and "cannot be checked" in refusal, shape


def test_aliases_and_casing_resolve_to_the_right_tables(data) -> None:
    refusal = guard(
        data, "SELECT SUM(X.CoinIn) FROM SESSIONS AS X JOIN Assets AS Y ON X.AssetId = Y.AssetId"
    )
    assert refusal and "repeats rows of x (sessions)" in refusal


def test_using_is_understood(data) -> None:
    refusal = guard(data, "SELECT SUM(coinIn) FROM sessions JOIN assets USING (assetId)")
    assert refusal and "repeats" in refusal


def test_measured_facts_are_reused_rather_than_remeasured(data) -> None:
    cache: dict = {}
    sql = "SELECT SUM(s.coinIn) FROM sessions s JOIN assets a ON s.assetId=a.assetId"
    relationships.preflight(data, sql, cache)
    assert len(cache) == 1
    relationships.preflight(data, sql, cache)
    assert len(cache) == 1, "the same join was measured twice"


# --- the single-table regression the live bank caught ---------------------------

ONE_TABLE = b"playerId,day,theoWin\n1,2024-01-01,10\n2,2024-06-01,20\n1,2024-07-01,30\n"


@pytest.fixture
def single():
    dataset = Dataset.load([CsvSource.from_upload("t.csv", ONE_TABLE)])
    try:
        yield dataset
    finally:
        dataset.close()


@pytest.mark.parametrize(
    "sql",
    [
        "WITH top AS (SELECT playerId FROM t GROUP BY playerId) "
        "SELECT sum(t.theoWin) FROM t JOIN top ON t.playerId = top.playerId",
        "WITH p AS (SELECT playerId FROM t) "
        "SELECT sum(t.theoWin) FROM t JOIN p ON t.playerId = p.playerId",
        "SELECT sum(x.theoWin) FROM t x JOIN t y ON x.playerId = y.playerId",
        "WITH g AS (SELECT playerId, row_number() OVER () rn FROM t) "
        "SELECT sum(t.theoWin) FROM t JOIN g ON t.playerId = g.playerId",
        "SELECT sum(theoWin) FROM t",
    ],
)
def test_one_loaded_table_never_reaches_the_join_guard(single, sql) -> None:
    """One CSV keeps the path it had, whatever shape the SQL takes.

    Found by the live bank rather than by reading. Gating on the shape of the query
    let a CTE over the single table reach a guard built for cross-table fan-out:
    44 refusals on a dataset with one table, and three questions exhausted their
    tool rounds. Two rounds of narrowing the shape rules only reduced it. The gate
    belongs on the dataset — fan-out inside one table is the grain guard's job, and
    that already runs.
    """
    assert relationships.preflight(single, sql, {})[0] is None


@pytest.mark.parametrize(
    ("label", "sql", "refused"),
    [
        (
            "a CTE reduced to one row per key",
            "WITH p AS (SELECT DISTINCT assetId FROM assets) "
            "SELECT sum(s.coinIn) FROM sessions s JOIN p ON s.assetId = p.assetId",
            False,
        ),
        (
            "a CTE whose grain cannot be proved",
            "WITH p AS (SELECT assetId FROM assets) "
            "SELECT sum(s.coinIn) FROM sessions s JOIN p ON s.assetId = p.assetId",
            True,
        ),
        (
            "a window inside a CTE, then a deduplicated join",
            "WITH ranked AS (SELECT assetId, row_number() OVER (PARTITION BY assetId) rn "
            "FROM assets), one AS (SELECT DISTINCT assetId FROM ranked WHERE rn = 1) "
            "SELECT sum(s.coinIn) FROM sessions s JOIN one ON s.assetId = one.assetId",
            False,
        ),
        (
            "a window that is not evidence of grain",
            "WITH g AS (SELECT assetId, row_number() OVER () rn FROM assets) "
            "SELECT sum(s.coinIn) FROM sessions s JOIN g ON s.assetId = g.assetId",
            True,
        ),
    ],
)
def test_a_derived_relation_is_credited_only_when_its_grain_is_visible(
    data, label, sql, refused
) -> None:
    """DISTINCT and GROUP BY prove a grain; a window function does not.

    A window adds columns and never multiplies rows, so it cannot cause fan-out —
    the guard first conflated "cannot prove grain with this" and "this is
    dangerous", and refused every query containing one. Undetermined is still
    refused, because allowing it was the same bypass in another costume.
    """
    verdict = relationships.preflight(data, sql, {})[0]
    assert bool(verdict) is refused, f"{label}: {verdict}"


# --- keys measured, and stated before anything joins ----------------------------


def test_a_single_column_key_is_found_and_reported(data) -> None:
    facts = relationships.discover_keys(data, "assets", {"assetId": 2, "day": 2, "manufacturer": 2})
    assert facts and all(f.unique for f in facts)


def test_a_composite_key_is_found_when_no_single_column_identifies_a_row() -> None:
    """The shape of the real asset file: one row per machine per day, and no single
    column comes close. This is the sentence that stops a join on assetId alone."""
    rows = ["assetId,day,note"]
    rows += [f"{asset},2024-01-{day:02d},x" for asset in (1, 2, 3) for day in range(1, 8)]
    dataset = Dataset.load([CsvSource.from_upload("a.csv", ("\n".join(rows) + "\n").encode())])
    try:
        facts = relationships.discover_keys(dataset, "a", {"assetId": 3, "day": 7, "note": 1})
        assert facts, "no key offered for a table that plainly has one"
        best = facts[0]
        assert best.unique and set(best.ref.columns) == {"assetId", "day"}
        assert "one row per" in best.describe()
    finally:
        dataset.close()


def test_a_measure_is_never_offered_as_a_key() -> None:
    """Money carries plenty of distinct values and identifies nothing. Ranked on
    count alone, a real asset table offered (grossWin, ticketOut) as its key."""
    rows = ["assetId,day,grossWin"]
    rows += [f"{a},2024-01-{d:02d},{a * 100 + d}.5" for a in (1, 2) for d in range(1, 6)]
    dataset = Dataset.load([CsvSource.from_upload("a.csv", ("\n".join(rows) + "\n").encode())])
    try:
        assert "grossWin" in relationships.measure_columns(dataset, "a")
        facts = relationships.discover_keys(dataset, "a", {"grossWin": 10, "assetId": 2, "day": 5})
        assert all("grossWin" not in f.ref.columns for f in facts), [f.ref for f in facts]
    finally:
        dataset.close()


def test_key_facts_count_nulls_apart_from_duplicates() -> None:
    dataset = Dataset.load([CsvSource.from_upload("a.csv", b"k,v\n1,a\n2,b\n,c\n")])
    try:
        facts = relationships.verify_key(dataset, relationships.Ref("a", ("k",)))
        assert facts.rows == 3 and facts.complete == 2
        assert facts.unique and facts.has_nulls
        assert "though 1 rows have none" in facts.describe()
    finally:
        dataset.close()


# --- resource behaviour: criteria 15, 18 and 19 ---------------------------------


def test_tables_with_nothing_in_common_produce_no_candidates_and_no_error(data) -> None:
    """Two unrelated files are a normal thing to load together."""
    unrelated = Dataset.load(
        [
            CsvSource.from_upload("weather.csv", b"city,degrees\nOslo,4\nCairo,31\n"),
            CsvSource.from_upload("stock.csv", b"ticker,price\nAAPL,190\nMSFT,410\n"),
        ]
    )
    try:
        found = relationships.validate(unrelated, [])
        assert (found.keys, found.joins, found.rejected) == ([], [], [])
        # And a query over both is simply not something the guard objects to.
        assert (
            relationships.preflight(
                unrelated,
                "SELECT sum(w.degrees) FROM weather w JOIN stock s ON w.city = s.ticker",
                {},
            )[0]
            is None
        )
    finally:
        unrelated.close()


def test_verification_is_bounded_by_the_query_deadline(data, monkeypatch) -> None:
    """Grouping two tables by their keys is cheap next to the join it prices, but
    cheap is not unbounded — a verification with no deadline could outlast the
    question that asked for it."""
    used = []
    original = data._deadline

    def watched():
        used.append(True)
        return original()

    monkeypatch.setattr(data, "_deadline", watched)
    relationships.verify_key(data, relationships.Ref("sessions", ("assetId",)))
    verify_join(data, ["assetId"], ["assetId"])
    assert used, "verification ran without the deadline every other query gets"


def test_measured_facts_do_not_outlive_the_dataset_that_produced_them(data) -> None:
    """The cache belongs to the tools, which belong to one loaded dataset. A reload
    builds both again, so a stale measurement cannot describe new rows."""
    from smart_data_studio.tools import AnalysisTools

    first = AnalysisTools(data)
    first.run_sql("SELECT sum(s.coinIn) FROM sessions s JOIN assets a ON s.assetId=a.assetId")
    assert first._join_facts, "nothing was cached"
    assert AnalysisTools(data)._join_facts == {}, "a new workspace inherited old measurements"


# --- taking a measure from the table nobody asked about -------------------------


def measure_tools(dataset):
    from smart_data_studio.tools import AnalysisTools

    tools = AnalysisTools(dataset)
    tools.shared_measures = {
        table: relationships.measure_columns(dataset, table) & {"coinIn"}
        for table in dataset.tables
    }
    return tools


@pytest.fixture
def measures():
    dataset = Dataset.load(
        [
            CsvSource.from_upload("sessions.csv", b"assetId,day,coinIn\n1,x,10.5\n2,y,20.5\n"),
            CsvSource.from_upload(
                "assets.csv", b"assetId,day,coinIn,paytableId\n1,x,900.5,7\n2,y,800.5,7\n"
            ),
        ]
    )
    try:
        yield dataset
    finally:
        dataset.close()


JOINED = "FROM sessions s JOIN assets a ON s.assetId=a.assetId AND s.day=a.day GROUP BY 1"


@pytest.mark.parametrize(
    ("label", "question", "sql", "warned"),
    [
        (
            "asks about sessions, totals the asset column",
            "Total session coin in by paytableId",
            f"SELECT a.paytableId, sum(a.coinIn) {JOINED}",
            True,
        ),
        (
            "asks about sessions, totals the session column",
            "Total session coin in by paytableId",
            f"SELECT a.paytableId, sum(s.coinIn) {JOINED}",
            False,
        ),
        (
            "names no table, so there is nothing to disagree with",
            "Total coin in by paytableId",
            f"SELECT a.paytableId, sum(a.coinIn) {JOINED}",
            False,
        ),
        (
            "asks about assets and totals the asset column",
            "Total asset coin in by paytableId",
            f"SELECT a.paytableId, sum(a.coinIn) {JOINED}",
            False,
        ),
    ],
)
def test_a_shared_measure_taken_from_the_wrong_table_is_flagged(
    measures, label, question, sql, warned
) -> None:
    """Asked for the game version with the most *session* coin in, a query summed
    coinIn from the asset table, which carries its own daily column of that name:
    $500,401,572 against a true $67,143,446. Nothing double-counted and no join was
    wrong, so neither the fan-out guard nor the filter guard has anything to catch
    — it is the wrong column, not a wrong join.

    Stating both totals in the profile was not enough on its own; the model still
    read the asset column. This is what corrected it.
    """
    import json

    tools = measure_tools(measures)
    tools.question = question
    payload = json.loads(tools.run_sql(sql))
    assert ("source_warning" in payload) is warned, f"{label}: {payload.get('source_warning')}"


@pytest.mark.parametrize(
    ("label", "sql", "refused", "noted"),
    [
        (
            "SUM over the repeated side always double counts",
            "SELECT sum(a.manufacturer_id) FROM sessions s JOIN assets a ON s.assetId=a.assetId",
            True,
            False,
        ),
        (
            "COUNT over the repeated side counts each row once per match",
            "SELECT count(a.assetId) FROM sessions s JOIN assets a ON s.assetId=a.assetId",
            True,
            False,
        ),
        (
            "AVG is reweighted, not corrupted, and may be exactly what was asked",
            "SELECT avg(a.manufacturer_id) FROM sessions s JOIN assets a ON s.assetId=a.assetId",
            False,
            True,
        ),
        (
            "MIN cannot change however often a row appears",
            "SELECT min(a.manufacturer_id) FROM sessions s JOIN assets a ON s.assetId=a.assetId",
            False,
            False,
        ),
        (
            "MAX likewise",
            "SELECT max(a.manufacturer_id) FROM sessions s JOIN assets a ON s.assetId=a.assetId",
            False,
            False,
        ),
    ],
)
def test_only_the_aggregates_repetition_actually_breaks_are_refused(
    label, sql, refused, noted
) -> None:
    """Refusing all of them cost a bank question its tool rounds for a query doing
    exactly what it was told: "for every session look up that machine's utilisation
    that day, then average over sessions" is a session-weighted average by
    construction, and 21.05 is the right answer to it.
    """
    dataset = Dataset.load(
        [
            CsvSource.from_upload("sessions.csv", b"assetId,coinIn\n1,10\n1,20\n2,30\n"),
            CsvSource.from_upload("assets.csv", b"assetId,manufacturer_id\n1,7\n1,9\n2,11\n"),
        ]
    )
    try:
        refusal, note = relationships.preflight(dataset, sql, {})
        assert bool(refusal) is refused, f"{label}: {refusal}"
        assert bool(note) is noted, f"{label}: {note}"
    finally:
        dataset.close()


# --- M4: shapes whose grain can now be proved -----------------------------------


@pytest.mark.parametrize(
    ("label", "sql", "refused"),
    [
        (
            "a self-join that multiplies is refused for that, not for being a self-join",
            "SELECT sum(x.coinIn) FROM sessions x JOIN sessions y ON x.assetId = y.assetId",
            True,
        ),
        (
            "a self-join on a key that does not repeat is allowed",
            "SELECT sum(x.coinIn) FROM sessions x "
            "JOIN sessions y ON x.assetId = y.assetId AND x.day = y.day",
            False,
        ),
        (
            "a right join is measured like any other",
            "SELECT sum(s.coinIn) FROM sessions s RIGHT JOIN assets a ON s.assetId = a.assetId",
            True,
        ),
        (
            "a full join likewise",
            "SELECT sum(s.coinIn) FROM sessions s FULL JOIN assets a ON s.assetId = a.assetId",
            True,
        ),
        (
            "and they are allowed when the key is complete",
            "SELECT sum(s.coinIn) FROM sessions s "
            "RIGHT JOIN assets a ON s.assetId = a.assetId AND s.day = a.day",
            False,
        ),
    ],
)
def test_shapes_whose_grain_can_be_proved_are_measured_rather_than_refused(
    data, label, sql, refused
) -> None:
    """A self-join was refused outright for being one, which is a real cost —
    period-over-period comparisons are written that way. Multiplication is a
    property of the join key, and it is measurable however many times a table
    appears, once each side is tracked by its alias rather than its name.

    RIGHT and FULL change which unmatched rows survive, not which rows repeat, and
    repetition is the whole question here.
    """
    verdict = guard(data, sql)
    assert bool(verdict) is refused, f"{label}: {verdict}"


def test_a_self_join_names_the_side_that_repeats(data) -> None:
    """Both sides are the same table, so the alias is the only thing that tells
    them apart — and it is what the query itself wrote."""
    refusal = guard(
        data, "SELECT sum(x.coinIn) FROM sessions x JOIN sessions y ON x.assetId = y.assetId"
    )
    assert "x (sessions)" in refusal, refusal


# --- review findings: five of these bypassed the guard entirely -----------------

FAN = b"assetId,day,coinIn\n1,x,10\n1,y,20\n2,x,30\n"
DAILY = b"assetId,day,fee\n1,x,100\n1,y,200\n2,x,300\n"


@pytest.fixture
def pair():
    dataset = Dataset.load(
        [
            CsvSource.from_upload("sessions.csv", FAN),
            CsvSource.from_upload("assets.csv", DAILY),
        ]
    )
    try:
        yield dataset
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("finding", "sql", "refused"),
    [
        (
            "a derived side unique on (assetId, day), joined on assetId alone",
            "SELECT sum(s.coinIn) FROM sessions s JOIN "
            "(SELECT assetId, day, sum(fee) f FROM assets GROUP BY assetId, day) d "
            "ON s.assetId = d.assetId",
            True,
        ),
        (
            "the same derived side joined on its whole grain",
            "SELECT sum(s.coinIn) FROM sessions s JOIN "
            "(SELECT assetId, day FROM assets GROUP BY assetId, day) d "
            "ON s.assetId = d.assetId AND s.day = d.day",
            False,
        ),
        (
            "a measure computed in the subquery, repeated once per matching session",
            "SELECT sum(d.f) FROM sessions s JOIN "
            "(SELECT assetId, sum(fee) f FROM assets GROUP BY assetId) d "
            "ON s.assetId = d.assetId",
            True,
        ),
        (
            "an aggregate sqlglot parses as an ordinary function",
            "SELECT total(s.coinIn) FROM sessions s JOIN assets a ON s.assetId = a.assetId",
            True,
        ),
        (
            "COUNT(*), which has no column to attribute and counts the join itself",
            "SELECT count(*) FROM sessions s JOIN assets a ON s.assetId = a.assetId",
            True,
        ),
        (
            "MIN, which no amount of repetition can change",
            "SELECT min(s.coinIn) FROM sessions s JOIN assets a ON s.assetId = a.assetId",
            False,
        ),
    ],
)
def test_the_ways_the_guard_could_be_walked_past(pair, finding, sql, refused) -> None:
    """Each of these produced a wrong number and no refusal.

    The derived-grain one was a reversed subset test: a subquery grouped by
    (assetId, day) is one row per pair, and joining on assetId alone still meets
    many of them — the check asked whether the join columns were *within* the
    grain rather than whether they *covered* it.

    A unique derived side cannot multiply the base side, but the base side repeats
    *it*, so a measure computed in the subquery is counted once per match.

    And listing the harmful aggregates rather than the harmless ones let total()
    through as though it were as safe as MIN, when it is a SUM by another spelling.
    """
    verdict, _ = relationships.preflight(pair, sql, {})
    assert bool(verdict) is refused, f"{finding}: {verdict}"


def test_an_inner_join_that_drops_rows_says_so(pair) -> None:
    """Nothing multiplies, so nothing is double counted — but a total over an
    inner join that leaves rows out is quietly short, and a short total looks
    exactly as reasonable as a correct one."""
    cache: dict = {}
    sql = (
        "SELECT sum(s.coinIn) FROM sessions s "
        "JOIN assets a ON s.assetId = a.assetId AND s.day = a.day"
    )
    relationships.preflight(pair, sql, cache)  # warm the measurement
    refusal, note = relationships.preflight(pair, sql, cache)
    assert refusal is None, "a complete join must still run"
    assert note is None or "leaves rows out" in note


def test_an_integer_that_names_itself_a_key_is_not_a_measure(pair) -> None:
    """movieId identifies and jackpots measures, and both are integers."""
    dataset = Dataset.load(
        [CsvSource.from_upload("t.csv", b"movieId,player_id,jackpots,rate\n1,7,3,0.5\n2,8,4,0.6\n")]
    )
    try:
        measures = relationships.measure_columns(dataset, "t")
        assert "jackpots" in measures and "rate" in measures
        assert "movieId" not in measures and "player_id" not in measures
    finally:
        dataset.close()


def test_candidate_keys_are_measured_in_two_passes_not_twenty_one(pair, monkeypatch) -> None:
    """Six singles and fifteen pairs used to be twenty-one scans of one table."""
    passes = []
    original = relationships._measure_keys

    def counted(dataset, table, candidates):
        passes.append(len(candidates))
        return original(dataset, table, candidates)

    monkeypatch.setattr(relationships, "_measure_keys", counted)
    relationships.discover_keys(pair, "assets", {"assetId": 2, "day": 2, "fee": 3})
    assert len(passes) <= 2, f"{len(passes)} passes over one table for {passes} candidates"


@pytest.mark.parametrize(
    ("finding", "sql", "refused"),
    [
        (
            "the join inside a subquery, aggregated outside under an alias",
            "SELECT sum(v) FROM (SELECT a.fee AS v FROM sessions s "
            "JOIN assets a ON s.assetId = a.assetId) w",
            True,
        ),
        (
            "SUM of a constant, which counts the joined rows",
            "SELECT sum(1) FROM sessions s JOIN assets a ON s.assetId = a.assetId",
            True,
        ),
        (
            "the same with a decimal constant",
            "SELECT sum(2.5) FROM sessions s JOIN assets a ON s.assetId = a.assetId",
            True,
        ),
        (
            "a traced column on a join that does not repeat",
            "SELECT sum(s.coinIn) FROM sessions s "
            "JOIN assets a ON s.assetId = a.assetId AND s.day = a.day",
            False,
        ),
        (
            "COUNT(*) where nothing multiplies",
            "SELECT count(*) FROM sessions s "
            "JOIN assets a ON s.assetId = a.assetId AND s.day = a.day",
            False,
        ),
        (
            "MIN, whatever it reads",
            "SELECT min(s.coinIn) FROM sessions s JOIN assets a ON s.assetId = a.assetId",
            False,
        ),
    ],
)
def test_an_aggregate_that_cannot_be_traced_is_not_thereby_safe(
    pair, finding, sql, refused
) -> None:
    """Three ways to read the joined output without naming a column of it:
    COUNT(*), SUM(1), and a column projected out of a subquery under an alias.
    Each reads exactly what fan-out inflates, and each walked past a guard that
    only looked at columns it could trace — treating "cannot tell" as "safe".
    """
    verdict, _ = relationships.preflight(pair, sql, {})
    assert bool(verdict) is refused, f"{finding}: {verdict}"


@pytest.fixture
def lopsided():
    """One asset-day with no session, one session with no asset."""
    dataset = Dataset.load(
        [
            CsvSource.from_upload("sessions.csv", b"assetId,day,coinIn\n1,x,10\n9,z,50\n"),
            CsvSource.from_upload("assets.csv", b"assetId,day,fee\n1,x,100\n7,q,700\n"),
        ]
    )
    try:
        yield dataset
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("label", "sql"),
    [
        (
            "an ordinary ON clause",
            "SELECT sum(s.coinIn) FROM sessions s "
            "JOIN assets a ON s.assetId = a.assetId AND s.day = a.day",
        ),
        (
            "a USING clause, which names no columns to parse",
            "SELECT sum(coinIn) FROM sessions JOIN assets USING (assetId, day)",
        ),
    ],
)
def test_an_inner_join_that_drops_rows_says_which_and_how_many(lopsided, label, sql) -> None:
    """Nothing multiplies, so nothing is double counted — the total is simply
    short, and a short total looks exactly as reasonable as a correct one.

    The previous version of this test asserted `note is None or isinstance(note,
    str)`, which no regression could ever fail. It also happened to use rows that
    all matched, so there was nothing to report either way.
    """
    cache: dict = {}
    relationships.preflight(lopsided, sql, cache)  # measure
    refusal, note = relationships.preflight(lopsided, sql, cache)
    assert refusal is None, f"{label}: a complete join must still run"
    assert note and "leaves rows out" in note, f"{label}: {note}"
    assert "1 rows of sessions" in note and "1 of assets" in note, note


def test_an_integer_can_be_part_of_a_composite_key() -> None:
    """A day stored as 20240101, an hour, a numbered stand. Excluding every integer
    measure from the key search hid composite keys made of them."""
    rows = ["assetId,day,fee"]
    rows += [f"{asset},2024010{day},1.5" for asset in (1, 2, 3) for day in range(1, 8)]
    dataset = Dataset.load([CsvSource.from_upload("a.csv", ("\n".join(rows) + "\n").encode())])
    try:
        found = relationships.discover_keys(dataset, "a", {"assetId": 3, "day": 7, "fee": 1})
        assert found, "no key offered for a table keyed on two integers"
        assert set(found[0].ref.columns) == {"assetId", "day"}, found[0].ref
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("label", "sql", "refused"),
    [
        (
            "an alias that only renames a column from a joined subquery",
            "SELECT sum(w.v) FROM (SELECT a.fee AS v FROM sessions s "
            "JOIN assets a ON s.assetId = a.assetId) w",
            True,
        ),
        (
            "the same subquery joined on its whole key",
            "SELECT sum(w.v) FROM (SELECT s.coinIn AS v FROM sessions s "
            "JOIN assets a ON s.assetId = a.assetId AND s.day = a.day) w",
            False,
        ),
        (
            "COUNT DISTINCT, which repetition cannot change",
            "SELECT count(DISTINCT s.assetId) FROM sessions s "
            "JOIN assets a ON s.assetId = a.assetId",
            False,
        ),
        (
            "SUM DISTINCT likewise",
            "SELECT sum(DISTINCT a.fee) FROM sessions s JOIN assets a ON s.assetId = a.assetId",
            False,
        ),
        (
            "a plain COUNT of the same column is not",
            "SELECT count(s.assetId) FROM sessions s JOIN assets a ON s.assetId = a.assetId",
            True,
        ),
    ],
)
def test_renaming_a_column_does_not_launder_it(pair, label, sql, refused) -> None:
    """`sum(w.v)` resolved `w`, so the column counted as traced — but `w` is a
    subquery that joins inside itself, and what it selects was never examined. An
    alias renames a column; it does not vouch for it.

    DISTINCT is the other half: repetition cannot change a distinct count, and
    refusing it turned the standard way of writing a safe count into an error.
    """
    verdict, _ = relationships.preflight(pair, sql, {})
    assert bool(verdict) is refused, f"{label}: {verdict}"


def test_an_identifier_outranks_a_measure_with_more_values() -> None:
    """Ranking on distinct count alone offered jackpots ahead of assetId; excluding
    every integer instead hid composite keys made of integers. The name decides the
    order rather than membership."""
    rows = ["assetId,day,jackpots"]
    rows += [f"{a},2024010{d},{a * 1000 + d}" for a in (1, 2, 3) for d in range(1, 8)]
    dataset = Dataset.load([CsvSource.from_upload("a.csv", ("\n".join(rows) + "\n").encode())])
    try:
        found = relationships.discover_keys(dataset, "a", {"assetId": 3, "day": 7, "jackpots": 21})
        assert found, "no key offered"
        assert set(found[0].ref.columns) == {"assetId", "day"}, found[0].ref
    finally:
        dataset.close()


def test_a_measure_that_happens_to_be_unique_is_not_called_a_key() -> None:
    """Naming it "one row per (jackpots)" reads as a key and invites a join on it,
    when all it says is that these values happen not to repeat in the rows loaded
    today. Saying nothing was worse — the column is still reported, described as
    what it is.
    """
    rows = ["region,jackpots"] + [f"{'N' if n % 2 else 'S'},{n * 7}" for n in range(12)]
    dataset = Dataset.load([CsvSource.from_upload("t.csv", ("\n".join(rows) + "\n").encode())])
    try:
        found = relationships.discover_keys(dataset, "t", {"jackpots": 12, "region": 2})
        assert found, "the column should still be reported"
        described = found[0].describe()
        assert "no column identifies a row" in described, described
        assert "one row per" not in described
        assert "measures rather than identifies" in described
    finally:
        dataset.close()


def test_a_real_identifier_key_is_still_stated_plainly() -> None:
    """The other half: the honest wording for a coincidence must not swallow the
    plain statement a genuine key deserves."""
    rows = ["assetId,day,jackpots"]
    rows += [f"{a},2024010{d},{a * 7 + d}" for a in (1, 2, 3) for d in range(1, 8)]
    dataset = Dataset.load([CsvSource.from_upload("a.csv", ("\n".join(rows) + "\n").encode())])
    try:
        found = relationships.discover_keys(dataset, "a", {"assetId": 3, "day": 7, "jackpots": 21})
        assert found[0].describe() == "one row per (assetId, day)"
    finally:
        dataset.close()
