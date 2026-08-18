"""Callable analysis tools exposed to the language model."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import sqlglot
from plotly.graph_objects import Figure
from sqlglot import exp

from smart_data_studio import analysis, timeseries
from smart_data_studio.charts import ChartSpec, make_figure
from smart_data_studio.config import (
    ANALYSIS_SAMPLE_SEED,
    MAX_ANALYSIS_CELLS,
    MAX_ANALYSIS_ROWS,
    MAX_CHART_ROWS,
    MAX_VALUE_MATCHES,
)
from smart_data_studio.dataset import Dataset, QueryResult, is_sensitive, quote_identifier


@dataclass
class AnalysisRecord:
    """An analysis tool's own output, kept so the UI can show it as evidence.

    Without this the model can quote a MAPE or a forecast the user has no way to
    check — the same failure the SQL panel exists to prevent. `subject` is free
    text because the tools describe different things: a series has a date and a
    value, a comparison has a dimension and a measure.
    """

    kind: str
    subject: str
    result: dict[str, object]


def _finite(value: object) -> object:
    """Strip NaN and Infinity, which are not valid JSON and say nothing to a reader."""
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dump(payload: dict[str, object]) -> str:
    return json.dumps(_finite(payload), default=str)


def _aggregates_by(tree: exp.Expression, key: str) -> bool:
    """Whether the query resolves to one row per key somewhere along the way.

    Three shapes reach entity grain and they are modelled differently. GROUP BY key
    and COUNT(DISTINCT key) carry the column inside their own node — the second on
    a Distinct that holds it. Plain SELECT DISTINCT does not: there the Distinct
    renders as the bare word and the columns sit on the Select above it.
    """
    wanted = key.lower()
    carries_key = (exp.Group, exp.Distinct)
    for node in tree.walk():
        if isinstance(node, carries_key) and wanted in node.sql().lower():
            return True
        selects_key = isinstance(node, exp.Select) and node.args.get("distinct")
        if selects_key and any(wanted in item.sql().lower() for item in node.expressions):
            return True
    return False


class AnalysisTools:
    """Tools shared across a whole conversation, so a later turn can chart an earlier result."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.results: list[QueryResult] = []
        self.analyses: list[AnalysisRecord] = []
        # Names the data had no value for. Anything the answer then says about one
        # rests on a mapping the model brought with it, which is worth recording
        # whether or not the mapping is right.
        self.unresolved: list[str] = []
        # Set per turn so a query answering a question about entities can be told
        # when it counted rows instead.
        self.entity_keys: dict[str, str] = {}
        self.question = ""
        self.chart: Figure | None = None
        self.chart_spec: ChartSpec | None = None
        self.chart_source: QueryResult | None = None
        self.visible_from = 0

    def reset_chart(self, keep_history: bool = True) -> None:
        """Charts belong to one question; results carry over so follow-ups can reuse them.

        In single-turn mode the earlier results are hidden as well, so a chart
        cannot quietly reach back into a question the model can no longer see.
        """
        self.chart = None
        self.chart_spec = None
        self.chart_source = None
        if not keep_history:
            self.visible_from = len(self.results)

    def run_sql(self, sql: str) -> str:
        """Run one read-only DuckDB SELECT against the loaded tables.

        Args:
          sql: A single DuckDB SELECT statement using only the tables in the schema.

        Returns:
          A JSON result containing columns, rows, total row count, and truncation status.
        """
        try:
            result = self.dataset.query(sql)
        except Exception as error:
            return _dump({"error": str(error)})
        self.results.append(result)
        payload = self.dataset.tool_payload(result)
        note = self._grain_note(sql)
        if note:
            payload = {**payload, "grain_warning": note}
        return _dump(payload)

    def _grain_note(self, sql: str) -> str | None:
        """Warn when a question about entities was answered by counting rows.

        A table whose playerId repeats has two grains, and picking the wrong one
        gives a number that is right for visits and wrong for players. Asked what
        share of players beat the tier above, one such query answered 4.96% where
        the answer per player is 30.54% — six times out, and reported as a
        percentage of players.

        Returned to the model rather than shown to the user, so it gets the chance
        to run the right query instead of the wrong one being labelled.
        """
        if not self.question:
            return None
        asked = self.question.lower()
        try:
            tree = sqlglot.parse_one(sql, dialect="duckdb")
        except Exception:
            return None
        for table, key in self.entity_keys.items():
            # playerId -> player. A bare "id" is too generic to match a question on.
            noun = re.sub(r"(_id|Id|ID)$", "", key).strip("_").lower()
            if len(noun) < 3 or noun not in asked or table.lower() not in sql.lower():
                continue
            if _aggregates_by(tree, key):
                continue
            return (
                f"This counts rows, and a row is one visit rather than one {noun}: {key} "
                f"repeats. The question asks about {noun}s, so aggregate to {key} first — "
                f"COUNT(DISTINCT {key}), or a subquery grouped by {key} — otherwise the "
                f"figure describes rows and will be quoted as though it described {noun}s."
            )
        return None

    def find_values(self, table: str, column: str, contains: str = "") -> str:
        """List every spelling a text column holds for a name, before filtering on it.

        Call this whenever a question names a place, segment or category and the
        answer needs a filter on it. Seeing one spelling in the profile is not
        evidence it is the only one: "NORTH LAS VEGAS" and "N LAS VEGAS" are the
        same city, and matching only the first misses a fifth of the visits while
        returning a number that looks entirely right.

        Args:
          table: The table to look in, exactly as it appears in the schema.
          column: The text column to search, exactly as it appears in the schema.
          contains: The name to look for. Each word is matched separately and
            without regard to case, so values sharing only some words are still
            returned — which is how an abbreviated spelling is found. Leave empty
            to list the column's most common values.

        Returns:
          A JSON object of matching values with their row counts, most common first.
        """
        if table not in self.dataset.tables:
            return _dump({"error": f"Unknown table: {table}"})
        columns = {name for name, _ in self.dataset.schema(table) if not is_sensitive(name)}
        if column not in columns:
            return _dump({"error": f"Unknown column: {column}"})
        quoted_table, quoted_column = quote_identifier(table), quote_identifier(column)
        value = f"CAST({quoted_column} AS VARCHAR)"
        # Scored word by word, not matched as one phrase. A phrase match cannot find
        # an abbreviation — "N LAS VEGAS" does not contain "north las vegas" — so
        # searching the whole phrase returns the one spelling already known and
        # hides the others, which is the entire failure this tool exists to catch.
        # Single characters are kept: dropping them turned "Gen X" into a search for
        # "Gen", which scores every Gen Z bucket as highly as the Gen X ones.
        words = contains.split()
        score = (
            " + ".join(f"CASE WHEN {value} ILIKE ? THEN 1 ELSE 0 END" for _ in words)
            if words
            else "0"
        )
        # Bound parameters throughout, so a value never becomes SQL.
        parameters = [f"%{word}%" for word in words]
        having = "HAVING score > 0" if words else ""
        try:
            rows = self.dataset.connection.execute(
                f"SELECT {value} AS value, count(*) AS rows, {score} AS score "
                f"FROM {quoted_table} WHERE {quoted_column} IS NOT NULL "
                f"GROUP BY 1, 3 {having} "
                f"ORDER BY score DESC, rows DESC LIMIT {MAX_VALUE_MATCHES * 4}",
                parameters,
            ).fetchall()
        except Exception as error:
            return _dump({"error": str(error)})

        # Keep only what scores near the best. Matching any single word drags in
        # values that merely share a fragment — DALLAS contains "las" — and burying
        # the real spellings under those defeats the point.
        if rows and words:
            # Rounded up, not to nearest: for a two-word name the nearest gives a
            # floor of 1, which is no filter at all and lets DALLAS back in.
            floor = max(1, math.ceil(max(row[2] for row in rows) * 0.6))
            rows = [row for row in rows if row[2] >= floor]
        more = len(rows) > MAX_VALUE_MATCHES
        found = [
            {"value": value, "rows": int(count)} for value, count, _ in rows[:MAX_VALUE_MATCHES]
        ]
        payload: dict[str, object] = {"column": column, "matches": found}
        if not found:
            payload["note"] = (
                f"No value in {column} matches {contains!r}. Answering about it anyway means "
                "using a mapping from outside this data — say so plainly in the answer."
            )
            if contains.strip():
                self.unresolved.append(f"{contains.strip()} (looked for in {column})")
        elif more:
            payload["note"] = (
                f"More than {MAX_VALUE_MATCHES} values match; only the commonest are shown."
            )
        return _dump(payload)

    def make_chart(
        self,
        kind: str,
        x: str,
        y: str | None = None,
        color: str | None = None,
        title: str | None = None,
    ) -> str:
        """Create a chart from the most recent SQL result.

        Args:
          kind: Chart type: bar, line, scatter, area, histogram, box, or pie.
          x: Column for the horizontal axis, categories, or pie labels.
          y: Optional numeric column for the vertical axis or pie values.
          color: Optional column used to split the series.
          title: Optional concise chart title.

        Returns:
          A short success message or a readable error.
        """
        reachable = self.results[self.visible_from :]
        if not reachable:
            return json.dumps({"error": "Run a SQL query before creating a chart."})

        source = reachable[-1]
        frame = source.frame
        if source.truncated:
            # The stored frame is capped for the prompt's sake. Charting it directly
            # would draw a fraction of the data and still look complete.
            full = self.dataset.query(source.sql, row_limit=MAX_CHART_ROWS)
            if full.truncated:
                return json.dumps(
                    {
                        "error": (
                            f"This result has {full.total_rows:,} rows, too many to chart. "
                            f"Aggregate or filter it to at most {MAX_CHART_ROWS:,} rows, "
                            "then chart that."
                        )
                    }
                )
            frame = full.frame

        spec = ChartSpec(kind=kind, x=x, y=y, color=color, title=title)
        try:
            self.chart = make_figure(frame, spec)
        except ValueError as error:
            return _dump({"error": str(error)})
        self.chart_spec = spec
        self.chart_source = source
        return json.dumps(
            {"status": "chart_created", "kind": kind, "rows_plotted": len(frame), "title": title}
        )

    def forecast(
        self, date_column: str, value_column: str, periods: int, coverage_column: str = ""
    ) -> str:
        """Forecast a time series from the most recent SQL result.

        Args:
          date_column: Column holding one row per whole period, such as a month.
          value_column: Numeric column to project forward.
          periods: How many future periods to forecast.
          coverage_column: Optional column holding how many days each period covers,
            for example count(DISTINCT day). Supply it and incomplete periods are
            proved rather than guessed at.

        Returns:
          JSON with the forecast, an 80% range per period, and an accuracy check
          against do-nothing baselines.
        """
        return self._on_series(
            "forecast",
            date_column,
            value_column,
            lambda series: timeseries.forecast(series, int(periods)),
            coverage_column,
        )

    def analyze_trend(self, date_column: str, value_column: str, coverage_column: str = "") -> str:
        """Split a time series into trend, seasonality and remainder.

        Args:
          date_column: Column holding one row per whole period, such as a month.
          value_column: Numeric column to analyse.
          coverage_column: Optional column holding how many days each period covers.

        Returns:
          JSON with direction, total change, and how much of the variation the
          trend and the season each explain.
        """
        return self._on_series(
            "trend", date_column, value_column, timeseries.decompose, coverage_column
        )

    def detect_anomalies(
        self, date_column: str, value_column: str, coverage_column: str = ""
    ) -> str:
        """Find periods that break the pattern of a time series.

        Args:
          date_column: Column holding one row per whole period, such as a month.
          value_column: Numeric column to check.
          coverage_column: Optional column holding how many days each period covers.

        Returns:
          JSON listing unusual periods with a score and direction.
        """
        return self._on_series(
            "anomalies", date_column, value_column, timeseries.anomalies, coverage_column
        )

    def compare_groups(self, dimension: str, measure: str) -> str:
        """Test whether groups genuinely differ on a measure, and by how much.

        Args:
          dimension: Column holding the groups to compare, such as a segment or tier.
          measure: Numeric column to compare across those groups.

        Returns:
          JSON with per-group summaries, p-values and — the part that matters at
          scale — effect size.
        """
        return self._on_frame(
            "comparison",
            f"{measure} across {dimension}",
            lambda frame: analysis.compare_groups(frame, dimension, measure),
        )

    def rank_drivers(self, measure: str, split: str) -> str:
        """Rank which dimensions explain a change in a measure between two sides.

        Args:
          measure: Numeric column whose change is being explained.
          split: Column holding exactly two values, such as period or segment labels.

        Returns:
          JSON ranking every usable dimension by how much movement it accounts for.
        """
        return self._on_frame(
            "drivers",
            f"{measure} between the two sides of {split}",
            lambda frame: analysis.rank_drivers(frame, measure, split),
            # This one sums. Means and correlations survive a sample; the difference
            # between two near-equal sums does not — it reported a change of
            # $1,091,057 where the truth was $526,870.
            may_sample=False,
        )

    def relate(self, target: str) -> str:
        """Rank every column by strength of association with a target column.

        Args:
          target: Numeric column to explain.

        Returns:
          JSON ranking columns by association strength on a comparable 0 to 1 scale.
        """
        return self._on_frame(
            "associations",
            f"what relates to {target}",
            lambda frame: analysis.relate(frame, target),
        )

    def _on_frame(self, kind: str, subject: str, analyse, may_sample: bool = True) -> str:
        """Run a whole-result analysis, on every row rather than the page shown."""
        if not self.results:
            return json.dumps({"error": "Run a SQL query before analysing it."})
        source = self.results[-1]
        if not may_sample and source.total_rows > self._affordable(source):
            return json.dumps(
                {
                    "error": (
                        f"This result has {source.total_rows:,} rows, too many to total "
                        "exactly, and sampled totals would be wrong. Aggregate first: "
                        "GROUP BY the dimensions and the two-sided split column, summing "
                        "the measure, then analyse that."
                    )
                }
            )
        frame, sampling = self._analysis_frame(source)
        try:
            outcome = analyse(frame)
        except analysis.NotAnalysable as error:
            return _dump({"error": str(error)})
        if sampling:
            outcome["sampled_rows"] = sampling
        self.analyses.append(AnalysisRecord(kind, subject, outcome))
        return _dump(outcome)

    def _analysis_frame(self, source: QueryResult):
        """The whole result, or a random sample of it when that is too large.

        Sampling beats refusing. Told a result was too big to analyse, the model
        reached for LIMIT — which takes the first rows in whatever order the scan
        produced, and skewed a group mean by 22% while looking perfectly ordinary.
        """
        if not source.truncated:
            return source.frame, None
        affordable = self._affordable(source)
        if source.total_rows <= affordable:
            return self.dataset.query(source.sql, row_limit=affordable).frame, None
        drawn = self.dataset.query(
            f"SELECT * FROM ({source.sql}) AS analysed USING SAMPLE "
            f"reservoir({affordable} ROWS) REPEATABLE ({ANALYSIS_SAMPLE_SEED})",
            row_limit=affordable,
        )
        return drawn.frame, (
            f"A random {len(drawn.frame):,} of {source.total_rows:,} rows were analysed."
        )

    @staticmethod
    def _affordable(source: QueryResult) -> int:
        """Rows that fit the memory budget, counted in cells so width matters."""
        return min(MAX_ANALYSIS_ROWS, MAX_ANALYSIS_CELLS // max(1, source.frame.shape[1]))

    def _on_series(
        self, kind: str, date_column: str, value_column: str, analyse, coverage_column: str = ""
    ) -> str:
        """Every series tool runs on the full result, not the page shown to the model."""
        if not self.results:
            return json.dumps({"error": "Run a SQL query before analysing a series."})
        source = self.results[-1]
        frame = source.frame
        if source.truncated:
            full = self.dataset.query(source.sql, row_limit=MAX_CHART_ROWS)
            if full.truncated:
                return json.dumps(
                    {
                        "error": (
                            f"This result has {full.total_rows:,} rows. Aggregate it to one row "
                            "per period (month, week, day) before analysing it as a series."
                        )
                    }
                )
            frame = full.frame
        try:
            outcome = analyse(
                timeseries.prepare(frame, date_column, value_column, coverage_column or None)
            )
        except timeseries.NotEnoughData as error:
            return _dump({"error": str(error)})
        self.analyses.append(AnalysisRecord(kind, f"{value_column} by {date_column}", outcome))
        return _dump(outcome)
