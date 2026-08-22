"""Callable analysis tools exposed to the language model."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import sqlglot
from plotly.graph_objects import Figure
from sqlglot import exp

from smart_data_studio import analysis, cohorts, joins, logs, timeseries
from smart_data_studio.charts import ChartSpec, make_figure
from smart_data_studio.config import (
    ANALYSIS_SAMPLE_SEED,
    COVERAGE_GAP,
    COVERAGE_NULL_FLOOR,
    MAX_ANALYSIS_CELLS,
    MAX_ANALYSIS_ROWS,
    MAX_CHART_ROWS,
    MAX_VALUE_MATCHES,
    MIN_COVERAGE_ROWS,
)
from smart_data_studio.dataset import Dataset, QueryResult, is_sensitive, quote_identifier


@dataclass
class AnalysisRecord:
    """An analysis tool's own output, kept so the UI can show it as evidence.

    `subject` is free text because the tools describe different things: a series
    has a date and a value, a comparison has a dimension and a measure.
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


AGGREGATE_NODES = joins.AGGREGATES


def _tables_in(tree: exp.Expression) -> set[str]:
    """Every table the query names, so a column is read against the right one."""
    return {table.name for table in tree.find_all(exp.Table) if table.name}


def _filtered_literals(tree: exp.Expression) -> dict[str, set[str]]:
    """Which string values each column is restricted to, from = and IN alone.

    Inequalities and LIKE are left out: they describe a range or a pattern rather
    than a chosen value, and cannot be compared against a name in the question.
    """
    found: dict[str, set[str]] = {}
    for node in tree.walk():
        if isinstance(node, exp.EQ):
            column, literal = node.left, node.right
            if isinstance(column, exp.Column) and isinstance(literal, exp.Literal):
                found.setdefault(column.name, set()).add(str(literal.this))
        elif isinstance(node, exp.In) and isinstance(node.this, exp.Column):
            for item in node.expressions:
                if isinstance(item, exp.Literal):
                    found.setdefault(node.this.name, set()).add(str(item.this))
    return found


def _aggregates_by(tree: exp.Expression, key: str) -> bool:
    """Whether the query resolves to one row per key somewhere along the way.

    GROUP BY key and COUNT(DISTINCT key) carry the column inside their own node.
    Plain SELECT DISTINCT does not: its Distinct renders as the bare word and the
    columns sit on the Select above it, so that shape is matched separately.
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
        # Names the data had no value for, so anything said about one came from the
        # model rather than the file.
        self.unresolved: list[str] = []
        self.entity_keys: dict[str, str] = {}
        # Measured join facts, kept for the dataset's lifetime so a join is checked
        # once however many questions reach for it.
        self.join_facts: dict = {}
        # table -> column -> values held. Keyed by table because two files may each
        # have a status column meaning different things.
        self.dimension_values: dict[str, dict[str, list[str]]] = {}
        # table -> measures another loaded table also carries under that name.
        self.shared_measures: dict[str, set[str]] = {}
        # table -> column -> how much of it is null, from the profile. Only used to
        # decide whether a coverage scan is worth running at all.
        self.null_shares: dict[str, dict[str, float]] = {}
        self.question = ""
        self.chart: Figure | None = None
        self.chart_spec: ChartSpec | None = None
        self.chart_source: QueryResult | None = None
        self.visible_from = 0

    def reset_chart(self, keep_history: bool = True) -> None:
        """Charts belong to one question; results carry over so follow-ups can reuse them.

        Single-turn hides the earlier results too, via visible_from.
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
        # Preflight before the query runs, so a join that would multiply is refused
        # rather than paid for and then explained.
        refusal, weighting = joins.preflight(self.dataset, sql, self.join_facts)
        if refusal:
            logs.event("join.refused")
            return _dump({"error": refusal})
        try:
            result = self.dataset.query(sql)
        except Exception as error:
            return _dump({"error": str(error)})
        self.results.append(result)
        payload = self.dataset.tool_payload(result)
        return _dump({**payload, **self._warnings(sql, weighting)})

    def _warnings(self, sql: str, weighting: str | None) -> dict[str, str]:
        """Every warning run_sql can attach, and the only place they are listed.

        The query is parsed once here and each check walks the same tree.
        """
        try:
            tree = sqlglot.parse_one(sql, dialect="duckdb")
        except Exception:
            return {"weighting_warning": weighting} if weighting else {}
        checks = (
            ("grain_warning", self._grain_note),
            ("filter_warning", self._substitution_note),
            ("source_warning", self._source_note),
            ("coverage_warning", self._coverage_note),
        )
        found = {name: note for name, check in checks if (note := check(tree))}
        if weighting:
            found["weighting_warning"] = weighting
        return found

    def _source_note(self, tree: exp.Expression) -> str | None:
        """Warn when a measure was totalled from a table the question is not about.

        Two tables carrying the same measure name hold different quantities, and
        nothing here double counts or joins, so no other guard has anything to see.
        """
        if not self.question or not self.shared_measures:
            return None
        named = self._tables_named_in_question()
        if len(named) != 1:
            return None  # nothing to disagree with
        wanted = named.pop()

        sources = joins.sources_in(tree, {name.lower() for name in self.dataset.tables})
        for node in tree.walk():
            if not isinstance(node, AGGREGATE_NODES):
                continue
            for column in node.find_all(exp.Column):
                owner = joins.column_owner(column, sources, self.dataset)
                if owner is None or owner == wanted:
                    continue
                if column.name in self.shared_measures.get(owner, set()):
                    return (
                        f"{column.name} exists on both {owner} and {wanted}, and they are "
                        f"different quantities. The question is about {wanted}, but this "
                        f"totals {owner}.{column.name}. Take the measure from {wanted}, "
                        f"joining to {owner} only for the columns {wanted} does not have."
                    )
        return None

    def _coverage_note(self, tree: exp.Expression) -> str | None:
        """Warn when a figure per group rests on very different amounts of data.

        A column 8% null overall reads as unremarkable, and the profile says so.
        Where those nulls sit in one group, that group's average covers 2% of it
        and still leads the answer as the best region — the count is in the
        result, below the number, and the ranking is made before anyone reaches it.
        """
        group = tree.args.get("group")
        if group is None or not self.null_shares:
            return None
        touched = {name.lower() for name in _tables_in(tree)}
        # One table. A joined result's coverage is the join guards' business.
        table = next((t for t in self.dataset.tables if t.lower() in touched), None)
        if table is None or len(touched) != 1:
            return None
        thin = {
            name: share
            for name, share in self.null_shares.get(table, {}).items()
            if share >= COVERAGE_NULL_FLOOR
        }
        if not thin:
            return None

        columns = {name for name, _ in self.dataset.schema(table)}
        selected = list(tree.expressions)
        dimensions, measures = [], []
        for node in tree.walk():
            if isinstance(node, (exp.Avg, exp.Sum)):
                measures += [c.name for c in node.find_all(exp.Column) if c.name in thin]
        for item in group.expressions:
            # GROUP BY 1 is as common as GROUP BY region, and points at the
            # select list rather than naming anything.
            if isinstance(item, exp.Literal) and item.is_int:
                index = int(item.this) - 1
                item = selected[index] if 0 <= index < len(selected) else item
            found = [c.name for c in item.find_all(exp.Column) if c.name in columns]
            dimensions += found
        if not measures or not dimensions:
            return None

        measure, dimension = measures[0], dimensions[0]
        try:
            rows = self.dataset.connection.execute(
                f"SELECT count(*) AS n, count({quote_identifier(measure)}) AS present "
                f"FROM {quote_identifier(table)} GROUP BY {quote_identifier(dimension)} "
                f"HAVING n >= {MIN_COVERAGE_ROWS}"
            ).fetchall()
        except Exception:
            return None  # a warning must never cost the answer
        shares = [present / total for total, present in rows if total]
        if len(shares) < 2:
            return None
        worst, best = min(shares), max(shares)
        if best - worst < COVERAGE_GAP:
            return None
        return (
            f"{measure} is not filled in evenly across {dimension}: the thinnest group has it "
            f"for {worst:.0%} of its rows where the best-covered has {best:.0%}. A figure "
            f"averaged over {worst:.0%} of a group is not comparable with one averaged over "
            f"{best:.0%}, so do not rank them against each other without saying which rest on "
            f"how much. Count the non-null values per group and lead with what the coverage "
            f"supports."
        )

    def _tables_named_in_question(self) -> set[str]:
        """Tables the question points at, matched on the part of each name that
        tells them apart rather than the prefix and suffix they all share."""
        asked = self.question.lower()
        parts = [set(re.split(r"[^a-z0-9]+", name.lower())) for name in self.dataset.tables]
        common = set.intersection(*parts) if parts else set()
        named = set()
        for name, tokens in zip(self.dataset.tables, parts, strict=True):
            for token in tokens - common:
                if token and (token in asked or token.rstrip("s") in asked):
                    named.add(name)
        return named

    def _substitution_note(self, tree: exp.Expression) -> str | None:
        """Warn when the query filtered on a value the question did not ask for.

        Only where the query filters that column: a question naming a value while
        grouping by the column is asking for a comparison, not a filter.
        """
        if not self.question or not self.dimension_values:
            return None
        filtered = _filtered_literals(tree)
        touched = {name.lower() for name in _tables_in(tree)}
        for column, used in filtered.items():
            known = next(
                (
                    values[column]
                    for table, values in self.dimension_values.items()
                    if table.lower() in touched and column in values
                ),
                None,
            )
            if not known:
                continue
            asked = [
                value
                for value in known
                if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", self.question, re.IGNORECASE)
            ]
            lowered = {value.lower() for value in used}
            missing = [value for value in asked if value.lower() not in lowered]
            if asked and missing:
                return (
                    f"The question names {', '.join(repr(v) for v in missing)} in {column}, but "
                    f"this query filtered {column} on {', '.join(sorted(used))}. Answer about "
                    f"what was asked, or say plainly which values the figure covers."
                )
        return None

    def _grain_note(self, tree: exp.Expression) -> str | None:
        """Warn when a question about entities was answered by counting rows.

        A table whose key repeats has two grains, and the wrong one gives a figure
        that is right per row and wrong per entity. Returned to the model rather
        than shown to the user, so it can run the right query instead.
        """
        if not self.question:
            return None
        asked = self.question.lower()
        touched = {name.lower() for name in _tables_in(tree)}
        for table, key in self.entity_keys.items():
            # playerId -> player. A bare "id" is too generic to match a question on.
            noun = re.sub(r"(_id|Id|ID)$", "", key).strip("_").lower()
            if len(noun) < 3 or noun not in asked or table.lower() not in touched:
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
        evidence it is the only one: a value written "NORTH DISTRICT" in some rows
        and "N DISTRICT" in others is one value, and matching only the first
        silently undercounts while returning a number that looks entirely right.

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
        # Word by word, not as one phrase: a phrase match cannot find an
        # abbreviation, which is the whole point of the tool. Single characters
        # count too, or "Gen X" searches for "Gen" and matches every Gen Z bucket.
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

        # Keep only what scores near the best, or values sharing one fragment bury
        # the real spellings.
        if rows and words:
            # Rounded up: for a two-word name the nearest gives a floor of 1, which
            # is no filter at all.
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

    def cohort_window(
        self,
        table: str,
        entity_column: str,
        cohort_column: str,
        activity_column: str,
        period: str = "month",
        horizon: int = 12,
    ) -> str:
        """Follow the entities that started in each period and see how many return.

        Use this for retention, repeat purchase, account vintage, readmission
        within a window, or any question of the form "of the ones that began then,
        how many are still here now". Do not build it out of run_sql: the base has
        to be the whole cohort, including entities whose first activity came later,
        and a query written by hand divides by the ones active in the first period
        instead — which is smaller, and gives a retention curve that is wrong
        while every figure in it is arithmetically correct.

        Args:
          table: The table holding one row per activity, as it appears in the schema.
          entity_column: The column identifying the thing being followed.
          cohort_column: The date each entity started — registration, signup, admission.
          activity_column: The date of the activity being counted.
          period: day, week, month, quarter or year.
          horizon: How many periods past its start each cohort is followed.

        Returns:
          JSON with each cohort's full size and, per period after it, how many were
          active and what share of the cohort that is.
        """
        try:
            outcome = cohorts.cohort_window(
                self.dataset,
                table,
                entity_column,
                cohort_column,
                activity_column,
                period,
                horizon,
            )
        except cohorts.NotCohortable as error:
            return _dump({"error": str(error)})
        except Exception as error:
            return _dump({"error": str(error)})
        self.analyses.append(
            AnalysisRecord("cohorts", f"{entity_column} by {cohort_column}", outcome)
        )
        return _dump(outcome)

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
            # The stored frame is capped for the prompt. Charting it would draw a
            # fraction of the data and still look complete.
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
            # between two near-equal sums does not.
            may_sample=False,
        )

    def find_outliers(self, dimension: str, measure: str) -> str:
        """Find which individual entities stand apart from the rest on a measure.

        Use this for "which machines, accounts, services or stores are behaving
        unusually". Ranking by the biggest number answers a different question —
        the busiest entity wins whatever it is doing — so this measures distance
        from the rest of the population instead.

        Prefer a rate or a ratio over a raw total: on a total, standing apart
        mostly means being large. If the result says the measure is skewed, divide
        it by whatever drives its size and call this again.

        Args:
          dimension: Column identifying the entities, such as an id or a name.
          measure: Numeric column to judge them on. A rate reads better than a total.

        Returns:
          JSON listing the entities furthest from the population median, each with
          a score in units of the median absolute deviation.
        """
        return self._on_frame(
            "outliers",
            f"{measure} across {dimension}",
            lambda frame: analysis.find_outliers(frame, dimension, measure),
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
        # visible_from, not the whole list: the tools outlive a single-turn reset, so
        # the newest result may belong to a turn this one cannot see.
        reachable = self.results[self.visible_from :]
        if not reachable:
            return json.dumps({"error": "Run a SQL query before analysing it."})
        source = reachable[-1]
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

        Sampling beats refusing: told a result was too big, the model reaches for
        LIMIT, which takes whatever order the scan produced and skews every mean.
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
        # visible_from, for the reason _on_frame gives.
        reachable = self.results[self.visible_from :]
        if not reachable:
            return json.dumps({"error": "Run a SQL query before analysing a series."})
        source = reachable[-1]
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
