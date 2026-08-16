"""Callable analysis tools exposed to the language model."""

from __future__ import annotations

import json

from plotly.graph_objects import Figure

from smart_data_studio.charts import ChartSpec, make_figure
from smart_data_studio.config import MAX_CHART_ROWS
from smart_data_studio.dataset import Dataset, QueryResult


class AnalysisTools:
    """Tools shared across a whole conversation, so a later turn can chart an earlier result."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.results: list[QueryResult] = []
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
            return json.dumps({"error": str(error)})
        self.results.append(result)
        return json.dumps(self.dataset.tool_payload(result), default=str)

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
            return json.dumps({"error": str(error)})
        self.chart_spec = spec
        self.chart_source = source
        return json.dumps(
            {"status": "chart_created", "kind": kind, "rows_plotted": len(frame), "title": title}
        )
