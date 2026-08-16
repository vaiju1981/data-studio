"""Small Ollama tool loop for analysis and automatic understanding."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ollama
from plotly.graph_objects import Figure

from smart_data_studio.config import (
    KEEP_TOOL_PAYLOADS,
    MAX_EXPLORE_ROUNDS,
    MAX_TOOL_ROUNDS,
    MODEL_ID,
    OLLAMA_HOST,
)
from smart_data_studio.dataset import Dataset, QueryResult
from smart_data_studio.profile import TableProfile
from smart_data_studio.tools import AnalysisTools

EXPLORE_PROMPT = """Explore this dataset with SQL before drawing any conclusion.
Run several queries: read real values, check how low-cardinality columns are distributed,
find the range of dates and numbers, and look for negatives, zeros or placeholder values
that change what a column means.
Establish the grain — what one row represents. Where a key repeats across rows, check which
columns stay constant within it and which vary; a column that varies is a per-row value, not a
property of the entity, even when its name suggests otherwise.
Then write at least five Markdown bullets. Every bullet must rest on a number you queried
or a profile fact, and should say what the columns mean and how the tables relate.
Do not guess at causes, trends, or business meaning the data does not establish."""

ANALYST_PROMPT = """You are the analysis engine inside Smart Data Studio.
Answer questions only from the loaded data. For every data question, call run_sql before answering.
Use DuckDB SQL and only the tables in the schema. Never guess a number.
Use make_chart after run_sql when a visualization materially helps. The chart must use exact column
names from the latest result. Mention truncation whenever the tool says truncated is true.
Keep the final answer concise, explain the important result, and never hide limitations.

Two rules that are easy to get wrong here:
- Anchor every relative time expression — "last 30 days", "recent", "lapsed 90 days" — on the
  latest date present in the data, never on CURRENT_DATE or today. The data ends before today.
- Group by an entity key alone. Adding a column that varies within that key splits one entity
  across several rows, each holding a partial total. Aggregate such columns with MAX or SUM
  instead. The profile findings say which columns are constant within each key."""

EXHAUSTED_MESSAGE = "I could not finish within the query limit. Try a narrower question."
OMITTED_PAYLOAD = json.dumps(
    {"note": "Earlier result omitted to save context. Re-run the query if you need it."}
)


@dataclass
class Answer:
    text: str
    results: list[QueryResult]
    chart: Figure | None = None


class DataAgent:
    def __init__(
        self,
        dataset: Dataset,
        profiles: list[TableProfile],
        client: Any | None = None,
    ):
        self.dataset = dataset
        self.profiles = profiles
        self.client = client or ollama.Client(host=OLLAMA_HOST)
        # One toolset for the whole conversation, so "now chart that" can reach the
        # result produced by an earlier turn.
        self.tools = AnalysisTools(dataset)
        self.understanding = ""
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]

    def build_understanding(self) -> str:
        """Explore the data with real queries before any question is asked.

        A profile says what a column contains, never what it means. A handful of
        queries can, and what they establish is folded into the chat context so the
        conversation starts already knowing the data.
        """
        explorer: list[dict[str, Any]] = [
            {"role": "system", "content": f"{self._data_context()}\n\n{EXPLORE_PROMPT}"},
            {"role": "user", "content": "Explore this data and report what you found."},
        ]
        # Charting is not part of understanding, so only the query tool is offered.
        self.understanding = self._run_loop(explorer, MAX_EXPLORE_ROUNDS, [self.tools.run_sql])
        self.messages[0] = {"role": "system", "content": self._system_prompt()}
        return self.understanding

    def ask(self, question: str, multi_turn: bool = True) -> Answer:
        if not multi_turn:
            # Start from the system prompt alone, so nothing from an earlier answer
            # reaches this one. The UI keeps showing the full history regardless.
            self.messages = [{"role": "system", "content": self._system_prompt()}]
        first_new_result = len(self.tools.results)
        self.tools.reset_chart(keep_history=multi_turn)
        self.messages.append({"role": "user", "content": question})
        text = self._run_loop(
            self.messages,
            MAX_TOOL_ROUNDS,
            [self.tools.run_sql, self.tools.make_chart],
        )
        self._trim_tool_payloads()
        return Answer(
            text=text,
            results=self._turn_results(first_new_result),
            chart=self.tools.chart,
        )

    def _run_loop(
        self, messages: list[dict[str, Any]], max_rounds: int, tools: list[Callable[..., str]]
    ) -> str:
        for _ in range(max_rounds):
            response = self.client.chat(model=MODEL_ID, messages=messages, tools=tools)
            message = response.message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return message.content or "Analysis complete."
            for call in message.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": call.function.name,
                        "content": self._invoke(call),
                    }
                )

        # Never leave the history ending on a tool message: the next turn would then
        # place a user message straight after a tool result, with no reply between.
        messages.append({"role": "assistant", "content": EXHAUSTED_MESSAGE})
        return EXHAUSTED_MESSAGE

    def _invoke(self, call: Any) -> str:
        name = call.function.name
        available = {"run_sql": self.tools.run_sql, "make_chart": self.tools.make_chart}
        function = available.get(name)
        if function is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return function(**dict(call.function.arguments))
        except Exception as error:
            # A malformed tool call is the model's mistake to correct on the next
            # round, not a reason to end the turn.
            return json.dumps({"error": f"Invalid call to {name}: {error}"})

    def _turn_results(self, first_new_result: int) -> list[QueryResult]:
        """This turn's results, plus the source behind a chart drawn from an earlier one."""
        results = self.tools.results[first_new_result:]
        source = self.tools.chart_source
        if source is not None and not any(result is source for result in results):
            return [source, *results]
        return results

    def _trim_tool_payloads(self) -> None:
        """Shrink all but the most recent tool results.

        Result payloads dominate the context. Recent ones are what follow-up questions
        build on; older ones can be queried again if they turn out to matter.
        """
        indexes = [
            index for index, message in enumerate(self.messages) if message["role"] == "tool"
        ]
        for index in indexes[:-KEEP_TOOL_PAYLOADS]:
            self.messages[index]["content"] = OMITTED_PAYLOAD

    def _date_bounds(self) -> str:
        """Spelled out separately because the anchoring rule is easy to miss in a stats table."""
        bounds = []
        for profile in self.profiles:
            for row in profile.stats.to_dict(orient="records"):
                kind = str(row["column_type"]).upper()
                if "DATE" in kind or "TIMESTAMP" in kind:
                    bounds.append(
                        f"{profile.table_name}.{row['column_name']}: {row['min']} to {row['max']}"
                    )
        if not bounds:
            return ""
        joined = "\n".join(f"- {bound}" for bound in bounds)
        return f"\n\nDate ranges present (anchor relative dates on these, not on today):\n{joined}"

    def _data_context(self) -> str:
        profile_text = "\n\n".join(profile.prompt_text() for profile in self.profiles)
        return (
            f"Schema:\n{self.dataset.schema_text()}\n\n"
            f"Profiles:\n{profile_text}{self._date_bounds()}\n\n"
            f"{self.dataset.sample_text()}"
        )

    def _system_prompt(self) -> str:
        learned = (
            f"\n\nWhat exploring this data established:\n{self.understanding}"
            if self.understanding
            else ""
        )
        return f"{ANALYST_PROMPT}\n\n{self._data_context()}{learned}"
