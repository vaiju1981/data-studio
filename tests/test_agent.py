from types import SimpleNamespace

from ollama import Message

from smart_data_studio.agent import EXHAUSTED_MESSAGE, OMITTED_PAYLOAD, DataAgent
from smart_data_studio.config import KEEP_TOOL_PAYLOADS, MODEL_ID
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_dataset

SALES = b"region,amount\nNorth,10\nSouth,20\nNorth,15\n"


class FakeClient:
    """Replays scripted assistant messages, cycling so each turn repeats the script."""

    def __init__(self, messages: list[Message]):
        self.scripted = list(messages)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=self.scripted[(len(self.calls) - 1) % len(self.scripted)])


def tool_call(name: str, **arguments) -> Message:
    return Message(
        role="assistant", tool_calls=[{"function": {"name": name, "arguments": arguments}}]
    )


def make_agent(client: FakeClient, data: bytes = SALES) -> tuple[DataAgent, Dataset]:
    dataset = Dataset.load([CsvSource.from_upload("sales.csv", data)])
    return DataAgent(dataset, profile_dataset(dataset), client=client), dataset


def test_agent_runs_sql_then_builds_chart() -> None:
    client = FakeClient(
        [
            tool_call(
                "run_sql", sql="SELECT region, SUM(amount) AS total FROM sales GROUP BY region"
            ),
            tool_call("make_chart", kind="bar", x="region", y="total", title="Sales by region"),
            Message(role="assistant", content="North totals 25 and South totals 20."),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("Chart total sales by region")
        assert answer.text.startswith("North totals 25")
        assert answer.results[0].total_rows == 2
        assert answer.chart is not None
        assert all(call["model"] == MODEL_ID for call in client.calls)
        assert callable(client.calls[0]["tools"][0])
    finally:
        dataset.close()


def test_understanding_queries_the_data_and_enters_the_chat_context() -> None:
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT DISTINCT region FROM sales"),
            Message(role="assistant", content="- region holds two values: North and South"),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        understanding = agent.build_understanding()
        assert "North and South" in understanding
        # The exploration actually ran SQL rather than only reading the profile.
        assert any(message["role"] == "tool" for message in client.calls[-1]["messages"])
        # What it learned is carried into the chat, along with real sample rows.
        system_prompt = agent.messages[0]["content"]
        assert "North and South" in system_prompt
        assert "Sample rows from sales" in system_prompt
    finally:
        dataset.close()


def test_malformed_tool_call_is_returned_to_the_model_instead_of_crashing() -> None:
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT 1 AS one", row_limit=5),
            Message(role="assistant", content="Recovered."),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("Anything")
        assert answer.text == "Recovered."
        tool_messages = [item for item in agent.messages if item["role"] == "tool"]
        assert "Invalid call to run_sql" in tool_messages[0]["content"]
    finally:
        dataset.close()


def test_chart_can_use_a_result_from_an_earlier_turn() -> None:
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT region, SUM(amount) AS total FROM sales GROUP BY 1"),
            Message(role="assistant", content="North 25, South 20."),
            tool_call("make_chart", kind="bar", x="region", y="total"),
            Message(role="assistant", content="Here is the chart."),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        agent.ask("totals by region")
        answer = agent.ask("now chart that")
        assert answer.chart is not None
        # The SQL behind the chart travels with the answer, even from an earlier turn.
        assert answer.results and "GROUP BY" in answer.results[0].sql
    finally:
        dataset.close()


def test_stale_chart_does_not_reappear_on_a_later_question() -> None:
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT region FROM sales"),
            tool_call("make_chart", kind="histogram", x="region"),
            Message(role="assistant", content="Charted."),
            Message(role="assistant", content="Just text this time."),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        assert agent.ask("chart regions").chart is not None
        assert agent.ask("and in words?").chart is None
    finally:
        dataset.close()


def test_exhausted_rounds_leave_the_history_on_an_assistant_message() -> None:
    client = FakeClient([tool_call("run_sql", sql="SELECT 1 AS one")])
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("loop forever")
        assert answer.text == EXHAUSTED_MESSAGE
        assert agent.messages[-1]["role"] == "assistant"
    finally:
        dataset.close()


def test_older_tool_payloads_are_replaced_to_bound_the_context() -> None:
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT region FROM sales"),
            Message(role="assistant", content="ok"),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        for _ in range(KEEP_TOOL_PAYLOADS + 2):
            agent.ask("regions?")
        payloads = [item["content"] for item in agent.messages if item["role"] == "tool"]
        assert payloads.count(OMITTED_PAYLOAD) == len(payloads) - KEEP_TOOL_PAYLOADS
        assert all(payload != OMITTED_PAYLOAD for payload in payloads[-KEEP_TOOL_PAYLOADS:])
    finally:
        dataset.close()


def test_single_turn_forgets_the_previous_question() -> None:
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT region, SUM(amount) AS total FROM sales GROUP BY 1"),
            Message(role="assistant", content="North 25, South 20."),
            tool_call("make_chart", kind="bar", x="region", y="total"),
            Message(role="assistant", content="Done."),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        agent.ask("totals by region", multi_turn=False)
        answer = agent.ask("now chart that", multi_turn=False)
        # The history is rebuilt from the system prompt, so only this turn is present.
        assert [item["role"] for item in agent.messages[:2]] == ["system", "user"]
        # And the chart cannot reach back to the result the model can no longer see.
        assert answer.chart is None
        tool_messages = [item for item in agent.messages if item["role"] == "tool"]
        assert "Run a SQL query before creating a chart" in tool_messages[-1]["content"]
    finally:
        dataset.close()


def test_context_states_the_date_range_to_anchor_relative_time_on() -> None:
    data = b"order_id,day,amount\n1,2024-01-05,10\n2,2024-03-09,20\n"
    client = FakeClient([Message(role="assistant", content="ok")])
    agent, dataset = make_agent(client, data=data)
    try:
        prompt = agent.messages[0]["content"]
        assert "Date ranges present" in prompt
        assert "2024-01-05" in prompt and "2024-03-09" in prompt
        assert "never on CURRENT_DATE" in prompt
    finally:
        dataset.close()


def test_metrics_live_in_the_system_prompt_and_survive_single_turn() -> None:
    client = FakeClient([Message(role="assistant", content="ok")])
    agent, dataset = make_agent(client)
    try:
        assert agent.set_metrics("avgBet = coinIn / handlePulls") is True
        assert agent.set_metrics("avgBet = coinIn / handlePulls") is False  # unchanged

        assert "avgBet = coinIn / handlePulls" in agent.messages[0]["content"]
        assert "apply the definition exactly as written" in agent.messages[0]["content"]

        # A single-turn question rebuilds the history, and the definitions must ride along.
        agent.ask("anything", multi_turn=False)
        assert "avgBet = coinIn / handlePulls" in agent.messages[0]["content"]
    finally:
        dataset.close()


def test_forecast_output_travels_to_the_ui_as_evidence() -> None:
    """The model can quote a MAPE; the user must be able to see where it came from."""
    rows = ["month,amount"]
    rows += [f"2024-{m:02d}-01,{100 + m}" for m in range(1, 13)]
    rows += [f"2025-{m:02d}-01,{110 + m}" for m in range(1, 13)]
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT month, amount FROM sales ORDER BY month"),
            tool_call("forecast", date_column="month", value_column="amount", periods=6),
            Message(role="assistant", content="Roughly flat."),
        ]
    )
    agent, dataset = make_agent(client, data=("\n".join(rows) + "\n").encode())
    try:
        answer = agent.ask("forecast the next six months")
        assert len(answer.analyses) == 1
        analysis = answer.analyses[0]
        assert analysis.kind == "forecast"
        assert analysis.value_column == "amount"
        # The accuracy comparison the model relied on is available to render.
        assert "model_mape_pct" in analysis.result["accuracy"]
        assert len(analysis.result["forecast"]) == 6
    finally:
        dataset.close()
