from types import SimpleNamespace

import ollama
import pytest
from ollama import Message

from smart_data_studio import agent as agent_module
from smart_data_studio.agent import EXHAUSTED_MESSAGE, OMITTED_PAYLOAD, DataAgent
from smart_data_studio.config import KEEP_TOOL_PAYLOADS, MODEL_ID, MODEL_RETRIES
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_dataset

SALES = b"region,amount\nNorth,10\nSouth,20\nNorth,15\n"


class FakeClient:
    """Replays scripted assistant messages, cycling so each turn repeats the script.

    Planning calls answer NONE without touching the script, so a test that scripts
    the direct path keeps testing the direct path. `deep=True` makes it plan
    instead, for the tests that want the investigating path.
    """

    def __init__(self, messages: list[Message], plan: list[str] | None = None):
        self.scripted = list(messages)
        self.plan = plan
        self.calls: list[dict] = []
        self.plan_calls = 0

    def chat(self, **kwargs):
        system = (kwargs.get("messages") or [{}])[0].get("content", "")
        if "Decide how this question should be answered" in system:
            self.plan_calls += 1
            reply = "\n".join(self.plan) if self.plan else "NONE"
            return SimpleNamespace(message=Message(role="assistant", content=reply))
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
        assert analysis.subject == "amount by month"
        # The accuracy comparison the model relied on is available to render.
        assert "model_wape_pct" in analysis.result["accuracy"]
        assert len(analysis.result["forecast"]) == 6
    finally:
        dataset.close()


def test_an_unsupported_claim_is_sent_back_for_evidence() -> None:
    """A number with no query behind it used to be captioned; now it is challenged."""
    client = FakeClient(
        [
            Message(role="assistant", content="Revenue was 4.2 million last quarter."),
            tool_call("run_sql", sql="SELECT SUM(amount) AS total FROM sales"),
            Message(role="assistant", content="Total is 45 across the loaded rows."),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("what was revenue?")
        assert answer.results, "the retry produced no query"
        assert "45" in answer.text
        asked = [item for item in agent.messages if item["role"] == "user"]
        assert any("no query ran" in item["content"] for item in asked)
    finally:
        dataset.close()


def test_a_refusal_is_left_alone() -> None:
    """Saying the data cannot answer is a good answer, and needs no SQL."""
    client = FakeClient(
        [Message(role="assistant", content="This data does not contain customer ages.")]
    )
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("how old are my customers?")
        assert "does not contain" in answer.text
        assert not any(
            "no query ran" in item["content"] for item in agent.messages if item["role"] == "user"
        )
    finally:
        dataset.close()


def test_a_direct_question_skips_the_investigation() -> None:
    """The planner answers NONE, and the question is worked in one pass."""
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT count(*) AS n FROM sales"),
            Message(role="assistant", content="Three rows."),
        ]
    )
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("how many rows are there?")
        assert answer.plan == []
        assert client.plan_calls == 1
    finally:
        dataset.close()


def test_a_judgement_question_is_worked_as_sub_questions_then_synthesised() -> None:
    """Each step runs on its own, and every query it ran stays as evidence."""
    steps = [
        "What is theo win per visit by geo type?",
        "What is theo win per player by geo type, which may point the other way?",
    ]
    client = FakeClient(
        [
            tool_call("run_sql", sql="SELECT region, avg(amount) AS a FROM sales GROUP BY 1"),
            Message(role="assistant", content="Per-visit favours one reading."),
        ],
        plan=steps,
    )
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("how should we grow the local segment?")
        assert answer.plan == steps
        # One query per step plus the synthesis pass, all kept as evidence.
        assert len(answer.results) >= len(steps)
        # The chat keeps the question and conclusion, not the working.
        assert sum(1 for item in agent.messages if item["role"] == "user") == 1
    finally:
        dataset.close()


def test_depth_can_be_forced_off() -> None:
    client = FakeClient([Message(role="assistant", content="ok")], plan=["a step", "another step"])
    agent, dataset = make_agent(client)
    try:
        assert agent.ask("how should we grow?", depth="never").plan == []
        assert client.plan_calls == 0
    finally:
        dataset.close()


class FlakyClient(FakeClient):
    """Fails the first `failures` calls, then behaves."""

    def __init__(self, messages, error: Exception, failures: int = 1):
        super().__init__(messages)
        self.error = error
        self.failures = failures
        self.attempts = 0

    def chat(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        return super().chat(**kwargs)


def test_a_blip_from_the_model_host_is_retried(monkeypatch) -> None:
    """A hosted model returns the occasional 500, and an investigation makes six to
    ten calls, so one blip would otherwise throw away a minute of work."""
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)
    client = FlakyClient(
        [
            tool_call("run_sql", sql="SELECT region, SUM(amount) AS total FROM sales GROUP BY 1"),
            Message(role="assistant", content="North leads."),
        ],
        ollama.ResponseError("Internal Server Error", 500),
    )
    agent, dataset = make_agent(client)
    try:
        assert agent.ask("which region leads?", depth="never").text == "North leads."
        assert client.attempts == 3  # two calls answered, one blip absorbed
    finally:
        dataset.close()


def test_a_rejected_request_is_not_retried(monkeypatch) -> None:
    """A 400 means the request itself is wrong — retrying only fails slower."""
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)
    client = FlakyClient(
        [Message(role="assistant", content="unused")],
        ollama.ResponseError("context too long", 400),
        failures=99,
    )
    agent, dataset = make_agent(client)
    try:
        with pytest.raises(ollama.ResponseError):
            agent.ask("which region leads?", depth="never")
        assert client.attempts == 1
    finally:
        dataset.close()


def test_retries_give_up_and_surface_the_error(monkeypatch) -> None:
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)
    client = FlakyClient(
        [Message(role="assistant", content="unused")], ConnectionError("refused"), failures=99
    )
    agent, dataset = make_agent(client)
    try:
        with pytest.raises(ConnectionError):
            agent.ask("which region leads?", depth="never")
        assert client.attempts == MODEL_RETRIES + 1
    finally:
        dataset.close()


# --- recovering rather than failing ---------------------------------------------


def history() -> list[dict]:
    """Two complete turns, each with a tool call and its result."""
    return [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "first question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "run_sql", "arguments": {"sql": "SELECT 1"}}}],
        },
        {"role": "tool", "tool_name": "run_sql", "content": "first payload"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "run_sql", "arguments": {"sql": "SELECT 1"}}}],
        },
        {"role": "tool", "tool_name": "run_sql", "content": "second payload"},
        {"role": "assistant", "content": "second answer"},
    ]


def test_shedding_drops_stale_payloads_before_anything_else() -> None:
    messages = history()
    assert agent_module._shed_context(messages) is True
    assert len(messages) == len(history())  # nothing removed yet
    payloads = [item["content"] for item in messages if item["role"] == "tool"]
    assert payloads == [OMITTED_PAYLOAD, "second payload"]


def test_shedding_removes_whole_turns_so_no_tool_result_is_orphaned() -> None:
    """Deleting an assistant message while keeping the tool results it called
    leaves a history the model cannot read."""
    messages = history()
    agent_module._shed_context(messages)  # payloads first
    assert agent_module._shed_context(messages) is True

    assert messages[0]["role"] == "system", "the system prompt must survive"
    assert messages[1]["content"] == "second question", "the newest turn must survive"
    for index, item in enumerate(messages):
        if item["role"] == "tool":
            caller = messages[index - 1]
            assert caller.get("tool_calls"), "a tool result outlived the call that made it"


def test_shedding_gives_up_when_nothing_is_left() -> None:
    assert agent_module._shed_context([{"role": "system", "content": "prompt"}]) is False


class OverContextClient(FakeClient):
    """Refuses any prompt bigger than `accepts_at` characters, as the host does."""

    def __init__(self, messages, accepts_at: int):
        super().__init__(messages)
        self.accepts_at = accepts_at
        self.sizes: list[int] = []

    def chat(self, **kwargs):
        sent = kwargs.get("messages") or []
        size = sum(len(str(item.get("content") or "")) for item in sent)
        self.sizes.append(size)
        if size > self.accepts_at:
            raise ollama.ResponseError(
                "The prompt is too long: 450028, model maximum context length: 262144", 400
            )
        return super().chat(**kwargs)


def test_a_conversation_past_the_context_window_is_shed_and_sent_again() -> None:
    """Verified against the host: an over-long prompt is a 400, so it is never
    retried as-is — but it is recoverable by making the prompt smaller."""
    client = OverContextClient([Message(role="assistant", content="Recovered.")], accepts_at=0)
    agent, dataset = make_agent(client)
    try:
        agent.messages = history()
        client.accepts_at = 60  # smaller than the full history, bigger than one turn
        answer = agent.ask("and now?", depth="never")
        assert answer.text == "Recovered."
        assert client.sizes[0] > client.sizes[-1], "it resent the same oversized prompt"
        # The trimming persists, so the next question starts from the smaller history.
        assert len(agent.messages) < len(history()) + 2
    finally:
        dataset.close()


class StepFailureClient(FakeClient):
    """Fails the numbered calls, counting only the ones that are not planning."""

    def __init__(self, messages, plan, fail_on: set[int]):
        super().__init__(messages, plan=plan)
        self.fail_on = fail_on
        self.attempts = 0

    def chat(self, **kwargs):
        system = (kwargs.get("messages") or [{}])[0].get("content", "")
        if "Decide how this question should be answered" in system:
            return super().chat(**kwargs)
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise ollama.ResponseError("bad request", 400)
        return super().chat(**kwargs)


def test_one_failed_sub_question_does_not_lose_the_investigation() -> None:
    client = StepFailureClient(
        [Message(role="assistant", content="Answered from what worked.")],
        plan=["first sub-question here", "second sub-question here"],
        fail_on={1},
    )
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("how should we grow?", depth="auto")
        assert answer.plan == client.plan, "the investigation was abandoned"
        assert answer.text == "Answered from what worked."
    finally:
        dataset.close()


def test_a_failed_investigation_falls_back_to_a_single_pass() -> None:
    """Investigating is the better answer, not the only one."""
    client = StepFailureClient(
        [Message(role="assistant", content="Direct answer.")],
        plan=["first sub-question here", "second sub-question here"],
        fail_on={1, 2},
    )
    agent, dataset = make_agent(client)
    try:
        answer = agent.ask("how should we grow?", depth="auto")
        assert answer.text == "Direct answer."
        assert answer.plan == [], "it reported an investigation it did not complete"
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConnectionError("refused"), "Could not reach Ollama"),
        (ollama.ResponseError("model not found", 404), "not available at"),
        (
            ollama.ResponseError("The prompt is too long: 1, model maximum context length: 2", 400),
            "context window",
        ),
        (ollama.ResponseError("upstream failure", 503), "failing to serve"),
        (ValueError("something else"), "could not be completed"),
    ],
)
def test_an_unrecoverable_failure_says_what_to_do_about_it(error, expected) -> None:
    assert expected in agent_module.explain_failure(error)
