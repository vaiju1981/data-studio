"""Streamlit entry point for Smart Data Studio."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from smart_data_studio import logs
from smart_data_studio.agent import Answer, DataAgent
from smart_data_studio.config import ALLOW_LOCAL_PATHS, MODEL_ID, OLLAMA_HOST
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_dataset
from smart_data_studio.ui import render

MULTI_TURN = "Multi-turn"
SINGLE_TURN = "Single turn"


def main() -> None:
    st.set_page_config(page_title="Smart Data Studio", page_icon="▦", layout="wide")
    logs.configure()
    if "session_id" not in st.session_state:
        st.session_state.session_id = logs.new_session()
        logs.event("session.started")
    logs.bind(session=st.session_state.session_id)
    render.styles()
    _initialize_state()

    with st.sidebar:
        _sidebar()

    if st.session_state.dataset is None:
        st.title("Smart Data Studio")
        st.caption("Load CSVs, understand their shape, and ask questions in plain English.")
        render.empty_state()
        return

    render.summary(st.session_state.profiles)
    render.profile_panel(st.session_state.profiles)
    _conversation()

    # Called at the top level of the script so Streamlit pins it to the bottom
    # of the page; inside a tab or container it would scroll away with the page.
    question = st.chat_input("Ask a question about the loaded data")
    if question:
        _answer(question)


def _initialize_state() -> None:
    defaults = {
        "dataset": None,
        "profiles": [],
        "understanding": "",
        "insight_error": "",
        "agent": None,
        "chat": [],
        "mode": MULTI_TURN,
        "metrics": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sidebar() -> None:
    st.header("Data sources")
    uploads = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Select one or more related CSV files.",
    )
    if ALLOW_LOCAL_PATHS:
        paths = st.text_area(
            "Or local CSV paths",
            placeholder="/Users/me/data/sales.csv\n/Users/me/data/regions.csv",
            help="One path per line. Paths are read by the machine running this app.",
        )
    else:
        paths = ""
        st.caption("Server paths are disabled on this deployment. Upload the file instead.")
    if st.button("Load and analyze", type="primary", use_container_width=True):
        _load(uploads, paths)

    if st.session_state.agent is not None:
        st.divider()
        _metrics_controls()

    st.divider()
    st.radio(
        "Conversation mode",
        [MULTI_TURN, SINGLE_TURN],
        key="mode",
        help=(
            "Multi-turn lets the model see earlier questions and their results, so "
            "follow-ups like 'now chart that' work. Single turn answers each question "
            "from the data alone. Either way you keep seeing the full history here."
        ),
    )

    st.divider()
    st.caption(
        f"Queries run locally in DuckDB. Schema, profile statistics and query results are sent "
        f"to `{MODEL_ID}` at `{OLLAMA_HOST}` when insights or chat are used."
    )


def _metrics_controls() -> None:
    st.subheader("Your metrics")
    metrics = st.text_area(
        "Definitions",
        key="metrics",
        height=140,
        placeholder=(
            "avgBet = 0 if handlePulls or coinIn is 0, else coinIn / handlePulls\n"
            "theo_last_90 = sum of theoWin over the last 90 days\n"
            "high bet = avgBet above the median, minimum 100 handle pulls"
        ),
        help=(
            "One definition per line, in plain English. Press ⌘/Ctrl+Enter or click away to "
            "apply. They are added to the model's instructions, so they survive reloads and "
            "apply in single-turn mode too. Pin anything ambiguous — a boundary like "
            "'including the cutoff day' is worth stating."
        ),
    )
    st.session_state.agent.set_metrics(metrics)

    if not metrics.strip():
        return
    # Naming the columns we recognised is how a typo surfaces: the misspelt one
    # simply will not be listed.
    known = st.session_state.dataset.columns_mentioned_in(metrics)
    if known:
        st.caption(f"Columns recognised: {', '.join(known)}")
    else:
        st.warning("No loaded column names found in these definitions — check the spelling.")


def _load(uploads: list[object], paths: str) -> None:
    try:
        sources = [CsvSource.from_upload(upload.name, upload.getvalue()) for upload in uploads]
        sources.extend(
            CsvSource.from_path(Path(line.strip())) for line in paths.splitlines() if line.strip()
        )
        with st.spinner("Loading and profiling your data…"):
            dataset = Dataset.load(sources)
            try:
                profiles = profile_dataset(dataset)
                agent = DataAgent(dataset, profiles)
                # Carry existing definitions onto the new dataset, so the exploration
                # below already knows them rather than learning them afterwards.
                agent.set_metrics(st.session_state.get("metrics", ""))
            except Exception:
                dataset.close()
                raise

        old_dataset = st.session_state.dataset
        if old_dataset is not None:
            old_dataset.close()
        st.session_state.dataset = dataset
        st.session_state.profiles = profiles
        st.session_state.agent = agent
        st.session_state.chat = []
        # Export caches are keyed by position in the chat, which restarts at zero
        # with the new conversation — stale entries would hand back the old data.
        for stale in [key for key in st.session_state if str(key).startswith("export-")]:
            del st.session_state[stale]
        st.session_state.insight_error = ""
        try:
            with st.spinner("Exploring your data…"):
                st.session_state.understanding = agent.build_understanding()
        except Exception as error:
            st.session_state.understanding = ""
            st.session_state.insight_error = str(error)
        st.rerun()
    except Exception as error:
        st.error(f"Could not load the CSV data: {error}")


def _conversation() -> None:
    if st.session_state.understanding or st.session_state.insight_error:
        with st.chat_message("assistant"):
            render.understanding(st.session_state.understanding, st.session_state.insight_error)
    for index, item in enumerate(st.session_state.chat):
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.markdown(item["content"])
            else:
                render.answer(item["answer"], str(index), st.session_state.dataset)


def _answer(question: str) -> None:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing…"):
                answer = st.session_state.agent.ask(
                    question, multi_turn=st.session_state.mode == MULTI_TURN
                )
        except Exception as error:
            answer = Answer(text=f"The analysis could not be completed: {error}", results=[])
        # Every user turn gets an assistant turn, so a failure cannot leave the
        # rendered history and the agent's own history out of step.
        render.answer(answer, str(len(st.session_state.chat)), st.session_state.dataset)
        st.session_state.chat.append({"role": "assistant", "answer": answer})


if __name__ == "__main__":
    main()
