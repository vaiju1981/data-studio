"""Streamlit entry point for Smart Data Studio."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from smart_data_studio import logs, recent, sessions
from smart_data_studio.agent import Answer, DataAgent, explain_failure
from smart_data_studio.config import ALLOW_LOCAL_PATHS, MODEL_ID, OLLAMA_HOST
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.profile import profile_dataset
from smart_data_studio.ui import render

MULTI_TURN = "Multi-turn"
SINGLE_TURN = "Single turn"
DEPTHS = {
    "When the question needs it": "auto",
    "Always investigate": "always",
    "Never — one pass": "never",
}


def main() -> None:
    st.set_page_config(page_title="Smart Data Studio", page_icon="▦", layout="wide")
    logs.configure()
    if "session_id" not in st.session_state:
        st.session_state.session_id = logs.new_session()
        logs.event("session.started")
    logs.bind(session=st.session_state.session_id)
    render.styles()
    _initialize_state()
    # After the state exists, because the first run of a tab has no workspace and
    # is not an eviction. A tab that had one and no longer does is.
    if not sessions.touch(st.session_state.session_id) and st.session_state.dataset is not None:
        _expire()

    with st.sidebar:
        _sidebar()

    if st.session_state.dataset is None:
        st.title("Smart Data Studio")
        st.caption("Load CSVs, understand their shape, and ask questions in plain English.")
        if st.session_state.expired:
            st.warning(
                "This workspace was released after sitting idle, so the data is gone from "
                "memory. Load the files again to carry on — nothing was sent anywhere."
            )
        render.empty_state()
        return

    render.summary(st.session_state.profiles)
    render.lineage_panel(st.session_state.dataset)
    render.relationship_panel(st.session_state.agent)
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
        "depth": next(iter(DEPTHS)),
        "metrics": "",
        "expired": False,
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
    chosen: list[str] = []
    if ALLOW_LOCAL_PATHS:
        known = recent.recall()
        # Kept across reruns so the sidebar always shows what the button will load,
        # and pruned to what still exists, since Streamlit refuses a selection
        # outside its options.
        st.session_state.chosen_paths = [
            path for path in st.session_state.get("chosen_paths", []) if path in known
        ]
        if known:
            chosen = st.multiselect(
                "Files you have loaded before",
                known,
                key="chosen_paths",
                format_func=lambda path: f"{Path(path).parent.name}/{Path(path).name}",
                help="Pick any number. Anything typed below is loaded along with them.",
            )
        paths = st.text_area(
            "Add a local CSV path" if known else "Or local CSV paths",
            placeholder="/Users/me/data/sales.csv\n/Users/me/data/regions.csv",
            help="One path per line. Paths are read by the machine running this app.",
        )
    else:
        paths = ""
        st.caption("Server paths are disabled on this deployment. Upload the file instead.")
    if st.button("Load and analyze", type="primary", use_container_width=True):
        _load(uploads, paths, chosen)

    if st.session_state.agent is not None:
        st.divider()
        _metrics_controls()

    st.divider()
    st.radio(
        "Investigate",
        list(DEPTHS),
        key="depth",
        help=(
            "A question of judgement — strategy, causes, 'how should we' — is broken into "
            "sub-questions and answered from all of them, including one asked at the grain "
            "the decision is made at. A lookup is answered in one pass either way."
        ),
    )

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

    if st.session_state.dataset is not None:
        st.divider()
        st.caption(
            "Your data lives in memory and is discarded when the session ends, is replaced, "
            "or is deleted here. Only the list of file paths is kept, on this machine, so "
            "you need not retype them; deleting clears that too."
        )
        if st.button("Delete my data", use_container_width=True):
            _forget()

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
        # Deliberately generic: naming one industry's measures makes a tool for any
        # CSV look like a tool for that industry.
        placeholder=(
            "unit price = 0 if quantity is 0, else revenue / quantity\n"
            "revenue_last_90 = sum of revenue over the last 90 days\n"
            "high value = unit price above the median, minimum 100 units"
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


def _clear_workspace() -> None:
    """Drop everything that describes the loaded data, keeping the tab's settings."""
    st.session_state.relationship_status = {}
    for key in ("dataset", "agent"):
        st.session_state[key] = None
    st.session_state.profiles = []
    st.session_state.chat = []
    st.session_state.understanding = ""
    st.session_state.insight_error = ""
    for stale in [key for key in st.session_state if str(key).startswith("export-")]:
        del st.session_state[stale]


def _expire() -> None:
    """The workspace was released for being idle while this tab still showed it.

    Cleared here rather than left to fail later: the connection is already closed,
    so every panel below is describing rows that no longer exist.
    """
    _clear_workspace()
    st.session_state.expired = True
    logs.event("session.expired")


def _forget() -> None:
    """Throw the workspace away on request, and prove it in the log."""
    sessions.release(st.session_state.session_id)
    recent.forget()
    _clear_workspace()
    st.session_state.chosen_paths = []
    st.session_state.expired = False
    logs.event("data.deleted", by="user")
    st.rerun()


def _load(uploads: list[object], paths: str, chosen: list[str] | None = None) -> None:
    try:
        local = [Path(line.strip()) for line in paths.splitlines() if line.strip()]
        # Remembered files and a newly typed one load together, so adding a second
        # table to a set you already use does not mean retyping the first.
        local = [Path(item) for item in chosen or []] + local
        sources = [CsvSource.from_upload(upload.name, upload.getvalue()) for upload in uploads]
        sources.extend(CsvSource.from_path(path) for path in local)
        # Before the load: refusing a large file once it is parsed and profiled
        # wastes the time and the memory both.
        sessions.check_capacity(st.session_state.session_id)
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

        recent.remember(local)

        # Registering before adopting means a host that is already full refuses
        # here, with the previous workspace still intact.
        sessions.register(st.session_state.session_id, dataset)
        old_dataset = st.session_state.dataset
        if old_dataset is not None and old_dataset is not dataset:
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
        st.session_state.expired = False
        # Verdicts describe the workspace that is going away, not the new one.
        st.session_state.relationship_status = {}
        try:
            with st.spinner("Exploring your data…"):
                st.session_state.understanding = agent.build_understanding()
                # After exploring, so the proposal sees what exploring established.
                agent.propose_relationships()
        except Exception as error:
            st.session_state.understanding = ""
            # The profile is already computed and shown; only the written summary
            # is lost, so the reason belongs on screen rather than in the log alone.
            st.session_state.insight_error = explain_failure(error)
        st.rerun()
    except sessions.TooManySessions as error:
        st.error(str(error))
    except Exception as error:
        logs.failure("load.failed")
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


def _progress(status, message: str) -> None:
    """Show what the agent is doing while it does it.

    The guard admits a single SELECT and nothing else, so a message starting with
    SELECT or WITH is SQL rather than a phase — worth a code block, and too long to
    put in the collapsed label.
    """
    if message.upper().startswith(("SELECT", "WITH")):
        status.code(message, language="sql", wrap_lines=True)
    else:
        status.write(message)
        # expanded is passed every time on purpose: update() clears the field when
        # it is omitted, which collapses the panel mid-question.
        status.update(label=message, expanded=True)


def _answer(question: str) -> None:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status("Working through it…", expanded=True) as status:
            try:
                answer = st.session_state.agent.ask(
                    question,
                    multi_turn=st.session_state.mode == MULTI_TURN,
                    depth=DEPTHS[st.session_state.depth],
                    progress=lambda message: _progress(status, message),
                )
                status.update(label="Done", state="complete", expanded=False)
            except Exception as error:
                logs.failure("answer.failed")
                answer = Answer(text=explain_failure(error), results=[])
                status.update(label="Could not finish", state="error", expanded=False)
        # Every user turn gets an assistant turn, so a failure cannot leave the
        # rendered history and the agent's own history out of step.
        render.answer(answer, str(len(st.session_state.chat)), st.session_state.dataset)
        st.session_state.chat.append({"role": "assistant", "answer": answer})


if __name__ == "__main__":
    main()
