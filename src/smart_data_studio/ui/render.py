"""Presentation helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from smart_data_studio.agent import Answer
from smart_data_studio.config import MAX_EXPORT_ROWS
from smart_data_studio.dataset import Dataset, QueryResult
from smart_data_studio.profile import TableProfile


def answer(item: Answer, key: str, dataset: Dataset) -> None:
    """Conclusion first, then the chart, then the evidence behind both."""
    st.markdown(item.text)
    if item.chart is not None:
        st.plotly_chart(item.chart, use_container_width=True, key=f"chart-{key}")
    if not item.results:
        # Never let an unsupported answer look like a verified one.
        st.caption("No query was run for this answer.")
        return

    count = len(item.results)
    st.caption(f"{count} quer{'y' if count == 1 else 'ies'} ran")
    for index, result in enumerate(item.results, start=1):
        with st.container(border=True):
            label = f"Query {index} · {result.total_rows:,} rows"
            if result.truncated:
                label += f" · showing the first {len(result.frame):,}"
            st.caption(label)
            # Generated SQL runs long; without wrapping the tail is simply clipped.
            st.code(result.sql, language="sql", wrap_lines=True)
            st.dataframe(result.frame, use_container_width=True, hide_index=True)
            _export(result, f"{key}-{index}", dataset)


def _export(result: QueryResult, key: str, dataset: Dataset) -> None:
    """Small results download straight away; large ones are rebuilt only on request."""
    if not result.truncated:
        st.download_button(
            "Download CSV",
            result.frame.to_csv(index=False).encode("utf-8"),
            file_name=f"query-{key}.csv",
            mime="text/csv",
            key=f"download-{key}",
        )
        return

    state_key = f"export-{key}"
    prepared = st.session_state.get(state_key)
    if prepared is None:
        exportable = min(result.total_rows, MAX_EXPORT_ROWS)
        if st.button(f"Prepare full export ({exportable:,} rows)", key=f"prepare-{key}"):
            with st.spinner("Building the export…"):
                full = dataset.query(result.sql, row_limit=MAX_EXPORT_ROWS)
            st.session_state[state_key] = full.frame.to_csv(index=False).encode("utf-8")
            st.rerun()
        if result.total_rows > MAX_EXPORT_ROWS:
            st.caption(
                f"Export is capped at {MAX_EXPORT_ROWS:,} of {result.total_rows:,} rows. "
                "Narrow the query to export everything."
            )
        return

    st.download_button(
        f"Download CSV ({len(prepared) / 1e6:.1f} MB)",
        prepared,
        file_name=f"query-{key}.csv",
        mime="text/csv",
        key=f"download-{key}",
    )


def understanding(text: str, error: str) -> None:
    if text:
        st.markdown(text)
        return
    st.warning("Written insights are unavailable, but the computed profile is ready.")
    if error:
        st.caption(error)


def summary(profiles: list[TableProfile]) -> None:
    # SUMMARIZE returns one row per column, so the profile already carries the
    # column count and no per-rerun DESCRIBE is needed.
    columns = st.columns(4)
    columns[0].metric("Tables", len(profiles))
    columns[1].metric("Rows", f"{sum(p.row_count for p in profiles):,}")
    columns[2].metric("Columns", f"{sum(len(p.stats) for p in profiles):,}")
    columns[3].metric("Profile flags", sum(len(p.findings) for p in profiles))


def profile_panel(profiles: list[TableProfile]) -> None:
    with st.expander("Data profile", expanded=False):
        for profile in profiles:
            st.markdown(f"**{profile.table_name}** · {profile.row_count:,} rows")
            for finding in profile.findings:
                st.markdown(f"- {finding}")
            st.dataframe(profile.stats, use_container_width=True, hide_index=True)


def styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8f5; }
        [data-testid="stSidebar"] { background: #17211d; color: #f6f1e7; }
        [data-testid="stSidebar"] * { color: inherit; }
        /* Widgets keep the light theme's surface colours, so the inherited cream
           text above lands on near-white and disappears. Darken the surfaces to
           match the sidebar rather than fighting the text colour. */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
        [data-testid="stSidebar"] textarea {
            background: #23302b; border: 1px solid #3d4d46;
        }
        [data-testid="stSidebar"] textarea::placeholder { color: #a3b3ab; opacity: 1; }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background: #f6f1e7; color: #17211d; border: none;
        }
        h1, h2, h3 { letter-spacing: -0.025em; }

        /* Leave room for the pinned composer so the last answer is never hidden. */
        [data-testid="stMainBlockContainer"] { padding-bottom: 7rem; max-width: 1100px; }
        [data-testid="stChatInput"] textarea { font-size: 0.97rem; }

        [data-testid="stChatMessage"] {
            background: transparent; padding: 0.35rem 0 1.1rem;
        }
        [data-testid="stChatMessage"] [data-testid="stCode"] { font-size: 0.8rem; }
        [data-testid="stMetric"] {
            background: white; border: 1px solid #dfe3de; border-radius: 14px; padding: 0.85rem;
        }
        .empty-card {
            margin: 4rem auto 0; max-width: 650px; padding: 3.5rem;
            text-align: center; border: 1px solid #d9ded8; border-radius: 24px;
            background: white; box-shadow: 0 18px 50px rgba(23, 33, 29, .07);
        }
        .empty-card p { color: #53605a; font-size: 1.05rem; line-height: 1.65; }
        .empty-mark { color: #d96b45; font-size: 2.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def empty_state() -> None:
    st.markdown(
        """
        <div class="empty-card">
          <div class="empty-mark">▦</div>
          <h2>Start with a CSV</h2>
          <p>Upload a file or enter a local path. Your data stays in an in-memory database,
          ready for fast profiling and read-only analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
