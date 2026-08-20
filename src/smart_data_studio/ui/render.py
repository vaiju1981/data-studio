"""Presentation helpers for the Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smart_data_studio.agent import Answer
from smart_data_studio.config import MAX_EXPORT_ROWS
from smart_data_studio.dataset import Dataset, QueryResult
from smart_data_studio.profile import TableProfile
from smart_data_studio.tools import AnalysisRecord


def answer(item: Answer, key: str, dataset: Dataset) -> None:
    """Conclusion first, then the chart, then the evidence behind both."""
    if item.plan:
        with st.expander(f"Investigated in {len(item.plan)} steps", expanded=False):
            for index, step in enumerate(item.plan, start=1):
                st.markdown(f"{index}. {step}")
    st.markdown(item.text)
    for assumption in item.assumptions:
        # Shown for the same reason the SQL is: an answer nobody can check is not
        # finished. The data held no value for this, so whatever was said about it
        # came from the model rather than the file.
        st.warning(
            f"**{assumption}** — this data holds no such value, so anything the answer says "
            "about it comes from the model's own knowledge, not from your file. Worth "
            "confirming before relying on it."
        )
    if item.chart is not None:
        st.plotly_chart(item.chart, use_container_width=True, key=f"chart-{key}")
    for index, analysis in enumerate(item.analyses, start=1):
        _analysis(analysis, f"{key}-{index}")

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


TITLES = {
    "forecast": "Forecast",
    "trend": "Trend",
    "anomalies": "Anomalies",
    "comparison": "Group comparison",
    "drivers": "Drivers",
    "associations": "Associations",
}


def _analysis(analysis: AnalysisRecord, key: str) -> None:
    """Show what the model was told, so a quoted MAPE or forecast can be checked."""
    result = analysis.result
    with st.container(border=True):
        periods = result.get("periods_used")
        st.caption(
            f"{TITLES.get(analysis.kind, analysis.kind)} · {analysis.subject}"
            + (f" · {periods} periods" if periods else "")
        )
        for note in result.get("notes") or []:
            st.warning(note)

        if analysis.kind == "forecast":
            accuracy = result.get("accuracy") or {}
            st.markdown(f"**Model** `{result.get('model', '')}`")
            if "model_wape_pct" in accuracy:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Method": "This model", "Error (WAPE %)": accuracy["model_wape_pct"]},
                            {
                                "Method": "Repeat last value",
                                "Error (WAPE %)": accuracy["repeat_last_value_wape_pct"],
                            },
                            {
                                "Method": "History average",
                                "Error (WAPE %)": accuracy["history_mean_wape_pct"],
                            },
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            if accuracy.get("verdict"):
                st.caption(accuracy["verdict"])
            st.dataframe(
                pd.DataFrame(result.get("forecast") or []),
                hide_index=True,
                use_container_width=True,
            )
        elif analysis.kind == "anomalies":
            found = result.get("anomalies") or []
            st.caption(result.get("method", ""))
            if found:
                st.dataframe(pd.DataFrame(found), hide_index=True, use_container_width=True)
            else:
                st.markdown("No period broke the pattern.")
        else:
            shown = {
                name: value
                for name, value in result.items()
                if name not in {"notes", "periods_used"}
            }
            st.dataframe(
                pd.DataFrame([{"Measure": k, "Value": v} for k, v in shown.items()]),
                hide_index=True,
                use_container_width=True,
            )


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


def lineage_panel(dataset: Dataset) -> None:
    """Where each table came from, and anything odd about how it parsed."""
    warnings = [(item.table, note) for item in dataset.lineage for note in item.warnings]
    outstanding = [pair for pair in warnings if "converted to a number" not in pair[1]]
    notices = len(outstanding) + len(dataset.rejected)
    label = "Source data" + (f" · {notices} parsing warning(s)" if notices else "")
    with st.expander(label, expanded=bool(notices)):
        # A file that was skipped is invisible in the table below, so it is named
        # here rather than leaving the user to notice one is missing.
        for rejection in dataset.rejected:
            st.error(f"Not loaded — {rejection}")
        for table, note in warnings:
            (st.success if "converted to a number" in note else st.warning)(f"**{table}** — {note}")
        _repair(dataset)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Table": item.table,
                        "From": item.source,
                        "Rows": f"{item.rows:,}",
                        "Columns": item.columns,
                        "Loaded (UTC)": item.loaded_at.replace("+00:00", ""),
                    }
                    for item in dataset.lineage
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        for table in dataset.tables:
            st.markdown(f"**{table}** columns as parsed")
            st.dataframe(
                pd.DataFrame(dataset.schema(table), columns=["Column", "Type"]),
                hide_index=True,
                use_container_width=True,
            )


def _repair(dataset: Dataset) -> None:
    """Offer to fix the type the panel just complained about.

    Reporting that a column was read as text and leaving the user to cast it in
    every query is half an answer.
    """
    convertible = [
        (table, column) for table in dataset.tables for column in dataset.text_columns(table)
    ]
    if not convertible:
        return
    st.markdown("**Read a text column as a number**")
    choice = st.selectbox(
        "Column",
        convertible,
        format_func=lambda pair: f"{pair[0]}.{pair[1]}",
        key="repair-choice",
        label_visibility="collapsed",
    )
    if st.button("Convert to number", key="repair-go"):
        table, column = choice
        try:
            st.session_state.repair_note = dataset.convert_to_number(table, column)
        except ValueError as error:
            st.session_state.repair_note = str(error)
        st.rerun()
    if st.session_state.get("repair_note"):
        st.caption(st.session_state.pop("repair_note"))


def relationship_panel(agent) -> None:
    """Candidates, what measuring them showed, and whether you agree.

    Structural and semantic are kept apart on purpose. A 100% match on a unique
    key still only proves what the join does to these rows — two unrelated id
    columns can coincide, and one pair here overlapped 85.7% by accident. So the
    wording is "structurally compatible", never "related", and confirming is
    yours to do.
    """
    found = getattr(agent, "relationships", None)
    if not found or not (found.joins or found.rejected):
        return
    with st.expander(f"How the tables relate · {len(found.joins)} candidate(s)", expanded=False):
        st.caption(
            "Proposed by the model, then measured here. Measuring shows what a join "
            "does to these rows; it cannot show that the columns mean the same thing."
        )
        for index, candidate in enumerate(found.joins):
            facts = agent.tools._join_facts.get((candidate.left, candidate.right))
            with st.container(border=True):
                st.markdown(f"**{candidate}**")
                if candidate.reason:
                    st.caption(candidate.reason)
                if facts is None:
                    st.caption("Not measured.")
                else:
                    multiplies = [
                        side.ref.table
                        for side, name in ((facts.left, "left"), (facts.right, "right"))
                        if facts.multiplies_side(name)
                    ]
                    st.markdown(
                        f"- {facts.cardinality}, producing {facts.joined_rows:,} rows"
                        + (f" — repeats {', '.join(multiplies)}" if multiplies else "")
                    )
                    if facts.partial:
                        st.markdown(
                            f"- {facts.left.unmatched:,} left and {facts.right.unmatched:,} "
                            "right rows match nothing"
                        )
                _confirmation(index, candidate)
        for reason in found.rejected:
            st.caption(f"Not usable — {reason}")


def _confirmation(index: int, candidate) -> None:
    """Yours to say. A rejected candidate stops being offered as a join path."""
    key = f"relationship-{index}"
    status = st.session_state.get("relationship_status", {}).get(str(candidate))
    if status:
        st.caption(f"You marked this **{status}**.")
        return
    left, right = st.columns(2)
    if left.button("Makes sense", key=f"{key}-yes", use_container_width=True):
        st.session_state.setdefault("relationship_status", {})[str(candidate)] = "meaningful"
        st.rerun()
    if right.button("Not meaningful", key=f"{key}-no", use_container_width=True):
        st.session_state.setdefault("relationship_status", {})[str(candidate)] = "rejected"
        st.rerun()


def profile_panel(profiles: list[TableProfile]) -> None:
    with st.expander("Data profile", expanded=False):
        for profile in profiles:
            st.markdown(f"**{profile.table_name}** · {profile.row_count:,} rows")
            for finding in profile.findings:
                st.markdown(f"- {finding}")
            if profile.dictionary:
                # Shown because the model is sent exactly this. Without it there is
                # no way to tell a column it never mentioned from one it never saw.
                st.markdown("**Values the model was given for each dimension**")
                for line in profile.dictionary:
                    st.markdown(f"- {line}")
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
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div,
        [data-testid="stSidebar"] textarea {
            background: #23302b; border: 1px solid #3d4d46;
        }
        [data-testid="stSidebar"] textarea::placeholder,
        [data-testid="stSidebar"] input::placeholder { color: #a3b3ab; opacity: 1; }
        /* Selected files render as accent-coloured chips. The inherited cream on
           that orange is 3.0:1, under the 4.5:1 small text needs; the ink colour
           on the same orange is 4.8:1, so the accent is kept and the text darkens. */
        [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] span {
            color: #17211d;
        }
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
