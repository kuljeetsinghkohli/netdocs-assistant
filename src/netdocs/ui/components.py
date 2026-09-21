"""
NetDocs UI — reusable Streamlit component functions.

Each function renders one logical section of the UI.
No internal netdocs modules are imported here.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from netdocs.ui.api_client import AgentResult, AskResult, Citation, HealthResult


# ---------------------------------------------------------------------------
# Sidebar: status panel
# ---------------------------------------------------------------------------


def render_status_panel(health: HealthResult) -> None:
    """Render the API status section in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔌 API Status")
    if health.reachable:
        st.sidebar.success(f"✅ API reachable — status: **{health.status}**")
        st.sidebar.metric(
            label="Indexed chunks",
            value=f"{health.vector_store_count:,}",
        )
    else:
        st.sidebar.error("❌ API unreachable")
        if health.error:
            st.sidebar.caption(f"Error: {health.error}")


# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------

DOC_TYPE_OPTIONS = ["design_doc", "runbook", "ticket", "config"]


def render_filter_panel() -> dict[str, Any]:
    """
    Render the filter widgets in the sidebar and return the current filter values.

    Returns a dict with keys: doc_types, site_id, date_from, date_to.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filters")

    doc_types = st.sidebar.multiselect(
        "Document types",
        options=DOC_TYPE_OPTIONS,
        default=[],
        help="Leave empty to search all document types.",
    )

    site_id = st.sidebar.text_input(
        "Site ID",
        value="",
        placeholder="e.g. LON-DC01",
        help="Filter results to a specific site.",
    )

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        date_from = st.date_input(
            "Date from",
            value=None,
            help="Lower bound for ticket change_date (YYYY-MM-DD).",
        )
    with col_b:
        date_to = st.date_input(
            "Date to",
            value=None,
            help="Upper bound for ticket change_date (YYYY-MM-DD).",
        )

    return {
        "doc_types": doc_types or None,
        "site_id": site_id.strip() or None,
        "date_from": str(date_from) if date_from else None,
        "date_to": str(date_to) if date_to else None,
    }


# ---------------------------------------------------------------------------
# Sidebar: mode toggle
# ---------------------------------------------------------------------------


def render_mode_toggle() -> str:
    """
    Render the mode toggle in the sidebar.

    Returns ``"ask"`` or ``"agent"``.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Mode")
    mode = st.sidebar.radio(
        "Query mode",
        options=["Ask docs", "Agent"],
        index=0,
        help=(
            "**Ask docs** — single-shot RAG; fast, grounded answers.\n\n"
            "**Agent** — multi-step tool-calling loop; can compare configs and look up tickets."
        ),
    )
    return "ask" if mode == "Ask docs" else "agent"


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------


def render_user_message(question: str) -> None:
    with st.chat_message("user"):
        st.markdown(question)


def render_ask_response(result: AskResult, *, show_debug: bool = False) -> None:
    """Render an /ask response including degraded banner, answer, and citations."""
    with st.chat_message("assistant"):
        if result.refused:
            st.warning(
                f"⚠️ **Response withheld** — {result.refusal_reason or 'insufficient evidence in indexed documents.'}",
                icon="⚠️",
            )
        else:
            # The answer already contains a rendered Sources block; render it
            # directly.  The separate _render_citations expander below provides
            # an interactive drill-down per source.
            st.markdown(result.answer)

        if result.citations:
            _render_citations(result.citations)

        if show_debug:
            with st.expander("🛠 Debug — confidence score", expanded=False):
                st.caption(f"Confidence: `{result.confidence:.4f}`")


def render_agent_response(
    result: AgentResult,
    *,
    show_trace: bool = False,
) -> None:
    """Render an /agent response including degraded banner, answer, and trace."""
    with st.chat_message("assistant"):
        if result.degraded:
            st.warning(
                "⚠️ **Degraded response** — the LLM fell back to extractive mode. "
                "Answer quality may be reduced; verify against source documents.",
                icon="⚠️",
            )

        if result.aborted:
            st.error(
                f"🚫 **Agent aborted** — {result.abort_reason or 'maximum steps reached.'}",
            )

        if result.answer:
            st.markdown(result.answer)

        if show_trace and result.steps:
            _render_agent_trace(result.steps)


def _render_citations(citations: list[Citation]) -> None:
    """Render each citation as an expandable source snippet.

    Uses the citation's ``number`` field (assigned by the generator in
    first-use order) so the ``[N]`` labels here match those in the answer text.
    Falls back to the list position if ``number`` is 0 (e.g. old API responses).
    """
    if not citations:
        return

    st.markdown("---")
    st.caption(f"📄 {len(citations)} source{'s' if len(citations) != 1 else ''}")

    for pos, cit in enumerate(citations, start=1):
        num = cit.number if cit.number else pos
        label = f"[{num}] {cit.source_file or cit.doc_id}  —  *{cit.section}*"
        with st.expander(label, expanded=False):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.caption("Doc ID")
                st.code(cit.doc_id, language=None)
            with col2:
                st.caption("Source file")
                st.code(cit.source_file or "(unknown)", language=None)
            st.caption("Document type")
            st.badge(cit.doc_type or "unknown")


def _render_agent_trace(steps: list) -> None:
    """Render agent tool calls and results as an expandable trace."""
    with st.expander("🔍 Agent trace", expanded=False):
        for i, step in enumerate(steps):
            if step.type == "tool_call":
                st.markdown(
                    f"**Step {i + 1}** — `{step.tool}` "
                    f"<span style='color:gray;font-size:0.8em'>({step.duration_ms:.0f} ms)</span>",
                    unsafe_allow_html=True,
                )
                if step.args:
                    with st.container():
                        st.caption("Arguments")
                        st.json(step.args)
                if step.result is not None:
                    with st.container():
                        st.caption("Result")
                        if isinstance(step.result, (dict, list)):
                            st.json(step.result)
                        else:
                            st.code(str(step.result), language=None)

            elif step.type == "error":
                st.markdown(f"**Step {i + 1}** — ❌ `{step.tool}` error: `{step.error_kind}`")
                if step.error:
                    st.code(step.error, language=None)

            elif step.type == "final_answer":
                st.markdown(f"**Step {i + 1}** — ✅ Final answer synthesised")

            st.divider()


# ---------------------------------------------------------------------------
# Degraded banner (standalone, outside chat bubble)
# ---------------------------------------------------------------------------


def render_degraded_banner() -> None:
    """Render a prominent page-level degraded banner."""
    st.error(
        "⚠️ **Degraded mode** — the LLM fell back to extractive summarisation. "
        "Answers are based on verbatim document passages and may lack coherence. "
        "Always verify against the cited source documents.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


def render_footer() -> None:
    """Render the AI-generated disclaimer footer."""
    st.markdown("---")
    st.caption(
        "🤖 Answers are AI-generated using retrieval-augmented generation and **must be verified "
        "against the cited source documents** before acting on them. "
        "This tool does not replace authoritative network documentation or change-management processes."
    )
