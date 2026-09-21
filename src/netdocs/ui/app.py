"""
NetDocs Assistant — Streamlit UI entry point.

Run with:
    streamlit run src/netdocs/ui/app.py

Or via Makefile:
    make ui

The UI communicates with the NetDocs FastAPI service exclusively over HTTP.
No internal netdocs modules are imported here beyond the ui package itself.
"""

from __future__ import annotations

import os

import streamlit as st

from netdocs.ui.api_client import NetDocsClient
from netdocs.ui.components import (
    render_agent_response,
    render_ask_response,
    render_degraded_banner,
    render_filter_panel,
    render_footer,
    render_mode_toggle,
    render_status_panel,
    render_user_message,
)

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NetDocs Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": ...}

if "health" not in st.session_state:
    st.session_state.health = None

if "show_trace" not in st.session_state:
    st.session_state.show_trace = False

# ---------------------------------------------------------------------------
# API client (base URL from env or sidebar)
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = os.getenv("NETDOCS_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🌐 NetDocs Assistant")
st.sidebar.caption("RAG-based Q&A over Contoso Global network documentation.")

api_url = st.sidebar.text_input(
    "API base URL",
    value=_DEFAULT_BASE_URL,
    help="URL of the running FastAPI service.",
)

client = NetDocsClient(base_url=api_url)

# --- Health check ----------------------------------------------------------
if st.sidebar.button("🔄 Refresh status", use_container_width=True):
    st.session_state.health = client.health()

if st.session_state.health is None:
    st.session_state.health = client.health()

render_status_panel(st.session_state.health)

# --- Mode toggle -----------------------------------------------------------
mode = render_mode_toggle()

# --- Filters ---------------------------------------------------------------
filters = render_filter_panel()

# --- Agent trace toggle (only relevant in agent mode) ----------------------
if mode == "agent":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Agent options")
    st.session_state.show_trace = st.sidebar.toggle(
        "Show agent trace",
        value=st.session_state.show_trace,
        help="Display each tool call and its result below the answer.",
    )
    max_steps = st.sidebar.slider(
        "Max agent steps",
        min_value=1,
        max_value=20,
        value=8,
        help="Maximum number of tool-call iterations the agent may take.",
    )
else:
    max_steps = 8

# --- Clear chat ------------------------------------------------------------
if st.sidebar.button("🗑 Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
# Main panel — page header
# ---------------------------------------------------------------------------

mode_label = "Ask docs" if mode == "ask" else "Agent"
st.title(f"🌐 NetDocs Assistant — {mode_label}")
st.caption(
    "Ask questions about Cisco SD-WAN/Viptela design docs, BGP/OSPF policies, "
    "runbooks, change tickets, and device configs."
)

# ---------------------------------------------------------------------------
# Re-render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        # assistant messages store the rendered payload
        payload = msg.get("payload")
        if payload is None:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        elif payload["type"] == "ask":
            from netdocs.ui.api_client import AskResult, Citation

            result = AskResult(
                answer=payload["answer"],
                citations=[Citation(**c) for c in payload["citations"]],
                refused=payload["refused"],
                refusal_reason=payload["refusal_reason"],
                confidence=payload["confidence"],
            )
            render_ask_response(result)
        elif payload["type"] == "agent":
            from netdocs.ui.api_client import AgentResult, AgentStep

            result = AgentResult(
                answer=payload["answer"],
                steps=[AgentStep(**s) for s in payload["steps"]],
                aborted=payload["aborted"],
                abort_reason=payload["abort_reason"],
                degraded=payload["degraded"],
            )
            render_agent_response(
                result,
                show_trace=st.session_state.show_trace,
            )
            if result.degraded:
                render_degraded_banner()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

question = st.chat_input(
    placeholder=(
        "Ask a question, e.g. 'What is the BGP hold timer for BT MPLS peers?'"
        if mode == "ask"
        else "e.g. 'Compare BGP configs on LON-DC01-RTR01 and LON-DC01-RTR02'"
    )
)

if question:
    # Append user message to history and render it immediately
    st.session_state.messages.append({"role": "user", "content": question})
    render_user_message(question)

    if not st.session_state.health.reachable:
        st.error(
            "❌ Cannot reach the API. Start the service with `make api` and refresh the status.",
        )
    else:
        with st.spinner("Thinking…"):
            if mode == "ask":
                try:
                    result = client.ask(
                        question,
                        doc_types=filters.get("doc_types"),
                        site_id=filters.get("site_id"),
                        date_from=filters.get("date_from"),
                        date_to=filters.get("date_to"),
                    )
                    render_ask_response(result)
                    # Persist serialisable payload for history re-render
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "payload": {
                            "type": "ask",
                            "answer": result.answer,
                            "citations": [
                                {
                                    "doc_id": c.doc_id,
                                    "section": c.section,
                                    "source_file": c.source_file,
                                    "doc_type": c.doc_type,
                                }
                                for c in result.citations
                            ],
                            "refused": result.refused,
                            "refusal_reason": result.refusal_reason,
                            "confidence": result.confidence,
                        },
                    })
                except Exception as exc:
                    st.error(f"❌ Request failed: {exc}")

            else:  # agent mode
                try:
                    result = client.agent(
                        question,
                        max_steps=max_steps,
                    )
                    render_agent_response(
                        result,
                        show_trace=st.session_state.show_trace,
                    )
                    if result.degraded:
                        render_degraded_banner()

                    # Persist serialisable payload for history re-render
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "payload": {
                            "type": "agent",
                            "answer": result.answer,
                            "steps": [
                                {
                                    "type": s.type,
                                    "tool": s.tool,
                                    "args": s.args,
                                    "result": s.result,
                                    "error": s.error,
                                    "error_kind": s.error_kind,
                                    "duration_ms": s.duration_ms,
                                    "answer": s.answer,
                                }
                                for s in result.steps
                            ],
                            "aborted": result.aborted,
                            "abort_reason": result.abort_reason,
                            "degraded": result.degraded,
                        },
                    })
                except Exception as exc:
                    st.error(f"❌ Request failed: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

render_footer()
