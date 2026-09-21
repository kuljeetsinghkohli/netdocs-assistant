from __future__ import annotations

"""
NetDocs FastAPI application.

Endpoints:
    POST /ask    — Answer a question using hybrid RAG.
    GET  /health — Liveness probe + vector store document count.

The retrieval pipeline (embedder, BM25 index, vector store, reranker) is
initialised once at startup via the FastAPI lifespan context manager and
stored on ``app.state`` so routes can access it without re-initialising
on every request.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from netdocs.api.schemas import (
    AgentRequest,
    AgentResponse,
    AgentStepOut,
    AskFilters,
    AskRequest,
    AskResponse,
    CitationOut,
    HealthResponse,
)
from netdocs.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — initialise heavy objects once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load models and indexes on startup; release on shutdown."""
    logger.info("NetDocs API starting up…")

    from netdocs.embeddings.embedder import get_embedder
    from netdocs.retriever.vector_store import VectorStore
    from netdocs.retriever.bm25_index import BM25Index
    from netdocs.retriever.hybrid import HybridRetriever
    from netdocs.retriever.reranker import get_reranker
    from netdocs.llm.client import get_llm_client

    embedder = get_embedder()
    store = VectorStore()
    bm25 = BM25Index.load_or_build(store)
    retriever = HybridRetriever(store, bm25, embedder)
    reranker = get_reranker()
    llm = get_llm_client()

    app.state.store = store
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.llm = llm

    logger.info("NetDocs API ready. Vector store: %d documents.", store.count())
    yield
    logger.info("NetDocs API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NetDocs Assistant API",
    description="RAG-based Q&A over Contoso Global network documentation.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe — returns OK and current vector store document count."""
    count = app.state.store.count() if hasattr(app.state, "store") else 0
    return HealthResponse(status="ok", vector_store_count=count)


@app.post("/ask", response_model=AskResponse, tags=["qa"])
async def ask(request: AskRequest) -> AskResponse:
    """Answer a question using hybrid retrieval and an LLM.

    - Retrieves relevant chunks using dense + BM25 search with RRF fusion.
    - Applies optional metadata filters (doc_type, site_id, date range).
    - Reranks with a cross-encoder.
    - Generates a cited answer; refuses if retrieval confidence is low.
    """
    from netdocs.llm.generator import generate_answer
    from netdocs.api.schemas import CitationOut as _CitOut

    retriever = app.state.retriever
    reranker = app.state.reranker
    llm = app.state.llm

    # Build filter dict
    filters: dict[str, Any] = {}
    if request.filters:
        f = request.filters
        if f.doc_type:
            filters["doc_type"] = f.doc_type
        if f.site_id:
            filters["site_id"] = f.site_id
        if f.date_from:
            filters["date_from"] = f.date_from
        if f.date_to:
            filters["date_to"] = f.date_to

    # Retrieve
    candidates = retriever.retrieve(
        request.question,
        top_k=(request.top_k or settings.retrieval_top_k) * 4,
        candidate_k=settings.retrieval_candidate_k,
        filters=filters or None,
    )

    # Rerank
    top_chunks = reranker.rerank(
        request.question,
        candidates,
        top_k=request.top_k or settings.retrieval_top_k,
    )

    # Apply doc-type boost to correct systematic mis-ordering for procedural queries
    from netdocs.retriever.hybrid import apply_doc_type_boost
    top_chunks = apply_doc_type_boost(top_chunks, request.question)

    # Generate
    result = generate_answer(request.question, top_chunks, llm)

    return AskResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                doc_id=c.doc_id,
                section=c.section,
                source_file=c.source_file,
                doc_type=c.doc_type,
            )
            for c in result.citations
        ],
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        confidence=result.confidence,
    )


@app.post("/agent", response_model=AgentResponse, tags=["agent"])
async def agent(request: AgentRequest) -> AgentResponse:
    """Answer a question using the agent tool-calling loop.

    The agent decides whether to answer directly from documents or to invoke
    one or more tools (parse_config, check_neighbor_state, draft_change_plan,
    find_related_tickets) before synthesising a final answer.

    State-changing tools (draft_change_plan) require ``require_approval=False``
    in the request body to auto-approve them in the API context.
    """
    from netdocs.agent.loop import AgentLoop, ToolCallStep, ErrorStep, FinalAnswerStep

    llm = app.state.llm

    # In the API, approval prompts are impossible — use a deny-all callback
    # when require_approval=True (so state-changing tools are skipped).
    def _api_deny_approval(tool_name: str, args: dict) -> bool:
        logger.warning(
            "/agent: auto-denying approval for state-changing tool %r "
            "(set require_approval=false to auto-approve)",
            tool_name,
        )
        return False

    loop = AgentLoop(
        llm_client=llm,
        max_steps=request.max_steps or 8,
        require_approval=request.require_approval,
        approval_callback=_api_deny_approval,
    )

    agent_result = loop.run(request.question)

    # Serialise steps
    steps_out: list[AgentStepOut] = []
    for step in agent_result.steps:
        stype = getattr(step, "type", "")
        if stype == "tool_call":
            assert isinstance(step, ToolCallStep)
            steps_out.append(AgentStepOut(
                type="tool_call",
                tool=step.tool,
                args=step.args,
                result=step.result,
                duration_ms=step.duration_ms,
            ))
        elif stype == "error":
            assert isinstance(step, ErrorStep)
            steps_out.append(AgentStepOut(
                type="error",
                tool=step.tool,
                args=step.args,
                error=step.error,
                error_kind=step.error_kind,
            ))
        elif stype == "final_answer":
            assert isinstance(step, FinalAnswerStep)
            steps_out.append(AgentStepOut(
                type="final_answer",
                answer=step.answer,
            ))

    return AgentResponse(
        answer=agent_result.final_answer,
        steps=steps_out,
        aborted=agent_result.aborted,
        abort_reason=agent_result.abort_reason,
        degraded=agent_result.degraded,
    )
