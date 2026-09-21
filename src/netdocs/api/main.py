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

from netdocs.api.schemas import AskFilters, AskRequest, AskResponse, CitationOut, HealthResponse
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
        confidence=result.confidence,
    )
