"""
Retrieval-only evaluation harness — no LLM calls.

Run with::

    python -m netdocs eval [--top-k N] [--out PATH] [--threshold FLOAT]

Metrics reported:
  Hit@1, Hit@3, Hit@5   — fraction of answerable questions where the expected
                          source file appears in the top-1/3/5 results.
  MRR                   — Mean Reciprocal Rank (answerable questions only).
  Refusal accuracy      — fraction of unanswerable questions for which ALL
                          top-k reranker scores are below the threshold.
  Per-question table    — every question, its rank (or "miss"), and whether
                          it was a refusal hit/miss.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    qid: str
    question: str
    unanswerable: bool
    expected_sources: list[str]
    retrieved_sources: list[str]          # ordered top-k source-file stems
    retrieved_scores: list[float]         # rerank_score for each retrieved chunk
    rank: int | None                      # 1-based rank of first hit, None if miss
    refusal_correct: bool | None          # None for answerable questions
    elapsed_s: float = 0.0

    @property
    def hit_at_1(self) -> bool:
        return self.rank is not None and self.rank <= 1

    @property
    def hit_at_3(self) -> bool:
        return self.rank is not None and self.rank <= 3

    @property
    def hit_at_5(self) -> bool:
        return self.rank is not None and self.rank <= 5

    @property
    def reciprocal_rank(self) -> float:
        if self.rank is None:
            return 0.0
        return 1.0 / self.rank


@dataclass
class EvalReport:
    results: list[QuestionResult] = field(default_factory=list)
    threshold: float = 0.10
    top_k: int = 5

    # Computed metrics
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    refusal_accuracy: float = 0.0
    n_answerable: int = 0
    n_unanswerable: int = 0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _stem(source_file: str) -> str:
    """Extract just the filename stem from a source_file path."""
    from pathlib import Path
    return Path(source_file).stem


def run_eval(
    *,
    top_k: int = 5,
    confidence_threshold: float | None = None,
    candidate_k: int | None = None,
) -> EvalReport:
    """Run the golden-set evaluation using the live retrieval pipeline.

    No LLM calls are made — only the retriever and reranker are exercised.

    Args:
        top_k:                 Number of top chunks to retrieve per question.
        confidence_threshold:  Score below which a result is considered a
                               refusal. Defaults to ``settings.retrieval_confidence_threshold``.
        candidate_k:           Hybrid retrieval candidate pool size.

    Returns:
        A populated :class:`EvalReport`.
    """
    from netdocs.config import settings
    from netdocs.embeddings.embedder import get_embedder
    from netdocs.retriever.vector_store import VectorStore
    from netdocs.retriever.bm25_index import BM25Index
    from netdocs.retriever.hybrid import HybridRetriever
    from netdocs.retriever.reranker import get_reranker
    from netdocs.eval.golden_set import GOLDEN_SET

    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.retrieval_confidence_threshold
    )
    k_cand = candidate_k or settings.retrieval_candidate_k

    logger.info("Initialising retrieval pipeline for eval…")
    embedder = get_embedder()
    store = VectorStore()
    bm25 = BM25Index.load_or_build(store)
    retriever = HybridRetriever(store, bm25, embedder)
    reranker = get_reranker()
    logger.info("Pipeline ready. Evaluating %d questions…", len(GOLDEN_SET))

    results: list[QuestionResult] = []

    for gq in GOLDEN_SET:
        t0 = time.perf_counter()

        # Retrieve + rerank (no LLM)
        candidates = retriever.retrieve(
            gq.question,
            top_k=top_k * 4,
            candidate_k=k_cand,
        )
        top_chunks = reranker.rerank(gq.question, candidates, top_k=top_k)

        # Apply doc-type boost (same as production pipeline)
        from netdocs.retriever.hybrid import apply_doc_type_boost
        top_chunks = apply_doc_type_boost(top_chunks, gq.question)

        elapsed = time.perf_counter() - t0

        # Extract ordered source stems from retrieved chunks
        retrieved_stems = [
            _stem(c.get("metadata", {}).get("source_file", c.get("id", "")))
            for c in top_chunks
        ]
        retrieved_scores = [
            float(c.get("rerank_score", c.get("rrf_score", 0.0)))
            for c in top_chunks
        ]

        # Find rank of first expected source
        rank: int | None = None
        if not gq.unanswerable and gq.expected_sources:
            for i, stem in enumerate(retrieved_stems, start=1):
                if any(exp in stem for exp in gq.expected_sources):
                    rank = i
                    break

        # Refusal accuracy for unanswerable questions
        refusal_correct: bool | None = None
        if gq.unanswerable:
            from netdocs.retriever.hybrid import is_out_of_scope
            top_score = max(retrieved_scores, default=0.0)
            refusal_correct = top_score < threshold or is_out_of_scope(gq.question)

        qr = QuestionResult(
            qid=gq.qid,
            question=gq.question,
            unanswerable=gq.unanswerable,
            expected_sources=gq.expected_sources,
            retrieved_sources=retrieved_stems,
            retrieved_scores=retrieved_scores,
            rank=rank,
            refusal_correct=refusal_correct,
            elapsed_s=elapsed,
        )
        results.append(qr)
        logger.debug(
            "%-4s  rank=%-4s  refusal_ok=%-5s  %s",
            gq.qid,
            str(rank) if rank else "miss",
            str(refusal_correct) if refusal_correct is not None else "n/a",
            gq.question[:60],
        )

    # Aggregate metrics
    answerable = [r for r in results if not r.unanswerable]
    unanswerable = [r for r in results if r.unanswerable]

    n_ans = len(answerable)
    n_unans = len(unanswerable)

    report = EvalReport(
        results=results,
        threshold=threshold,
        top_k=top_k,
        n_answerable=n_ans,
        n_unanswerable=n_unans,
    )

    if n_ans:
        report.hit_at_1 = sum(r.hit_at_1 for r in answerable) / n_ans
        report.hit_at_3 = sum(r.hit_at_3 for r in answerable) / n_ans
        report.hit_at_5 = sum(r.hit_at_5 for r in answerable) / n_ans
        report.mrr = sum(r.reciprocal_rank for r in answerable) / n_ans

    if n_unans:
        report.refusal_accuracy = sum(
            1 for r in unanswerable if r.refusal_correct
        ) / n_unans

    return report
