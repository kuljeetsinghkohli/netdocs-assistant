from __future__ import annotations

"""
Cross-encoder reranker.

Takes a query and a list of candidate chunks and scores them with a
cross-encoder (a.k.a. bi-directional attention) model so that
relevance is judged on the (query, passage) pair rather than
independently.

Model used by default: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
  — ~90 MB, runs on CPU in ~50 ms for 20 candidates.
"""

import logging
from typing import Any

from netdocs.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker over a list of candidate chunks.

    Args:
        model_name: HuggingFace cross-encoder model identifier.
    """

    def __init__(self, model_name: str = _DEFAULT_RERANK_MODEL) -> None:
        from sentence_transformers import CrossEncoder  # lazy import

        logger.info("Loading cross-encoder reranker: %s", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("Reranker ready.")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score and rerank candidates using the cross-encoder.

        Args:
            query:      The user query.
            candidates: List of dicts, each must have key ``"text"``.
            top_k:      Number of results to return after reranking.
                        Defaults to ``settings.retrieval_top_k``.

        Returns:
            List of candidate dicts with an added ``"rerank_score"`` key,
            sorted descending by ``"rerank_score"``.  Scores are raw logits
            from the cross-encoder (higher = more relevant).
        """
        k = top_k or settings.retrieval_top_k
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        raw_scores: list[float] = self._model.predict(pairs).tolist()

        for candidate, score in zip(candidates, raw_scores):
            candidate["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:k]


# Module-level singleton — avoids re-loading the model on each request.
_reranker_instance: Reranker | None = None


def get_reranker(model_name: str = _DEFAULT_RERANK_MODEL) -> Reranker:
    """Return the reranker singleton, loading the model on first call."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(model_name)
    return _reranker_instance
