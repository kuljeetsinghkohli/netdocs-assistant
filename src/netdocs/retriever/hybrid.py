from __future__ import annotations

"""
Hybrid retrieval: dense vector search + BM25, fused via Reciprocal Rank Fusion.

Pipeline:
    1. Expand the query (append procedure-domain keywords when phrasing is
       procedural — "runbook", "procedure", "how do I", "steps").
    2. Embed the (expanded) query.
    3. Fetch ``candidate_k`` results from ChromaDB (dense ANN).
    4. Fetch ``candidate_k`` results from the BM25 index (keyword).
    5. Fuse rankings with RRF.
    6. Apply metadata pre-filters to the fused set.
    7. Return the top ``top_k`` results.

Metadata filters accepted (all optional, all ANDed together):
    doc_type:   One or more of  "design_doc" | "runbook" | "ticket" | "config"
    site_id:    Site identifier string, e.g. "LON-DC01"
    date_from:  ISO date string "YYYY-MM-DD" — filters ``change_date >= date_from``
    date_to:    ISO date string "YYYY-MM-DD" — filters ``change_date <= date_to``
"""

import logging
import re
from typing import Any

from netdocs.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query normalisation and expansion
# ---------------------------------------------------------------------------

# Phrases that signal "I want a procedure / runbook answer"
_PROCEDURE_TRIGGERS = re.compile(
    r"\b(procedure|runbook|steps?|how do i|how to|what (?:is|are) the steps?|"
    r"what should i do|diagnos|troubleshoot|recover|remediat|fix|resolve|"
    r"when (?:a|the) \w+ (?:goes?|went|is|flap|fail|down)|"
    r"what (?:is|are) the (?:runbook|procedure))\b",
    re.IGNORECASE,
)

# Expansion suffix injected for BM25 (dense embedding already covers semantics)
_PROCEDURE_EXPANSION = "runbook procedure steps diagnosis recovery"

# Ticket-content words that should NOT outrank runbooks for procedural queries
_TICKET_NOISE_WORDS = re.compile(
    r"\b(rollback plan|rollback|implementation plan|test plan|outcome|change window)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Out-of-scope (OOD) topic detection
# ---------------------------------------------------------------------------

# Topics that are provably NOT covered in the NetDocs corpus.
# Queries that match these patterns should be refused regardless of reranker
# score, because a high cross-encoder score can arise from partial keyword
# overlap with in-scope documents (e.g. "SD-WAN vendor" matching SD-WAN docs
# even though vendor SLA details are absent).
_OOD_PATTERNS = re.compile(
    r"\b(snmp community|snmp community string|wifi password|wi-?fi password|"
    r"wireless password|guest (?:wifi|network) password|"
    r"sla for .{0,40}(?:vendor|support|contract)|"
    r"vendor support (?:sla|contract|agreement)|"
    r"(?:support contract|maintenance contract) sla|"
    r"configure (?:cisco )?aci|aci fabric)\b",
    re.IGNORECASE,
)


def is_out_of_scope(query: str) -> bool:
    """Return True when the query matches a known out-of-corpus topic.

    This guard prevents a high cross-encoder score arising from superficial
    keyword overlap (e.g. "SD-WAN vendor SLA" scoring against SD-WAN design
    docs) from leaking through to the user as a confident answer.

    Args:
        query: The raw user query.

    Returns:
        True if the query is provably unanswerable from the current corpus.
    """
    return bool(_OOD_PATTERNS.search(query))


# ---------------------------------------------------------------------------
# Query synonym normalisation
# ---------------------------------------------------------------------------

# Maps surface-form phrasing variants to canonical retrieval terms.
# Applied before BM25 so that e.g. "flapping" and "flaps" both hit "flap"
# and "session flaps" expands to include "BGP peer flapping keepalive".
_SYNONYM_MAP: list[tuple[re.Pattern[str], str]] = [
    # BGP session instability variants
    (re.compile(r"\bflap(?:ping|s)?\b", re.IGNORECASE), "flap flapping"),
    # "goes down" / "went down" → common runbook phrasing
    (re.compile(r"\bgo(?:es|ne)?\s+down\b", re.IGNORECASE), "down failure"),
    (re.compile(r"\bwent\s+down\b", re.IGNORECASE), "down failure"),
    # "neighbour" vs "neighbor" normalisation
    (re.compile(r"\bneighbour\b", re.IGNORECASE), "neighbor neighbour"),
    # "recover" / "recovery" → standard runbook keyword
    (re.compile(r"\brecover(?:y|ing)?\b", re.IGNORECASE), "recover recovery"),
    # "renew" / "renewal" → cert runbook keyword
    (re.compile(r"\brenew(?:al|ing)?\b", re.IGNORECASE), "renew renewal"),
]


def normalize_query(query: str) -> str:
    """Apply synonym expansion to the query for BM25 retrieval.

    This is applied *in addition to* the procedure-trigger expansion so that
    phrasing variants such as "session flaps" or "neighbour goes down" receive
    the same BM25 token weight as the canonical phrasing present in runbook
    headings.

    Args:
        query: The raw (or procedure-expanded) query string.

    Returns:
        Query with synonym expansions appended (originals preserved).
    """
    extras: list[str] = []
    for pattern, replacement in _SYNONYM_MAP:
        if pattern.search(query):
            extras.append(replacement)
    if extras:
        expanded = f"{query} {' '.join(extras)}"
        logger.debug("Synonym expansion: %r → %r", query[:60], expanded[:80])
        return expanded
    return query


def _expand_query(query: str) -> str:
    """Return an expanded query string for BM25 retrieval.

    Applies two layers of expansion in sequence:
      1. Procedure-trigger expansion (appends "runbook procedure steps …").
      2. Synonym normalisation (appends variant forms of key terms).

    The dense embedding receives the *original* query unchanged; only BM25
    benefits from the expansion.

    Args:
        query: The raw user query.

    Returns:
        The query with expansion suffixes appended when triggered, otherwise
        the original query unchanged.
    """
    expanded = query
    if _PROCEDURE_TRIGGERS.search(query):
        logger.debug("Query expansion triggered for: %r", query[:80])
        expanded = f"{query} {_PROCEDURE_EXPANSION}"
    return normalize_query(expanded)


# ---------------------------------------------------------------------------
# Doc-type relevance boost
# ---------------------------------------------------------------------------

# Score boosts applied *after* reranking to correct systematic mis-ranking.
# Values are absolute additions to the cross-encoder rerank_score.
#   +0.5  runbook/design_doc chunks for procedural queries
#   −0.5  ticket chunks whose text matches ticket-noise patterns for proc queries
_RUNBOOK_BOOST = 0.5
_TICKET_PENALTY = 0.5


def apply_doc_type_boost(
    chunks: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Apply doc-type relevance boosts to reranked chunks.

    For procedural queries:
      * Runbook and design-doc chunks get a +{_RUNBOOK_BOOST} bonus on
        ``rerank_score`` (or ``rrf_score`` when rerank_score is absent).
      * Ticket chunks whose text contains ticket-boilerplate patterns
        (rollback plan, implementation plan, etc.) get a penalty.

    Re-sorts the list by the adjusted score.

    Args:
        chunks: List of result dicts with ``rerank_score`` or ``rrf_score``.
        query:  The original user query.

    Returns:
        Re-sorted list with ``rerank_score`` adjusted in-place.
    """
    if not _PROCEDURE_TRIGGERS.search(query):
        return chunks  # no boost for non-procedural queries

    for chunk in chunks:
        doc_type = chunk.get("metadata", {}).get("doc_type", "")
        score_key = "rerank_score" if "rerank_score" in chunk else "rrf_score"
        current = chunk.get(score_key, 0.0)
        text = chunk.get("text", "")

        if doc_type == "runbook":
            chunk[score_key] = current + _RUNBOOK_BOOST
        elif doc_type == "ticket" and _TICKET_NOISE_WORDS.search(text):
            chunk[score_key] = current - _TICKET_PENALTY

    # Re-sort by the adjusted score
    score_key_for_sort = "rerank_score" if any("rerank_score" in c for c in chunks) else "rrf_score"
    chunks.sort(key=lambda x: x.get(score_key_for_sort, 0.0), reverse=True)
    return chunks


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def _rrf_score(rank: int, k: int) -> float:
    """Reciprocal Rank Fusion score for a single rank."""
    return 1.0 / (k + rank + 1)


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        dense_results:  List of dicts from vector store query.
                        Each must have key ``"id"``.
        sparse_results: List of dicts from BM25 query.
                        Each must have key ``"id"``.
        k:              RRF constant (defaults to ``settings.rrf_k``).

    Returns:
        List of dicts with keys ``"id"`` and ``"rrf_score"``,
        sorted descending by ``"rrf_score"``.
    """
    rrf_k = k if k is not None else settings.rrf_k
    scores: dict[str, float] = {}

    for rank, item in enumerate(dense_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank, rrf_k)

    for rank, item in enumerate(sparse_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank, rrf_k)

    fused = [{"id": doc_id, "rrf_score": score} for doc_id, score in scores.items()]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def _build_chroma_where(filters: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a user filter dict into a ChromaDB ``where`` clause.

    Supported keys: ``doc_type`` (str or list[str]), ``site_id`` (str).
    Date filtering (``date_from``, ``date_to``) is applied post-retrieval
    because ChromaDB does not support lexicographic range queries on strings
    without an integer representation.
    """
    clauses: list[dict[str, Any]] = []

    if "doc_type" in filters:
        val = filters["doc_type"]
        if isinstance(val, list):
            if len(val) == 1:
                clauses.append({"doc_type": {"$eq": val[0]}})
            elif len(val) > 1:
                clauses.append({"doc_type": {"$in": val}})
        else:
            clauses.append({"doc_type": {"$eq": val}})

    if "site_id" in filters:
        clauses.append({"site_id": {"$eq": filters["site_id"]}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _passes_date_filter(
    metadata: dict[str, Any],
    date_from: str | None,
    date_to: str | None,
) -> bool:
    """Return True if the chunk's change_date is within [date_from, date_to]."""
    if not date_from and not date_to:
        return True
    change_date = metadata.get("change_date", "")
    if not change_date:
        # Chunks without a date (e.g. configs, design docs) always pass
        return True
    if date_from and change_date < date_from:
        return False
    if date_to and change_date > date_to:
        return False
    return True


class HybridRetriever:
    """Combines dense + BM25 retrieval with RRF fusion and metadata filtering.

    Args:
        vector_store: A :class:`~netdocs.retriever.vector_store.VectorStore` instance.
        bm25_index:   A :class:`~netdocs.retriever.bm25_index.BM25Index` instance.
        embedder:     A :class:`~netdocs.embeddings.embedder.BaseEmbedder` instance.
    """

    def __init__(self, vector_store: Any, bm25_index: Any, embedder: Any) -> None:
        self._store = vector_store
        self._bm25 = bm25_index
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-K chunks most relevant to ``query``.

        Args:
            query:       Natural-language query string.
            top_k:       Number of final results to return. Defaults to
                         ``settings.retrieval_top_k``.
            candidate_k: Number of candidates fetched from each source before
                         fusion. Defaults to ``settings.retrieval_candidate_k``.
            filters:     Optional metadata filter dict. Supported keys:
                         ``doc_type``, ``site_id``, ``date_from``, ``date_to``.

        Returns:
            List of result dicts with keys:
            ``id``, ``text``, ``metadata``, ``rrf_score``.
            Sorted descending by ``rrf_score``.
        """
        k_top = top_k or settings.retrieval_top_k
        k_cand = candidate_k or settings.retrieval_candidate_k
        filt = filters or {}

        # 1. Embed the original query (unmodified — dense semantics are good as-is)
        vec = self._embedder.embed([query])[0]

        # 2. Dense retrieval (with optional Chroma where-filter)
        chroma_where = _build_chroma_where(filt)
        dense = self._store.query(
            query_embedding=vec,
            n_results=k_cand,
            where=chroma_where,
        )

        # 3. BM25 retrieval — use an expanded query to boost runbook/procedure terms
        expanded_query = _expand_query(query)
        sparse = self._bm25.query(expanded_query, n_results=k_cand)

        # 4. RRF fusion
        fused = reciprocal_rank_fusion(dense, sparse)

        # 5. Hydrate: attach text + metadata from the vector store result map
        id_to_dense = {d["id"]: d for d in dense}

        # For IDs only in BM25, we need to fetch from the store
        bm25_only_ids = [
            f["id"] for f in fused if f["id"] not in id_to_dense
        ]
        if bm25_only_ids:
            extra = self._store.get_by_ids(bm25_only_ids)
            id_to_dense.update({e["id"]: e for e in extra})

        hydrated: list[dict[str, Any]] = []
        date_from = filt.get("date_from")
        date_to = filt.get("date_to")

        for item in fused:
            doc_id = item["id"]
            base = id_to_dense.get(doc_id)
            if base is None:
                continue
            meta = base.get("metadata", {})
            # Apply doc_type / site_id filter again (BM25 has no pre-filter)
            if "doc_type" in filt:
                allowed = filt["doc_type"]
                if isinstance(allowed, str):
                    allowed = [allowed]
                if meta.get("doc_type") not in allowed:
                    continue
            if "site_id" in filt and meta.get("site_id") != filt["site_id"]:
                continue
            if not _passes_date_filter(meta, date_from, date_to):
                continue
            hydrated.append(
                {
                    "id": doc_id,
                    "text": base.get("text", ""),
                    "metadata": meta,
                    "rrf_score": item["rrf_score"],
                }
            )
            if len(hydrated) >= k_top * 4:  # over-fetch for reranker
                break

        logger.debug(
            "HybridRetriever: query=%r  expanded=%r  dense=%d  sparse=%d  fused=%d  hydrated=%d",
            query[:60],
            expanded_query[:60] if expanded_query != query else "(unchanged)",
            len(dense),
            len(sparse),
            len(fused),
            len(hydrated),
        )
        return hydrated
