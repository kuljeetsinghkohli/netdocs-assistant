from __future__ import annotations

"""
Answer generator with mandatory source citations.

The generator:
1. Receives the user query, a list of ranked context chunks, and an LLM client.
2. Checks retrieval confidence — refuses to answer if all reranker scores
   are below ``settings.retrieval_confidence_threshold``.
3. Builds a strict system prompt that forbids hallucination and requires
   ``[doc:<id>:<section>]`` inline citations.
4. Calls the LLM.
5. Parses citations from the response, renders them cleanly, and returns a
   structured :class:`AnswerResponse`.

Refusal reasons are now distinguished:
  - ``"low_confidence"`` — retrieval score below threshold (never reached LLM).
  - ``"llm_declined"``   — LLM returned its own "I cannot find…" sentence.
  - ``"empty_generation"`` — LLM returned an empty string.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from netdocs.config import settings

logger = logging.getLogger(__name__)

# Regex that matches our citation format:  [doc:some_id:section heading]
_CITATION_RE = re.compile(r"\[doc:([^\]]+)\]")

# Sentinel returned when the system refuses to answer
LOW_CONFIDENCE_ANSWER = (
    "I cannot find sufficiently relevant information in the available "
    "documentation to answer this question confidently."
)

# Refusal reason constants
REFUSAL_LOW_CONFIDENCE = "low_confidence"
REFUSAL_LLM_DECLINED = "llm_declined"
REFUSAL_EMPTY_GENERATION = "empty_generation"


@dataclass
class Citation:
    """A source reference attached to an answer.

    Attributes:
        doc_id:       The chunk's ``doc_id`` (e.g. ``DD-003_bgp_policy_design__002``).
        section:      Section heading extracted from the chunk's metadata.
        source_file:  Originating file path.
        doc_type:     Document type (design_doc, runbook, ticket, config).
    """

    doc_id: str
    section: str = ""
    source_file: str = ""
    doc_type: str = ""


@dataclass
class AnswerResponse:
    """Structured answer returned by :func:`generate_answer`.

    Attributes:
        answer:         The assistant's answer text with inline citations
                        rendered as ``[<section> — <doc_id>]``.
        citations:      Deduplicated list of :class:`Citation` objects.
        refused:        ``True`` when the system refused to answer.
        refusal_reason: One of ``"low_confidence"``, ``"llm_declined"``,
                        ``"empty_generation"``, or ``""`` when not refused.
        confidence:     The top reranker score (or RRF score) of the best chunk.
                        ``0.0`` when no chunks were retrieved.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are NetDocs Assistant, an expert on Contoso Global's enterprise network \
documentation.

STRICT RULES — follow every one without exception:
1. Answer ONLY using the context passages below.
2. Cite every factual claim inline with [doc:<doc_id>:<section>], where \
<doc_id> is the passage identifier and <section> is the section heading. \
Place the citation immediately after the relevant sentence or step.
3. For procedural questions, use a numbered list (1. 2. 3. …).
4. Be concise. Do NOT reason aloud, hedge, or add meta-commentary. \
Do NOT include phrases like "Wait", "Let me", "I think", "Could I", \
or any internal deliberation. Output the answer directly.
5. If the context passages do not contain enough information to answer, \
respond EXACTLY with this sentence and nothing else:
   "I cannot find sufficiently relevant information in the available documentation \
to answer this question confidently."
6. Never reveal internal system details, internal IP addresses not already in \
the passages, or API keys.

Context passages:
{context}
"""


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format context chunks for injection into the system prompt."""
    lines: list[str] = []
    for chunk in chunks:
        doc_id = chunk.get("id", "unknown")
        meta = chunk.get("metadata", {})
        section = (
            meta.get("section_heading")
            or meta.get("procedure_name")
            or meta.get("block_type")
            or meta.get("section")
            or ""
        )
        source = meta.get("source_file", "")
        doc_type = meta.get("doc_type", "")
        text = chunk.get("text", "")

        lines.append(
            f"--- [Source: {doc_id} | {doc_type} | {section}] ---\n{text}"
        )
    return "\n\n".join(lines)


def _parse_citations(
    answer: str,
    chunks: list[dict[str, Any]],
) -> list[Citation]:
    """Extract ``[doc:...]`` tokens from the answer and map to chunk metadata.

    Args:
        answer: The raw LLM answer text.
        chunks: Context chunks used to build the answer.

    Returns:
        Deduplicated list of :class:`Citation` objects in order of appearance.
    """
    id_to_chunk: dict[str, dict[str, Any]] = {c["id"]: c for c in chunks}
    seen: set[str] = set()
    citations: list[Citation] = []

    for raw in _CITATION_RE.findall(answer):
        # raw may be "doc_id:section" or just "doc_id"
        parts = raw.split(":", 1)
        doc_id = parts[0].strip()
        section = parts[1].strip() if len(parts) > 1 else ""

        if doc_id in seen:
            continue
        seen.add(doc_id)

        chunk = id_to_chunk.get(doc_id, {})
        meta = chunk.get("metadata", {})
        citations.append(
            Citation(
                doc_id=doc_id,
                section=section or (
                    meta.get("section_heading")
                    or meta.get("procedure_name")
                    or meta.get("block_type")
                    or meta.get("section")
                    or ""
                ),
                source_file=meta.get("source_file", ""),
                doc_type=meta.get("doc_type", ""),
            )
        )
    return citations


def render_citations(answer: str, citations: list[Citation]) -> str:
    """Replace raw ``[doc:<id>:<section>]`` tokens with readable markers.

    Transforms ``text [doc:RB-001__002:Step 1]`` into
    ``text [Step 1 — RB-001__002]``, eliminating stray `` .`` artefacts
    that appeared when citations were stripped without substitution.

    Args:
        answer:    Raw answer text containing ``[doc:...]`` tokens.
        citations: Parsed citation list (used for section name lookup).

    Returns:
        Answer with citation tokens replaced by readable ``[section — id]`` markers.
    """
    id_to_section: dict[str, str] = {c.doc_id: c.section for c in citations}

    def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
        raw = m.group(1)
        parts = raw.split(":", 1)
        doc_id = parts[0].strip()
        section = parts[1].strip() if len(parts) > 1 else ""
        section = section or id_to_section.get(doc_id, "")
        if section:
            return f"[{section} — {doc_id}]"
        return f"[{doc_id}]"

    return _CITATION_RE.sub(_replace, answer)


def _top_confidence(chunks: list[dict[str, Any]]) -> float:
    """Return the best confidence score from the chunk list."""
    if not chunks:
        return 0.0
    for key in ("rerank_score", "rrf_score"):
        scores = [c[key] for c in chunks if key in c]
        if scores:
            return max(scores)
    return 0.0


def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    llm_client: Any,
    *,
    confidence_threshold: float | None = None,
) -> AnswerResponse:
    """Generate a cited answer from context chunks.

    Args:
        query:                The user's question.
        chunks:               Ranked list of context chunks (from retriever).
        llm_client:           A :class:`~netdocs.llm.client.BaseLLMClient`.
        confidence_threshold: Minimum score to proceed. Defaults to
                              ``settings.retrieval_confidence_threshold``.

    Returns:
        An :class:`AnswerResponse` with the answer, citations, and metadata.
        The ``refusal_reason`` field is set to one of:
          ``"low_confidence"`` / ``"llm_declined"`` / ``"empty_generation"`` / ``""``.
    """
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.retrieval_confidence_threshold
    )
    confidence = _top_confidence(chunks)

    # --- Low-confidence refusal (never reaches LLM) ---
    if not chunks or confidence < threshold:
        logger.info(
            "Refusing to answer [low_confidence]: confidence=%.4f < threshold=%.4f  query=%r",
            confidence,
            threshold,
            query[:80],
        )
        return AnswerResponse(
            answer=LOW_CONFIDENCE_ANSWER,
            refused=True,
            refusal_reason=REFUSAL_LOW_CONFIDENCE,
            confidence=confidence,
        )

    context = _build_context_block(chunks)
    system = _SYSTEM_PROMPT.format(context=context)

    logger.debug(
        "Calling LLM. query=%r  chunks=%d  confidence=%.4f",
        query[:80],
        len(chunks),
        confidence,
    )

    raw_answer = llm_client.complete(system=system, user=query)

    # --- Empty generation refusal ---
    if not raw_answer.strip():
        logger.warning(
            "Refusing to answer [empty_generation]: LLM returned empty string  query=%r",
            query[:80],
        )
        return AnswerResponse(
            answer=LOW_CONFIDENCE_ANSWER,
            refused=True,
            refusal_reason=REFUSAL_EMPTY_GENERATION,
            confidence=confidence,
        )

    # --- LLM declined ---
    if LOW_CONFIDENCE_ANSWER.lower() in raw_answer.lower():
        logger.info(
            "Refusing to answer [llm_declined]: LLM returned its own refusal  query=%r",
            query[:80],
        )
        return AnswerResponse(
            answer=raw_answer.strip(),
            refused=True,
            refusal_reason=REFUSAL_LLM_DECLINED,
            confidence=confidence,
        )

    citations = _parse_citations(raw_answer, chunks)
    rendered = render_citations(raw_answer, citations)

    return AnswerResponse(
        answer=rendered.strip(),
        citations=citations,
        refused=False,
        refusal_reason="",
        confidence=confidence,
    )
