"""
NetDocs UI — HTTP API client.

All communication with the FastAPI service goes through this module.
No internal netdocs modules are imported here; everything is over HTTP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed response containers
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    doc_id: str
    section: str
    source_file: str
    doc_type: str
    number: int = 0


@dataclass
class AskResult:
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str
    confidence: float


@dataclass
class AgentStep:
    type: str
    tool: str = ""
    args: dict = field(default_factory=dict)
    result: Any = None
    error: str = ""
    error_kind: str = ""
    duration_ms: float = 0.0
    answer: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    aborted: bool
    abort_reason: str
    degraded: bool


@dataclass
class HealthResult:
    reachable: bool
    status: str = ""
    vector_store_count: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class NetDocsClient:
    """Thin synchronous HTTP client for the NetDocs FastAPI service."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------

    def health(self) -> HealthResult:
        """Check API reachability and return vector store stats."""
        try:
            r = httpx.get(f"{self._base}/health", timeout=self._timeout)
            r.raise_for_status()
            data = r.json()
            return HealthResult(
                reachable=True,
                status=data.get("status", ""),
                vector_store_count=data.get("vector_store_count", 0),
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("Health check HTTP error: %s", exc)
            return HealthResult(reachable=False, error=str(exc))
        except Exception as exc:  # network error, timeout, etc.
            logger.warning("Health check failed: %s", exc)
            return HealthResult(reachable=False, error=str(exc))

    # ------------------------------------------------------------------
    # POST /ask
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        doc_types: Optional[list[str]] = None,
        site_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> AskResult:
        """POST /ask and return a structured result."""
        filters: dict[str, Any] = {}
        if doc_types:
            filters["doc_type"] = doc_types
        if site_id:
            filters["site_id"] = site_id
        if date_from:
            filters["date_from"] = date_from
        if date_to:
            filters["date_to"] = date_to

        payload: dict[str, Any] = {"question": question}
        if filters:
            payload["filters"] = filters
        if top_k is not None:
            payload["top_k"] = top_k

        r = httpx.post(
            f"{self._base}/ask",
            json=payload,
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()

        citations = [
            Citation(
                doc_id=c["doc_id"],
                section=c.get("section", ""),
                source_file=c.get("source_file", ""),
                doc_type=c.get("doc_type", ""),
                number=c.get("number", 0),
            )
            for c in data.get("citations", [])
        ]

        return AskResult(
            answer=data["answer"],
            citations=citations,
            refused=data.get("refused", False),
            refusal_reason=data.get("refusal_reason", ""),
            confidence=float(data.get("confidence", 0.0)),
        )

    # ------------------------------------------------------------------
    # POST /agent
    # ------------------------------------------------------------------

    def agent(
        self,
        question: str,
        *,
        max_steps: int = 8,
        require_approval: bool = False,
    ) -> AgentResult:
        """POST /agent and return a structured result."""
        payload: dict[str, Any] = {
            "question": question,
            "max_steps": max_steps,
            "require_approval": require_approval,
        }

        r = httpx.post(
            f"{self._base}/agent",
            json=payload,
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()

        steps = [
            AgentStep(
                type=s.get("type", ""),
                tool=s.get("tool", ""),
                args=s.get("args", {}),
                result=s.get("result"),
                error=s.get("error", ""),
                error_kind=s.get("error_kind", ""),
                duration_ms=float(s.get("duration_ms", 0.0)),
                answer=s.get("answer", ""),
            )
            for s in data.get("steps", [])
        ]

        return AgentResult(
            answer=data["answer"],
            steps=steps,
            aborted=data.get("aborted", False),
            abort_reason=data.get("abort_reason", ""),
            degraded=data.get("degraded", False),
        )
