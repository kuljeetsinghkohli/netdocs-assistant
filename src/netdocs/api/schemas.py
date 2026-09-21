from __future__ import annotations

"""
Pydantic request / response schemas for the NetDocs API.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

class AskFilters(BaseModel):
    """Optional metadata filters for the /ask endpoint."""

    doc_type: Optional[list[str]] = Field(
        default=None,
        description=(
            "Restrict retrieval to these document types. "
            "Allowed values: design_doc, runbook, ticket, config."
        ),
        examples=[["runbook", "design_doc"]],
    )
    site_id: Optional[str] = Field(
        default=None,
        description="Restrict to a specific site, e.g. 'LON-DC01'.",
    )
    date_from: Optional[str] = Field(
        default=None,
        description="ISO date lower bound for ticket change_date (YYYY-MM-DD).",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="ISO date upper bound for ticket change_date (YYYY-MM-DD).",
    )


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer.",
        examples=["What is the BGP hold timer configured for BT MPLS peers?"],
    )
    filters: Optional[AskFilters] = Field(
        default=None,
        description="Optional metadata filters applied before retrieval.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve (default: 5).",
    )


class CitationOut(BaseModel):
    """A single source citation."""

    doc_id: str
    section: str
    source_file: str
    doc_type: str


class AskResponse(BaseModel):
    """Response body for POST /ask."""

    answer: str = Field(description="The generated answer, with inline [doc:...] citations.")
    citations: list[CitationOut] = Field(default_factory=list)
    refused: bool = Field(description="True when the system refused to answer.")
    confidence: float = Field(description="Retrieval confidence score of the top result.")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    vector_store_count: int
