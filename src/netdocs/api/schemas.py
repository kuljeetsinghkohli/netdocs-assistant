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

    answer: str = Field(description="The generated answer, with rendered inline citations.")
    citations: list[CitationOut] = Field(default_factory=list)
    refused: bool = Field(description="True when the system refused to answer.")
    refusal_reason: str = Field(
        default="",
        description=(
            "Reason for refusal when refused=true. "
            "One of: 'low_confidence', 'llm_declined', 'empty_generation', or '' when not refused."
        ),
    )
    confidence: float = Field(description="Retrieval confidence score of the top result.")


# ---------------------------------------------------------------------------
# /agent
# ---------------------------------------------------------------------------

class AgentStepOut(BaseModel):
    """A single step in the agent's execution trace."""

    type: str = Field(description="Step type: 'tool_call', 'error', or 'final_answer'.")
    tool: str = Field(default="", description="Tool name (for tool_call and error steps).")
    args: dict = Field(default_factory=dict)
    result: Any = Field(default=None, description="Tool result (for tool_call steps).")
    error: str = Field(default="", description="Error message (for error steps).")
    error_kind: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    answer: str = Field(default="", description="Final answer text (for final_answer steps).")


class AgentRequest(BaseModel):
    """Request body for POST /agent."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The question for the agent to answer.",
        examples=["Is the BGP session with BT MPLS on LON-DC01-RTR01 up?"],
    )
    max_steps: Optional[int] = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum tool-call iterations (default: 8).",
    )
    require_approval: bool = Field(
        default=False,
        description=(
            "When True, state-changing tools (draft_change_plan) are NOT auto-approved "
            "and will be skipped in the API context."
        ),
    )


class AgentResponse(BaseModel):
    """Response body for POST /agent."""

    answer: str = Field(description="The agent's final answer.")
    steps: list[AgentStepOut] = Field(default_factory=list)
    aborted: bool = Field(default=False)
    abort_reason: str = Field(default="")
    degraded: bool = Field(default=False, description="True when LLM fell back to extractive.")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    vector_store_count: int
