from __future__ import annotations

"""
System and decision prompts for the NetDocs agent loop.

These are deliberately concise — the loop uses an explicit tool-calling
protocol, not free-form ReAct, so the prompts only need to steer the LLM
to emit well-formed JSON tool calls or a final answer.
"""

# ---------------------------------------------------------------------------
# Decision prompt — ask LLM to choose next action
# ---------------------------------------------------------------------------

DECISION_SYSTEM = """\
You are NetDocs Agent, an expert assistant for Contoso Global network operations.
You have access to the following tools:

{tool_descriptions}

STRICT OUTPUT RULES — pick exactly one of these two formats:

A) Call a tool:
   {{
     "action": "tool_call",
     "tool": "<tool_name>",
     "args": {{ ... }}
   }}

B) Give a final answer (when you have enough information):
   {{
     "action": "final_answer",
     "answer": "<your complete answer here>"
   }}

Do NOT output anything outside the JSON object.
Do NOT call a tool you have already called with the same arguments.
If you cannot answer and no tool will help, output a final_answer explaining why.

DEVICE DISCOVERY RULE:
If the user's question refers to "all routers", "which routers", "all devices",
or does NOT name a specific device or filename, you MUST call list_configs first
to discover the available devices before calling parse_config or check_neighbor_state.
Never ask the user for a filename or device name that you can discover with list_configs.
"""

DECISION_USER = """\
Question: {question}

Conversation history:
{history}

What is the next action?
"""

# ---------------------------------------------------------------------------
# Extractive fallback decision prompt
# (used when provider is "extractive" — no real LLM reasoning)
# ---------------------------------------------------------------------------

EXTRACTIVE_DECISION_TEMPLATE = """\
Based on the question, determine whether a tool call is needed.
Question keywords that suggest tools:
  - which routers / all devices / all routers / generic (no device named) → list_configs
  - config / neighbor / ASN / route-map / prefix-list → parse_config
  - neighbor state / session / bgp state / up/down → check_neighbor_state
  - change plan / CR / rollout / implement → draft_change_plan
  - ticket / CRQ / incident / INC → find_related_tickets

Question: {question}
"""

# ---------------------------------------------------------------------------
# Final synthesis prompt — combine tool outputs into a readable answer
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """\
You are NetDocs Agent. Synthesise the tool results below into a clear,
concise answer for the user. Cite sources where relevant.

RULES:
1. Use ONLY the information from the tool results.
2. Be direct and concrete. No meta-commentary.
3. If the data is insufficient, say so clearly.
"""

SYNTHESIS_USER = """\
Original question: {question}

Tool results:
{tool_results_text}

Write the final answer.
"""


def format_tool_descriptions(tool_registry: dict) -> str:
    """Render tool registry entries as a numbered list for the decision prompt."""
    lines = []
    for i, (name, spec) in enumerate(tool_registry.items(), 1):
        approval_note = " [REQUIRES HUMAN APPROVAL]" if spec.requires_approval else ""
        lines.append(f"{i}. {name}{approval_note}: {spec.description}")
    return "\n".join(lines)


def format_history(steps: list[dict]) -> str:
    """Render the agent's step history for injection into the decision prompt."""
    if not steps:
        return "(none)"
    parts: list[str] = []
    for step in steps:
        kind = step.get("type", "?")
        if kind == "tool_call":
            parts.append(
                f"Tool call: {step['tool']}({step.get('args', {})})\n"
                f"Result: {str(step.get('result', ''))[:300]}"
            )
        elif kind == "final_answer":
            parts.append(f"Final answer: {step.get('answer', '')[:200]}")
        elif kind == "error":
            parts.append(f"Error in {step.get('tool', '?')}: {step.get('error', '')}")
    return "\n---\n".join(parts)


def format_tool_results(steps: list[dict]) -> str:
    """Render all tool call results for the synthesis prompt."""
    parts: list[str] = []
    for step in steps:
        if step.get("type") == "tool_call":
            parts.append(
                f"[{step['tool']}]\n{json_safe(step.get('result', {}))}"
            )
    return "\n\n".join(parts) if parts else "(no tool results)"


def json_safe(obj) -> str:  # type: ignore[no-untyped-def]
    """Convert any object to a JSON string, falling back to repr."""
    import json
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return repr(obj)
