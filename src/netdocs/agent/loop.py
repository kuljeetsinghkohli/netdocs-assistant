from __future__ import annotations

"""
NetDocs Agent Loop — explicit tool-calling loop with no heavy framework.

Design decisions
----------------
* **No LangGraph / LangChain** — just a plain Python ``while`` loop.
* **Structured logging** — every tool call is logged at INFO with a JSON
  summary.  The full step trace is available on :attr:`AgentResult.steps`.
* **Max-step guard** — the loop aborts after ``max_steps`` iterations and
  returns whatever partial answer it has.  Default: 8.
* **Human-approval flag** — tools whose :attr:`~netdocs.agent.tools.ToolSpec.requires_approval`
  is ``True`` are gated behind a ``human_approval`` callback when
  ``require_approval=True`` (the default).  In non-interactive / API mode
  the loop caller may pass ``require_approval=False`` or supply an
  auto-approving callback.
* **Tool timeout** — each tool call is wrapped in a ``concurrent.futures``
  executor with the per-tool ``timeout_seconds``.  On timeout a
  ``ToolTimeoutError`` is recorded and the loop continues.
* **LLM decision parsing** — the loop asks the LLM to emit JSON with either
  ``{"action":"tool_call","tool":"...","args":{}}`` or
  ``{"action":"final_answer","answer":"..."}``.  When the LLM is the
  ``extractive`` provider it falls back to a keyword-based heuristic.
* **Loop guard** — repeated identical ``(tool, args)`` pairs count as one
  extra step; the loop breaks if it detects the same call twice.

Usage
-----
::

    from netdocs.agent.loop import run_agent

    result = run_agent("Is the BGP session with BT MPLS on LON-DC01-RTR01 up?")
    print(result.final_answer)
    print(result.steps)
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from netdocs.agent.prompts import (
    DECISION_SYSTEM,
    DECISION_USER,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
    format_history,
    format_tool_descriptions,
    format_tool_results,
)
from netdocs.agent.tools import TOOL_REGISTRY, ToolSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolTimeoutError(Exception):
    """Raised when a tool exceeds its allowed execution time."""


class ToolNotFoundError(Exception):
    """Raised when the LLM requests a tool that is not registered."""


class MaxStepsExceeded(Exception):
    """Raised (internally) when the loop hits its step limit."""


# ---------------------------------------------------------------------------
# Step records — structured log of every agent action
# ---------------------------------------------------------------------------

@dataclass
class ToolCallStep:
    """A successfully executed tool call."""
    type: str = "tool_call"
    tool: str = ""
    args: dict = field(default_factory=dict)
    result: Any = None
    duration_ms: float = 0.0
    approved: bool = True


@dataclass
class ErrorStep:
    """A failed tool call or parse error."""
    type: str = "error"
    tool: str = ""
    args: dict = field(default_factory=dict)
    error: str = ""
    error_kind: str = ""  # "timeout" | "not_found" | "bad_args" | "execution"


@dataclass
class FinalAnswerStep:
    """The loop's terminal action."""
    type: str = "final_answer"
    answer: str = ""


Step = Union[ToolCallStep, ErrorStep, FinalAnswerStep]


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Complete result returned by :func:`run_agent` / :class:`AgentLoop`.

    Attributes:
        final_answer: The agent's final answer text.
        steps:        Ordered list of all steps taken (tool calls, errors,
                      final answer).
        aborted:      True when the loop was cut short (max steps, approval
                      denied, unrecoverable error).
        abort_reason: Human-readable reason when ``aborted=True``.
        degraded:     True when the LLM fell back to the extractive provider.
    """
    final_answer: str = ""
    steps: list[Step] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""
    degraded: bool = False


# ---------------------------------------------------------------------------
# Default approval callback — interactive CLI
# ---------------------------------------------------------------------------

def _cli_approval(tool_name: str, args: dict[str, Any]) -> bool:
    """Ask the user on stdin whether to approve a state-changing tool call."""
    print(
        f"\n[APPROVAL REQUIRED] Tool: {tool_name!r}\n"
        f"  Args: {json.dumps(args, indent=2, default=str)}\n"
        "Approve? [y/N] ",
        end="",
        flush=True,
    )
    try:
        answer = input().strip().lower()
    except (EOFError, OSError):
        answer = "n"
    return answer in {"y", "yes"}


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    """Explicit tool-calling agent loop.

    Args:
        llm_client:      A :class:`~netdocs.llm.client.BaseLLMClient` instance.
        tool_registry:   Dict of tool name → :class:`~netdocs.agent.tools.ToolSpec`.
                         Defaults to :data:`~netdocs.agent.tools.TOOL_REGISTRY`.
        max_steps:       Maximum tool calls before the loop aborts. Default: 8.
        require_approval: When True, tools with ``requires_approval=True`` are
                         gated behind *approval_callback*. Default: True.
        approval_callback: Callable(tool_name, args) → bool.  Defaults to
                          interactive CLI prompt.
    """

    def __init__(
        self,
        llm_client: Any,
        tool_registry: dict[str, ToolSpec] | None = None,
        max_steps: int = 8,
        require_approval: bool = True,
        approval_callback: Callable[[str, dict], bool] | None = None,
    ) -> None:
        self._llm = llm_client
        self._tools = tool_registry if tool_registry is not None else TOOL_REGISTRY
        self._max_steps = max_steps
        self._require_approval = require_approval
        self._approval_cb = approval_callback or _cli_approval

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, question: str) -> AgentResult:
        """Run the agent loop for *question* and return an :class:`AgentResult`.

        The loop alternates between:
          1. Asking the LLM for the next action (tool call or final answer).
          2. Executing the tool (with approval gate + timeout).
          3. Recording the step and continuing.

        The loop terminates on:
          * A ``final_answer`` action from the LLM.
          * ``max_steps`` tool calls (loop guard).
          * Approval denial for a required-approval tool.
        """
        steps: list[Step] = []
        seen_calls: set[tuple[str, str]] = set()  # (tool, args_json) dedup guard
        result = AgentResult()

        tool_desc = format_tool_descriptions(self._tools)

        for step_num in range(self._max_steps + 1):
            if step_num == self._max_steps:
                # Hard limit reached — synthesise from what we have
                logger.warning(
                    "AgentLoop: max_steps=%d reached for question=%r",
                    self._max_steps, question[:80],
                )
                result.aborted = True
                result.abort_reason = f"max_steps ({self._max_steps}) exceeded"
                break

            history_text = format_history([_step_to_dict(s) for s in steps])
            system = DECISION_SYSTEM.format(tool_descriptions=tool_desc)
            user = DECISION_USER.format(question=question, history=history_text)

            raw = self._llm.complete(system=system, user=user)

            # Detect degraded response (extractive fallback after 503 retry)
            from netdocs.llm.client import DEGRADED_PREFIX
            if raw.startswith(DEGRADED_PREFIX):
                result.degraded = True
                raw = raw[len(DEGRADED_PREFIX):]

            # Parse the LLM's JSON response
            action = _parse_llm_action(raw, question, self._tools)

            if action is None:
                # Unparseable — treat as final answer with the raw text
                logger.warning(
                    "AgentLoop: could not parse LLM action from response, "
                    "treating as final answer."
                )
                answer_step = FinalAnswerStep(answer=raw.strip())
                steps.append(answer_step)
                result.final_answer = raw.strip()
                break

            if action["action"] == "final_answer":
                answer_step = FinalAnswerStep(answer=action.get("answer", ""))
                steps.append(answer_step)
                result.final_answer = action.get("answer", "")
                logger.info("AgentLoop: final_answer reached after %d steps.", step_num)
                break

            # --- Tool call ---
            tool_name = action.get("tool", "")
            tool_args = action.get("args", {})

            # Loop guard — deduplicate identical calls
            call_key = (tool_name, json.dumps(tool_args, sort_keys=True, default=str))
            if call_key in seen_calls:
                logger.warning(
                    "AgentLoop: duplicate tool call detected (%s), forcing final_answer.",
                    tool_name,
                )
                result.aborted = True
                result.abort_reason = f"duplicate tool call: {tool_name}"
                break
            seen_calls.add(call_key)

            # Validate tool exists
            spec = self._tools.get(tool_name)
            if spec is None:
                err = ErrorStep(
                    tool=tool_name,
                    args=tool_args,
                    error=f"Tool {tool_name!r} is not registered",
                    error_kind="not_found",
                )
                steps.append(err)
                logger.error("AgentLoop: unknown tool %r", tool_name)
                _log_tool_call(tool_name, tool_args, None, error=err.error)
                continue

            # Approval gate
            if self._require_approval and spec.requires_approval:
                approved = self._approval_cb(tool_name, tool_args)
                if not approved:
                    result.aborted = True
                    result.abort_reason = f"human approval denied for {tool_name!r}"
                    logger.info("AgentLoop: approval denied for %r", tool_name)
                    break

            # Execute tool with timeout
            t0 = time.perf_counter()
            try:
                tool_result = _run_tool_with_timeout(spec, tool_args)
                duration_ms = (time.perf_counter() - t0) * 1000
                call_step = ToolCallStep(
                    tool=tool_name,
                    args=tool_args,
                    result=tool_result,
                    duration_ms=duration_ms,
                    approved=not spec.requires_approval,
                )
                steps.append(call_step)
                _log_tool_call(tool_name, tool_args, tool_result)

            except ToolTimeoutError as exc:
                err = ErrorStep(
                    tool=tool_name,
                    args=tool_args,
                    error=str(exc),
                    error_kind="timeout",
                )
                steps.append(err)
                _log_tool_call(tool_name, tool_args, None, error=str(exc))

            except (TypeError, ValueError, KeyError) as exc:
                err = ErrorStep(
                    tool=tool_name,
                    args=tool_args,
                    error=str(exc),
                    error_kind="bad_args",
                )
                steps.append(err)
                _log_tool_call(tool_name, tool_args, None, error=str(exc))

            except Exception as exc:
                err = ErrorStep(
                    tool=tool_name,
                    args=tool_args,
                    error=str(exc),
                    error_kind="execution",
                )
                steps.append(err)
                _log_tool_call(tool_name, tool_args, None, error=str(exc))

        # If we broke out of the loop without a final_answer, synthesise one
        if not result.final_answer and not result.aborted:
            result.final_answer = self._synthesise(question, steps)
        elif not result.final_answer and result.aborted:
            # Still try to synthesise from whatever we gathered
            synthesised = self._synthesise(question, steps)
            result.final_answer = synthesised or (
                f"Agent aborted: {result.abort_reason}. "
                "Insufficient information to answer the question."
            )

        result.steps = steps
        return result

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesise(self, question: str, steps: list[Step]) -> str:
        """Call the LLM to synthesise tool results into a final answer."""
        tool_results_text = format_tool_results([_step_to_dict(s) for s in steps])
        system = SYNTHESIS_SYSTEM
        user = SYNTHESIS_USER.format(
            question=question,
            tool_results_text=tool_results_text,
        )
        try:
            raw = self._llm.complete(system=system, user=user)
            from netdocs.llm.client import DEGRADED_PREFIX
            if raw.startswith(DEGRADED_PREFIX):
                raw = raw[len(DEGRADED_PREFIX):]
            return raw.strip()
        except Exception as exc:
            logger.error("AgentLoop._synthesise failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    *,
    llm_client: Any | None = None,
    max_steps: int = 8,
    require_approval: bool = False,
) -> AgentResult:
    """Run the agent loop with the default LLM client.

    This is the simplest way to invoke the agent from the CLI or tests.

    Args:
        question:        The question to answer.
        llm_client:      Override the LLM client (defaults to
                         :func:`~netdocs.llm.client.get_llm_client`).
        max_steps:       Maximum tool-call iterations.
        require_approval: Gate state-changing tools behind human approval.
                         Default ``False`` for non-interactive use.

    Returns:
        :class:`AgentResult`
    """
    if llm_client is None:
        from netdocs.llm.client import get_llm_client
        llm_client = get_llm_client()

    loop = AgentLoop(
        llm_client=llm_client,
        max_steps=max_steps,
        require_approval=require_approval,
    )
    return loop.run(question)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_tool_with_timeout(spec: ToolSpec, args: dict[str, Any]) -> Any:
    """Execute *spec.fn(args)* in a thread, raising :exc:`ToolTimeoutError` on timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(spec.fn, args)
        try:
            return future.result(timeout=spec.timeout_seconds)
        except FuturesTimeout:
            raise ToolTimeoutError(
                f"Tool {spec.name!r} timed out after {spec.timeout_seconds}s"
            )


def _parse_llm_action(raw: str, question: str, tools: dict[str, ToolSpec]) -> dict | None:
    """Parse the LLM's raw output into an action dict.

    Tries JSON extraction first.  Falls back to a keyword heuristic when the
    extractive provider returns free-form text.

    Returns ``None`` when parsing fails completely.
    """
    # 1. Try strict JSON
    action = _try_json_parse(raw)
    if action and action.get("action") in {"tool_call", "final_answer"}:
        return action

    # 2. Try to find a JSON object embedded in surrounding text
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        action = _try_json_parse(m.group(0))
        if action and action.get("action") in {"tool_call", "final_answer"}:
            return action

    # 3. Extractive / keyword fallback — pick first matching tool or answer
    q_lower = question.lower()

    # If no specific device is named in the question, prefer list_configs so the
    # agent can discover available devices rather than asking the user.
    _no_device_named = not _guess_device_from_question(question)

    heuristics = [
        # Generic "which routers…" or device-less BGP state questions → discover first
        (["which router", "which routers", "all router", "all routers",
          "all device", "all devices", "have bgp", "bgp peers that are",
          "bgp neighbors that are", "routers have bgp", "routers with bgp"],
         "list_configs",
         lambda q: {}),  # noqa: ARG005
        (["parse config", "bgp config", "prefix-list", "route-map", "asn", "config file"],
         "parse_config",
         lambda q: {"filename": _guess_device_from_question(q)}),
        (["neighbor state", "bgp session", "bgp state", "session up", "session down",
          "peer up", "peer down", "is the bgp"],
         "check_neighbor_state",
         lambda q: {"device": _guess_device_from_question(q)}),
        (["change plan", "rollout plan", "implementation plan", "cr for", "draft change"],
         "draft_change_plan",
         lambda q: {"intent": q, "device": _guess_device_from_question(q)}),
        (["ticket", "crq", "incident", "inc-", "related ticket"],
         "find_related_tickets",
         lambda q: {"query": _guess_device_from_question(q) or q}),
    ]

    for keywords, tool_name, args_fn in heuristics:
        if tool_name not in tools:
            continue
        if any(kw in q_lower for kw in keywords):
            return {"action": "tool_call", "tool": tool_name, "args": args_fn(question)}

    # 4. Treat the raw response as a final answer
    if raw.strip():
        return {"action": "final_answer", "answer": raw.strip()}

    return None


def _try_json_parse(text: str) -> dict | None:
    """Attempt to parse *text* as JSON, returning None on failure."""
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _guess_device_from_question(question: str) -> str:
    """Extract a device hostname from the question using a simple pattern."""
    # Match patterns like LON-DC01-RTR01, NYC-BR03-VE01, AWS-USE1-COR, etc.
    m = re.search(
        r"\b([A-Z]{2,5}-[A-Z]{2,5}\d{0,4}-[A-Z]{2,6}\d{0,4}(?:-\w+)?)\b",
        question,
        re.IGNORECASE,
    )
    return m.group(1).upper() if m else ""


def _step_to_dict(step: Step) -> dict[str, Any]:
    """Convert a step dataclass to a plain dict for prompt formatting."""
    return {k: getattr(step, k) for k in step.__dataclass_fields__}


def _log_tool_call(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    *,
    error: str | None = None,
) -> None:
    """Emit a structured INFO log entry for every tool invocation."""
    entry: dict[str, Any] = {
        "tool": tool_name,
        "args": args,
    }
    if error is not None:
        entry["error"] = error
        logger.info("agent_tool_call %s", json.dumps(entry, default=str))
    else:
        # Summarise the result to avoid flooding logs
        result_summary = _summarise_result(result)
        entry["result_summary"] = result_summary
        logger.info("agent_tool_call %s", json.dumps(entry, default=str))


def _summarise_result(result: Any) -> Any:
    """Return a log-safe summary of a tool result."""
    if isinstance(result, dict):
        # Show top-level keys and any list lengths
        summary: dict[str, Any] = {}
        for k, v in result.items():
            if isinstance(v, list):
                summary[k] = f"[{len(v)} items]"
            elif isinstance(v, str) and len(v) > 120:
                summary[k] = v[:120] + "…"
            else:
                summary[k] = v
        return summary
    if isinstance(result, str) and len(result) > 200:
        return result[:200] + "…"
    return result
