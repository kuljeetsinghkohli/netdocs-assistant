"""
Tests for the Gemini model fallback chain.

All tests are fully offline — no real google-genai SDK calls are made.
The GeminiClient is instantiated with __init__ bypassed and _call_with_model
monkey-patched so we can inject controlled failures.

Covered scenarios (per spec):
  1. Quota (429) on primary → success on next model.
  2. 404 NOT_FOUND on primary → success on next model.
  3. 400 INVALID_ARGUMENT on primary → retry same model without thinking_config
     → success.
  4. All models fail → RuntimeError propagates → RetryingLLMClient degrades to
     extractive (DEGRADED prefix, no exception).
  5. A model marked failed on request N is skipped on request N+1.
  6. 400 INVALID_ARGUMENT, same-model retry also fails → moves to next model.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers: build a bare GeminiClient without touching google-genai
# ---------------------------------------------------------------------------

def _bare_client(primary_model: str = "model-primary") -> "GeminiClient":
    """Return a GeminiClient with __init__ bypassed."""
    from netdocs.llm.client import GeminiClient

    obj = object.__new__(GeminiClient)
    obj._model = primary_model
    obj._temperature = 0.0
    obj._max_tokens = 4096
    # _client is never called in tests because _call_with_model is patched.
    obj._client = None
    return obj


def _fake_response(answer_text: str = "The answer.", finish: str = "STOP") -> object:
    """Return a minimal fake Gemini response object."""

    class _Part:
        def __init__(self, t):
            self.text = t
            self.thought = False

    class _Usage:
        prompt_token_count = 5
        candidates_token_count = 10
        thoughts_token_count = 0
        total_token_count = 15

    part = _Part(answer_text)
    fr = type("FR", (), {"name": finish})()
    content = type("C", (), {"parts": [part]})()
    candidate = type("Candidate", (), {"content": content, "finish_reason": fr})()
    usage = _Usage()

    class _Resp:
        pass

    resp = _Resp()
    resp.candidates = [candidate]
    resp.usage_metadata = usage
    resp.text = answer_text
    return resp


class _FakeException(Exception):
    """Generic exception with an optional status_code attribute."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Fixture: reset failed-model set + singleton between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset():
    from netdocs.llm.client import reset_llm_client
    reset_llm_client()
    yield
    reset_llm_client()


# ---------------------------------------------------------------------------
# Helper: patch fallback list
# ---------------------------------------------------------------------------

def _patch_fallbacks(monkeypatch, models: list[str]) -> None:
    """Override _gemini_fallback_models() to return *models*."""
    import netdocs.llm.client as m
    monkeypatch.setattr(m, "_gemini_fallback_models", lambda: models)


# ===========================================================================
# Scenario 1: Quota (429) on primary → success on second model
# ===========================================================================

class TestQuotaFallback:
    def test_quota_429_falls_to_next_model(self, monkeypatch):
        """Daily quota on primary → next model succeeds → no exception."""
        from netdocs.llm.client import GeminiClient

        client = _bare_client("model-a")
        _patch_fallbacks(monkeypatch, ["model-b"])

        call_log: list[tuple[str, bool]] = []

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            call_log.append((model, use_thinking))
            if model == "model-a":
                raise _FakeException("429 RESOURCE_EXHAUSTED per_day quota", status_code=429)
            return _fake_response("Answer from model-b")

        client._call_with_model = fake_call_with_model  # type: ignore

        result = client.complete("sys", "user")

        assert result == "Answer from model-b"
        # model-a was tried once, model-b once
        assert call_log[0] == ("model-a", True)
        assert call_log[1] == ("model-b", True)

    def test_quota_model_skipped_on_next_call(self, monkeypatch):
        """After quota on model-a, the second call skips model-a entirely."""
        import netdocs.llm.client as m

        client = _bare_client("model-a")
        _patch_fallbacks(monkeypatch, ["model-b"])

        call_log: list[str] = []

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            call_log.append(model)
            if model == "model-a":
                raise _FakeException("429 daily quota", status_code=429)
            return _fake_response("ok")

        client._call_with_model = fake_call_with_model  # type: ignore

        # First call — model-a fails, model-b serves
        client.complete("sys", "user1")
        assert "model-a" in m._failed_gemini_models

        call_log.clear()
        # Second call — model-a must be skipped
        client.complete("sys", "user2")
        assert "model-a" not in call_log
        assert "model-b" in call_log


# ===========================================================================
# Scenario 2: 404 NOT_FOUND on primary → success on next model
# ===========================================================================

class TestNotFoundFallback:
    def test_404_falls_to_next_model(self, monkeypatch):
        """404 NOT_FOUND (retired model) → next model succeeds."""
        client = _bare_client("model-a")
        _patch_fallbacks(monkeypatch, ["model-b"])

        call_log: list[str] = []

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            call_log.append(model)
            if model == "model-a":
                raise _FakeException("404 NOT_FOUND model not found")
            return _fake_response("Answer from model-b")

        client._call_with_model = fake_call_with_model  # type: ignore

        result = client.complete("sys", "user")
        assert result == "Answer from model-b"
        assert call_log == ["model-a", "model-b"]

    def test_404_model_skipped_on_next_call(self, monkeypatch):
        import netdocs.llm.client as m

        client = _bare_client("model-retired")
        _patch_fallbacks(monkeypatch, ["model-good"])

        call_log: list[str] = []

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            call_log.append(model)
            if model == "model-retired":
                raise _FakeException("404 not_found")
            return _fake_response("ok")

        client._call_with_model = fake_call_with_model  # type: ignore

        client.complete("sys", "u1")
        assert "model-retired" in m._failed_gemini_models

        call_log.clear()
        client.complete("sys", "u2")
        assert "model-retired" not in call_log


# ===========================================================================
# Scenario 3: 400 INVALID_ARGUMENT → retry same model without thinking_config
# ===========================================================================

class TestInvalidArgumentRetry:
    def test_400_retries_without_thinking_then_succeeds(self, monkeypatch):
        """400 on model-a (thinking) → retry model-a (no thinking) → success."""
        client = _bare_client("model-a")
        _patch_fallbacks(monkeypatch, ["model-b"])

        call_log: list[tuple[str, bool]] = []

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            call_log.append((model, use_thinking))
            if model == "model-a" and use_thinking:
                raise _FakeException("400 INVALID_ARGUMENT thinking_config not supported")
            # model-a without thinking_config succeeds
            return _fake_response("Answer without thinking")

        client._call_with_model = fake_call_with_model  # type: ignore

        result = client.complete("sys", "user")
        assert result == "Answer without thinking"
        # First call: model-a with thinking; second: model-a without thinking
        assert call_log[0] == ("model-a", True)
        assert call_log[1] == ("model-a", False)
        # model-b should NOT be called
        assert all(m == "model-a" for m, _ in call_log)

    def test_400_both_retries_fail_moves_to_next_model(self, monkeypatch):
        """400 on model-a (thinking) → 400 on model-a (no thinking) → model-b."""
        client = _bare_client("model-a")
        _patch_fallbacks(monkeypatch, ["model-b"])

        call_log: list[tuple[str, bool]] = []

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            call_log.append((model, use_thinking))
            if model == "model-a":
                raise _FakeException("400 INVALID_ARGUMENT")
            return _fake_response("Answer from model-b")

        client._call_with_model = fake_call_with_model  # type: ignore

        result = client.complete("sys", "user")
        assert result == "Answer from model-b"
        # model-a tried twice (thinking + no thinking), then model-b once
        assert call_log[0] == ("model-a", True)
        assert call_log[1] == ("model-a", False)
        assert call_log[2] == ("model-b", True)


# ===========================================================================
# Scenario 4: All models fail → degraded extractive path, no 500
# ===========================================================================

class TestAllModelsFail:
    def test_all_fail_gives_degraded_not_exception(self, monkeypatch):
        """When every model fails, RetryingLLMClient catches the error and
        returns a DEGRADED-prefixed extractive answer — no exception escapes."""
        import time
        from netdocs.llm.client import RetryingLLMClient, DEGRADED_PREFIX

        client = _bare_client("model-a")
        _patch_fallbacks(monkeypatch, ["model-b"])

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            raise _FakeException(f"429 quota exhausted", status_code=429)

        client._call_with_model = fake_call_with_model  # type: ignore

        # RetryingLLMClient wraps the GeminiClient; _sleep is stubbed out.
        retrying = RetryingLLMClient(client, max_attempts=1, _sleep=lambda s: None)

        # The system prompt must contain a Source passage for ExtractiveClient.
        system = (
            "--- [Source: DD-001__000 | design_doc | Overview] ---\n"
            "BGP hold timer is 90 seconds.\n"
        )
        result = retrying.complete(system, "What is the BGP hold timer?")

        assert result.startswith(DEGRADED_PREFIX), (
            f"Expected DEGRADED prefix, got: {result[:60]!r}"
        )
        # The extractive fallback should include the passage text
        assert "90 seconds" in result or "Overview" in result


# ===========================================================================
# Scenario 5: Failed model is skipped on next call (cross-request persistence)
# ===========================================================================

class TestFailedModelSkipped:
    def test_failed_model_skipped_on_subsequent_request(self, monkeypatch):
        """Model marked failed in request 1 is skipped in request 2."""
        import netdocs.llm.client as m

        client = _bare_client("model-x")
        _patch_fallbacks(monkeypatch, ["model-y"])

        call_log: list[str] = []

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            call_log.append(model)
            if model == "model-x":
                raise _FakeException("404 not found")
            return _fake_response("ok")

        client._call_with_model = fake_call_with_model  # type: ignore

        # Request 1 — model-x fails, model-y serves
        r1 = client.complete("sys", "q1")
        assert r1 == "ok"
        assert "model-x" in m._failed_gemini_models

        call_log.clear()
        # Request 2 — model-x must not appear in call_log
        r2 = client.complete("sys", "q2")
        assert r2 == "ok"
        assert "model-x" not in call_log
        assert call_log == ["model-y"]


# ===========================================================================
# AllModelsFailedError — sentinel exception raised by GeminiClient
# ===========================================================================

class TestAllModelsFailedError:
    def test_all_models_fail_raises_all_models_failed_error(self, monkeypatch):
        """GeminiClient raises AllModelsFailedError (not bare RuntimeError) when
        the entire fallback chain is exhausted."""
        from netdocs.llm.client import AllModelsFailedError

        client = _bare_client("model-a")
        _patch_fallbacks(monkeypatch, ["model-b"])

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            raise _FakeException("429 quota exhausted", status_code=429)

        client._call_with_model = fake_call_with_model  # type: ignore

        with pytest.raises(AllModelsFailedError):
            client.complete("sys", "user")

    def test_all_models_failed_error_is_runtime_error_subclass(self):
        """AllModelsFailedError is a subclass of RuntimeError for backward compat."""
        from netdocs.llm.client import AllModelsFailedError

        err = AllModelsFailedError("boom")
        assert isinstance(err, RuntimeError)

    def test_retrying_client_catches_all_models_failed_degrades(self, monkeypatch):
        """RetryingLLMClient catches AllModelsFailedError and returns DEGRADED answer."""
        from netdocs.llm.client import RetryingLLMClient, AllModelsFailedError, DEGRADED_PREFIX

        client = _bare_client("model-a")
        _patch_fallbacks(monkeypatch, ["model-b"])

        def fake_call_with_model(model, prompt, temp, mtok, *, use_thinking=True):
            raise _FakeException("429 quota exhausted", status_code=429)

        client._call_with_model = fake_call_with_model  # type: ignore

        retrying = RetryingLLMClient(client, max_attempts=1, _sleep=lambda s: None)

        system = (
            "--- [Source: DD-001__000 | design_doc | Overview] ---\n"
            "BGP hold timer is 90 seconds.\n"
        )
        result = retrying.complete(system, "What is the BGP hold timer?")

        assert result.startswith(DEGRADED_PREFIX), (
            f"Expected DEGRADED prefix, got: {result[:60]!r}"
        )
        assert "90 seconds" in result or "Overview" in result

    def test_agent_loop_degrades_on_all_models_failed(self, monkeypatch):
        """AgentLoop returns degraded=True (no exception) when GeminiClient
        raises AllModelsFailedError on every LLM call."""
        from netdocs.llm.client import AllModelsFailedError, BaseLLMClient
        from netdocs.agent.loop import AgentLoop
        from netdocs.agent.tools import ToolSpec

        def _mock_neighbor(args: dict) -> dict:
            return {"device": args.get("device", "TEST"), "neighbors": []}

        registry = {
            "check_neighbor_state": ToolSpec(
                name="check_neighbor_state",
                description="check BGP neighbor state",
                fn=_mock_neighbor,
                timeout_seconds=5.0,
            ),
        }

        class _ChainExhaustedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise AllModelsFailedError("All Gemini models failed")

        loop = AgentLoop(
            _ChainExhaustedLLM(),
            tool_registry=registry,
            max_steps=5,
            require_approval=False,
        )
        result = loop.run("Is the BGP session on LON-DC01-RTR01 up?")

        assert result.degraded is True, "Expected degraded=True when chain is exhausted"
        assert result.final_answer.strip() != "", "Expected non-empty answer in degraded mode"

    def test_agent_loop_chain_exhausted_tool_calls_in_trace(self, monkeypatch):
        """When GeminiClient raises AllModelsFailedError, tool calls executed via
        the heuristic fallback are still present in the step trace."""
        from netdocs.llm.client import AllModelsFailedError, BaseLLMClient
        from netdocs.agent.loop import AgentLoop
        from netdocs.agent.tools import ToolSpec

        def _mock_neighbor(args: dict) -> dict:
            return {"device": args.get("device", "TEST"), "neighbors": []}

        registry = {
            "check_neighbor_state": ToolSpec(
                name="check_neighbor_state",
                description="check BGP neighbor state",
                fn=_mock_neighbor,
                timeout_seconds=5.0,
            ),
        }

        class _ChainExhaustedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise AllModelsFailedError("All Gemini models failed")

        loop = AgentLoop(
            _ChainExhaustedLLM(),
            tool_registry=registry,
            max_steps=5,
            require_approval=False,
        )
        result = loop.run("Is the BGP session on LON-DC01-RTR01 up?")

        tool_steps = [s for s in result.steps if getattr(s, "type", "") == "tool_call"]
        assert tool_steps, "Expected at least one tool_call step in the trace"
        assert any(s.tool == "check_neighbor_state" for s in tool_steps)

    def test_ask_path_degrades_on_all_models_failed(self):
        """generate_answer returns degraded=True (no exception) when the LLM
        raises AllModelsFailedError."""
        from netdocs.llm.client import AllModelsFailedError, BaseLLMClient
        from netdocs.llm.generator import generate_answer

        class _ChainExhaustedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise AllModelsFailedError("All Gemini models failed")

        chunks = [
            {
                "id": "DD-001__000",
                "text": "BGP hold timer is 90 seconds.",
                "metadata": {
                    "source_file": "design_doc.md",
                    "doc_type": "design_doc",
                    "section_heading": "BGP Timers",
                },
                "rerank_score": 0.9,
            }
        ]
        result = generate_answer(
            "What is the BGP hold timer?",
            chunks,
            _ChainExhaustedLLM(),
            confidence_threshold=0.0,
        )

        assert result.degraded is True, "Expected degraded=True when chain is exhausted"
        assert result.answer.strip() != "", "Expected non-empty answer in degraded mode"
        # No exception should have propagated

    def test_ask_path_chain_exhausted_200_no_500(self):
        """generate_answer must not raise when AllModelsFailedError occurs —
        the caller (API route) must be able to return 200."""
        from netdocs.llm.client import AllModelsFailedError, BaseLLMClient
        from netdocs.llm.generator import generate_answer

        class _ChainExhaustedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise AllModelsFailedError("All Gemini models failed")

        chunks = [
            {
                "id": "RB-001__000",
                "text": "Clear BGP session using: clear ip bgp <neighbor>.",
                "metadata": {
                    "source_file": "runbook.md",
                    "doc_type": "runbook",
                    "section_heading": "BGP Reset Procedure",
                },
                "rerank_score": 0.85,
            }
        ]
        # Must complete without raising
        result = generate_answer(
            "How do I reset a BGP session?",
            chunks,
            _ChainExhaustedLLM(),
            confidence_threshold=0.0,
        )
        assert result is not None
        assert result.degraded is True
