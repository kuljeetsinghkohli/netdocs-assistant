from __future__ import annotations

"""
Provider-agnostic LLM client.

Supported providers (configured via ``NETDOCS_LLM_PROVIDER`` in ``.env``):

    openai      — OpenAI Chat Completions API.
                  Requires ``OPENAI_API_KEY``.
    ollama      — Any OpenAI-compatible local server (Ollama, LM Studio, …).
                  Requires ``NETDOCS_LLM_BASE_URL`` pointing at the server.
    gemini      — Google Gemini via the google-genai SDK.
                  Requires ``GEMINI_API_KEY``.
    extractive  — No LLM or API key needed.  Returns the top reranked chunks
                  verbatim with [doc:id:section] citations as a plain answer.
    fake        — Deterministic stub that echoes the system prompt.  Used in
                  tests so no network or API key is needed.

All providers expose the same ``BaseLLMClient.complete()`` interface.

Provider selection order (when ``NETDOCS_LLM_PROVIDER`` is not explicitly set
or is left as the legacy default "openai"):
  1. If ``GEMINI_API_KEY`` is set  → gemini
  2. Otherwise                     → extractive

Retry behaviour
---------------
The :class:`RetryingLLMClient` wrapper adds automatic retry with exponential
backoff for transient HTTP errors (429 Rate-Limit, 503 Service-Unavailable).
It wraps any ``BaseLLMClient`` and, after **3 failed attempts**, falls back to
:class:`ExtractiveClient` and marks the response as "degraded" by prepending
``[DEGRADED] `` to the returned text.  No exception is raised to the caller.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from netdocs.config import settings

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant's text.

        Args:
            system:      System prompt.
            user:        User message.
            temperature: Override the configured temperature.
            max_tokens:  Override the configured max_tokens.

        Returns:
            The assistant reply as a plain string.
        """


# ---------------------------------------------------------------------------
# OpenAI provider (also used for Ollama via base_url override)
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions client.

    Works with the real OpenAI API and with any OpenAI-compatible server
    (Ollama, LM Studio) by setting ``base_url``.

    Args:
        model:       Model name. Defaults to ``settings.llm_model``.
        api_key:     API key. Defaults to ``settings.openai_api_key``.
        base_url:    Override base URL. Defaults to the official OpenAI URL.
                     Set to ``settings.llm_base_url`` for Ollama.
        temperature: Sampling temperature. Defaults to ``settings.llm_temperature``.
        max_tokens:  Max completion tokens. Defaults to ``settings.llm_max_tokens``.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # lazy import
        except ImportError:
            raise ImportError(
                "The 'openai' package is not installed. "
                "Run: pip install 'netdocs[openai]'  or  pip install openai"
            )

        self._model = model or settings.llm_model
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens or settings.llm_max_tokens

        key = api_key or settings.openai_api_key or None
        if not key:
            raise ValueError(
                "OpenAI provider requires OPENAI_API_KEY. "
                "Set it in .env or switch provider: NETDOCS_LLM_PROVIDER=gemini"
            )
        self._client = OpenAI(api_key=key, base_url=base_url or None)
        logger.info(
            "OpenAIClient ready. model=%s  base_url=%s",
            self._model,
            base_url or "(openai default)",
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self._temperature
        mtok = max_tokens or self._max_tokens
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=mtok,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

# Tokens reserved so that thinking budget does not crowd out the answer.
_GEMINI_THINKING_BUDGET = 0          # 0 = disabled / minimal
_GEMINI_DEFAULT_MAX_OUTPUT = 4096    # raised from 1024 to avoid truncation
_GEMINI_RETRY_MAX_OUTPUT = 8192      # retry limit when first call is truncated

# ---------------------------------------------------------------------------
# Gemini fallback-chain helpers
# ---------------------------------------------------------------------------

#: Process-lifetime set of Gemini model IDs that have permanently failed.
#: Populated by GeminiClient.complete(); avoids re-trying a dead model.
_failed_gemini_models: set[str] = set()


def _gemini_fallback_models() -> list[str]:
    """Return the ordered fallback model list from settings (deduplicated)."""
    raw = settings.gemini_model_fallbacks
    return [m.strip() for m in raw.split(",") if m.strip()]


def _is_quota_error(exc: Exception) -> bool:
    """Return True for 429 / RESOURCE_EXHAUSTED daily-quota errors.

    Only matches the specific phrases Gemini uses for quota exhaustion.
    Avoids false positives on generic error messages that mention "quota".
    """
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "quota_exceeded" in msg
        or "per_day" in msg
        or "daily limit" in msg
        or "daily quota" in msg
    )


def _is_model_not_found_error(exc: Exception) -> bool:
    """Return True for 404 NOT_FOUND (model retired / unavailable)."""
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 404:
        return True
    return "404" in msg or "not_found" in msg or "not found" in msg


def _is_invalid_argument_error(exc: Exception) -> bool:
    """Return True for 400 INVALID_ARGUMENT (e.g. model rejects thinking_config)."""
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 400:
        return True
    return "400" in msg or "invalid_argument" in msg


class GeminiClient(BaseLLMClient):
    """Google Gemini client using the google-genai SDK.

    Reads ``GEMINI_API_KEY`` from the environment / .env.
    Model is resolved in priority order:
      1. ``model`` constructor argument
      2. ``GEMINI_MODEL`` env-var / ``settings.gemini_model``
         (default: ``gemini-2.0-flash``)
      3. ``NETDOCS_LLM_MODEL`` if it starts with "gemini"

    Thinking models (e.g. gemini-2.5-*) are handled safely:
    * ``thinking_config`` is set to ``budget_tokens=0`` to minimise / disable
      internal reasoning, keeping output tokens for the actual answer.
    * Response parts are filtered to include only non-thought text, so leaked
      reasoning fragments never appear in the returned answer.
    * ``finish_reason`` and token usage are logged at DEBUG level.
    * If ``finish_reason`` is MAX_TOKENS **or** the answer is empty, the call
      is retried once with a doubled ``max_output_tokens``.  If still empty, a
      :class:`RuntimeError` is raised with a clear message.

    Args:
        model:       Gemini model name. See resolution order above.
        api_key:     API key. Defaults to ``settings.gemini_api_key``.
        temperature: Sampling temperature.
        max_tokens:  Max output tokens (default: 4096).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        try:
            import google.genai as genai  # type: ignore
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is not installed. "
                "Run: pip install google-genai"
            )

        key = api_key or settings.gemini_api_key or ""
        if not key:
            raise ValueError(
                "Gemini provider requires GEMINI_API_KEY. "
                "Set it in .env, or use NETDOCS_LLM_PROVIDER=extractive for keyless operation."
            )

        # Model resolution: explicit arg > GEMINI_MODEL > llm_model if gemini-*
        if model:
            self._model = model
        elif settings.gemini_model:
            self._model = settings.gemini_model
        elif settings.llm_model.startswith("gemini"):
            self._model = settings.llm_model
        else:
            self._model = "gemini-2.0-flash"

        self._temperature = temperature if temperature is not None else settings.llm_temperature
        # Use the larger default; caller's llm_max_tokens is often 1024 which
        # is too small when a thinking model eats tokens internally.
        configured = max_tokens or settings.llm_max_tokens
        self._max_tokens = max(configured, _GEMINI_DEFAULT_MAX_OUTPUT)

        self._client = genai.Client(api_key=key)
        logger.info("GeminiClient ready. model=%s  max_output_tokens=%d",
                    self._model, self._max_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(self, prompt: str, temp: float, mtok: int) -> Any:
        """Single generate_content call using self._model."""
        return self._call_with_model(self._model, prompt, temp, mtok, use_thinking=True)

    def _call_with_model(
        self,
        model: str,
        prompt: str,
        temp: float,
        mtok: int,
        *,
        use_thinking: bool = True,
    ) -> Any:
        """Single generate_content call for *model*; returns the raw response object.

        Args:
            model:         Gemini model ID to call.
            prompt:        Full prompt string.
            temp:          Sampling temperature.
            mtok:          Max output tokens.
            use_thinking:  When False, thinking_config is omitted entirely.
                           Used on the INVALID_ARGUMENT retry for the same model.
        """
        from google.genai import types  # type: ignore

        # Build ThinkingConfig defensively — field names changed between SDK
        # versions.  v1.x uses ``thinking_budget``; earlier releases used
        # ``budget_tokens``.  We try the current name first and fall back.
        thinking_cfg: Any = None
        if use_thinking:
            try:
                # SDK >= 1.x: thinking_budget=0 disables/minimises reasoning.
                # include_thoughts=False ensures thought parts are not returned.
                thinking_cfg = types.ThinkingConfig(
                    thinking_budget=_GEMINI_THINKING_BUDGET,
                    include_thoughts=False,
                )
            except Exception:
                # Older SDK or model that doesn't support thinking config — skip.
                thinking_cfg = None

        config_kwargs: dict[str, Any] = dict(
            temperature=temp,
            max_output_tokens=mtok,
        )
        if thinking_cfg is not None:
            config_kwargs["thinking_config"] = thinking_cfg

        return self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Return only non-thought text parts concatenated.

        The google-genai SDK may return multiple ``Part`` objects per
        ``Candidate``.  Thought parts carry ``thought=True`` (or appear as
        ``thought_signature`` parts) and must be excluded from the answer.
        """
        try:
            parts = response.candidates[0].content.parts
        except (AttributeError, IndexError):
            # Fallback: use the SDK's .text property if parts are unavailable
            return response.text or ""

        text_parts: list[str] = []
        for part in parts:
            # Skip thought / reasoning parts
            if getattr(part, "thought", False):
                continue
            # Skip parts that have no text (e.g. inline_data, thought_signature)
            t = getattr(part, "text", None)
            if t:
                text_parts.append(t)

        return "".join(text_parts)

    @staticmethod
    def _finish_reason(response: Any) -> str:
        """Extract finish_reason string from the first candidate (best-effort)."""
        try:
            reason = response.candidates[0].finish_reason
            # SDK may return an enum or a string
            return str(reason.name) if hasattr(reason, "name") else str(reason)
        except (AttributeError, IndexError):
            return "UNKNOWN"

    @staticmethod
    def _token_usage(response: Any) -> dict[str, int]:
        """Extract token usage metadata (best-effort)."""
        try:
            um = response.usage_metadata
            return {
                "prompt": getattr(um, "prompt_token_count", 0) or 0,
                "candidates": getattr(um, "candidates_token_count", 0) or 0,
                "thoughts": getattr(um, "thoughts_token_count", 0) or 0,
                "total": getattr(um, "total_token_count", 0) or 0,
            }
        except AttributeError:
            return {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _complete_with_model(
        self,
        model: str,
        prompt: str,
        temp: float,
        mtok: int,
        *,
        use_thinking: bool = True,
    ) -> str:
        """Run one generate_content round-trip for *model* and return answer text.

        Handles the MAX_TOKENS / empty-answer retry internally.
        Raises the underlying SDK exception unchanged so the caller can classify it.
        """
        response = self._call_with_model(model, prompt, temp, mtok, use_thinking=use_thinking)

        finish = self._finish_reason(response)
        usage = self._token_usage(response)
        logger.debug(
            "Gemini response. model=%s  finish_reason=%s  tokens=%s",
            model, finish, usage,
        )

        answer = self._extract_text(response)

        # Retry once if truncated or empty
        if finish == "MAX_TOKENS" or not answer.strip():
            retry_mtok = max(mtok * 2, _GEMINI_RETRY_MAX_OUTPUT)
            logger.warning(
                "Gemini answer truncated or empty (finish_reason=%s, tokens=%s). "
                "Retrying with max_output_tokens=%d.",
                finish, usage, retry_mtok,
            )
            response = self._call_with_model(model, prompt, temp, retry_mtok, use_thinking=use_thinking)
            finish = self._finish_reason(response)
            usage = self._token_usage(response)
            logger.debug(
                "Gemini retry response. finish_reason=%s  tokens=%s",
                finish, usage,
            )
            answer = self._extract_text(response)

        if not answer.strip():
            raise RuntimeError(
                f"Gemini returned an empty answer after retry. "
                f"model={model!r}  finish_reason={finish}  tokens={usage}. "
                "Check your GEMINI_MODEL setting and API quota."
            )

        return answer

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self._temperature
        mtok = max_tokens if max_tokens is not None else self._max_tokens
        prompt = f"{system}\n\nUser question: {user}"

        # Build the full ordered candidate list: primary first, then fallbacks.
        primary = self._model
        fallbacks = _gemini_fallback_models()
        # Deduplicate: if primary already appears in fallbacks, skip repeats.
        seen: set[str] = set()
        candidates: list[str] = []
        for m in [primary] + fallbacks:
            if m not in seen:
                seen.add(m)
                candidates.append(m)

        last_exc: Exception | None = None
        for model in candidates:
            if model in _failed_gemini_models:
                logger.debug("Gemini: skipping previously-failed model %r.", model)
                continue
            try:
                answer = self._complete_with_model(model, prompt, temp, mtok)
                logger.info("Gemini: served by model=%r.", model)
                return answer
            except Exception as exc:
                if _is_invalid_argument_error(exc):
                    # (c) 400 INVALID_ARGUMENT — retry the SAME model without
                    # thinking_config before giving up on it.
                    logger.warning(
                        "Gemini model %r rejected thinking_config (400 INVALID_ARGUMENT: %s). "
                        "Retrying once without thinking_config.",
                        model, exc,
                    )
                    try:
                        answer = self._complete_with_model(
                            model, prompt, temp, mtok, use_thinking=False
                        )
                        logger.info("Gemini: served by model=%r (no thinking_config).", model)
                        return answer
                    except Exception as exc2:
                        logger.warning(
                            "Gemini model %r also failed without thinking_config (%s). "
                            "Marking as failed and moving to next model.",
                            model, exc2,
                        )
                        _failed_gemini_models.add(model)
                        last_exc = exc2
                elif _is_quota_error(exc):
                    # (a) daily quota 429 — skip permanently for this process.
                    logger.warning(
                        "Gemini model %r hit daily quota (429: %s). "
                        "Skipping for the rest of this process.",
                        model, exc,
                    )
                    _failed_gemini_models.add(model)
                    last_exc = exc
                elif _is_model_not_found_error(exc):
                    # (b) 404 NOT_FOUND — model retired / unavailable.
                    logger.warning(
                        "Gemini model %r returned 404 NOT_FOUND (%s). "
                        "Skipping for the rest of this process.",
                        model, exc,
                    )
                    _failed_gemini_models.add(model)
                    last_exc = exc
                else:
                    # Any other error is not a fallback trigger — re-raise.
                    raise

        # All candidates failed — propagate to the RetryingLLMClient which
        # will fall back to extractive (degraded) mode.  Never return a 500.
        raise RuntimeError(
            f"All Gemini models in the fallback chain failed. "
            f"Last error: {last_exc}. "
            "Falling back to extractive provider."
        ) from last_exc


# ---------------------------------------------------------------------------
# Extractive provider (no LLM, no API key)
# ---------------------------------------------------------------------------

class ExtractiveClient(BaseLLMClient):
    """Zero-dependency extractive 'LLM' that needs no API key.

    Instead of calling an LLM it scans the user message for injected context
    passages (``[Source: <id> | ...]`` markers produced by the generator's
    ``_build_context_block``), assembles the top passages as the answer, and
    appends ``[doc:<id>:<section>]`` citations so the citation parser works
    normally downstream.

    This is useful for:
    * Offline / air-gapped environments.
    * Quick smoke-tests without spending API quota.
    * Situations where the retrieved context is self-explanatory.
    """

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        import re

        # The system prompt contains context passages in the format:
        #   --- [Source: <id> | <doc_type> | <section>] ---\n<text>
        passage_re = re.compile(
            r"--- \[Source: ([^|]+)\|([^|]*)\|([^\]]*)\] ---\n(.*?)(?=\n--- \[Source:|\Z)",
            re.DOTALL,
        )

        passages = passage_re.findall(system)
        if not passages:
            return (
                "I cannot find sufficiently relevant information in the available "
                "documentation to answer this question confidently."
            )

        lines: list[str] = ["**Relevant passages from the documentation:**\n"]
        for doc_id, doc_type, section, text in passages:
            doc_id = doc_id.strip()
            section = section.strip()
            excerpt = text.strip()
            # Truncate very long passages
            words = excerpt.split()
            if len(words) > 120:
                excerpt = " ".join(words[:120]) + "…"
            lines.append(f"• {excerpt}  [doc:{doc_id}:{section}]")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fake provider (tests)
# ---------------------------------------------------------------------------

class FakeLLMClient(BaseLLMClient):
    """Deterministic stub for offline testing.

    Returns a canned answer that includes every ``[doc:...]`` citation
    token found in the user message so citation-extraction tests pass.

    Args:
        canned_answer: Fixed string to return. When ``None`` the client
                       echoes citation tokens discovered in the user message.
    """

    def __init__(self, canned_answer: str | None = None) -> None:
        self._canned = canned_answer

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self._canned is not None:
            return self._canned

        # Extract doc_id tokens from the injected context so the generator
        # can parse citations even from this fake client.
        import re
        citations = re.findall(r"\[Source: ([^\]]+)\]", user)
        if citations:
            cite_str = "  ".join(f"[doc:{c}]" for c in citations[:2])
            return f"Fake answer based on context. {cite_str}"
        return "Fake answer: no relevant documentation found."


# ---------------------------------------------------------------------------
# Retrying wrapper — 429 / 503 with exponential backoff + extractive fallback
# ---------------------------------------------------------------------------

#: HTTP status codes that trigger a retry.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 503})
#: Maximum number of attempts (first call + 2 retries = 3 total).
_RETRY_MAX_ATTEMPTS: int = 3
#: Base delay in seconds for the first retry; doubles each attempt.
_RETRY_BASE_DELAY: float = 1.0

#: Sentinel prefix added when the extractive fallback is used.
DEGRADED_PREFIX = "[DEGRADED] "


#: Substrings that indicate a per-day (non-transient) quota exhaustion.
#: These must NOT be retried — fall back immediately.
_DAILY_QUOTA_PHRASES: tuple[str, ...] = (
    "resource_exhausted",
    "daily",
    "per_day",
    "quota_exceeded",
    "daily limit",
    "daily quota",
)


def _is_daily_quota_error(exc: Exception) -> bool:
    """Return True if *exc* signals a non-transient daily quota exhaustion.

    Gemini returns gRPC status ``RESOURCE_EXHAUSTED`` with quota IDs that
    contain "per_day" or "daily".  We detect this from the string
    representation so the retry loop can skip backoff and degrade immediately.
    """
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _DAILY_QUOTA_PHRASES)


def _is_retryable_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a 429 or 503 HTTP error."""
    msg = str(exc).lower()
    # Handle openai, httpx, and generic HTTP client exceptions by inspecting
    # the status_code attribute or the message text.
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in _RETRYABLE_STATUS_CODES:
        return True
    # Fallback: keyword scan of the string representation
    return "429" in msg or "503" in msg or "rate limit" in msg or "service unavailable" in msg


class RetryingLLMClient(BaseLLMClient):
    """Wraps any :class:`BaseLLMClient` with retry + graceful degradation.

    On a 429 or 503 error the call is retried up to ``max_attempts`` times
    with exponential backoff (``base_delay * 2**attempt`` seconds).  After all
    retries are exhausted the call is delegated to :class:`ExtractiveClient`
    and the returned text is prefixed with :data:`DEGRADED_PREFIX` so callers
    can detect the degraded state.

    Args:
        inner:        The real LLM client to wrap.
        max_attempts: Total attempts including the first call (default: 3).
        base_delay:   Seconds before the first retry (doubles each time).
        _sleep:       Override ``time.sleep`` — used in tests to avoid delays.
    """

    def __init__(
        self,
        inner: BaseLLMClient,
        max_attempts: int = _RETRY_MAX_ATTEMPTS,
        base_delay: float = _RETRY_BASE_DELAY,
        *,
        _sleep=time.sleep,
    ) -> None:
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._sleep = _sleep
        self._fallback = ExtractiveClient()

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return self._inner.complete(
                    system, user, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:
                if not _is_retryable_error(exc):
                    raise
                # Daily quota errors must not be retried — degrade immediately.
                if _is_daily_quota_error(exc):
                    logger.error(
                        "RetryingLLMClient: daily quota exhausted (%s). "
                        "Falling back to extractive provider immediately (degraded mode).",
                        exc,
                    )
                    result = self._fallback.complete(
                        system, user, temperature=temperature, max_tokens=max_tokens
                    )
                    return DEGRADED_PREFIX + result
                last_exc = exc
                # Only sleep if there is a subsequent attempt to make
                if attempt + 1 < self._max_attempts:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        "RetryingLLMClient: attempt %d/%d failed (%s). Retrying in %.1fs.",
                        attempt + 1,
                        self._max_attempts,
                        exc,
                        delay,
                    )
                    self._sleep(delay)
                else:
                    logger.warning(
                        "RetryingLLMClient: attempt %d/%d failed (%s). No more retries.",
                        attempt + 1,
                        self._max_attempts,
                        exc,
                    )

        # All retries exhausted — fall back to extractive
        logger.error(
            "RetryingLLMClient: all %d attempts failed (%s). "
            "Falling back to extractive provider (degraded mode).",
            self._max_attempts,
            last_exc,
        )
        result = self._fallback.complete(system, user, temperature=temperature, max_tokens=max_tokens)
        return DEGRADED_PREFIX + result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_llm_instance: BaseLLMClient | None = None


def _resolve_provider() -> str:
    """Determine the effective provider.

    If the configured provider is "openai" (the legacy default) but openai
    is not actually available or no key is set, fall back automatically:
      • GEMINI_API_KEY set  → "gemini"
      • otherwise           → "extractive"
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        # Auto-downgrade when the openai package or key is missing
        try:
            import openai as _openai_pkg  # noqa: F401
            has_openai_pkg = True
        except ImportError:
            has_openai_pkg = False

        has_openai_key = bool(settings.openai_api_key)

        if not has_openai_pkg or not has_openai_key:
            gemini_key = settings.gemini_api_key
            if gemini_key:
                logger.info(
                    "openai package/key unavailable; auto-switching to 'gemini' provider."
                )
                return "gemini"
            logger.info(
                "openai package/key unavailable and no GEMINI_API_KEY; "
                "auto-switching to 'extractive' provider."
            )
            return "extractive"

    return provider


def get_llm_client() -> BaseLLMClient:
    """Return the configured LLM client singleton.

    Provider is controlled by ``settings.llm_provider`` (or auto-resolved):
        - ``"openai"``     → :class:`OpenAIClient` with official base URL.
        - ``"ollama"``     → :class:`OpenAIClient` with ``settings.llm_base_url``.
        - ``"gemini"``     → :class:`GeminiClient`.
        - ``"extractive"`` → :class:`ExtractiveClient` (no key needed).
        - ``"fake"``       → :class:`FakeLLMClient`.

    Returns:
        A :class:`BaseLLMClient` instance.

    Raises:
        SystemExit: Never — missing packages / keys produce a friendly message.
    """
    global _llm_instance
    if _llm_instance is None:
        provider = _resolve_provider()
        try:
            if provider == "openai":
                _llm_instance = OpenAIClient()
            elif provider == "ollama":
                _llm_instance = OpenAIClient(base_url=settings.llm_base_url, api_key="ollama")
            elif provider == "gemini":
                _llm_instance = GeminiClient()
            elif provider == "extractive":
                _llm_instance = ExtractiveClient()
            elif provider == "fake":
                _llm_instance = FakeLLMClient()
            else:
                raise ValueError(
                    f"Unknown LLM_PROVIDER: '{provider}'. "
                    "Choose 'openai', 'ollama', 'gemini', 'extractive', or 'fake'."
                )
        except (ImportError, ValueError) as exc:
            # Surface a friendly one-liner instead of a traceback
            import sys
            print(f"[netdocs] LLM provider error: {exc}", file=sys.stderr)
            # Fall back to extractive so the pipeline still works
            logger.warning("Falling back to extractive provider: %s", exc)
            _llm_instance = ExtractiveClient()
    return _llm_instance


def reset_llm_client() -> None:
    """Clear the singleton and the failed-model set — useful in tests."""
    global _llm_instance
    _llm_instance = None
    _failed_gemini_models.clear()
