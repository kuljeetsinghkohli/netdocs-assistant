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
"""

import logging
import os
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
        """Single generate_content call; returns the raw response object."""
        from google.genai import types  # type: ignore

        # Build ThinkingConfig defensively — field names changed between SDK
        # versions.  v1.x uses ``thinking_budget``; earlier releases used
        # ``budget_tokens``.  We try the current name first and fall back.
        thinking_cfg: Any = None
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
            model=self._model,
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

        response = self._call(prompt, temp, mtok)

        finish = self._finish_reason(response)
        usage = self._token_usage(response)
        logger.debug(
            "Gemini response. model=%s  finish_reason=%s  tokens=%s",
            self._model, finish, usage,
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
            response = self._call(prompt, temp, retry_mtok)
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
                f"model={self._model!r}  finish_reason={finish}  tokens={usage}. "
                "Check your GEMINI_MODEL setting and API quota."
            )

        return answer


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
    """Clear the singleton — useful in tests to swap providers between cases."""
    global _llm_instance
    _llm_instance = None
