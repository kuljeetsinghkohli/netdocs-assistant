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

class GeminiClient(BaseLLMClient):
    """Google Gemini client using the google-genai SDK.

    Reads ``GEMINI_API_KEY`` from the environment / .env.
    Model defaults to ``gemini-2.0-flash`` (fast, free tier available).

    Args:
        model:       Gemini model name. Defaults to ``settings.llm_model``
                     if it looks like a Gemini model, else ``gemini-2.0-flash``.
        api_key:     API key. Defaults to ``settings.gemini_api_key``.
        temperature: Sampling temperature.
        max_tokens:  Max output tokens.
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

        # Pick a sensible default model for Gemini
        cfg_model = settings.llm_model
        if model:
            self._model = model
        elif cfg_model.startswith("gemini"):
            self._model = cfg_model
        else:
            self._model = "gemini-3.6-flash"

        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens or settings.llm_max_tokens

        self._client = genai.Client(api_key=key)
        logger.info("GeminiClient ready. model=%s", self._model)

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from google.genai import types  # type: ignore

        temp = temperature if temperature is not None else self._temperature
        mtok = max_tokens or self._max_tokens

        response = self._client.models.generate_content(
            model=self._model,
            contents=f"{system}\n\nUser question: {user}",
            config=types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=mtok,
            ),
        )
        return response.text or ""


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
