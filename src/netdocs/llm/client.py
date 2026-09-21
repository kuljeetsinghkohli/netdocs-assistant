from __future__ import annotations

"""
Provider-agnostic LLM client.

Supported providers (configured via ``NETDOCS_LLM_PROVIDER`` in ``.env``):

    openai   — OpenAI Chat Completions API (default).
               Requires ``OPENAI_API_KEY``.
    ollama   — Any OpenAI-compatible local server (Ollama, LM Studio, …).
               Requires ``NETDOCS_LLM_BASE_URL`` pointing at the server.
    fake     — Deterministic stub that echoes the system prompt.  Used in
               tests so no network or API key is needed.

All providers expose the same ``BaseLLMClient.complete()`` interface.
"""

import logging
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
        from openai import OpenAI  # lazy import

        self._model = model or settings.llm_model
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens or settings.llm_max_tokens

        key = api_key or settings.openai_api_key or None
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


def get_llm_client() -> BaseLLMClient:
    """Return the configured LLM client singleton.

    Provider is controlled by ``settings.llm_provider``:
        - ``"openai"``  → :class:`OpenAIClient` with official base URL.
        - ``"ollama"``  → :class:`OpenAIClient` with ``settings.llm_base_url``.
        - ``"fake"``    → :class:`FakeLLMClient`.

    Returns:
        A :class:`BaseLLMClient` instance.
    """
    global _llm_instance
    if _llm_instance is None:
        provider = settings.llm_provider.lower()
        if provider == "openai":
            _llm_instance = OpenAIClient()
        elif provider == "ollama":
            _llm_instance = OpenAIClient(base_url=settings.llm_base_url, api_key="ollama")
        elif provider == "fake":
            _llm_instance = FakeLLMClient()
        else:
            raise ValueError(
                f"Unknown llm_provider: '{provider}'. "
                "Choose 'openai', 'ollama', or 'fake'."
            )
    return _llm_instance


def reset_llm_client() -> None:
    """Clear the singleton — useful in tests to swap providers between cases."""
    global _llm_instance
    _llm_instance = None
