from __future__ import annotations

"""
Embedder — abstraction over OpenAI and sentence-transformers backends.

The active backend is selected by ``settings.embed_backend``:
    "sentence-transformers"  →  local CPU/GPU model (default, no API key needed)
    "openai"                 →  OpenAI text-embedding-3-small (requires OPENAI_API_KEY)

Usage::

    from netdocs.embeddings.embedder import get_embedder

    embedder = get_embedder()
    vectors = embedder.embed(["hello world", "BGP peer down"])
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from netdocs.config import settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedder interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return a list of float vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder backed by a local sentence-transformers model.

    Args:
        model_name: HuggingFace model name or local path.
                    Defaults to ``settings.embed_model``.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        name = model_name or settings.embed_model
        logger.info("Loading sentence-transformers model: %s", name)
        self._model = SentenceTransformer(name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Model loaded. Embedding dimension: %d", self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the local sentence-transformers model.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors.
        """
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine similarity → dot product
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self._dim


class OpenAIEmbedder(BaseEmbedder):
    """Embedder backed by OpenAI embeddings API.

    Requires ``OPENAI_API_KEY`` in the environment.

    Args:
        model: OpenAI embedding model name. Defaults to ``settings.openai_embed_model``.
    """

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # lazy import

        self._model = model or settings.openai_embed_model
        self._client = OpenAI()
        self._dim = 1536 if "3-small" in self._model else 3072
        logger.info("OpenAI embedder ready. Model: %s  Dim: %d", self._model, self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the OpenAI Embeddings API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors.
        """
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dim


# --- Factory ---

_embedder_instance: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """Return the configured embedder singleton.

    Creates the instance on first call; subsequent calls return the same object.
    Backend is controlled by ``settings.embed_backend``.
    """
    global _embedder_instance
    if _embedder_instance is None:
        backend = settings.embed_backend.lower()
        if backend == "openai":
            _embedder_instance = OpenAIEmbedder()
        elif backend == "sentence-transformers":
            _embedder_instance = SentenceTransformerEmbedder()
        else:
            raise ValueError(
                f"Unknown embed_backend: '{backend}'. "
                "Choose 'openai' or 'sentence-transformers'."
            )
    return _embedder_instance
