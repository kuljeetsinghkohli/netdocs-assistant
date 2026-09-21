from __future__ import annotations

"""
BM25 index over the ChromaDB corpus.

Builds a rank_bm25 (Okapi BM25) index from the texts stored in the vector
store.  The index is serialised to disk so it survives restarts without
rebuilding from scratch on every startup.

Usage::

    from netdocs.retriever.bm25_index import BM25Index
    from netdocs.retriever.vector_store import VectorStore

    store = VectorStore()
    idx = BM25Index.load_or_build(store)
    results = idx.query("BGP peer flapping", n_results=20)
    # [{"id": "...", "score": 3.45, "rank": 0}, ...]
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

from netdocs.config import settings

logger = logging.getLogger(__name__)

_TOKENISE_RE = re.compile(r"[a-zA-Z0-9]+")

# Disk cache paths (sibling to the ChromaDB directory)
_DEFAULT_CACHE_DIR = Path("chroma_db")
_BM25_PICKLE = "bm25_index.pkl"
_BM25_IDS = "bm25_ids.json"


def _tokenise(text: str) -> list[str]:
    """Lowercase word-tokeniser; keeps alphanumeric tokens only."""
    return _TOKENISE_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 index over a fixed corpus of documents.

    Args:
        ids:    Ordered list of document IDs (parallel to ``corpus``).
        corpus: List of raw text strings, one per document.
        k1:     BM25 k1 term-frequency saturation parameter.
        b:      BM25 b length-normalisation parameter.
    """

    def __init__(
        self,
        ids: list[str],
        corpus: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi  # lazy import
        except ImportError as exc:
            raise ImportError(
                "rank_bm25 is required for BM25 search. "
                "Install it with: pip install rank-bm25"
            ) from exc

        if len(ids) != len(corpus):
            raise ValueError("ids and corpus must have the same length.")

        self._ids = ids
        tokenised = [_tokenise(t) for t in corpus]
        self._bm25 = BM25Okapi(tokenised, k1=k1, b=b)
        logger.info("BM25 index built: %d documents", len(ids))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query: str, n_results: int = 20) -> list[dict[str, Any]]:
        """Score all documents against ``query`` and return top-N.

        Args:
            query:     Query string.
            n_results: Number of results to return.

        Returns:
            List of dicts with keys ``id``, ``score``, ``rank`` (0-based),
            sorted descending by score.  Entries with score 0 are excluded.
        """
        tokens = _tokenise(query)
        if not tokens:
            return []

        scores: list[float] = self._bm25.get_scores(tokens).tolist()
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(ranked[:n_results]):
            if score <= 0.0:
                break
            results.append({"id": self._ids[idx], "score": score, "rank": rank})
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, cache_dir: Path | None = None) -> None:
        """Serialise the index and ID list to disk.

        Args:
            cache_dir: Directory to write cache files. Defaults to ``chroma_db/``.
        """
        out = cache_dir or _DEFAULT_CACHE_DIR
        out.mkdir(parents=True, exist_ok=True)
        with open(out / _BM25_PICKLE, "wb") as f:
            pickle.dump(self._bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out / _BM25_IDS, "w") as f:
            json.dump(self._ids, f)
        logger.debug("BM25 index saved to %s", out)

    @classmethod
    def load(cls, cache_dir: Path | None = None) -> "BM25Index":
        """Load a previously saved index from disk.

        Args:
            cache_dir: Directory containing the cache files.

        Returns:
            A :class:`BM25Index` instance restored from disk.

        Raises:
            FileNotFoundError: If the cache files do not exist.
        """
        src = cache_dir or _DEFAULT_CACHE_DIR
        pkl_path = src / _BM25_PICKLE
        ids_path = src / _BM25_IDS
        if not pkl_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"BM25 cache not found at {src}")

        with open(pkl_path, "rb") as f:
            bm25_obj = pickle.load(f)
        with open(ids_path) as f:
            ids = json.load(f)

        obj = cls.__new__(cls)
        obj._ids = ids
        obj._bm25 = bm25_obj
        logger.info("BM25 index loaded from %s (%d docs)", src, len(ids))
        return obj

    @classmethod
    def load_or_build(
        cls,
        vector_store: Any,
        cache_dir: Path | None = None,
        force_rebuild: bool = False,
    ) -> "BM25Index":
        """Return a cached index or build one from the vector store.

        Args:
            vector_store: A :class:`~netdocs.retriever.vector_store.VectorStore`
                          instance.  Must expose ``get_all_texts()`` and
                          ``get_all_ids()``.
            cache_dir:    Cache directory. Defaults to ``chroma_db/``.
            force_rebuild: Skip the cache and always rebuild.

        Returns:
            A ready :class:`BM25Index`.
        """
        src = cache_dir or _DEFAULT_CACHE_DIR
        if not force_rebuild:
            try:
                return cls.load(src)
            except FileNotFoundError:
                pass

        logger.info("Building BM25 index from vector store…")
        ids = vector_store.get_all_ids()
        texts = vector_store.get_all_texts()
        idx = cls(
            ids=ids,
            corpus=texts,
            k1=settings.bm25_k1,
            b=settings.bm25_b,
        )
        idx.save(src)
        return idx
