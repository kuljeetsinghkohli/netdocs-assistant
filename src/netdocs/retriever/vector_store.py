from __future__ import annotations

"""
ChromaDB vector store wrapper.

Handles:
- Persistent collection creation / retrieval.
- Idempotent upsert of chunks (re-ingest = update, not duplicate).
- Simple similarity query used during retrieval.
"""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from netdocs.config import settings
from netdocs.ingestion.models import Chunk

logger = logging.getLogger(__name__)

# Metadata values must be str/int/float/bool in ChromaDB.
# We store all extra fields as-is but ensure none are None.
_CHROMA_NONE_SENTINEL = ""


def _sanitise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Replace None values with empty string (ChromaDB rejects None)."""
    return {k: (v if v is not None else _CHROMA_NONE_SENTINEL) for k, v in meta.items()}


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection.

    Args:
        collection_name: ChromaDB collection name. Defaults to ``settings.chroma_collection``.
        persist_dir:     Path to ChromaDB persistence directory.
                         Defaults to ``settings.chroma_path``.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | None = None,
    ) -> None:
        col_name = collection_name or settings.chroma_collection
        path = str(persist_dir or settings.chroma_path)

        logger.info("Initialising ChromaDB at %s  collection=%s", path, col_name)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready. Current count: %d",
            col_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks and their embeddings into the collection.

        Uses the ``chunk.doc_id`` as the ChromaDB document ID.  If a document
        with the same ID already exists it is overwritten (idempotent).

        Args:
            chunks:     List of :class:`~netdocs.ingestion.models.Chunk` objects.
            embeddings: Pre-computed embedding vectors, one per chunk.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length."
            )

        ids = [c.doc_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            _sanitise_metadata(
                {
                    "source_file": c.source_file,
                    "doc_type": c.doc_type,
                    **c.metadata,
                }
            )
            for c in chunks
        ]

        # ChromaDB upsert in batches of 500 to avoid request size limits
        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            logger.debug("Upserted batch %d:%d", start, end)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a nearest-neighbour similarity query.

        Args:
            query_embedding: Query vector.
            n_results:       Number of results to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            List of dicts with keys: ``id``, ``text``, ``metadata``, ``distance``.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output: list[dict[str, Any]] = []
        for doc_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return output

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def get_all_texts(self) -> list[str]:
        """Return all document texts (used for BM25 index construction).

        Warning: loads entire collection into memory.  Use only on startup.
        """
        result = self._collection.get(include=["documents"])
        return result["documents"] or []

    def get_all_ids(self) -> list[str]:
        """Return all document IDs."""
        result = self._collection.get(include=[])
        return result["ids"] or []

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch documents by their IDs.

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of dicts with keys ``id``, ``text``, ``metadata``.
        """
        if not ids:
            return []
        result = self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        output: list[dict[str, Any]] = []
        for doc_id, text, meta in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        ):
            output.append({"id": doc_id, "text": text, "metadata": meta})
        return output
