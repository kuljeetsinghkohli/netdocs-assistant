from __future__ import annotations

"""
Document loader — discovers files in ``data/raw`` and dispatches
each file to the correct parser based on its sub-directory.

Directory-to-doctype mapping (from ARCHITECTURE.md §6):
    configs/      → config
    design_docs/  → design_doc
    runbooks/     → runbook
    tickets/      → ticket
"""

import logging
from pathlib import Path
from typing import Iterator

from netdocs.config import settings
from netdocs.ingestion.models import Chunk
from netdocs.ingestion.parsers.config_parser import parse_config
from netdocs.ingestion.parsers.prose_parser import parse_prose
from netdocs.ingestion.parsers.runbook_parser import parse_runbook
from netdocs.ingestion.parsers.ticket_parser import parse_ticket

logger = logging.getLogger(__name__)

# Map sub-directory name → (doc_type, parser function)
_DIR_MAP: dict[str, tuple[str, callable]] = {
    "configs": ("config", parse_config),
    "design_docs": ("design_doc", parse_prose),
    "runbooks": ("runbook", parse_runbook),
    "tickets": ("ticket", parse_ticket),
}

# File extensions we will attempt to parse
_SUPPORTED_EXTENSIONS = {".md", ".txt", ".cfg", ".conf", ".text"}


def iter_source_files(data_dir: Path | None = None) -> Iterator[tuple[Path, str, callable]]:
    """Yield ``(file_path, doc_type, parser_fn)`` for every supported file.

    Args:
        data_dir: Root of the raw-document tree.
                  Defaults to ``settings.data_dir``.

    Yields:
        Tuples of ``(Path, doc_type_str, parser_callable)``.
    """
    root = data_dir or settings.data_dir

    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    for subdir, (doc_type, parser_fn) in _DIR_MAP.items():
        subdir_path = root / subdir
        if not subdir_path.exists():
            logger.warning("Sub-directory not found, skipping: %s", subdir_path)
            continue

        for file_path in sorted(subdir_path.iterdir()):
            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                logger.debug("Skipping unsupported file type: %s", file_path.name)
                continue
            logger.debug("Queuing %s  [%s]", file_path.name, doc_type)
            yield file_path, doc_type, parser_fn


def load_all(data_dir: Path | None = None) -> list[Chunk]:
    """Load and chunk every document in the raw data directory.

    Args:
        data_dir: Root of the raw-document tree.

    Returns:
        A flat list of :class:`~netdocs.ingestion.models.Chunk` objects,
        ordered by file path within each document type.
    """
    chunks: list[Chunk] = []
    file_count = 0

    for file_path, doc_type, parser_fn in iter_source_files(data_dir):
        try:
            file_chunks = parser_fn(file_path)
            chunks.extend(file_chunks)
            file_count += 1
            logger.info(
                "Parsed %-30s  type=%-10s  chunks=%d",
                file_path.name,
                doc_type,
                len(file_chunks),
            )
        except Exception:
            logger.exception("Failed to parse %s — skipping", file_path)

    logger.info(
        "Load complete: %d files → %d chunks total", file_count, len(chunks)
    )
    return chunks
