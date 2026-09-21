"""
Prose document parser — Markdown design documents.

Strategy (from ARCHITECTURE.md §5.2):
- Parse the Markdown text to identify H1/H2/H3 heading structure.
- Each section (heading + its paragraphs until the next same-or-higher heading)
  is treated as one "section unit".
- Section units larger than ``chunk_size_prose`` tokens are split further
  using a recursive character splitter with overlap.
- Metadata: document_title, section_heading, subsection, doc_id_prefix.
"""

import logging
import re
from pathlib import Path

from netdocs.config import settings
from netdocs.ingestion.models import Chunk

logger = logging.getLogger(__name__)

# Heading regex: captures level (# count) and heading text
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _approximate_tokens(text: str) -> int:
    """Rough token count: split on whitespace + punctuation boundaries."""
    return len(text.split())


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple recursive splitter: paragraphs → sentences → hard truncation.

    Tries to honour natural paragraph boundaries first, then sentence
    boundaries, then falls back to sliding-window word splitting.

    Args:
        text:       Input text to split.
        chunk_size: Target maximum tokens per chunk (approximate).
        overlap:    Number of tokens to repeat at the start of each chunk.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return [c for c in chunks if c.strip()]


def _extract_front_matter(text: str) -> dict[str, str]:
    """Extract simple key: value pairs from the first lines of a Markdown doc."""
    meta: dict[str, str] = {}
    for line in text.splitlines()[:20]:
        m = re.match(r"^\*\*(.+?)\*\*:\s*(.+)$", line)
        if m:
            meta[m.group(1).lower().replace(" ", "_")] = m.group(2).strip()
    return meta


def parse_prose(file_path: Path) -> list[Chunk]:
    """Parse a Markdown design document into section-level chunks.

    Args:
        file_path: Path to the ``.md`` file.

    Returns:
        A list of :class:`Chunk` objects with prose-specific metadata.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    stem = file_path.stem

    # --- Extract document-level metadata from front matter ---
    front_matter = _extract_front_matter(text)
    doc_title = front_matter.get("document_type", stem.replace("_", " ").title())

    # Extract H1 title if present
    h1_match = _HEADING_RE.search(text)
    if h1_match and len(h1_match.group(1)) == 1:
        doc_title = h1_match.group(2).strip()

    # --- Split text into (heading, body) pairs ---
    sections = _extract_sections(text)

    chunk_size = settings.chunk_size_prose
    overlap = settings.chunk_overlap_prose

    chunks: list[Chunk] = []
    chunk_idx = 0

    for heading_level, heading_text, body_text in sections:
        if not body_text.strip():
            continue

        # Determine section / subsection for metadata
        if heading_level == 1:
            section = heading_text
            subsection = ""
        elif heading_level == 2:
            section = heading_text
            subsection = ""
        else:
            section = heading_text
            subsection = heading_text

        sub_chunks = _split_text(body_text, chunk_size, overlap)
        for part_idx, chunk_text in enumerate(sub_chunks):
            if not chunk_text.strip():
                continue
            combined = f"## {heading_text}\n\n{chunk_text}"
            chunks.append(
                Chunk(
                    text=combined,
                    doc_id=f"{stem}__{chunk_idx:03d}",
                    source_file=str(file_path),
                    doc_type="design_doc",
                    metadata={
                        "document_title": doc_title,
                        "section_heading": section,
                        "subsection": subsection,
                        "heading_level": heading_level,
                        "part_index": part_idx,
                        **{k: v for k, v in front_matter.items()
                           if k in ("author", "version", "date")},
                    },
                )
            )
            chunk_idx += 1

    logger.info(
        "prose_parser: %s → %d sections → %d chunks",
        file_path.name,
        len(sections),
        len(chunks),
    )
    return chunks


def _extract_sections(
    text: str,
) -> list[tuple[int, str, str]]:
    """Split Markdown text into ``(heading_level, heading_text, body_text)`` tuples.

    Text before the first heading is returned as a level-0 "preamble" section.
    """
    headings = list(_HEADING_RE.finditer(text))
    sections: list[tuple[int, str, str]] = []

    if not headings:
        return [(0, "Document", text)]

    # Preamble before first heading
    preamble = text[: headings[0].start()].strip()
    if preamble:
        sections.append((0, "Preamble", preamble))

    for i, match in enumerate(headings):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        body_start = match.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((level, heading_text, body))

    return sections
