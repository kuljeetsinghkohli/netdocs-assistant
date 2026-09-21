"""
Runbook parser — structured Markdown runbooks with numbered steps.

Strategy (from ARCHITECTURE.md §5.2):
- Treat each top-level H2/H3 section (procedure step or group) as a unit.
- Code fences (CLI commands) are kept intact within their section — never split.
- Sections that exceed ``chunk_size_runbook`` tokens are split at paragraph
  boundaries; code fences are always kept whole.
- Metadata: procedure_name, step_number, device_role, runbook_id.
"""

import logging
import re
from pathlib import Path

from netdocs.config import settings
from netdocs.ingestion.models import Chunk
from netdocs.ingestion.parsers.prose_parser import _extract_sections

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^(?:Step\s+)?(\d+)\s*[—–-]?\s*(.+)", re.IGNORECASE)
_RUNBOOK_ID_RE = re.compile(r"^RB-\d+", re.IGNORECASE)


def _extract_runbook_id(stem: str) -> str:
    """Extract runbook ID like 'RB-001' from filename stem."""
    m = re.match(r"(RB-\d+)", stem, re.IGNORECASE)
    return m.group(1).upper() if m else stem


def _split_preserving_code_fences(
    text: str, chunk_size: int
) -> list[str]:
    """Split ``text`` at paragraph boundaries, keeping code fences intact.

    Args:
        text:       Input text.
        chunk_size: Approximate max token count per chunk.

    Returns:
        List of text segments.
    """
    # Split on double newlines (paragraphs), but keep code fences together
    paragraphs: list[str] = []
    in_fence = False
    current: list[str] = []

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
        elif in_fence:
            current.append(line)
        elif line.strip() == "":
            if current:
                paragraphs.append("\n".join(current))
                current = []
        else:
            current.append(line)

    if current:
        paragraphs.append("\n".join(current))

    # Now bin paragraphs into chunks
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(para.split())
        if current_tokens + para_tokens > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c for c in chunks if c.strip()]


def _build_step_toc(sections: list[tuple[int, str, str]]) -> str:
    """Build a compact table-of-contents string listing numbered procedure steps.

    Used to augment the preamble chunk so that when it ranks highest the LLM
    can still see the full step sequence and cite the runbook.

    Args:
        sections: List of ``(level, heading_text, body_text)`` tuples from
                  :func:`~netdocs.ingestion.parsers.prose_parser._extract_sections`.

    Returns:
        Newline-separated list of step headings, e.g.::

            Procedure steps:
            1. Confirm the Flap and Identify the Peer
            2. Check Underlying Connectivity
            ...
    """
    steps: list[str] = []
    for _, heading_text, _ in sections:
        m = _STEP_RE.match(heading_text)
        if m:
            steps.append(f"{m.group(1)}. {m.group(2).strip()}")
    if not steps:
        return ""
    return "Procedure steps:\n" + "\n".join(steps)


def parse_runbook(file_path: Path) -> list[Chunk]:
    """Parse a runbook Markdown file into step-level chunks.

    The first chunk (index 000) is the document preamble (title, metadata,
    prerequisites).  It is augmented with a compact table-of-contents of the
    numbered procedure steps so that when it ranks highest in retrieval the LLM
    still receives the full step outline and can produce a cited answer.

    Args:
        file_path: Path to the ``.md`` runbook file.

    Returns:
        A list of :class:`Chunk` objects with runbook-specific metadata.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    stem = file_path.stem
    runbook_id = _extract_runbook_id(stem)

    # Extract document title from H1
    doc_title = stem
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if h1_match:
        doc_title = h1_match.group(1).strip()

    # Detect device_role from title (best-effort)
    device_role = _infer_device_role(doc_title)

    sections = _extract_sections(text)
    chunk_size = settings.chunk_size_runbook

    # Build step TOC once; injected into preamble chunks (step_number == 0
    # and part_index == 0) so they remain useful when retrieved alone.
    step_toc = _build_step_toc(sections)

    chunks: list[Chunk] = []
    chunk_idx = 0
    procedure_name = doc_title

    for level, heading_text, body_text in sections:
        if not body_text.strip():
            continue

        # Extract step number if present
        step_match = _STEP_RE.match(heading_text)
        step_number = int(step_match.group(1)) if step_match else 0
        if step_match:
            procedure_name = step_match.group(2).strip()

        sub_chunks = _split_preserving_code_fences(body_text, chunk_size)
        for part_idx, chunk_text in enumerate(sub_chunks):
            if not chunk_text.strip():
                continue
            # Prefix with doc-type and source title so the embedding model
            # encodes document-type semantics alongside the content.  This
            # makes phrasing like "runbook procedure" rank runbook chunks
            # higher even when the body text does not repeat those words.
            #
            # For preamble chunks (step_number == 0, part_index == 0) also
            # append the step TOC so the LLM can produce a cited overview
            # answer when this chunk ranks first.
            is_preamble = step_number == 0 and part_idx == 0
            toc_suffix = f"\n\n{step_toc}" if (is_preamble and step_toc) else ""
            combined = (
                f"[DOC_TYPE: runbook] [SOURCE: {doc_title}]\n"
                f"### {heading_text}\n\n{chunk_text}{toc_suffix}"
            )
            chunks.append(
                Chunk(
                    text=combined,
                    doc_id=f"{stem}__{chunk_idx:03d}",
                    source_file=str(file_path),
                    doc_type="runbook",
                    metadata={
                        "runbook_id": runbook_id,
                        "document_title": doc_title,
                        "procedure_name": procedure_name,
                        "step_number": step_number,
                        "heading_level": level,
                        "part_index": part_idx,
                        "device_role": device_role,
                    },
                )
            )
            chunk_idx += 1

    logger.info(
        "runbook_parser: %s → %d sections → %d chunks",
        file_path.name,
        len(sections),
        len(chunks),
    )
    return chunks


def _infer_device_role(title: str) -> str:
    """Infer device role from runbook title."""
    title_lower = title.lower()
    if "bgp" in title_lower:
        return "hub_router"
    if "ospf" in title_lower:
        return "dc_router"
    if "omp" in title_lower or "tunnel" in title_lower or "vedge" in title_lower or "ztp" in title_lower:
        return "vedge"
    if "vmanage" in title_lower or "cert" in title_lower:
        return "vmanage"
    return "any"
