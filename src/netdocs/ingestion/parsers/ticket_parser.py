"""
Change ticket parser — structured key-value header + free-text body.

Strategy (from ARCHITECTURE.md §5.2):
- Parse the structured header fields (TICKET_ID, STATUS, AFFECTED_DEVICE, …)
  into chunk metadata.
- Split the free-text sections (DESCRIPTION, IMPLEMENTATION PLAN,
  ROLLBACK PLAN, TEST PLAN, OUTCOME) into separate chunks or group them
  depending on token count.
- Never split within a labelled section.
- Metadata: ticket_id, status, priority, risk_level, affected_device,
  affected_site, change_date, approver, section.
"""

import logging
import re
from pathlib import Path

from netdocs.config import settings
from netdocs.ingestion.models import Chunk

logger = logging.getLogger(__name__)

# Header fields to extract into metadata
_HEADER_FIELDS = {
    "TICKET_ID", "TITLE", "STATUS", "PRIORITY", "RISK_LEVEL",
    "AFFECTED_DEVICE", "AFFECTED_SITE", "CHANGE_DATE", "REQUESTED_BY",
    "APPROVED_BY", "IMPLEMENTED_BY", "TICKET_TYPE",
}

# Body sections (free-text)
_SECTION_RE = re.compile(
    r"^(DESCRIPTION|IMPLEMENTATION PLAN|ROLLBACK PLAN|TEST PLAN|OUTCOME)\s*:\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_HEADER_LINE_RE = re.compile(r"^([A-Z_]+)\s*:\s*(.*)$")


def _parse_header(lines: list[str]) -> tuple[dict[str, str], int]:
    """Parse the structured header block.

    Returns:
        Tuple of ``(header_dict, last_header_line_index)``.
    """
    header: dict[str, str] = {}
    last_idx = 0

    for i, line in enumerate(lines):
        m = _HEADER_LINE_RE.match(line.strip())
        if m and m.group(1) in _HEADER_FIELDS:
            header[m.group(1).lower()] = m.group(2).strip()
            last_idx = i
        elif line.strip() == "" and header:
            # First blank line after header signals end of header block
            break

    return header, last_idx


def _parse_sections(text_after_header: str) -> list[tuple[str, str]]:
    """Split the body into ``(section_name, section_text)`` tuples.

    Text before the first section label is included as an unnamed 'body' section.
    """
    matches = list(_SECTION_RE.finditer(text_after_header))
    if not matches:
        return [("body", text_after_header.strip())]

    sections: list[tuple[str, str]] = []

    # Pre-section text
    pre = text_after_header[: matches[0].start()].strip()
    if pre:
        sections.append(("body", pre))

    for i, match in enumerate(matches):
        section_name = match.group(1).lower().replace(" ", "_")
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text_after_header)
        body = text_after_header[body_start:body_end].strip()
        if body:
            sections.append((section_name, body))

    return sections


def parse_ticket(file_path: Path) -> list[Chunk]:
    """Parse a change ticket text file into section-level chunks.

    Args:
        file_path: Path to the ``.txt`` ticket file.

    Returns:
        A list of :class:`Chunk` objects with ticket-specific metadata.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    stem = file_path.stem
    lines = text.splitlines()

    # --- Parse header ---
    header, last_header_line = _parse_header(lines)
    body_text = "\n".join(lines[last_header_line + 1 :])

    ticket_id = header.get("ticket_id", stem)
    title = header.get("title", "")

    # Base metadata shared across all chunks of this ticket
    base_meta: dict[str, str] = {
        "ticket_id": ticket_id,
        "title": title,
        "status": header.get("status", ""),
        "priority": header.get("priority", ""),
        "risk_level": header.get("risk_level", ""),
        "affected_device": header.get("affected_device", ""),
        "affected_site": header.get("affected_site", ""),
        "change_date": header.get("change_date", ""),
        "approver": header.get("approved_by", ""),
        "ticket_type": header.get("ticket_type", ""),
    }

    # --- Parse body sections ---
    sections = _parse_sections(body_text)

    # Group small sections together if they are below threshold
    chunk_size = settings.chunk_size_ticket
    chunks: list[Chunk] = []
    chunk_idx = 0

    # Always start with a header summary chunk
    header_summary = (
        f"[DOC_TYPE: ticket] [SOURCE: {title}]\n"
        f"Ticket: {ticket_id}\nTitle: {title}\n"
        f"Status: {header.get('status', '')}\n"
        f"Risk: {header.get('risk_level', '')}\n"
        f"Device: {header.get('affected_device', '')}\n"
        f"Date: {header.get('change_date', '')}\n"
        f"Approver: {header.get('approved_by', '')}"
    )
    chunks.append(
        Chunk(
            text=header_summary,
            doc_id=f"{stem}__{chunk_idx:03d}",
            source_file=str(file_path),
            doc_type="ticket",
            metadata={**base_meta, "section": "header"},
        )
    )
    chunk_idx += 1

    for section_name, section_text in sections:
        if not section_text.strip():
            continue
        # Prepend doc-type tag and section label for retrieval context
        full_text = f"[DOC_TYPE: ticket] [SECTION: {section_name.upper().replace('_', ' ')}]\n{section_text}"
        # If section is large, split it
        words = full_text.split()
        if len(words) <= chunk_size:
            sub_chunks = [full_text]
        else:
            # Sliding window split
            sub_chunks = []
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                sub_chunks.append(" ".join(words[start:end]))
                start += chunk_size - 20  # small overlap

        for part in sub_chunks:
            chunks.append(
                Chunk(
                    text=part,
                    doc_id=f"{stem}__{chunk_idx:03d}",
                    source_file=str(file_path),
                    doc_type="ticket",
                    metadata={**base_meta, "section": section_name},
                )
            )
            chunk_idx += 1

    logger.info(
        "ticket_parser: %s → %d sections → %d chunks",
        file_path.name,
        len(sections),
        len(chunks),
    )
    return chunks
