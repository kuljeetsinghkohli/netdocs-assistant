"""
Config file parser — IOS-XE and vEdge (Viptela) style.

Strategy (from ARCHITECTURE.md §5.2):
- Parse the raw config text line by line.
- Detect logical block boundaries by looking for block-start keywords
  at the root indent level (e.g. ``interface``, ``router bgp``,
  ``route-map``, ``ip prefix-list``, ``vpn``, ``system``).
- Each top-level block becomes one chunk.
- Lines between blocks (global commands like ``hostname``, ``logging``)
  are collected into a "global" preamble chunk.
- No splitting within a block — blocks are self-contained.
- Metadata: device_name, hostname, block_type, block_name.
"""

import logging
import re
from pathlib import Path

from netdocs.ingestion.models import Chunk

logger = logging.getLogger(__name__)

# Keywords that signal the START of a top-level config block.
# Ordered from most specific to least to avoid false matches.
_BLOCK_STARTERS = re.compile(
    r"^(?:"
    r"interface|"
    r"router\s+(?:bgp|ospf|isis|rip|eigrp)|"
    r"route-map|"
    r"ip\s+(?:prefix-list|community-list|access-list|vrf)|"
    r"policy-map|"
    r"class-map|"
    r"vrf\s+definition|"
    r"crypto\s+(?:ikev2|ipsec|keyring|pki)|"
    r"vpn\s+\d+|"          # vEdge VPN block
    r"system$|"            # vEdge system block
    r"omp$|"               # vEdge OMP
    r"sdwan$|"             # IOS-XE SD-WAN stanza
    r"bfd$|"
    r"ntp$|"               # vEdge ntp block
    r"aaa$"
    r")",
    re.IGNORECASE,
)

# Lines that are comments or blank — not part of config content
_SKIP_LINE = re.compile(r"^\s*(?:!|$)")


def _extract_device_name(text: str, stem: str) -> str:
    """Try to extract hostname from config text; fall back to filename stem."""
    for line in text.splitlines():
        m = re.match(r"^\s*(?:hostname|host-name)\s+(\S+)", line, re.IGNORECASE)
        if m:
            return m.group(1)
    return stem


def _classify_block(header_line: str) -> tuple[str, str]:
    """Return ``(block_type, block_name)`` for a block header line.

    Examples:
        ``interface GigabitEthernet0/0/0`` → ``("interface", "GigabitEthernet0/0/0")``
        ``router bgp 65001``               → ``("router_bgp", "65001")``
        ``route-map RM-FROM-BT-IN permit`` → ``("route_map", "RM-FROM-BT-IN")``
        ``vpn 0``                          → ``("vpn", "0")``
    """
    header_line = header_line.strip()
    parts = header_line.split()
    if not parts:
        return "unknown", ""

    first = parts[0].lower()

    if first == "interface":
        return "interface", parts[1] if len(parts) > 1 else ""
    if first == "router":
        proto = parts[1].lower() if len(parts) > 1 else ""
        name = parts[2] if len(parts) > 2 else ""
        return f"router_{proto}", name
    if first == "route-map":
        return "route_map", parts[1] if len(parts) > 1 else ""
    if first == "ip" and len(parts) > 1:
        sub = parts[1].lower()
        name = parts[2] if len(parts) > 2 else ""
        return f"ip_{sub}", name
    if first == "vpn":
        return "vpn", parts[1] if len(parts) > 1 else ""
    if first == "system":
        return "system", "global"

    return first, " ".join(parts[1:]) if len(parts) > 1 else ""


def _split_into_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split config lines into ``(header, body_lines)`` blocks.

    Lines before the first block header go into a special ``__preamble__`` block.
    """
    blocks: list[tuple[str, list[str]]] = []
    current_header = "__preamble__"
    current_body: list[str] = []

    for line in lines:
        stripped = line.rstrip()
        # Skip pure comment lines and blanks for block detection only
        if _SKIP_LINE.match(stripped):
            current_body.append(stripped)
            continue
        # Check for a new top-level block (no leading whitespace for IOS-XE,
        # or one-space indent for vEdge)
        if _BLOCK_STARTERS.match(stripped) and not stripped.startswith("  "):
            if current_body or current_header != "__preamble__":
                blocks.append((current_header, current_body))
            current_header = stripped
            current_body = [stripped]
        else:
            current_body.append(stripped)

    # Flush last block
    if current_body:
        blocks.append((current_header, current_body))

    return blocks


def parse_config(file_path: Path) -> list[Chunk]:
    """Parse a device config file into a list of :class:`Chunk` objects.

    Each top-level config block becomes one chunk.  Blocks that are pure
    comments or empty after stripping are discarded.

    Args:
        file_path: Path to the ``.cfg`` or ``.conf`` file.

    Returns:
        A list of :class:`Chunk` objects with config-specific metadata.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    stem = file_path.stem
    device_name = _extract_device_name(text, stem)

    blocks = _split_into_blocks(lines)
    chunks: list[Chunk] = []

    for idx, (header, body_lines) in enumerate(blocks):
        # Reconstruct block text
        block_text = "\n".join(body_lines).strip()

        # Drop near-empty blocks (only comments / whitespace)
        meaningful = [
            l for l in body_lines
            if l.strip() and not l.strip().startswith("!")
        ]
        if len(meaningful) < 2:
            continue

        block_type, block_name = _classify_block(header)

        chunk = Chunk(
            text=block_text,
            doc_id=f"{stem}__{idx:03d}",
            source_file=str(file_path),
            doc_type="config",
            metadata={
                "device_name": device_name,
                "hostname": device_name,
                "block_type": block_type,
                "block_name": block_name,
                "block_index": idx,
                "site_id": _infer_site_id(device_name),
            },
        )
        chunks.append(chunk)
        logger.debug(
            "Config block: %-30s  type=%-15s  tokens≈%d",
            block_name or header[:30],
            block_type,
            len(block_text.split()),
        )

    logger.info(
        "config_parser: %s → %d blocks / %d chunks",
        file_path.name,
        len(blocks),
        len(chunks),
    )
    return chunks


def _infer_site_id(device_name: str) -> str:
    """Extract a site identifier from a device name (best-effort).

    Examples:
        LON-DC01-RTR01 → LON-DC01
        SGP-BR02-VE01  → SGP-BR02
    """
    parts = device_name.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return device_name
