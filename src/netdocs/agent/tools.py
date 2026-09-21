from __future__ import annotations

"""
Agent tool implementations for NetDocs.

Four tools are registered:

    parse_config          — extract BGP neighbors, ASNs, route-maps, and
                            prefix-lists from a config file in data/raw/configs/.
    check_neighbor_state  — look up live (mock) BGP neighbor state from the
                            inventory JSON in data/mock/neighbor_inventory.json.
    draft_change_plan     — build a structured change plan from retrieved runbooks
                            and existing ticket context.
    find_related_tickets  — search ticket files for references to a device or
                            keyword.

Each tool is implemented as a plain function that accepts a ``dict`` of
arguments and returns a ``dict`` result.  The :mod:`netdocs.agent.loop`
module dispatches calls to these functions.

State-changing tools (``draft_change_plan``) carry ``requires_approval=True``
so the loop will pause and request human confirmation before executing them
when running in approval mode.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path("data/raw/configs")
_TICKETS_DIR = Path("data/raw/tickets")
_MOCK_INVENTORY = Path("data/mock/neighbor_inventory.json")


# ---------------------------------------------------------------------------
# Tool: parse_config
# ---------------------------------------------------------------------------

def parse_config(args: dict[str, Any]) -> dict[str, Any]:
    """Extract BGP neighbors, ASNs, route-maps, and prefix-lists from a config file.

    Args (in ``args`` dict):
        filename (str): Config filename inside data/raw/configs/ (e.g.
                        ``"LON-DC01-RTR01.cfg"``). May omit the ``.cfg`` suffix.

    Returns:
        dict with keys:
            device (str), asn (int | None), neighbors (list[dict]),
            route_maps (list[str]), prefix_lists (list[str]),
            raw_bgp_block (str)
    """
    filename: str = args.get("filename", "")
    if not filename:
        raise ValueError("parse_config requires 'filename' argument")

    # Accept bare hostname or full filename
    if not filename.endswith(".cfg"):
        filename = filename + ".cfg"

    cfg_path = _CONFIGS_DIR / filename
    if not cfg_path.exists():
        available = [p.name for p in _CONFIGS_DIR.glob("*.cfg")]
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. "
            f"Available: {available}"
        )

    text = cfg_path.read_text(encoding="utf-8", errors="replace")

    # --- Extract local ASN ---
    asn: int | None = None
    asn_m = re.search(r"^router bgp (\d+)", text, re.MULTILINE)
    if asn_m:
        asn = int(asn_m.group(1))

    # --- Extract BGP block ---
    bgp_block_lines: list[str] = []
    in_bgp = False
    for line in text.splitlines():
        if re.match(r"^router bgp \d+", line):
            in_bgp = True
        if in_bgp:
            bgp_block_lines.append(line)
            if line.strip() == "!" and len(bgp_block_lines) > 1:
                break

    bgp_block = "\n".join(bgp_block_lines)

    # --- Extract neighbors ---
    neighbors: list[dict[str, Any]] = []
    # Collect all IPs that appear in a "neighbor <ip> remote-as" line
    remote_as_re = re.compile(r"^ neighbor (\S+) remote-as (\d+)", re.MULTILINE)
    desc_re = re.compile(r"^ neighbor (\S+) description (.+)", re.MULTILINE)
    rm_in_re = re.compile(r"^ neighbor (\S+) route-map (\S+) in", re.MULTILINE)
    rm_out_re = re.compile(r"^ neighbor (\S+) route-map (\S+) out", re.MULTILINE)

    neighbor_map: dict[str, dict[str, Any]] = {}
    for m in remote_as_re.finditer(text):
        ip, remote_asn = m.group(1), int(m.group(2))
        neighbor_map[ip] = {"ip": ip, "remote_as": remote_asn}

    for m in desc_re.finditer(text):
        ip, desc = m.group(1), m.group(2).strip()
        if ip in neighbor_map:
            neighbor_map[ip]["description"] = desc

    for m in rm_in_re.finditer(text):
        ip, rm = m.group(1), m.group(2)
        if ip in neighbor_map:
            neighbor_map[ip]["route_map_in"] = rm

    for m in rm_out_re.finditer(text):
        ip, rm = m.group(1), m.group(2)
        if ip in neighbor_map:
            neighbor_map[ip]["route_map_out"] = rm

    neighbors = list(neighbor_map.values())

    # --- Extract route-maps ---
    route_maps = sorted(set(re.findall(r"^route-map (\S+)", text, re.MULTILINE)))

    # --- Extract prefix-lists ---
    prefix_lists = sorted(set(re.findall(r"^ip prefix-list (\S+)", text, re.MULTILINE)))

    logger.info(
        "parse_config: file=%s  asn=%s  neighbors=%d  route_maps=%d  prefix_lists=%d",
        filename, asn, len(neighbors), len(route_maps), len(prefix_lists),
    )

    return {
        "device": filename.replace(".cfg", ""),
        "asn": asn,
        "neighbors": neighbors,
        "route_maps": route_maps,
        "prefix_lists": prefix_lists,
        "raw_bgp_block": bgp_block,
    }


# ---------------------------------------------------------------------------
# Tool: check_neighbor_state
# ---------------------------------------------------------------------------

def check_neighbor_state(args: dict[str, Any]) -> dict[str, Any]:
    """Look up BGP neighbor state from the mock inventory.

    Args (in ``args`` dict):
        device (str):      Device hostname (e.g. ``"LON-DC01-RTR01"``).
        neighbor_ip (str): Optional — filter to a specific neighbor IP.
                           When omitted, returns all neighbors for the device.

    Returns:
        dict with keys:
            device (str), neighbors (list[dict])
    """
    device: str = args.get("device", "")
    if not device:
        raise ValueError("check_neighbor_state requires 'device' argument")

    neighbor_ip: str | None = args.get("neighbor_ip")

    if not _MOCK_INVENTORY.exists():
        raise FileNotFoundError(
            f"Mock inventory not found at {_MOCK_INVENTORY}. "
            "Create data/mock/neighbor_inventory.json first."
        )

    inventory = json.loads(_MOCK_INVENTORY.read_text())
    all_neighbors: list[dict[str, Any]] = inventory.get("neighbors", [])

    # Filter by device (case-insensitive)
    matches = [
        n for n in all_neighbors
        if n.get("device", "").lower() == device.lower()
    ]

    if not matches:
        # Return a "device not found" result rather than raising — caller can handle
        logger.warning("check_neighbor_state: device %r not found in inventory", device)
        return {"device": device, "neighbors": [], "warning": f"Device {device!r} not found in inventory"}

    if neighbor_ip:
        matches = [n for n in matches if n.get("neighbor_ip") == neighbor_ip]

    logger.info(
        "check_neighbor_state: device=%s  filter_ip=%s  results=%d",
        device, neighbor_ip or "*", len(matches),
    )

    return {"device": device, "neighbors": matches}


# ---------------------------------------------------------------------------
# Tool: draft_change_plan
# ---------------------------------------------------------------------------

def draft_change_plan(args: dict[str, Any]) -> dict[str, Any]:
    """Build a structured change plan from retrieved runbook context.

    This is a **state-changing** tool — ``requires_approval = True``.

    Args (in ``args`` dict):
        intent (str):        Human-readable description of the change
                             (e.g. "Reset BGP session on LON-DC01-RTR01 with BT peer").
        device (str):        Target device hostname.
        runbook_context (str): Relevant runbook text retrieved from the vector
                               store (injected by the agent loop).
        ticket_refs (list[str]): Optional list of related ticket IDs.

    Returns:
        dict with keys:
            intent (str), device (str),
            pre_checks (list[str]), steps (list[str]),
            validation (list[str]), rollback (list[str]),
            ticket_refs (list[str])
    """
    intent: str = args.get("intent", "")
    device: str = args.get("device", "")
    runbook_context: str = args.get("runbook_context", "")
    ticket_refs: list[str] = args.get("ticket_refs", [])

    if not intent:
        raise ValueError("draft_change_plan requires 'intent' argument")
    if not device:
        raise ValueError("draft_change_plan requires 'device' argument")

    # Build plan sections from runbook context (or supply sensible defaults)
    pre_checks = _extract_section(runbook_context, "pre.check", [
        f"Confirm change window is approved for {device}",
        "Verify no other changes are in-flight on this device",
        "Take a pre-change config backup: show running-config",
        "Check current BGP summary: show ip bgp summary",
    ])

    steps = _extract_section(runbook_context, "implementation|step", [
        f"SSH to {device}",
        "Enter privileged exec mode: enable",
        f"Apply change per approved CR for {device}",
        "Save configuration: write memory",
    ])

    validation = _extract_section(runbook_context, "validation|test plan|post.change", [
        "Verify BGP sessions are Established: show ip bgp summary",
        "Confirm routing table is correct: show ip route",
        "Check interface states: show interfaces status",
        "Monitor syslog for 15 minutes",
    ])

    rollback = _extract_section(runbook_context, "rollback", [
        "Revert to pre-change configuration",
        "Re-apply previous running-config from backup",
        "Confirm BGP sessions recover after rollback",
    ])

    logger.info(
        "draft_change_plan: intent=%r  device=%s  ticket_refs=%s",
        intent[:80], device, ticket_refs,
    )

    return {
        "intent": intent,
        "device": device,
        "pre_checks": pre_checks,
        "steps": steps,
        "validation": validation,
        "rollback": rollback,
        "ticket_refs": ticket_refs,
    }


def _extract_section(context: str, section_pattern: str, default: list[str]) -> list[str]:
    """Extract bullet/numbered items from a named section in runbook text."""
    if not context:
        return default

    # Find the section heading
    section_re = re.compile(
        rf"(?i)(?:^|\n)##?\s*(?:{section_pattern})[^\n]*\n(.*?)(?=\n##|\Z)",
        re.DOTALL,
    )
    m = section_re.search(context)
    if not m:
        return default

    block = m.group(1)
    # Extract numbered or bulleted items
    items = re.findall(r"(?m)^[ \t]*(?:\d+[.)]\s*|[-*•]\s*)(.+)", block)
    return items if items else default


# ---------------------------------------------------------------------------
# Tool: find_related_tickets
# ---------------------------------------------------------------------------

def find_related_tickets(args: dict[str, Any]) -> dict[str, Any]:
    """Search ticket files for references to a device, site, or keyword.

    Args (in ``args`` dict):
        query (str):  Search term (device name, ticket ID, keyword).

    Returns:
        dict with keys:
            query (str), tickets (list[dict])
            Each ticket dict has: ticket_id, title, status, affected_device,
            affected_site, change_date, description_excerpt.
    """
    query: str = args.get("query", "")
    if not query:
        raise ValueError("find_related_tickets requires 'query' argument")

    if not _TICKETS_DIR.exists():
        return {"query": query, "tickets": []}

    results: list[dict[str, Any]] = []
    query_lower = query.lower()

    ticket_files = sorted(_TICKETS_DIR.glob("*.txt"))
    for ticket_path in ticket_files:
        try:
            text = ticket_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        if query_lower not in text.lower():
            continue

        ticket: dict[str, Any] = {"ticket_id": ticket_path.stem}
        for field, pattern in [
            ("title", r"^TITLE:\s*(.+)"),
            ("status", r"^STATUS:\s*(.+)"),
            ("affected_device", r"^AFFECTED_DEVICE:\s*(.+)"),
            ("affected_site", r"^AFFECTED_SITE:\s*(.+)"),
            ("change_date", r"^CHANGE_DATE:\s*(.+)"),
        ]:
            m = re.search(pattern, text, re.MULTILINE)
            ticket[field] = m.group(1).strip() if m else ""

        # Pull a short excerpt around the first match
        idx = text.lower().find(query_lower)
        start = max(0, idx - 80)
        end = min(len(text), idx + 160)
        excerpt = text[start:end].replace("\n", " ").strip()
        ticket["description_excerpt"] = f"…{excerpt}…"

        results.append(ticket)

    logger.info(
        "find_related_tickets: query=%r  matches=%d", query, len(results)
    )

    return {"query": query, "tickets": results}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolSpec:
    """Metadata for a single registered tool."""

    __slots__ = ("name", "description", "fn", "requires_approval", "timeout_seconds")

    def __init__(
        self,
        name: str,
        description: str,
        fn,
        *,
        requires_approval: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.requires_approval = requires_approval
        self.timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return (
            f"ToolSpec(name={self.name!r}, "
            f"requires_approval={self.requires_approval}, "
            f"timeout={self.timeout_seconds}s)"
        )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "parse_config": ToolSpec(
        name="parse_config",
        description=(
            "Extract BGP neighbors, ASNs, route-maps, and prefix-lists from "
            "a device config file. Arg: filename (str)."
        ),
        fn=parse_config,
        requires_approval=False,
        timeout_seconds=5.0,
    ),
    "check_neighbor_state": ToolSpec(
        name="check_neighbor_state",
        description=(
            "Query live BGP neighbor state from the mock inventory. "
            "Args: device (str), neighbor_ip (str, optional)."
        ),
        fn=check_neighbor_state,
        requires_approval=False,
        timeout_seconds=5.0,
    ),
    "draft_change_plan": ToolSpec(
        name="draft_change_plan",
        description=(
            "Draft a structured change plan (pre-checks, steps, validation, "
            "rollback) from runbook context. REQUIRES HUMAN APPROVAL before "
            "execution. Args: intent (str), device (str), "
            "runbook_context (str, optional), ticket_refs (list[str], optional)."
        ),
        fn=draft_change_plan,
        requires_approval=True,
        timeout_seconds=10.0,
    ),
    "find_related_tickets": ToolSpec(
        name="find_related_tickets",
        description=(
            "Search change tickets for a device name, ticket ID, or keyword. "
            "Arg: query (str)."
        ),
        fn=find_related_tickets,
        requires_approval=False,
        timeout_seconds=5.0,
    ),
}
