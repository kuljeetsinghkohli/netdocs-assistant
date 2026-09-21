"""
Tests for the NetDocs agent layer and LLM retry logic.

Coverage:
  - parse_config: happy path, missing file, bare hostname, ASN extraction
  - check_neighbor_state: device found, device not found, neighbor_ip filter
  - draft_change_plan: happy path, missing required args
  - find_related_tickets: match found, no match, missing query arg
  - AgentLoop: max-steps guard, bad-tool-name error, tool timeout,
               duplicate-call guard, approval-denied abort, extractive+heuristic
  - RetryingLLMClient: succeeds on second attempt after 503,
                       falls back to extractive after 3 failures (degraded),
                       non-retryable error propagates immediately
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def configs_dir(tmp_path: Path) -> Path:
    """Return a temp directory with a minimal IOS-XE-style config file."""
    d = tmp_path / "configs"
    d.mkdir()
    cfg = d / "TEST-RTR01.cfg"
    cfg.write_text(
        """\
hostname TEST-RTR01
!
router bgp 65100
 bgp router-id 10.0.0.1
 neighbor 192.168.1.1 remote-as 12345
 neighbor 192.168.1.1 description ISP-PEER
 neighbor 192.168.1.1 route-map RM-FROM-ISP-IN in
 neighbor 192.168.1.1 route-map RM-TO-ISP-OUT out
 neighbor 10.0.0.2 remote-as 65100
 neighbor 10.0.0.2 description iBGP-PEER
!
route-map RM-FROM-ISP-IN permit 10
 set local-preference 200
!
route-map RM-TO-ISP-OUT permit 10
!
ip prefix-list PL-SUMMARY seq 10 permit 10.0.0.0/8
ip prefix-list PL-CLOUD seq 10 permit 10.100.0.0/16 le 24
!
end
"""
    )
    return d


@pytest.fixture()
def mock_inventory(tmp_path: Path) -> Path:
    """Return a path to a minimal mock inventory JSON."""
    d = tmp_path / "mock"
    d.mkdir()
    inv_path = d / "neighbor_inventory.json"
    inv_path.write_text(
        json.dumps(
            {
                "neighbors": [
                    {
                        "device": "TEST-RTR01",
                        "neighbor_ip": "192.168.1.1",
                        "description": "ISP-PEER",
                        "remote_as": 12345,
                        "session_state": "Established",
                        "uptime": "10d 00:00:00",
                        "prefixes_received": 500,
                        "prefixes_sent": 2,
                        "hold_time": 90,
                        "keepalive": 30,
                        "bfd_state": "Up",
                        "last_reset": "2025-01-01",
                        "last_reset_reason": "Admin",
                    },
                    {
                        "device": "TEST-RTR01",
                        "neighbor_ip": "10.0.0.2",
                        "description": "iBGP-PEER",
                        "remote_as": 65100,
                        "session_state": "Idle",
                        "uptime": "0d 00:00:00",
                        "prefixes_received": 0,
                        "prefixes_sent": 0,
                        "hold_time": 90,
                        "keepalive": 30,
                        "bfd_state": "Down",
                        "last_reset": "2025-01-20",
                        "last_reset_reason": "Hold timer expired",
                    },
                ]
            }
        )
    )
    return inv_path


@pytest.fixture()
def tickets_dir(tmp_path: Path) -> Path:
    """Return a temp directory with two synthetic ticket files."""
    d = tmp_path / "tickets"
    d.mkdir()

    (d / "CRQ-TEST-001.txt").write_text(
        """\
TICKET_ID: CRQ-TEST-001
TITLE: BGP Reset on TEST-RTR01
STATUS: Closed
AFFECTED_DEVICE: TEST-RTR01
AFFECTED_SITE: TEST-SITE
CHANGE_DATE: 2025-01-15

DESCRIPTION:
Reset BGP session on TEST-RTR01 to clear stale routes.
OUTCOME: Successful.
"""
    )
    (d / "CRQ-TEST-002.txt").write_text(
        """\
TICKET_ID: CRQ-TEST-002
TITLE: OSPF Area Migration NYC
STATUS: Open
AFFECTED_DEVICE: NYC-DC01-RTR01
AFFECTED_SITE: NYC-DC01
CHANGE_DATE: 2025-02-01

DESCRIPTION:
Migrate OSPF area 10 to area 20 on NYC-DC01-RTR01.
"""
    )
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_paths(monkeypatch, configs_dir: Path, mock_inv: Path, tickets_dir: Path):
    """Patch the path constants in the tools module."""
    import netdocs.agent.tools as tools_mod
    monkeypatch.setattr(tools_mod, "_CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(tools_mod, "_MOCK_INVENTORY", mock_inv)
    monkeypatch.setattr(tools_mod, "_TICKETS_DIR", tickets_dir)


# ---------------------------------------------------------------------------
# parse_config
# ---------------------------------------------------------------------------

class TestParseConfig:
    def test_happy_path(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import parse_config

        result = parse_config({"filename": "TEST-RTR01.cfg"})
        assert result["asn"] == 65100
        assert result["device"] == "TEST-RTR01"
        # Two neighbors declared with remote-as
        assert len(result["neighbors"]) == 2
        ips = {n["ip"] for n in result["neighbors"]}
        assert "192.168.1.1" in ips
        assert "10.0.0.2" in ips
        # Route-maps
        assert "RM-FROM-ISP-IN" in result["route_maps"]
        assert "RM-TO-ISP-OUT" in result["route_maps"]
        # Prefix lists
        assert "PL-SUMMARY" in result["prefix_lists"]
        assert "PL-CLOUD" in result["prefix_lists"]

    def test_bare_hostname_no_extension(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import parse_config

        result = parse_config({"filename": "TEST-RTR01"})
        assert result["asn"] == 65100

    def test_missing_file_raises(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import parse_config

        with pytest.raises(FileNotFoundError, match="DOES-NOT-EXIST.cfg"):
            parse_config({"filename": "DOES-NOT-EXIST.cfg"})

    def test_missing_filename_arg_raises(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import parse_config

        with pytest.raises(ValueError, match="filename"):
            parse_config({})

    def test_neighbor_descriptions_and_routemaps(
        self, monkeypatch, configs_dir, mock_inventory, tickets_dir
    ):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import parse_config

        result = parse_config({"filename": "TEST-RTR01"})
        isp_n = next(n for n in result["neighbors"] if n["ip"] == "192.168.1.1")
        assert isp_n["description"] == "ISP-PEER"
        assert isp_n["route_map_in"] == "RM-FROM-ISP-IN"
        assert isp_n["route_map_out"] == "RM-TO-ISP-OUT"


# ---------------------------------------------------------------------------
# check_neighbor_state
# ---------------------------------------------------------------------------

class TestCheckNeighborState:
    def test_all_neighbors_for_device(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import check_neighbor_state

        result = check_neighbor_state({"device": "TEST-RTR01"})
        assert result["device"] == "TEST-RTR01"
        assert len(result["neighbors"]) == 2

    def test_filter_by_neighbor_ip(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import check_neighbor_state

        result = check_neighbor_state({"device": "TEST-RTR01", "neighbor_ip": "192.168.1.1"})
        assert len(result["neighbors"]) == 1
        assert result["neighbors"][0]["session_state"] == "Established"

    def test_device_not_found_returns_empty(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import check_neighbor_state

        result = check_neighbor_state({"device": "GHOST-RTR99"})
        assert result["neighbors"] == []
        assert "warning" in result

    def test_missing_device_arg_raises(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import check_neighbor_state

        with pytest.raises(ValueError, match="device"):
            check_neighbor_state({})


# ---------------------------------------------------------------------------
# draft_change_plan
# ---------------------------------------------------------------------------

class TestDraftChangePlan:
    def test_happy_path_no_context(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import draft_change_plan

        result = draft_change_plan({
            "intent": "Reset BGP session",
            "device": "TEST-RTR01",
        })
        assert result["intent"] == "Reset BGP session"
        assert result["device"] == "TEST-RTR01"
        # All four plan sections must be present and non-empty
        assert len(result["pre_checks"]) > 0
        assert len(result["steps"]) > 0
        assert len(result["validation"]) > 0
        assert len(result["rollback"]) > 0

    def test_missing_intent_raises(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import draft_change_plan

        with pytest.raises(ValueError, match="intent"):
            draft_change_plan({"device": "TEST-RTR01"})

    def test_missing_device_raises(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import draft_change_plan

        with pytest.raises(ValueError, match="device"):
            draft_change_plan({"intent": "Reset BGP"})

    def test_ticket_refs_passed_through(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import draft_change_plan

        result = draft_change_plan({
            "intent": "Clear route table",
            "device": "NYC-RTR01",
            "ticket_refs": ["CRQ-2025-001", "CRQ-2025-002"],
        })
        assert result["ticket_refs"] == ["CRQ-2025-001", "CRQ-2025-002"]


# ---------------------------------------------------------------------------
# find_related_tickets
# ---------------------------------------------------------------------------

class TestFindRelatedTickets:
    def test_match_by_device(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import find_related_tickets

        result = find_related_tickets({"query": "TEST-RTR01"})
        assert len(result["tickets"]) == 1
        assert result["tickets"][0]["ticket_id"] == "CRQ-TEST-001"

    def test_no_match_returns_empty(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import find_related_tickets

        result = find_related_tickets({"query": "GHOST-DEVICE-99"})
        assert result["tickets"] == []

    def test_match_by_keyword(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import find_related_tickets

        result = find_related_tickets({"query": "OSPF"})
        assert any(t["ticket_id"] == "CRQ-TEST-002" for t in result["tickets"])

    def test_missing_query_raises(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import find_related_tickets

        with pytest.raises(ValueError, match="query"):
            find_related_tickets({})

    def test_excerpt_present(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import find_related_tickets

        result = find_related_tickets({"query": "BGP Reset"})
        assert result["tickets"]
        assert "description_excerpt" in result["tickets"][0]


# ---------------------------------------------------------------------------
# AgentLoop — failure paths
# ---------------------------------------------------------------------------

def _make_fake_llm(responses: list[str]) -> Any:
    """Return a simple fake LLM that cycles through *responses* in order."""
    from netdocs.llm.client import BaseLLMClient

    class _Fake(BaseLLMClient):
        def __init__(self, resps):
            self._resps = list(resps)
            self._idx = 0

        def complete(self, system, user, *, temperature=None, max_tokens=None):
            if self._idx < len(self._resps):
                r = self._resps[self._idx]
                self._idx += 1
                return r
            return json.dumps({"action": "final_answer", "answer": "Done."})

    return _Fake(responses)


class TestAgentLoopFailurePaths:
    def _minimal_tool_registry(self):
        """Return a minimal tool registry with a fast no-op tool."""
        from netdocs.agent.tools import ToolSpec

        def _noop(args):
            return {"ok": True}

        return {
            "noop_tool": ToolSpec(
                name="noop_tool",
                description="does nothing",
                fn=_noop,
                timeout_seconds=5.0,
            )
        }

    def test_max_steps_guard(self):
        """Loop should abort after max_steps iterations."""
        from netdocs.agent.loop import AgentLoop

        # LLM always asks to call the noop_tool (different args each time to avoid dedup guard)
        responses = [
            json.dumps({"action": "tool_call", "tool": "noop_tool", "args": {"n": i}})
            for i in range(20)
        ]
        llm = _make_fake_llm(responses)
        registry = self._minimal_tool_registry()

        loop = AgentLoop(llm, tool_registry=registry, max_steps=3, require_approval=False)
        result = loop.run("test question")

        assert result.aborted
        assert "max_steps" in result.abort_reason

    def test_unknown_tool_name_records_error(self):
        """Requesting a tool that doesn't exist produces an error step."""
        from netdocs.agent.loop import AgentLoop

        responses = [
            json.dumps({"action": "tool_call", "tool": "GHOST_TOOL", "args": {}}),
            json.dumps({"action": "final_answer", "answer": "Could not find ghost tool."}),
        ]
        llm = _make_fake_llm(responses)

        loop = AgentLoop(llm, max_steps=5, require_approval=False)
        result = loop.run("ghost question")

        error_steps = [s for s in result.steps if getattr(s, "type", "") == "error"]
        assert error_steps, "Expected at least one error step"
        assert error_steps[0].error_kind == "not_found"  # type: ignore[union-attr]

    def test_bad_tool_args_records_error(self):
        """A tool that raises ValueError on bad args produces a bad_args error step."""
        from netdocs.agent.loop import AgentLoop
        from netdocs.agent.tools import ToolSpec

        def _strict_tool(args: dict) -> dict:
            if "required_key" not in args:
                raise ValueError("required_key is missing")
            return {"ok": True}

        registry = {
            "strict_tool": ToolSpec(
                name="strict_tool",
                description="needs required_key",
                fn=_strict_tool,
                timeout_seconds=5.0,
            )
        }
        responses = [
            json.dumps({"action": "tool_call", "tool": "strict_tool", "args": {}}),
            json.dumps({"action": "final_answer", "answer": "Got an error."}),
        ]
        llm = _make_fake_llm(responses)
        loop = AgentLoop(llm, tool_registry=registry, max_steps=5, require_approval=False)
        result = loop.run("question with bad args")

        error_steps = [s for s in result.steps if getattr(s, "type", "") == "error"]
        assert error_steps
        assert error_steps[0].error_kind == "bad_args"  # type: ignore[union-attr]

    def test_tool_timeout_records_error(self):
        """A tool that exceeds its timeout produces a timeout error step."""
        import time
        from netdocs.agent.loop import AgentLoop
        from netdocs.agent.tools import ToolSpec

        def _slow_tool(args: dict) -> dict:
            time.sleep(10)  # will be killed by timeout
            return {"ok": True}

        registry = {
            "slow_tool": ToolSpec(
                name="slow_tool",
                description="slow",
                fn=_slow_tool,
                timeout_seconds=0.1,  # very short timeout
            )
        }
        responses = [
            json.dumps({"action": "tool_call", "tool": "slow_tool", "args": {}}),
            json.dumps({"action": "final_answer", "answer": "Timed out."}),
        ]
        llm = _make_fake_llm(responses)
        loop = AgentLoop(llm, tool_registry=registry, max_steps=5, require_approval=False)
        result = loop.run("slow question")

        error_steps = [s for s in result.steps if getattr(s, "type", "") == "error"]
        assert error_steps
        assert error_steps[0].error_kind == "timeout"  # type: ignore[union-attr]

    def test_duplicate_call_guard(self):
        """The same (tool, args) pair twice should abort the loop."""
        from netdocs.agent.loop import AgentLoop

        identical_call = json.dumps({"action": "tool_call", "tool": "noop_tool", "args": {"x": 1}})
        responses = [identical_call, identical_call, identical_call]
        llm = _make_fake_llm(responses)
        registry = self._minimal_tool_registry()

        loop = AgentLoop(llm, tool_registry=registry, max_steps=10, require_approval=False)
        result = loop.run("repeat question")

        assert result.aborted
        assert "duplicate" in result.abort_reason

    def test_approval_denied_aborts(self):
        """When the human denies approval the loop should abort."""
        from netdocs.agent.loop import AgentLoop
        from netdocs.agent.tools import ToolSpec

        def _state_changer(args: dict) -> dict:
            return {"changed": True}

        registry = {
            "state_changer": ToolSpec(
                name="state_changer",
                description="changes state",
                fn=_state_changer,
                requires_approval=True,
                timeout_seconds=5.0,
            )
        }
        responses = [
            json.dumps({"action": "tool_call", "tool": "state_changer", "args": {}}),
        ]
        llm = _make_fake_llm(responses)
        # Auto-deny approval
        loop = AgentLoop(
            llm,
            tool_registry=registry,
            max_steps=5,
            require_approval=True,
            approval_callback=lambda name, args: False,
        )
        result = loop.run("change something")

        assert result.aborted
        assert "approval denied" in result.abort_reason

    def test_extractive_llm_with_heuristic_routing(self, monkeypatch, tmp_path):
        """With extractive LLM and a device keyword, the loop routes to check_neighbor_state."""
        from netdocs.llm.client import ExtractiveClient
        from netdocs.agent.loop import AgentLoop
        from netdocs.agent.tools import ToolSpec

        captured: list[dict] = []

        def _mock_neighbor(args: dict) -> dict:
            captured.append(args)
            return {"device": args.get("device", ""), "neighbors": []}

        registry = {
            "check_neighbor_state": ToolSpec(
                name="check_neighbor_state",
                description="check neighbor",
                fn=_mock_neighbor,
                timeout_seconds=5.0,
            )
        }
        llm = ExtractiveClient()
        loop = AgentLoop(llm, tool_registry=registry, max_steps=3, require_approval=False)
        result = loop.run("Is the BGP session on LON-DC01-RTR01 up?")

        # The heuristic should have picked check_neighbor_state
        assert any(
            getattr(s, "tool", "") == "check_neighbor_state"
            for s in result.steps
        )


# ---------------------------------------------------------------------------
# list_configs tool
# ---------------------------------------------------------------------------

class TestListConfigs:
    def test_returns_all_configs(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import list_configs

        result = list_configs({})
        assert len(result["devices"]) == 1
        assert result["devices"][0]["hostname"] == "TEST-RTR01"
        assert result["devices"][0]["filename"] == "TEST-RTR01.cfg"

    def test_site_id_extracted(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import list_configs

        result = list_configs({})
        device = result["devices"][0]
        # "TEST-RTR01" splits into ["TEST", "RTR01"] → site_id = "TEST-RTR01"
        assert device["site_id"] == "TEST-RTR01"

    def test_empty_dir_returns_empty(self, monkeypatch, tmp_path, mock_inventory, tickets_dir):
        empty_dir = tmp_path / "empty_configs"
        empty_dir.mkdir()
        import netdocs.agent.tools as tools_mod
        monkeypatch.setattr(tools_mod, "_CONFIGS_DIR", empty_dir)
        from netdocs.agent.tools import list_configs

        result = list_configs({})
        assert result["devices"] == []

    def test_missing_dir_returns_empty(self, monkeypatch, tmp_path, mock_inventory, tickets_dir):
        import netdocs.agent.tools as tools_mod
        monkeypatch.setattr(tools_mod, "_CONFIGS_DIR", tmp_path / "nonexistent")
        from netdocs.agent.tools import list_configs

        result = list_configs({})
        assert result["devices"] == []


# ---------------------------------------------------------------------------
# AgentLoop — list_configs triggered by generic question
# ---------------------------------------------------------------------------

class TestListConfigsDiscovery:
    """Verify that a generic question (no device named) triggers list_configs first."""

    def test_generic_bgp_question_triggers_list_configs(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        """When the LLM is a fake that returns list_configs first, the loop executes it."""
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.loop import AgentLoop
        from netdocs.agent.tools import ToolSpec, TOOL_REGISTRY

        list_configs_called: list[dict] = []
        check_neighbor_called: list[dict] = []

        def _mock_list_configs(args: dict) -> dict:
            list_configs_called.append(args)
            return {
                "devices": [
                    {"hostname": "TEST-RTR01", "filename": "TEST-RTR01.cfg",
                     "site_id": "TEST-RTR", "vendor": "cisco-ios-xe"},
                ]
            }

        def _mock_check_neighbor(args: dict) -> dict:
            check_neighbor_called.append(args)
            return {"device": args.get("device", ""), "neighbors": []}

        registry = {
            "list_configs": ToolSpec(
                name="list_configs",
                description="list configs",
                fn=_mock_list_configs,
                timeout_seconds=5.0,
            ),
            "check_neighbor_state": ToolSpec(
                name="check_neighbor_state",
                description="check neighbor state",
                fn=_mock_check_neighbor,
                timeout_seconds=5.0,
            ),
        }

        # Fake LLM: first call → list_configs, second → check_neighbor_state, third → final_answer
        responses = [
            json.dumps({"action": "tool_call", "tool": "list_configs", "args": {}}),
            json.dumps({"action": "tool_call", "tool": "check_neighbor_state",
                        "args": {"device": "TEST-RTR01"}}),
            json.dumps({"action": "final_answer", "answer": "No BGP peers are down."}),
        ]
        llm = _make_fake_llm(responses)
        loop = AgentLoop(llm, tool_registry=registry, max_steps=5, require_approval=False)
        result = loop.run("Which routers have BGP peers that are down?")

        # list_configs must have been called
        assert list_configs_called, "list_configs was not called for a generic question"

        # list_configs must be the FIRST tool call in the step trace
        tool_steps = [s for s in result.steps if getattr(s, "type", "") == "tool_call"]
        assert tool_steps, "No tool calls recorded"
        assert tool_steps[0].tool == "list_configs", (
            f"First tool called was {tool_steps[0].tool!r}, expected 'list_configs'"
        )

    def test_extractive_generic_question_triggers_list_configs(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        """With the extractive (heuristic) LLM, a generic BGP question routes to list_configs."""
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.llm.client import ExtractiveClient
        from netdocs.agent.loop import AgentLoop
        from netdocs.agent.tools import ToolSpec

        captured: list[dict] = []

        def _mock_list_configs(args: dict) -> dict:
            captured.append({"tool": "list_configs", **args})
            return {"devices": []}

        registry = {
            "list_configs": ToolSpec(
                name="list_configs",
                description="list configs",
                fn=_mock_list_configs,
                timeout_seconds=5.0,
            ),
        }

        llm = ExtractiveClient()
        loop = AgentLoop(llm, tool_registry=registry, max_steps=3, require_approval=False)
        result = loop.run("Which routers have BGP peers that are down?")

        assert any(
            getattr(s, "tool", "") == "list_configs"
            for s in result.steps
        ), "Expected list_configs in steps for generic question"


# ---------------------------------------------------------------------------
# RetryingLLMClient — 503 retry and degraded fallback
# ---------------------------------------------------------------------------

class TestRetryingLLMClient:
    def test_succeeds_on_second_attempt_after_503(self):
        """The wrapped client fails once with 503, then succeeds on retry."""
        from netdocs.llm.client import RetryingLLMClient, BaseLLMClient

        call_count = 0

        class _FailOnceLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    err = RuntimeError("503 Service Unavailable")
                    err.status_code = 503  # type: ignore[attr-defined]
                    raise err
                return "success on retry"

        client = RetryingLLMClient(
            _FailOnceLLM(), max_attempts=3, base_delay=0.0, _sleep=lambda _: None
        )
        result = client.complete("sys", "usr")
        assert result == "success on retry"
        assert call_count == 2

    def test_falls_back_to_extractive_after_all_retries_fail(self):
        """After 3 503 failures the client falls back to extractive (degraded)."""
        from netdocs.llm.client import RetryingLLMClient, BaseLLMClient, DEGRADED_PREFIX

        class _AlwaysFailLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                err = RuntimeError("503 Service Unavailable")
                err.status_code = 503  # type: ignore[attr-defined]
                raise err

        client = RetryingLLMClient(
            _AlwaysFailLLM(), max_attempts=3, base_delay=0.0, _sleep=lambda _: None
        )
        result = client.complete("system prompt text", "user text")
        # Result should be prefixed with DEGRADED_PREFIX
        assert result.startswith(DEGRADED_PREFIX)

    def test_non_retryable_error_propagates_immediately(self):
        """A non-503/429 error is re-raised without retry."""
        from netdocs.llm.client import RetryingLLMClient, BaseLLMClient

        call_count = 0

        class _KeyErrorLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                nonlocal call_count
                call_count += 1
                raise ValueError("Bad request — invalid model name")

        client = RetryingLLMClient(
            _KeyErrorLLM(), max_attempts=3, base_delay=0.0, _sleep=lambda _: None
        )
        with pytest.raises(ValueError, match="Bad request"):
            client.complete("sys", "usr")

        # Must not have retried
        assert call_count == 1

    def test_429_rate_limit_is_retried(self):
        """A 429 error is treated as retryable."""
        from netdocs.llm.client import RetryingLLMClient, BaseLLMClient

        call_count = 0

        class _RateLimitLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    err = RuntimeError("429 Too Many Requests — rate limit exceeded")
                    err.status_code = 429  # type: ignore[attr-defined]
                    raise err
                return "ok after rate limit"

        client = RetryingLLMClient(
            _RateLimitLLM(), max_attempts=3, base_delay=0.0, _sleep=lambda _: None
        )
        result = client.complete("sys", "usr")
        assert result == "ok after rate limit"
        assert call_count == 3

    def test_exponential_backoff_delays(self):
        """Sleep durations follow base * 2^attempt pattern."""
        from netdocs.llm.client import RetryingLLMClient, BaseLLMClient

        delays: list[float] = []

        class _AlwaysFail(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                err = RuntimeError("503")
                err.status_code = 503  # type: ignore[attr-defined]
                raise err

        client = RetryingLLMClient(
            _AlwaysFail(),
            max_attempts=3,
            base_delay=2.0,
            _sleep=lambda d: delays.append(d),
        )
        client.complete("sys", "usr")  # exhausts retries, returns degraded

        # 3 attempts → 2 sleep calls (between attempt 1→2 and 2→3; no sleep after final attempt)
        assert delays == pytest.approx([2.0, 4.0])

    def test_degraded_prefix_stripped_by_agent_loop(self):
        """AgentLoop detects and strips DEGRADED_PREFIX from LLM responses."""
        from netdocs.llm.client import DEGRADED_PREFIX, BaseLLMClient
        from netdocs.agent.loop import AgentLoop

        class _DegradedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                # Return a degraded response with a valid JSON final_answer
                payload = json.dumps({"action": "final_answer", "answer": "degraded answer"})
                return DEGRADED_PREFIX + payload

        loop = AgentLoop(_DegradedLLM(), max_steps=3, require_approval=False)
        result = loop.run("any question")
        assert result.degraded is True
        # The answer should be the content, not the prefix
        assert "DEGRADED" not in result.final_answer


# ---------------------------------------------------------------------------
# AgentLoop — LLM quota / rate-limit degradation
# ---------------------------------------------------------------------------

def _make_quota_error(message: str = "429 Too Many Requests — rate limit exceeded") -> Exception:
    """Return a retryable HTTP error mimicking a Gemini/OpenAI 429."""
    err = RuntimeError(message)
    err.status_code = 429  # type: ignore[attr-defined]
    return err


def _make_daily_quota_error() -> Exception:
    """Return an error mimicking Gemini RESOURCE_EXHAUSTED per-day quota."""
    return RuntimeError(
        "429 RESOURCE_EXHAUSTED: You exceeded your current quota, daily per_day limit reached."
    )


class TestAgentLoopLLMDegradation:
    """AgentLoop must degrade gracefully when the LLM raises quota/rate errors."""

    def _minimal_registry_with_neighbor(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        """Return a registry with a single check_neighbor_state stub."""
        _patch_paths(monkeypatch, configs_dir, mock_inventory, tickets_dir)
        from netdocs.agent.tools import ToolSpec

        def _mock_neighbor(args: dict) -> dict:
            return {"device": args.get("device", "TEST"), "neighbors": []}

        return {
            "check_neighbor_state": ToolSpec(
                name="check_neighbor_state",
                description="check BGP neighbor state",
                fn=_mock_neighbor,
                timeout_seconds=5.0,
            ),
        }

    def test_429_returns_degraded_true(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        """When the LLM raises a 429, the loop returns degraded=True (not an exception)."""
        from netdocs.llm.client import BaseLLMClient
        from netdocs.agent.loop import AgentLoop

        registry = self._minimal_registry_with_neighbor(
            monkeypatch, configs_dir, mock_inventory, tickets_dir
        )

        class _RateLimitedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise _make_quota_error()

        loop = AgentLoop(
            _RateLimitedLLM(),
            tool_registry=registry,
            max_steps=5,
            require_approval=False,
        )
        result = loop.run("Is the BGP session on LON-DC01-RTR01 up?")

        assert result.degraded is True

    def test_429_answer_is_non_empty(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        """The degraded answer must contain a non-empty explanation."""
        from netdocs.llm.client import BaseLLMClient
        from netdocs.agent.loop import AgentLoop

        registry = self._minimal_registry_with_neighbor(
            monkeypatch, configs_dir, mock_inventory, tickets_dir
        )

        class _RateLimitedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise _make_quota_error()

        loop = AgentLoop(
            _RateLimitedLLM(),
            tool_registry=registry,
            max_steps=5,
            require_approval=False,
        )
        result = loop.run("Is the BGP session on LON-DC01-RTR01 up?")

        assert result.final_answer.strip() != ""

    def test_429_tool_calls_appear_in_trace(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        """Tool calls executed before/during degradation must still appear in steps."""
        from netdocs.llm.client import BaseLLMClient
        from netdocs.agent.loop import AgentLoop

        registry = self._minimal_registry_with_neighbor(
            monkeypatch, configs_dir, mock_inventory, tickets_dir
        )

        class _RateLimitedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise _make_quota_error()

        loop = AgentLoop(
            _RateLimitedLLM(),
            tool_registry=registry,
            max_steps=5,
            require_approval=False,
        )
        result = loop.run("Is the BGP session on LON-DC01-RTR01 up?")

        # The heuristic should have routed to check_neighbor_state
        tool_steps = [s for s in result.steps if getattr(s, "type", "") == "tool_call"]
        assert tool_steps, "Expected at least one tool_call step in the trace"
        assert any(s.tool == "check_neighbor_state" for s in tool_steps)

    def test_503_returns_degraded_true(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        """A 503 Service Unavailable error also triggers graceful degradation."""
        from netdocs.llm.client import BaseLLMClient
        from netdocs.agent.loop import AgentLoop

        registry = self._minimal_registry_with_neighbor(
            monkeypatch, configs_dir, mock_inventory, tickets_dir
        )

        class _ServiceUnavailableLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                err = RuntimeError("503 Service Unavailable")
                err.status_code = 503  # type: ignore[attr-defined]
                raise err

        loop = AgentLoop(
            _ServiceUnavailableLLM(),
            tool_registry=registry,
            max_steps=5,
            require_approval=False,
        )
        result = loop.run("Is the BGP session on LON-DC01-RTR01 up?")

        assert result.degraded is True

    def test_non_quota_exception_propagates(self):
        """A non-retryable exception (e.g. ValueError) must NOT be swallowed."""
        from netdocs.llm.client import BaseLLMClient
        from netdocs.agent.loop import AgentLoop

        class _BrokenLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise ValueError("Invalid model configuration")

        loop = AgentLoop(_BrokenLLM(), max_steps=3, require_approval=False)
        with pytest.raises(ValueError, match="Invalid model configuration"):
            loop.run("any question")

    def test_degraded_note_in_answer(self, monkeypatch, configs_dir, mock_inventory, tickets_dir):
        """The final answer must contain the user-facing rate-limit note."""
        from netdocs.llm.client import BaseLLMClient
        from netdocs.agent.loop import AgentLoop, _DEGRADED_NOTE

        registry = self._minimal_registry_with_neighbor(
            monkeypatch, configs_dir, mock_inventory, tickets_dir
        )

        class _RateLimitedLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise _make_quota_error()

        loop = AgentLoop(
            _RateLimitedLLM(),
            tool_registry=registry,
            max_steps=5,
            require_approval=False,
        )
        result = loop.run("Is the BGP session on LON-DC01-RTR01 up?")

        assert _DEGRADED_NOTE in result.final_answer


# ---------------------------------------------------------------------------
# RetryingLLMClient — daily quota skips retries
# ---------------------------------------------------------------------------

class TestDailyQuotaDegradation:
    def test_daily_quota_falls_back_immediately_without_retry(self):
        """RESOURCE_EXHAUSTED / per-day quota must NOT be retried; degrade immediately."""
        from netdocs.llm.client import RetryingLLMClient, BaseLLMClient, DEGRADED_PREFIX

        call_count = 0

        class _DailyQuotaLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                nonlocal call_count
                call_count += 1
                raise _make_daily_quota_error()

        client = RetryingLLMClient(
            _DailyQuotaLLM(), max_attempts=3, base_delay=0.0, _sleep=lambda _: None
        )
        result = client.complete("system prompt", "user text")

        # Must have fallen back to extractive (degraded prefix present)
        assert result.startswith(DEGRADED_PREFIX)
        # Must NOT have retried — only one call to the inner LLM
        assert call_count == 1

    def test_daily_quota_no_sleep_delays(self):
        """Daily quota fallback must not sleep at all."""
        from netdocs.llm.client import RetryingLLMClient, BaseLLMClient

        delays: list[float] = []

        class _DailyQuotaLLM(BaseLLMClient):
            def complete(self, system, user, **kwargs):
                raise _make_daily_quota_error()

        client = RetryingLLMClient(
            _DailyQuotaLLM(),
            max_attempts=3,
            base_delay=2.0,
            _sleep=lambda d: delays.append(d),
        )
        client.complete("sys", "usr")

        assert delays == [], f"Expected no sleep delays for daily quota, got {delays}"
