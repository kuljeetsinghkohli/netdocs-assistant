"""
Tests for the NetDocs UI HTTP API client (src/netdocs/ui/api_client.py).

All tests are fully offline — HTTP calls are intercepted with httpx's
built-in transport mocking (no external libraries required beyond httpx,
which is already in the dev dependencies).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from netdocs.ui.api_client import (
    AgentResult,
    AgentStep,
    AskResult,
    Citation,
    HealthResult,
    NetDocsClient,
)


# ---------------------------------------------------------------------------
# Helpers — build mock httpx responses
# ---------------------------------------------------------------------------


def _json_response(
    data: dict[str, Any],
    status_code: int = 200,
    url: str = "http://testserver/",
) -> httpx.Response:
    """Return a real httpx.Response with JSON body, without hitting the network."""
    response = httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(data).encode(),
    )
    # httpx.Response.raise_for_status() requires ._request to be set.
    response.request = httpx.Request("GET", url)
    return response


def _transport_from_map(routes: dict[str, httpx.Response]) -> httpx.MockTransport:
    """
    Return an httpx.MockTransport that dispatches requests by URL.

    ``routes`` maps a URL substring to the desired response.
    The first matching route is used.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for fragment, response in routes.items():
            if fragment in url:
                return response
        raise ValueError(f"Unexpected request: {url}")

    return httpx.MockTransport(handler)


def _client_with(routes: dict[str, httpx.Response]) -> NetDocsClient:
    """Return a NetDocsClient whose underlying httpx calls are intercepted."""
    transport = _transport_from_map(routes)

    # Monkey-patch the module-level httpx functions used by the client.
    # The client calls ``httpx.get(...)`` and ``httpx.post(...)`` directly
    # (stateless calls, not a shared Client instance), so we patch them to
    # use an httpx.Client with our mock transport.
    _http_client = httpx.Client(transport=transport)

    import netdocs.ui.api_client as _mod

    original_get = _mod.httpx.get  # type: ignore[attr-defined]
    original_post = _mod.httpx.post  # type: ignore[attr-defined]

    # Replace module-level httpx.get / httpx.post with bound methods of
    # the mock client so all requests flow through the mock transport.
    _mod.httpx.get = _http_client.get  # type: ignore[attr-defined]
    _mod.httpx.post = _http_client.post  # type: ignore[attr-defined]

    client = NetDocsClient(base_url="http://testserver")
    client._restore = lambda: setattr(_mod, "httpx", httpx)  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_httpx():
    """Ensure httpx module is restored after each test that patches it."""
    import netdocs.ui.api_client as _mod

    saved_get = _mod.httpx.get  # type: ignore[attr-defined]
    saved_post = _mod.httpx.post  # type: ignore[attr-defined]
    yield
    _mod.httpx.get = saved_get  # type: ignore[attr-defined]
    _mod.httpx.post = saved_post  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests: health()
# ---------------------------------------------------------------------------


class TestHealth:
    def test_reachable_returns_health_result(self):
        client = _client_with({
            "/health": _json_response({"status": "ok", "vector_store_count": 1234}),
        })
        result = client.health()

        assert isinstance(result, HealthResult)
        assert result.reachable is True
        assert result.status == "ok"
        assert result.vector_store_count == 1234
        assert result.error == ""

    def test_network_error_returns_unreachable(self):
        import netdocs.ui.api_client as _mod

        def _raise_get(url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        _mod.httpx.get = _raise_get  # type: ignore[attr-defined]

        client = NetDocsClient(base_url="http://testserver")
        result = client.health()

        assert result.reachable is False
        assert "Connection refused" in result.error

    def test_http_error_returns_unreachable(self):
        client = _client_with({
            "/health": _json_response({"detail": "Internal Server Error"}, status_code=500),
        })
        result = client.health()

        assert result.reachable is False
        assert result.error != ""


# ---------------------------------------------------------------------------
# Tests: ask()
# ---------------------------------------------------------------------------


_ASK_PAYLOAD = {
    "answer": "The BGP hold timer is 90 seconds.",
    "citations": [
        {
            "doc_id": "DD-003__002",
            "section": "BGP Configuration",
            "source_file": "data/raw/design_doc/DD-003.md",
            "doc_type": "design_doc",
        }
    ],
    "refused": False,
    "refusal_reason": "",
    "confidence": 4.75,
}


class TestAsk:
    def test_returns_ask_result(self):
        client = _client_with({"/ask": _json_response(_ASK_PAYLOAD)})
        result = client.ask("What is the BGP hold timer?")

        assert isinstance(result, AskResult)
        assert result.answer == "The BGP hold timer is 90 seconds."
        assert result.refused is False
        assert result.confidence == pytest.approx(4.75)

    def test_citations_parsed(self):
        client = _client_with({"/ask": _json_response(_ASK_PAYLOAD)})
        result = client.ask("What is the BGP hold timer?")

        assert len(result.citations) == 1
        cit = result.citations[0]
        assert isinstance(cit, Citation)
        assert cit.doc_id == "DD-003__002"
        assert cit.section == "BGP Configuration"
        assert cit.doc_type == "design_doc"

    def test_refused_flag_propagated(self):
        payload = {**_ASK_PAYLOAD, "refused": True, "refusal_reason": "low_confidence", "citations": []}
        client = _client_with({"/ask": _json_response(payload)})
        result = client.ask("What is the SNMP password?")

        assert result.refused is True
        assert result.refusal_reason == "low_confidence"
        assert result.citations == []

    def test_filters_included_in_request(self):
        """Verify that doc_types, site_id, date_from, date_to appear in the POST body."""
        captured: dict = {}

        import netdocs.ui.api_client as _mod

        def fake_post(url, *, json=None, timeout=None):
            captured.update(json or {})
            return _json_response(_ASK_PAYLOAD)

        _mod.httpx.post = fake_post  # type: ignore[attr-defined]

        client = NetDocsClient(base_url="http://testserver")
        client.ask(
            "question",
            doc_types=["runbook", "ticket"],
            site_id="LON-DC01",
            date_from="2024-01-01",
            date_to="2024-12-31",
        )

        filters = captured.get("filters", {})
        assert filters["doc_type"] == ["runbook", "ticket"]
        assert filters["site_id"] == "LON-DC01"
        assert filters["date_from"] == "2024-01-01"
        assert filters["date_to"] == "2024-12-31"

    def test_no_filters_omits_filters_key(self):
        """When no filters are given, the 'filters' key must not appear in the payload."""
        captured: dict = {}

        import netdocs.ui.api_client as _mod

        def fake_post(url, *, json=None, timeout=None):
            captured.update(json or {})
            return _json_response(_ASK_PAYLOAD)

        _mod.httpx.post = fake_post  # type: ignore[attr-defined]

        client = NetDocsClient(base_url="http://testserver")
        client.ask("question")

        assert "filters" not in captured

    def test_http_error_propagates(self):
        client = _client_with({"/ask": _json_response({"detail": "not found"}, status_code=422)})
        with pytest.raises(httpx.HTTPStatusError):
            client.ask("bad question")

    def test_empty_citations_list(self):
        payload = {**_ASK_PAYLOAD, "citations": []}
        client = _client_with({"/ask": _json_response(payload)})
        result = client.ask("question")

        assert result.citations == []


# ---------------------------------------------------------------------------
# Tests: agent()
# ---------------------------------------------------------------------------


_AGENT_PAYLOAD = {
    "answer": "The agent found the BGP config.",
    "steps": [
        {
            "type": "tool_call",
            "tool": "search_docs",
            "args": {"query": "BGP hold timer"},
            "result": {"chunks": ["chunk1"]},
            "error": "",
            "error_kind": "",
            "duration_ms": 123.4,
            "answer": "",
        },
        {
            "type": "final_answer",
            "tool": "",
            "args": {},
            "result": None,
            "error": "",
            "error_kind": "",
            "duration_ms": 0.0,
            "answer": "The agent found the BGP config.",
        },
    ],
    "aborted": False,
    "abort_reason": "",
    "degraded": False,
}


class TestAgent:
    def test_returns_agent_result(self):
        client = _client_with({"/agent": _json_response(_AGENT_PAYLOAD)})
        result = client.agent("What is the BGP config?")

        assert isinstance(result, AgentResult)
        assert result.answer == "The agent found the BGP config."
        assert result.aborted is False
        assert result.degraded is False

    def test_steps_parsed(self):
        client = _client_with({"/agent": _json_response(_AGENT_PAYLOAD)})
        result = client.agent("question")

        assert len(result.steps) == 2
        step0 = result.steps[0]
        assert isinstance(step0, AgentStep)
        assert step0.type == "tool_call"
        assert step0.tool == "search_docs"
        assert step0.duration_ms == pytest.approx(123.4)

    def test_degraded_flag_propagated(self):
        payload = {**_AGENT_PAYLOAD, "degraded": True}
        client = _client_with({"/agent": _json_response(payload)})
        result = client.agent("question")

        assert result.degraded is True

    def test_aborted_flag_propagated(self):
        payload = {**_AGENT_PAYLOAD, "aborted": True, "abort_reason": "max_steps"}
        client = _client_with({"/agent": _json_response(payload)})
        result = client.agent("question")

        assert result.aborted is True
        assert result.abort_reason == "max_steps"

    def test_max_steps_sent_in_payload(self):
        captured: dict = {}

        import netdocs.ui.api_client as _mod

        def fake_post(url, *, json=None, timeout=None):
            captured.update(json or {})
            return _json_response(_AGENT_PAYLOAD)

        _mod.httpx.post = fake_post  # type: ignore[attr-defined]

        client = NetDocsClient(base_url="http://testserver")
        client.agent("question", max_steps=12)

        assert captured["max_steps"] == 12

    def test_http_error_propagates(self):
        client = _client_with({"/agent": _json_response({"detail": "error"}, status_code=500)})
        with pytest.raises(httpx.HTTPStatusError):
            client.agent("question")
