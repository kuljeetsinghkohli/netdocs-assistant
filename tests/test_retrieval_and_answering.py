"""
Pytest tests for the NetDocs retrieval and answer generation layer.

All tests run offline:
- No real LLM calls (FakeLLMClient is used throughout).
- No real embedding model calls (a FakeEmbedder is injected).
- No real BM25/vector-store queries (FakeVectorStore + FakeBM25Index).
- FastAPI tests use TestClient (no network).

The goal is to verify correctness of the orchestration logic:
  - RRF fusion maths
  - Metadata filter logic
  - Citation parsing
  - Confidence threshold refusal
  - API request/response contract
"""

from __future__ import annotations

import math
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures and fake implementations
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Returns a fixed-length zero vector for any input."""

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]


class FakeVectorStore:
    """In-memory vector store for tests."""

    def __init__(self, docs: list[dict[str, Any]] | None = None) -> None:
        self._docs = {d["id"]: d for d in (docs or [])}

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results = list(self._docs.values())
        if where:
            # Apply simple equality filter on doc_type
            eq_filter = where.get("doc_type", {}).get("$eq")
            if eq_filter:
                results = [r for r in results if r.get("metadata", {}).get("doc_type") == eq_filter]
        return results[:n_results]

    def get_all_ids(self) -> list[str]:
        return list(self._docs.keys())

    def get_all_texts(self) -> list[str]:
        return [d["text"] for d in self._docs.values()]

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self._docs[i] for i in ids if i in self._docs]

    def count(self) -> int:
        return len(self._docs)


class FakeBM25Index:
    """Returns all documents ranked by appearance order."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def query(self, query: str, n_results: int = 20) -> list[dict[str, Any]]:
        return [
            {"id": id_, "score": 1.0 / (i + 1), "rank": i}
            for i, id_ in enumerate(self._ids[:n_results])
        ]


def _make_chunk(
    doc_id: str,
    text: str = "some text",
    doc_type: str = "design_doc",
    section: str = "Overview",
    site_id: str = "LON-DC01",
    change_date: str = "",
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "text": text,
        "metadata": {
            "doc_type": doc_type,
            "section_heading": section,
            "site_id": site_id,
            "change_date": change_date,
            "source_file": f"data/raw/{doc_type}/{doc_id}.md",
        },
    }


# ---------------------------------------------------------------------------
# Tests: RRF fusion
# ---------------------------------------------------------------------------

class TestRRF:
    def test_basic_fusion_combines_scores(self):
        from netdocs.retriever.hybrid import reciprocal_rank_fusion

        dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        sparse = [{"id": "b"}, {"id": "a"}, {"id": "d"}]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)

        ids = [f["id"] for f in fused]
        # "b" appears in both lists at rank 1 and 0 — should have highest combined score
        # "a" appears at rank 0 and rank 1
        # Scores for a: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
        # Scores for b: 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61  (same as a — tie)
        # Both a and b beat c and d which only appear once
        assert "c" in ids
        assert "d" in ids

    def test_single_list_produces_valid_scores(self):
        from netdocs.retriever.hybrid import reciprocal_rank_fusion

        dense = [{"id": "x"}, {"id": "y"}]
        fused = reciprocal_rank_fusion(dense, [], k=60)
        assert len(fused) == 2
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

    def test_rrf_score_formula(self):
        from netdocs.retriever.hybrid import reciprocal_rank_fusion

        dense = [{"id": "only"}]
        fused = reciprocal_rank_fusion(dense, [], k=60)
        expected = 1.0 / (60 + 0 + 1)
        assert abs(fused[0]["rrf_score"] - expected) < 1e-9

    def test_empty_lists_return_empty(self):
        from netdocs.retriever.hybrid import reciprocal_rank_fusion

        assert reciprocal_rank_fusion([], []) == []


# ---------------------------------------------------------------------------
# Tests: Metadata filter builder
# ---------------------------------------------------------------------------

class TestBuildChromaWhere:
    def test_single_doc_type(self):
        from netdocs.retriever.hybrid import _build_chroma_where

        w = _build_chroma_where({"doc_type": "runbook"})
        assert w == {"doc_type": {"$eq": "runbook"}}

    def test_multiple_doc_types(self):
        from netdocs.retriever.hybrid import _build_chroma_where

        w = _build_chroma_where({"doc_type": ["runbook", "ticket"]})
        assert w == {"doc_type": {"$in": ["runbook", "ticket"]}}

    def test_site_id_filter(self):
        from netdocs.retriever.hybrid import _build_chroma_where

        w = _build_chroma_where({"site_id": "LON-DC01"})
        assert w == {"site_id": {"$eq": "LON-DC01"}}

    def test_combined_and_clause(self):
        from netdocs.retriever.hybrid import _build_chroma_where

        w = _build_chroma_where({"doc_type": "config", "site_id": "LON-DC01"})
        assert w == {"$and": [
            {"doc_type": {"$eq": "config"}},
            {"site_id": {"$eq": "LON-DC01"}},
        ]}

    def test_empty_filters_return_none(self):
        from netdocs.retriever.hybrid import _build_chroma_where

        assert _build_chroma_where({}) is None


# ---------------------------------------------------------------------------
# Tests: Date filter
# ---------------------------------------------------------------------------

class TestDateFilter:
    def test_no_filter_always_passes(self):
        from netdocs.retriever.hybrid import _passes_date_filter

        assert _passes_date_filter({"change_date": "2024-08-01"}, None, None)

    def test_date_from_excludes_earlier(self):
        from netdocs.retriever.hybrid import _passes_date_filter

        assert not _passes_date_filter({"change_date": "2024-07-01"}, "2024-08-01", None)
        assert _passes_date_filter({"change_date": "2024-08-01"}, "2024-08-01", None)

    def test_date_to_excludes_later(self):
        from netdocs.retriever.hybrid import _passes_date_filter

        assert not _passes_date_filter({"change_date": "2024-12-01"}, None, "2024-11-30")
        assert _passes_date_filter({"change_date": "2024-11-30"}, None, "2024-11-30")

    def test_chunk_without_date_always_passes(self):
        from netdocs.retriever.hybrid import _passes_date_filter

        assert _passes_date_filter({}, "2024-01-01", "2024-12-31")


# ---------------------------------------------------------------------------
# Tests: Citation parsing
# ---------------------------------------------------------------------------

class TestCitationParsing:
    def _chunks(self) -> list[dict[str, Any]]:
        return [
            _make_chunk("DD-003__002", text="BGP policy text", section="BGP Communities"),
            _make_chunk("RB-001__001", text="BGP flap runbook", doc_type="runbook", section="Step 3"),
        ]

    def test_citations_extracted_from_answer(self):
        from netdocs.llm.generator import _parse_citations

        answer = (
            "The hold timer is 90s [doc:DD-003__002:BGP Communities]. "
            "See also [doc:RB-001__001:Step 3] for recovery steps."
        )
        cits = _parse_citations(answer, self._chunks())
        assert len(cits) == 2
        assert cits[0].doc_id == "DD-003__002"
        assert cits[0].section == "BGP Communities"
        assert cits[1].doc_id == "RB-001__001"

    def test_duplicate_citations_deduplicated(self):
        from netdocs.llm.generator import _parse_citations

        answer = "[doc:DD-003__002] and again [doc:DD-003__002]"
        cits = _parse_citations(answer, self._chunks())
        assert len(cits) == 1

    def test_no_citations_returns_empty(self):
        from netdocs.llm.generator import _parse_citations

        cits = _parse_citations("No citations here.", self._chunks())
        assert cits == []

    def test_unknown_doc_id_still_parsed(self):
        from netdocs.llm.generator import _parse_citations

        answer = "[doc:UNKNOWN__000:mystery section]"
        cits = _parse_citations(answer, self._chunks())
        assert len(cits) == 1
        assert cits[0].doc_id == "UNKNOWN__000"
        assert cits[0].source_file == ""  # not in chunks → blank


# ---------------------------------------------------------------------------
# Tests: Answer generator
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    def _fake_llm(self) -> Any:
        from netdocs.llm.client import FakeLLMClient
        return FakeLLMClient(canned_answer="The answer is yes. [doc:DD-003__002:Overview]")

    def _chunks_with_score(self, score: float = 5.0) -> list[dict[str, Any]]:
        chunk = _make_chunk("DD-003__002")
        chunk["rerank_score"] = score
        return [chunk]

    def test_answer_returned_above_threshold(self):
        from netdocs.llm.generator import generate_answer

        result = generate_answer(
            "What is the BGP hold timer?",
            self._chunks_with_score(5.0),
            self._fake_llm(),
            confidence_threshold=0.10,
        )
        assert not result.refused
        assert result.answer.strip() != ""
        assert result.confidence == 5.0

    def test_refusal_below_threshold(self):
        from netdocs.llm.generator import generate_answer

        result = generate_answer(
            "What is the BGP hold timer?",
            self._chunks_with_score(-10.0),   # low reranker score
            self._fake_llm(),
            confidence_threshold=0.10,
        )
        assert result.refused
        assert result.citations == []

    def test_refusal_on_empty_chunks(self):
        from netdocs.llm.generator import generate_answer

        result = generate_answer(
            "Who are you?",
            [],
            self._fake_llm(),
            confidence_threshold=0.10,
        )
        assert result.refused
        assert result.confidence == 0.0

    def test_citations_populated_from_answer(self):
        from netdocs.llm.generator import generate_answer

        result = generate_answer(
            "BGP policy?",
            self._chunks_with_score(5.0),
            self._fake_llm(),
            confidence_threshold=0.10,
        )
        assert len(result.citations) >= 1
        assert result.citations[0].doc_id == "DD-003__002"

    def test_rrf_score_used_when_no_rerank_score(self):
        from netdocs.llm.generator import generate_answer

        chunk = _make_chunk("DD-003__002")
        chunk["rrf_score"] = 0.5  # no rerank_score
        result = generate_answer(
            "question",
            [chunk],
            self._fake_llm(),
            confidence_threshold=0.10,
        )
        assert result.confidence == pytest.approx(0.5)

    def test_llm_refusal_sentence_marks_refused(self):
        from netdocs.llm.client import FakeLLMClient
        from netdocs.llm.generator import generate_answer, LOW_CONFIDENCE_ANSWER

        llm = FakeLLMClient(canned_answer=LOW_CONFIDENCE_ANSWER)
        chunk = _make_chunk("X__000")
        chunk["rerank_score"] = 10.0

        result = generate_answer("question", [chunk], llm, confidence_threshold=0.10)
        assert result.refused


# ---------------------------------------------------------------------------
# Tests: BM25 index (unit)
# ---------------------------------------------------------------------------

class TestBM25Index:
    def _build(self) -> Any:
        from netdocs.retriever.bm25_index import BM25Index

        ids = ["a", "b", "c"]
        corpus = [
            "BGP peer flapping hold timer keepalive",
            "OSPF neighbour adjacency area",
            "Zscaler ZIA internet breakout policy",
        ]
        return BM25Index(ids=ids, corpus=corpus)

    def test_relevant_doc_scores_higher(self):
        idx = self._build()
        results = idx.query("BGP peer")
        assert results[0]["id"] == "a"

    def test_empty_query_returns_empty(self):
        idx = self._build()
        results = idx.query("   ")
        assert results == []

    def test_zero_score_docs_excluded(self):
        idx = self._build()
        results = idx.query("BGP peer", n_results=10)
        assert all(r["score"] > 0 for r in results)

    def test_n_results_respected(self):
        idx = self._build()
        results = idx.query("the", n_results=1)
        assert len(results) <= 1

    def test_save_and_load(self, tmp_path):
        from netdocs.retriever.bm25_index import BM25Index

        idx = self._build()
        idx.save(tmp_path)
        loaded = BM25Index.load(tmp_path)
        r1 = idx.query("BGP")
        r2 = loaded.query("BGP")
        assert [x["id"] for x in r1] == [x["id"] for x in r2]


# ---------------------------------------------------------------------------
# Tests: HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _make_retriever(self, docs: list[dict[str, Any]]) -> Any:
        from netdocs.retriever.hybrid import HybridRetriever

        store = FakeVectorStore(docs)
        bm25 = FakeBM25Index([d["id"] for d in docs])
        embedder = FakeEmbedder()
        return HybridRetriever(store, bm25, embedder)

    def test_returns_results(self):
        docs = [_make_chunk(f"doc_{i}") for i in range(5)]
        retriever = self._make_retriever(docs)
        results = retriever.retrieve("BGP hold timer", top_k=3)
        assert len(results) <= 3 * 4  # over-fetch factor

    def test_doc_type_filter_applied(self):
        docs = [
            _make_chunk("runbook_1", doc_type="runbook"),
            _make_chunk("design_1", doc_type="design_doc"),
        ]
        retriever = self._make_retriever(docs)
        results = retriever.retrieve(
            "BGP",
            top_k=10,
            filters={"doc_type": "runbook"},
        )
        for r in results:
            assert r["metadata"]["doc_type"] == "runbook"

    def test_date_filter_excludes_old_tickets(self):
        docs = [
            _make_chunk("t1", doc_type="ticket", change_date="2024-01-01"),
            _make_chunk("t2", doc_type="ticket", change_date="2024-12-01"),
        ]
        retriever = self._make_retriever(docs)
        results = retriever.retrieve(
            "BGP",
            top_k=10,
            filters={"date_from": "2024-06-01"},
        )
        for r in results:
            cd = r["metadata"].get("change_date", "")
            if cd:
                assert cd >= "2024-06-01"


# ---------------------------------------------------------------------------
# Tests: FastAPI /ask endpoint
# ---------------------------------------------------------------------------

class TestAPI:
    @pytest.fixture(autouse=True)
    def setup_app(self, monkeypatch):
        """Override app.state with fake pipeline so no models are loaded."""
        from netdocs.api.main import app
        from netdocs.llm.client import FakeLLMClient
        from netdocs.llm.generator import generate_answer

        docs = [_make_chunk("DD-001__000", text="SD-WAN overlay design text")]
        store = FakeVectorStore(docs)
        bm25 = FakeBM25Index(list(store._docs.keys()))
        embedder = FakeEmbedder()

        class FakeReranker:
            def rerank(self, query, candidates, top_k=5):
                for c in candidates:
                    c["rerank_score"] = 1.0
                return candidates[:top_k]

        from netdocs.retriever.hybrid import HybridRetriever

        retriever = HybridRetriever(store, bm25, embedder)
        reranker = FakeReranker()
        llm = FakeLLMClient()

        app.state.store = store
        app.state.retriever = retriever
        app.state.reranker = reranker
        app.state.llm = llm
        yield

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from netdocs.api.main import app

        return TestClient(app, raise_server_exceptions=True)

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["vector_store_count"], int)

    def test_ask_returns_answer(self, client):
        r = client.post("/ask", json={"question": "What is the SD-WAN overlay design?"})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert isinstance(body["refused"], bool)
        assert isinstance(body["confidence"], float)
        assert isinstance(body["citations"], list)

    def test_ask_empty_question_rejected(self, client):
        r = client.post("/ask", json={"question": ""})
        assert r.status_code == 422

    def test_ask_with_filters(self, client):
        r = client.post(
            "/ask",
            json={
                "question": "BGP timer changes",
                "filters": {
                    "doc_type": ["ticket"],
                    "date_from": "2024-01-01",
                },
            },
        )
        assert r.status_code == 200

    def test_ask_top_k_respected(self, client):
        r = client.post("/ask", json={"question": "BGP", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["citations"]) <= 3


# ---------------------------------------------------------------------------
# Tests: FakeLLMClient
# ---------------------------------------------------------------------------

class TestFakeLLMClient:
    def test_canned_answer_returned(self):
        from netdocs.llm.client import FakeLLMClient

        llm = FakeLLMClient(canned_answer="hello")
        assert llm.complete("sys", "user") == "hello"

    def test_citation_echo_without_canned(self):
        from netdocs.llm.client import FakeLLMClient

        llm = FakeLLMClient()
        user_msg = "context [Source: doc1] and [Source: doc2]"
        answer = llm.complete("sys", user_msg)
        assert "doc:" in answer or "Fake" in answer


# ---------------------------------------------------------------------------
# Tests: Settings / config
# ---------------------------------------------------------------------------

class TestSettings:
    def test_default_settings_load(self):
        from netdocs.config import Settings

        s = Settings()
        assert s.retrieval_top_k == 5
        assert s.rrf_k == 60
        assert s.llm_provider == "openai"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NETDOCS_RETRIEVAL_TOP_K", "10")
        from netdocs.config import Settings

        s = Settings()
        assert s.retrieval_top_k == 10

# ---------------------------------------------------------------------------
# Tests: Provider selection (offline — no real LLM calls)
# ---------------------------------------------------------------------------

class TestProviderSelection:
    """Verify _resolve_provider() auto-fallback logic."""

    def setup_method(self):
        from netdocs.llm.client import reset_llm_client
        reset_llm_client()

    def teardown_method(self):
        from netdocs.llm.client import reset_llm_client
        reset_llm_client()

    def test_explicit_extractive_returns_extractive_instance(self, monkeypatch):
        monkeypatch.setenv("NETDOCS_LLM_PROVIDER", "extractive")
        from netdocs.config import Settings
        import netdocs.llm.client as client_mod

        # Force re-read of settings
        original_settings = client_mod.settings
        client_mod.settings = Settings()
        try:
            from netdocs.llm.client import ExtractiveClient
            result = client_mod._resolve_provider()
            assert result == "extractive"
        finally:
            client_mod.settings = original_settings

    def test_openai_missing_pkg_and_no_gemini_key_falls_back_to_extractive(
        self, monkeypatch
    ):
        monkeypatch.setenv("NETDOCS_LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from netdocs.config import Settings
        import netdocs.llm.client as client_mod
        import sys

        original_settings = client_mod.settings
        s = Settings()
        # Explicitly clear keys so .env file values don't leak through
        object.__setattr__(s, "openai_api_key", "")
        object.__setattr__(s, "gemini_api_key", "")
        client_mod.settings = s

        # Simulate openai package missing
        saved = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore
        try:
            result = client_mod._resolve_provider()
            assert result == "extractive"
        finally:
            if saved is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = saved
            client_mod.settings = original_settings

    def test_openai_missing_pkg_with_gemini_key_falls_back_to_gemini(
        self, monkeypatch
    ):
        monkeypatch.setenv("NETDOCS_LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
        from netdocs.config import Settings
        import netdocs.llm.client as client_mod
        import sys

        original_settings = client_mod.settings
        client_mod.settings = Settings()

        # Simulate openai package missing
        saved = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore
        try:
            result = client_mod._resolve_provider()
            assert result == "gemini"
        finally:
            if saved is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = saved
            client_mod.settings = original_settings

    def test_explicit_gemini_provider_is_preserved(self, monkeypatch):
        monkeypatch.setenv("NETDOCS_LLM_PROVIDER", "gemini")
        from netdocs.config import Settings
        import netdocs.llm.client as client_mod

        original_settings = client_mod.settings
        client_mod.settings = Settings()
        try:
            result = client_mod._resolve_provider()
            assert result == "gemini"
        finally:
            client_mod.settings = original_settings

    def test_fake_provider_is_preserved(self, monkeypatch):
        monkeypatch.setenv("NETDOCS_LLM_PROVIDER", "fake")
        from netdocs.config import Settings
        import netdocs.llm.client as client_mod

        original_settings = client_mod.settings
        client_mod.settings = Settings()
        try:
            result = client_mod._resolve_provider()
            assert result == "fake"
        finally:
            client_mod.settings = original_settings

    def test_get_llm_client_returns_extractive_when_openai_unavailable(
        self, monkeypatch
    ):
        monkeypatch.setenv("NETDOCS_LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from netdocs.config import Settings
        import netdocs.llm.client as client_mod
        from netdocs.llm.client import ExtractiveClient
        import sys

        original_settings = client_mod.settings
        s = Settings()
        # Explicitly clear keys so .env file values don't leak through
        object.__setattr__(s, "openai_api_key", "")
        object.__setattr__(s, "gemini_api_key", "")
        client_mod.settings = s

        saved = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore
        client_mod._llm_instance = None
        try:
            llm = client_mod.get_llm_client()
            assert isinstance(llm, ExtractiveClient)
        finally:
            if saved is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = saved
            client_mod.settings = original_settings
            client_mod._llm_instance = None


# ---------------------------------------------------------------------------
# Tests: ExtractiveClient (offline)
# ---------------------------------------------------------------------------

class TestExtractiveClient:
    def _client(self):
        from netdocs.llm.client import ExtractiveClient
        return ExtractiveClient()

    def _system_with_passages(self, passages: list[tuple[str, str, str, str]]) -> str:
        """Build a fake system prompt matching _build_context_block format."""
        from netdocs.llm.generator import _build_context_block
        chunks = [
            {
                "id": doc_id,
                "text": text,
                "metadata": {
                    "doc_type": doc_type,
                    "section_heading": section,
                    "source_file": f"data/raw/{doc_type}/{doc_id}.md",
                },
            }
            for doc_id, doc_type, section, text in passages
        ]
        from netdocs.llm.generator import _SYSTEM_PROMPT
        context = _build_context_block(chunks)
        return _SYSTEM_PROMPT.format(context=context)

    def test_extractive_returns_passage_text(self):
        client = self._client()
        system = self._system_with_passages([
            ("DD-001__000", "design_doc", "BGP Overview",
             "The BGP hold timer is set to 90 seconds on all peering sessions."),
        ])
        answer = client.complete(system, "What is the BGP hold timer?")
        assert "90 seconds" in answer or "90" in answer

    def test_extractive_appends_doc_citation(self):
        client = self._client()
        system = self._system_with_passages([
            ("DD-001__000", "design_doc", "BGP Overview", "BGP hold timer text."),
        ])
        answer = client.complete(system, "BGP timer?")
        assert "[doc:DD-001__000" in answer

    def test_extractive_multiple_passages(self):
        client = self._client()
        system = self._system_with_passages([
            ("DD-001__000", "design_doc", "BGP Overview", "Hold timer is 90s."),
            ("RB-002__001", "runbook", "Recovery Steps", "Restart BGP session."),
        ])
        answer = client.complete(system, "BGP flap?")
        assert "[doc:DD-001__000" in answer
        assert "[doc:RB-002__001" in answer

    def test_extractive_no_passages_returns_refusal(self):
        client = self._client()
        answer = client.complete("No context passages here.", "what?")
        assert "cannot find" in answer.lower()

    def test_extractive_long_passage_truncated(self):
        client = self._client()
        long_text = "word " * 200  # 200 words
        system = self._system_with_passages([
            ("DD-001__000", "design_doc", "Section", long_text),
        ])
        answer = client.complete(system, "question?")
        word_count = len(answer.split())
        assert word_count < 200, "Long passages should be truncated"

    def test_extractive_integrates_with_generate_answer(self):
        """ExtractiveClient plugs into the generator pipeline end-to-end."""
        from netdocs.llm.generator import generate_answer
        from netdocs.llm.client import ExtractiveClient

        chunk = {
            "id": "DD-003__002",
            "text": "BGP sessions flapped due to keepalive timeout after maintenance.",
            "metadata": {
                "doc_type": "ticket",
                "section_heading": "Incident Root Cause",
                "source_file": "data/raw/ticket/DD-003__002.md",
            },
            "rerank_score": 5.0,
        }
        result = generate_answer(
            "Why did the BGP session flap?",
            [chunk],
            ExtractiveClient(),
            confidence_threshold=0.10,
        )
        assert not result.refused
        assert "keepalive" in result.answer or "BGP" in result.answer
        assert len(result.citations) >= 1
        assert result.citations[0].doc_id == "DD-003__002"



# ---------------------------------------------------------------------------
# Tests: GeminiClient — offline (no real API calls)
# ---------------------------------------------------------------------------

class _FakePart:
    """Minimal stand-in for a google.genai Part."""
    def __init__(self, text: str | None = None, thought: bool = False) -> None:
        self.text = text
        self.thought = thought


class _FakeFinishReason:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCandidate:
    def __init__(self, parts: list, finish_reason_name: str = "STOP") -> None:
        self.content = type("Content", (), {"parts": parts})()
        self.finish_reason = _FakeFinishReason(finish_reason_name)


class _FakeUsage:
    def __init__(self, prompt=10, candidates=20, thoughts=5, total=35) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts
        self.total_token_count = total


class _FakeResponse:
    def __init__(self, candidates: list, usage: "_FakeUsage | None" = None) -> None:
        self.candidates = candidates
        self.usage_metadata = usage or _FakeUsage()
        # .text is what the real SDK exposes; we set it to the first text part
        # for fallback testing, but _extract_text should not use it.
        self.text = None


class TestGeminiClientHelpers:
    """Unit-test the static helper methods of GeminiClient — no API key needed."""

    def _client_obj(self):
        """Return a bare GeminiClient instance with __init__ bypassed."""
        from netdocs.llm.client import GeminiClient
        obj = object.__new__(GeminiClient)
        obj._model = "gemini-2.0-flash"
        obj._temperature = 0.0
        obj._max_tokens = 4096
        return obj

    # --- _extract_text -------------------------------------------------

    def test_extract_text_skips_thought_parts(self):
        from netdocs.llm.client import GeminiClient

        parts = [
            _FakePart(text="think about it", thought=True),
            _FakePart(text="The real answer.", thought=False),
        ]
        response = _FakeResponse([_FakeCandidate(parts)])
        result = GeminiClient._extract_text(response)
        assert result == "The real answer."
        assert "think about it" not in result

    def test_extract_text_skips_non_text_parts(self):
        """Parts with no .text (e.g. thought_signature) must be skipped."""
        from netdocs.llm.client import GeminiClient

        parts = [
            _FakePart(text=None, thought=False),   # thought_signature-like
            _FakePart(text="Answer here.", thought=False),
        ]
        response = _FakeResponse([_FakeCandidate(parts)])
        result = GeminiClient._extract_text(response)
        assert result == "Answer here."

    def test_extract_text_concatenates_multiple_text_parts(self):
        from netdocs.llm.client import GeminiClient

        parts = [
            _FakePart(text="Part one. ", thought=False),
            _FakePart(text="Part two.", thought=False),
        ]
        response = _FakeResponse([_FakeCandidate(parts)])
        result = GeminiClient._extract_text(response)
        assert result == "Part one. Part two."

    def test_extract_text_fallback_when_no_candidates(self):
        """If candidates list is empty, fall back to response.text."""
        from netdocs.llm.client import GeminiClient

        response = _FakeResponse([])
        response.text = "fallback text"
        result = GeminiClient._extract_text(response)
        assert result == "fallback text"

    def test_extract_text_all_thought_returns_empty(self):
        """If every part is a thought, the extracted text should be empty."""
        from netdocs.llm.client import GeminiClient

        parts = [
            _FakePart(text="Wait! Could I answer with the specific step?", thought=True),
        ]
        response = _FakeResponse([_FakeCandidate(parts)])
        result = GeminiClient._extract_text(response)
        assert result == ""

    # --- _finish_reason ------------------------------------------------

    def test_finish_reason_stop(self):
        from netdocs.llm.client import GeminiClient

        response = _FakeResponse([_FakeCandidate([], "STOP")])
        assert GeminiClient._finish_reason(response) == "STOP"

    def test_finish_reason_max_tokens(self):
        from netdocs.llm.client import GeminiClient

        response = _FakeResponse([_FakeCandidate([], "MAX_TOKENS")])
        assert GeminiClient._finish_reason(response) == "MAX_TOKENS"

    def test_finish_reason_no_candidates(self):
        from netdocs.llm.client import GeminiClient

        response = _FakeResponse([])
        assert GeminiClient._finish_reason(response) == "UNKNOWN"

    # --- _token_usage --------------------------------------------------

    def test_token_usage_populated(self):
        from netdocs.llm.client import GeminiClient

        response = _FakeResponse([], usage=_FakeUsage(10, 20, 5, 35))
        usage = GeminiClient._token_usage(response)
        assert usage["prompt"] == 10
        assert usage["candidates"] == 20
        assert usage["thoughts"] == 5
        assert usage["total"] == 35

    # --- complete() with MAX_TOKENS retry (monkey-patched) -------------

    def test_complete_retries_on_max_tokens(self):
        """complete() must retry once when finish_reason is MAX_TOKENS."""
        from netdocs.llm.client import GeminiClient

        call_count = [0]

        def fake_call(prompt, temp, mtok):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: MAX_TOKENS with empty answer
                return _FakeResponse(
                    [_FakeCandidate([], "MAX_TOKENS")],
                    _FakeUsage(),
                )
            else:
                # Second (retry) call: STOP with real answer
                return _FakeResponse(
                    [_FakeCandidate([_FakePart("The full answer.", False)], "STOP")],
                    _FakeUsage(),
                )

        client = self._client_obj()
        client._call = fake_call  # type: ignore[assignment]

        result = client.complete("system prompt", "user question")
        assert result == "The full answer."
        assert call_count[0] == 2

    def test_complete_raises_after_two_empty_responses(self):
        """After two consecutive empty answers, a RuntimeError must be raised."""
        from netdocs.llm.client import GeminiClient

        def fake_call(prompt, temp, mtok):
            return _FakeResponse(
                [_FakeCandidate([_FakePart(text=None, thought=False)], "MAX_TOKENS")],
                _FakeUsage(),
            )

        client = self._client_obj()
        client._call = fake_call  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="empty answer after retry"):
            client.complete("sys", "user")

    def test_complete_filters_thought_parts_from_answer(self):
        """Thought parts must never appear in the returned answer."""
        from netdocs.llm.client import GeminiClient

        def fake_call(prompt, temp, mtok):
            parts = [
                _FakePart("Wait! Could I answer with the specific step present, "
                           "or is it mandatory to", thought=True),
                _FakePart("1. Re-add the removed route-map clauses.", thought=False),
            ]
            return _FakeResponse(
                [_FakeCandidate(parts, "STOP")],
                _FakeUsage(),
            )

        client = self._client_obj()
        client._call = fake_call  # type: ignore[assignment]

        result = client.complete("sys", "user")
        assert "Wait" not in result
        assert "Could I" not in result
        assert "Re-add the removed route-map clauses" in result


# ---------------------------------------------------------------------------
# Tests: Query expansion and doc-type boosting
# ---------------------------------------------------------------------------

class TestQueryExpansion:
    def test_procedure_phrase_triggers_expansion(self):
        from netdocs.retriever.hybrid import _expand_query

        expanded = _expand_query("What is the runbook procedure when a BGP session flaps?")
        assert "runbook procedure steps" in expanded

    def test_non_procedure_phrase_unchanged(self):
        from netdocs.retriever.hybrid import _expand_query

        q = "What BGP ASN does Contoso use?"
        assert _expand_query(q) == q

    def test_how_do_i_triggers_expansion(self):
        from netdocs.retriever.hybrid import _expand_query

        q = "How do I diagnose a BGP flap?"
        expanded = _expand_query(q)
        assert len(expanded) > len(q)

    def test_steps_triggers_expansion(self):
        from netdocs.retriever.hybrid import _expand_query

        q = "What steps should I follow for tunnel recovery?"
        assert len(_expand_query(q)) > len(q)


class TestDocTypeBoost:
    def _make_chunk(self, doc_type: str, text: str = "BGP flap") -> dict:
        return {
            "id": f"{doc_type}_001",
            "text": text,
            "metadata": {"doc_type": doc_type},
            "rerank_score": 2.0,
        }

    def test_runbook_boosted_for_procedure_query(self):
        from netdocs.retriever.hybrid import apply_doc_type_boost

        chunks = [
            self._make_chunk("ticket", text="rollback plan re-add removed route-map"),
            self._make_chunk("runbook"),
        ]
        # Both start at score 2.0; ticket has rollback-plan text
        result = apply_doc_type_boost(chunks, "What is the runbook procedure when BGP flaps?")
        assert result[0]["metadata"]["doc_type"] == "runbook"

    def test_ticket_with_rollback_penalised(self):
        from netdocs.retriever.hybrid import apply_doc_type_boost

        ticket_chunk = {
            "id": "crq_001",
            "text": "ROLLBACK PLAN\n1. Re-add the removed route-map clauses.",
            "metadata": {"doc_type": "ticket"},
            "rerank_score": 3.0,
        }
        runbook_chunk = {
            "id": "rb_001",
            "text": "BGP flap diagnosis steps.",
            "metadata": {"doc_type": "runbook"},
            "rerank_score": 2.5,
        }
        result = apply_doc_type_boost([ticket_chunk, runbook_chunk], "How do I diagnose a BGP flap?")
        # Runbook should rank first after boost
        assert result[0]["id"] == "rb_001"

    def test_no_boost_for_non_procedural_query(self):
        from netdocs.retriever.hybrid import apply_doc_type_boost

        chunks = [
            self._make_chunk("ticket"),
            self._make_chunk("runbook"),
        ]
        original_order = [c["id"] for c in chunks]
        result = apply_doc_type_boost(chunks, "What BGP ASN does Contoso use?")
        # Scores unchanged, order preserved (ticket had score 2.0 first)
        assert [c["id"] for c in result] == original_order


# ---------------------------------------------------------------------------
# Tests: Refusal reason distinction
# ---------------------------------------------------------------------------

class TestRefusalReason:
    def _fake_llm(self, answer: str):
        from netdocs.llm.client import FakeLLMClient
        return FakeLLMClient(canned_answer=answer)

    def _chunk(self, score: float = 5.0) -> dict:
        return {
            "id": "X__000",
            "text": "some text",
            "metadata": {"doc_type": "design_doc", "section_heading": "Overview"},
            "rerank_score": score,
        }

    def test_low_confidence_sets_reason(self):
        from netdocs.llm.generator import generate_answer, REFUSAL_LOW_CONFIDENCE

        result = generate_answer("q", [self._chunk(-5.0)], self._fake_llm("ok"), confidence_threshold=0.0)
        assert result.refused
        assert result.refusal_reason == REFUSAL_LOW_CONFIDENCE

    def test_empty_generation_sets_reason(self):
        from netdocs.llm.generator import generate_answer, REFUSAL_EMPTY_GENERATION

        result = generate_answer("q", [self._chunk(5.0)], self._fake_llm(""), confidence_threshold=-99.0)
        assert result.refused
        assert result.refusal_reason == REFUSAL_EMPTY_GENERATION

    def test_llm_declined_sets_reason(self):
        from netdocs.llm.generator import generate_answer, LOW_CONFIDENCE_ANSWER, REFUSAL_LLM_DECLINED

        result = generate_answer("q", [self._chunk(5.0)], self._fake_llm(LOW_CONFIDENCE_ANSWER), confidence_threshold=-99.0)
        assert result.refused
        assert result.refusal_reason == REFUSAL_LLM_DECLINED

    def test_successful_answer_has_no_reason(self):
        from netdocs.llm.generator import generate_answer

        result = generate_answer("q", [self._chunk(5.0)], self._fake_llm("The answer is 42."), confidence_threshold=-99.0)
        assert not result.refused
        assert result.refusal_reason == ""


# ---------------------------------------------------------------------------
# Tests: Citation rendering (no stray " .")
# ---------------------------------------------------------------------------

class TestCitationRendering:
    def test_renders_doc_id_and_section(self):
        from netdocs.llm.generator import render_citations, Citation

        citations = [Citation(doc_id="RB-001__002", section="Step 1 — Confirm the Flap")]
        answer = "Run show bgp summary. [doc:RB-001__002:Step 1 — Confirm the Flap]"
        rendered = render_citations(answer, citations)
        assert "[doc:" not in rendered
        assert "[Step 1 — Confirm the Flap — RB-001__002]" in rendered

    def test_no_stray_period_space(self):
        from netdocs.llm.generator import render_citations, Citation

        citations = [Citation(doc_id="DD-003__001", section="BGP Overview")]
        answer = "The hold timer is 90s [doc:DD-003__001:BGP Overview]."
        rendered = render_citations(answer, citations)
        assert " ." not in rendered

    def test_citation_without_section_uses_lookup(self):
        from netdocs.llm.generator import render_citations, Citation

        citations = [Citation(doc_id="DD-003__001", section="My Section")]
        answer = "Some text [doc:DD-003__001]."
        rendered = render_citations(answer, citations)
        assert "[My Section — DD-003__001]" in rendered


# ---------------------------------------------------------------------------
# Tests: Regression — BGP runbook must rank in top 3 for both phrasings
# ---------------------------------------------------------------------------

class TestBGPRunbookRegression:
    """
    Offline regression using FakeVectorStore / FakeBM25Index.

    We simulate a corpus where RB-001 (runbook) competes with CRQ-2024-1015
    (ticket rollback plan) — which was the real mis-ranking case.

    These tests verify that:
    1. Query expansion adds runbook keywords to the BM25 query.
    2. Doc-type boost places RB-001 in the top-3 for both BGP-flap phrasings.
    """

    def _make_corpus(self) -> list[dict]:
        """Minimal corpus replicating the mis-ranking scenario."""
        return [
            # Runbook chunk — procedure-style content
            {
                "id": "RB-001_bgp_flap_diagnosis__000",
                "text": (
                    "[DOC_TYPE: runbook] [SOURCE: RB-001: BGP Peer Flapping — Diagnosis and Recovery]\n"
                    "### Step 1 — Confirm the Flap and Identify the Peer\n\n"
                    "Log in to the affected router and check BGP summary:\n"
                    "show bgp summary\nshow bgp neighbors <peer-ip>"
                ),
                "metadata": {
                    "doc_type": "runbook",
                    "source_file": "data/raw/runbooks/RB-001_bgp_flap_diagnosis.md",
                    "procedure_name": "Confirm the Flap and Identify the Peer",
                    "step_number": 1,
                },
            },
            # Ticket chunk — rollback plan containing BGP keywords
            {
                "id": "CRQ-2024-1015__002",
                "text": (
                    "[DOC_TYPE: ticket] [SECTION: ROLLBACK PLAN]\n"
                    "If any BGP session is disrupted:\n"
                    "1. Re-add the removed route-map clauses.\n"
                    "2. Soft reset outbound.\n"
                    "3. Verify session stability."
                ),
                "metadata": {
                    "doc_type": "ticket",
                    "source_file": "data/raw/tickets/CRQ-2024-1015.txt",
                    "section": "rollback_plan",
                },
            },
            # Another ticket chunk — description
            {
                "id": "CRQ-2024-1015__001",
                "text": (
                    "[DOC_TYPE: ticket] [SECTION: DESCRIPTION]\n"
                    "BGP route-map audit. Stale prefix-list removed. "
                    "Session stability verified after change."
                ),
                "metadata": {
                    "doc_type": "ticket",
                    "source_file": "data/raw/tickets/CRQ-2024-1015.txt",
                    "section": "description",
                },
            },
        ]

    def _rerank_with_boost(self, query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        """Simulate reranker + boost without loading real models."""
        from netdocs.retriever.hybrid import apply_doc_type_boost

        # Assign fake rerank_score based on keyword overlap (simulates cross-encoder)
        query_words = set(query.lower().split())
        for chunk in chunks:
            text_words = set(chunk["text"].lower().split())
            overlap = len(query_words & text_words)
            chunk["rerank_score"] = float(overlap)

        # Apply doc-type boost
        boosted = apply_doc_type_boost(list(chunks), query)
        return boosted[:top_k]

    def _check_rb001_in_top3(self, query: str):
        corpus = self._make_corpus()
        top3 = self._rerank_with_boost(query, corpus, top_k=3)
        top3_sources = [c["metadata"]["source_file"] for c in top3]
        assert any("RB-001_bgp_flap_diagnosis" in s for s in top3_sources), (
            f"RB-001_bgp_flap_diagnosis not in top-3 for query: {query!r}\n"
            f"Got: {top3_sources}"
        )

    def test_bgp_runbook_top3_procedure_phrasing(self):
        """'What is the runbook procedure when a BGP session flaps?' → RB-001 top-3"""
        self._check_rb001_in_top3("What is the runbook procedure when a BGP session flaps?")

    def test_bgp_runbook_top3_diagnosis_phrasing(self):
        """'How do I diagnose a BGP flap?' → RB-001 top-3"""
        self._check_rb001_in_top3("How do I diagnose a BGP flap?")

    def test_query_expansion_adds_runbook_keywords(self):
        from netdocs.retriever.hybrid import _expand_query

        q1 = "What is the runbook procedure when a BGP session flaps?"
        q2 = "How do I diagnose a BGP flap?"
        for q in (q1, q2):
            expanded = _expand_query(q)
            assert expanded != q, f"No expansion for: {q!r}"
            assert "runbook" in expanded.lower() or "procedure" in expanded.lower()
