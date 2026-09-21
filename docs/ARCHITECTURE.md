# NetDocs Assistant — Architecture

> **Status:** Approved design baseline  
> **Scope:** RAG-based Q&A assistant over synthetic enterprise network documentation  
> **Document types in scope:** Cisco SD-WAN/Viptela design docs, BGP/OSPF policy docs, runbooks, change tickets, device configs

---

## 1. Users

| Persona | Description | Primary Need |
|---|---|---|
| **Network Engineer** | Day-to-day operator, owns device configs and runbooks | "Why is this BGP peer down? What does the runbook say?" |
| **Solutions Architect** | Designs topology, writes policy docs | "What SD-WAN policies apply to site X? What changed last quarter?" |
| **NOC Analyst** | Monitors alerts, works change tickets | "Is there an open ticket for this device? What's the approved change window?" |
| **Auditor / Compliance** | Reviews configs for policy adherence | "Show all configs where BGP MED is overridden" |

---

## 2. Functional Requirements

### Ingestion Pipeline
- FR-1: Ingest documents from a local directory (initial) and an S3 bucket (later).
- FR-2: Parse and chunk each document type with a type-appropriate strategy (see §4).
- FR-3: Attach rich metadata to every chunk: `doc_type`, `source_file`, `device_name`, `site_id`, `date`, `ticket_id`, `section_heading`.
- FR-4: Embed chunks and upsert into a vector store with idempotent doc IDs (re-ingest = update, not duplicate).
- FR-5: Support incremental ingestion (only process new/changed files).

### Retrieval
- FR-6: Hybrid search: dense vector similarity + BM25 keyword search, fused via Reciprocal Rank Fusion (RRF).
- FR-7: Metadata pre-filtering before vector search (e.g., filter by `doc_type=config`, `site_id=LON-01`).
- FR-8: Cross-encoder reranking of the top-K candidates before answer generation.

### Answer Generation
- FR-9: Answers must cite the source chunk(s): filename, section, and page/line where available.
- FR-10: The LLM must be instructed to say "I don't know" rather than hallucinate when evidence is absent.
- FR-11: Support multi-turn conversation with session memory (last N turns in context).

### Agent Layer
- FR-12: An agent can invoke tools beyond retrieval: `search_docs`, `lookup_device_config`, `get_open_tickets`, `diff_configs`.
- FR-13: The agent decides whether a question needs a single retrieval call or a multi-step plan (ReAct loop).

### Evaluation
- FR-14: An offline evaluation harness runs a golden Q&A dataset and reports: faithfulness, answer relevance, context recall (RAGAS metrics).
- FR-15: A retrieval-only benchmark reports MRR@10 and Recall@10 against labelled queries.

### API & UI
- FR-16: A REST API (`/chat`, `/ingest`, `/health`) accepts and returns JSON.
- FR-17: A minimal Streamlit UI for demo and exploratory use.

---

## 3. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | End-to-end query latency ≤ 5 s (p95) on a single GPU or fast CPU inference |
| NFR-2 | Ingestion of 500 synthetic documents completes in < 10 minutes |
| NFR-3 | Vector store supports at least 100 k chunks without degraded recall |
| NFR-4 | All secrets (API keys, credentials) loaded from `.env`, never committed |
| NFR-5 | The system runs fully offline (local LLM + local embeddings) as a fallback |
| NFR-6 | Reproducible environment: `pyproject.toml` + `uv` lockfile |
| NFR-7 | Evaluation suite is deterministic and can be run in CI |

---

## 4. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Chunking config files naively loses structural context (interface block split across chunks) | High | High | Use structure-aware parsing for configs (§4 chunking strategy) |
| LLM hallucination on policy details | Medium | High | Strict system prompt + citation enforcement + eval harness |
| Embedding model doesn't understand network jargon (BGP, OSPF, vEdge) | Medium | Medium | Evaluate domain-specific vs general embeddings; include keyword search as fallback |
| Retrieval returns irrelevant chunks from wrong doc type | Medium | Medium | Mandatory metadata pre-filter on `doc_type` |
| Evaluation golden set becomes stale as docs evolve | Low | Medium | Pin eval dataset to a specific corpus version hash |
| Over-engineered agent adds latency with no benefit for simple queries | Medium | Low | Default to single-shot RAG; only escalate to agent on explicit tool triggers |

---

## 5. Architecture

### 5.1 High-Level Data Flow

```
Documents (local/S3)
        │
        ▼
┌──────────────────┐
│  Ingestion       │  parse → chunk → embed → upsert
│  Pipeline        │
└────────┬─────────┘
         │ chunks + metadata
         ▼
┌──────────────────┐
│  Vector Store    │  ChromaDB (local) / Qdrant (prod)
│  + BM25 Index    │  tantivy / rank_bm25
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  Retrieval Layer                         │
│  1. Metadata pre-filter                  │
│  2. Dense ANN search  ─┐                 │
│  3. BM25 keyword search ├─ RRF fusion    │
│  4. Cross-encoder rerank                 │
└────────┬─────────────────────────────────┘
         │ top-N reranked chunks
         ▼
┌──────────────────┐
│  Agent / RAG     │  ReAct loop (optional)
│  Orchestrator    │  tools: search_docs, lookup_config,
│  (LangGraph)     │         get_tickets, diff_configs
└────────┬─────────┘
         │ prompt + context
         ▼
┌──────────────────┐
│  LLM             │  OpenAI GPT-4o (default)
│                  │  or local Ollama (fallback)
└────────┬─────────┘
         │ answer + citations
         ▼
┌──────────────────┐     ┌──────────────────┐
│  FastAPI         │     │  Streamlit UI    │
│  REST API        │     │  (demo/explore)  │
└──────────────────┘     └──────────────────┘
         │
         ▼
┌──────────────────┐
│  Eval Harness    │  RAGAS + custom retrieval metrics
│  (offline)       │
└──────────────────┘
```

### 5.2 Chunking Strategy by Document Type

Each document type has different information density and structural patterns. A single strategy will not work.

#### Device Configs (IOS-XE, vEdge, YANG)
- **Parse** with a config-aware parser (ciscoconfparse or custom regex) to extract logical blocks: `interface`, `router bgp`, `policy-map`, `vrf`, `template`.
- **Chunk** = one logical block. Never split a block mid-way.
- **Metadata**: `device_name`, `hostname`, `block_type` (interface/bgp/policy), `interface_name`.
- **Overlap**: none — config blocks are self-contained. Adjacent blocks are linked via `parent_block` metadata.
- **Target size**: 200–600 tokens per block.

#### Prose Documents (Design Docs, Policy Docs)
- **Parse** with `unstructured` or `pypdf` to extract headings and paragraphs.
- **Chunk** with a recursive character splitter respecting heading boundaries first, then paragraph, then sentence.
- **Overlap**: 10–15% overlap (e.g., 50 tokens on a 400-token chunk) to preserve cross-sentence context.
- **Metadata**: `section_heading`, `subsection`, `page_number`, `document_title`.
- **Target size**: 350–500 tokens.

#### Runbooks
- Runbooks are structured prose with numbered steps and code blocks.
- **Parse** heading hierarchy; treat each top-level procedure as a "document unit".
- **Chunk** = one procedure step or a group of 2–3 short steps. Keep code blocks (CLI commands) intact — never split a code fence.
- **Metadata**: `procedure_name`, `step_number`, `device_role`.
- **Target size**: 250–400 tokens.

#### Change Tickets
- Structured key-value header (ticket ID, date, approver, CI affected) + free-text description/resolution.
- **Parse** header fields into metadata; chunk only the free-text body.
- **Metadata**: `ticket_id`, `status`, `affected_device`, `change_date`, `approver`.
- **Chunk** the description and resolution separately (two chunks per ticket).
- **Target size**: 200–350 tokens (tickets are short).

### 5.3 Embeddings

| Option | Model | Notes |
|---|---|---|
| **Default** | `text-embedding-3-small` (OpenAI) | Best recall/cost ratio for English prose |
| **Fallback / offline** | `BAAI/bge-base-en-v1.5` via sentence-transformers | Runs on CPU, strong on technical text |
| **Experiment** | `nomic-embed-text` via Ollama | Fully local, good for structured text |

Dimension: 768 (bge) or 1536 (OpenAI). Stored in ChromaDB with cosine distance.

### 5.4 Vector Store

- **Development**: ChromaDB (persistent, local, zero-ops).
- **Production path**: Qdrant (supports payload filtering, scalar quantisation, on-disk index).
- An abstraction layer (`retriever/vector_store.py`) isolates the backend so switching is a config change.

### 5.5 Retrieval: Hybrid Search + Reranking

```
Query
  ├─► Dense ANN (ChromaDB)  → top-50 chunks
  ├─► BM25 (rank_bm25)      → top-50 chunks
  └─► RRF fusion            → top-20 merged
         └─► Cross-encoder rerank (ms-marco-MiniLM-L-6-v2) → top-5 final context
```

- RRF formula: `score(d) = Σ 1 / (k + rank_i(d))`, k=60.
- Reranker keeps latency manageable (~50 ms on CPU for 20 candidates).
- Metadata pre-filter applied before both searches (e.g., `doc_type IN [config, runbook]`).

### 5.6 Answer Generation with Citations

- System prompt instructs the LLM to:
  1. Answer using **only** the provided context chunks.
  2. Cite every factual claim with `[Source: <filename>, <section>]`.
  3. Respond with "I cannot find this in the available documentation" when context is insufficient.
- The response schema is a Pydantic model: `{ answer: str, citations: List[Citation], confidence: float }`.
- Citations are rendered as clickable links in the UI.

### 5.7 Agent Layer

- Implemented with **LangGraph** (lightweight, inspectable, avoids LangChain's abstraction overhead).
- **Default path**: single `search_docs` tool call (standard RAG). No multi-step planning for simple questions.
- **Agent escalation triggers**: question contains comparative language ("compare", "diff", "changed since"), or references multiple devices/sites, or asks for open tickets.
- **Tools**:

  | Tool | Input | Output |
  |---|---|---|
  | `search_docs` | query, filters | ranked chunks + citations |
  | `lookup_device_config` | device_name, block_type | raw config block |
  | `get_open_tickets` | device_name or site_id | list of ticket summaries |
  | `diff_configs` | device_a, device_b, block_type | unified diff string |

- Agent state is a simple dict passed through the LangGraph node chain. No persistent agent memory beyond the current session window.

### 5.8 Evaluation Harness

- **Tool**: RAGAS (faithfulness, answer relevance, context recall, context precision).
- **Golden dataset**: `eval/golden_qa.json` — 50 hand-crafted Q&A pairs with labelled source chunks.
- **Retrieval benchmark**: `eval/retrieval_bench.py` — queries against labelled docs, reports MRR@10 and Recall@10.
- **Runner**: `make eval` triggers `eval/run_eval.py`, writes results to `eval/results/`.
- CI integration: GitHub Actions runs eval on push to `main`; fails if faithfulness drops below 0.80.

### 5.9 API

```
POST /chat          { session_id, message, filters? }  →  { answer, citations, session_id }
POST /ingest        { source_path, doc_type? }         →  { chunks_added, chunks_updated }
GET  /health        →  { status, vector_store_count }
```

- **FastAPI** with Pydantic v2 request/response models.
- Session state stored in-process dict (dev) / Redis (prod).

### 5.10 UI

- **Streamlit** — single `app.py`. Not a production UI; exists for demos and exploratory testing.
- Features: chat interface, source document expander per citation, sidebar filters (doc_type, site_id).

---

## 6. Folder Structure

```
netdocs-assistant/
├── docs/
│   └── ARCHITECTURE.md
├── data/
│   └── raw/                   # synthetic source documents (gitignored bulk)
│       ├── configs/
│       ├── design_docs/
│       ├── runbooks/
│       └── tickets/
├── src/
│   └── netdocs/
│       ├── __init__.py
│       ├── config.py            # pydantic-settings: loads .env, all config constants
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loader.py        # file discovery, dispatch by doc_type
│       │   ├── parsers/
│       │   │   ├── config_parser.py    # ciscoconfparse / block extractor
│       │   │   ├── prose_parser.py     # unstructured / pypdf
│       │   │   ├── runbook_parser.py   # heading + step extractor
│       │   │   └── ticket_parser.py    # key-value header + body splitter
│       │   └── chunker.py       # type-dispatched chunking + metadata assembly
│       │
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── embedder.py      # abstraction over OpenAI / sentence-transformers
│       │
│       ├── retriever/
│       │   ├── __init__.py
│       │   ├── vector_store.py  # ChromaDB wrapper (swappable to Qdrant)
│       │   ├── bm25_index.py    # rank_bm25 index build + query
│       │   ├── hybrid.py        # RRF fusion logic
│       │   └── reranker.py      # cross-encoder reranker wrapper
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── graph.py         # LangGraph state machine definition
│       │   ├── tools.py         # tool implementations
│       │   └── prompts.py       # system prompt templates
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py          # FastAPI app + lifespan
│       │   ├── routes.py        # /chat, /ingest, /health
│       │   └── schemas.py       # Pydantic request/response models
│       │
│       └── eval/
│           ├── __init__.py
│           ├── run_eval.py      # RAGAS pipeline runner
│           ├── retrieval_bench.py
│           └── golden_qa.json   # labelled Q&A pairs
│
├── app.py                       # Streamlit entry point
├── pyproject.toml
├── uv.lock
├── Makefile                     # make ingest / make eval / make serve
└── .env.example
```

**Module boundaries (strict):**
- `ingestion` has no knowledge of `retriever` or `agent`. It produces `Chunk` objects.
- `retriever` has no knowledge of `agent`. It accepts a query + filters and returns ranked `Chunk` objects.
- `agent` depends on `retriever` and `embeddings` only through the tool interfaces in `tools.py`.
- `api` depends on `agent` only. It never calls retriever or embeddings directly.
- `eval` is a standalone harness; it calls the API over HTTP, not internal modules.

---

## 7. 3-Day Milestone Plan

### Day 1 — Foundation: Ingestion + Retrieval
**Goal:** Data flows from raw files to queryable vector store.

- Set up `pyproject.toml` with all dependencies; confirm environment loads.
- Implement `config.py` with pydantic-settings.
- Implement all four parsers (`config_parser`, `prose_parser`, `runbook_parser`, `ticket_parser`).
- Implement `chunker.py` with type dispatch and metadata assembly.
- Implement `embedder.py` (OpenAI primary, bge fallback).
- Implement `vector_store.py` (ChromaDB) and `bm25_index.py`.
- Run ingestion on the full synthetic corpus; verify chunk counts and metadata.
- **Exit criterion:** `make ingest` completes cleanly; ChromaDB contains all expected chunks with correct metadata.

### Day 2 — Retrieval Quality + Agent + API
**Goal:** Retrieval pipeline is tuned; agent answers questions via API.

- Implement `hybrid.py` (RRF) and `reranker.py`.
- Manually spot-check 10 queries: verify top-5 context is relevant.
- Implement `agent/graph.py`, `tools.py`, `prompts.py`.
- Implement FastAPI `routes.py` and `schemas.py`.
- Write smoke tests: `pytest tests/smoke/` covering `/chat` and `/health`.
- **Exit criterion:** `POST /chat` returns a grounded answer with citations for a set of 5 representative queries.

### Day 3 — Evaluation + UI + Polish
**Goal:** Measurable quality baseline; demo-ready UI.

- Author `eval/golden_qa.json` (50 Q&A pairs).
- Implement `run_eval.py`; run RAGAS; establish baseline scores.
- Implement `retrieval_bench.py`; establish MRR@10 baseline.
- Build `app.py` Streamlit UI (chat + citation expander + sidebar filters).
- Write `Makefile` targets: `make ingest`, `make eval`, `make serve`.
- Write `.env.example`.
- **Exit criterion:** `make eval` produces a results file; Streamlit UI loads and answers a question end-to-end.

---

## 8. Principal Engineer Critique

### What will break

1. **Config parser brittleness.** `ciscoconfparse` handles IOS-XE reasonably but will choke on YANG/XML configs and non-standard vEdge indentation. A single regex-based fallback is needed from day 1, not later.

2. **BM25 index is in-memory and ephemeral.** `rank_bm25` builds the index in RAM at startup from all stored chunks. At 100 k chunks this is ~200 MB and takes ~10 s. Acceptable for a prototype; becomes a cold-start problem in production. Plan to serialise the index to disk.

3. **Session memory is a global dict.** The in-process session store will silently lose all history on server restart and does not scale past one process. Fine for a demo; must be replaced before any real use.

4. **Golden eval set is hand-crafted bias.** 50 Q&A pairs written by the same engineer who built the system will overfit to the retrieval patterns already in the system. An independent reviewer should author at least half the eval set.

5. **Reranker adds ~50–150 ms per query.** On a CPU-only machine with 20 candidates this is acceptable. But if the BM25 or ANN stages return noisy results and the candidate set grows, latency will blow past NFR-1. Cap candidates hard at 20 before reranking.

### What is over-engineered for a 3-day prototype

1. **The agent / ReAct layer on Day 1.** The value of a multi-step agent is unproven until retrieval quality is validated. The agent adds complexity (state machine, tool routing, extra LLM calls) before the simpler RAG path is even tested. **Cut:** implement single-shot RAG first; add the agent layer only after eval baselines exist.

2. **`diff_configs` tool.** Computing a diff requires two device configs to already be in the store with consistent metadata keys. This is fiddly to get right and is a low-frequency query. **Cut for Day 1–2;** add in a follow-up sprint.

3. **Qdrant migration abstraction.** Spending time on a clean `vector_store.py` abstraction that swaps between ChromaDB and Qdrant is premature. ChromaDB's API is simple; abstract only when migration is actually planned.

4. **RAGAS requires an LLM to judge answers** (it calls GPT-4 internally by default). Running full RAGAS in CI on every push will cost money and slow the loop. **Cut:** use retrieval-only metrics (MRR@10, Recall@10) in CI; run RAGAS manually or on a schedule.

### What should be cut entirely from the first version

- Redis session store (use in-process dict until needed)
- S3 ingestion source (use local directory only)
- `diff_configs` tool
- Per-user auth / API keys
- Streaming LLM responses (adds websocket complexity; polling is fine for now)

---

## 9. Revised Plan (Post-Critique)

The following changes are applied to the baseline above:

| Change | Rationale |
|---|---|
| Agent layer moved to Day 2 (after retrieval spot-check, not before) | Retrieval quality must be validated before adding agent complexity |
| `diff_configs` tool deferred to post-sprint backlog | Low-frequency, high-implementation-cost |
| `vector_store.py` abstraction simplified: only ChromaDB, no Qdrant shim | YAGNI; abstract when migration is real |
| CI eval uses retrieval metrics only (MRR@10, Recall@10); RAGAS is manual | Cost and speed |
| BM25 index serialised to disk on first build | Avoids 10 s cold-start penalty |
| Config parser includes a regex fallback path for non-IOS-XE formats | Prevents hard failure on YANG/vEdge configs |
| Session store is explicitly documented as dev-only in-process dict | Prevents future misuse |

The folder structure, module boundaries, chunking strategies, retrieval pipeline, and milestone plan in §3–§7 reflect this revised scope.

---

*Generated by NetDocs Assistant architecture session.*
