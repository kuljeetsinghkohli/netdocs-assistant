# NetDocs Assistant

RAG-based Q&A assistant over synthetic enterprise network documentation (Cisco SD-WAN/Viptela design docs, BGP/OSPF policies, runbooks, change tickets, and device configs).

---

## Quick start

### Prerequisites

- Python 3.9+
- [`uv`](https://docs.astral.sh/uv/) (or plain `pip`)
- An OpenAI API key — set `OPENAI_API_KEY` in `.env` (see [`.env.example`](.env.example))

### Install dependencies

```bash
# with uv (recommended)
uv sync --extra dev

# or with pip
pip install -e ".[dev]"
```

### Ingest documents

```bash
make ingest
```

This parses all documents in `data/raw/`, chunks them, embeds them, and loads them into the local ChromaDB vector store.

---

## Running the API and UI

The API and UI are separate processes. Open two terminal tabs.

### 1 — Start the FastAPI service

```bash
make api
```

This starts `uvicorn` on **http://localhost:8000** with `--reload` enabled.

- Interactive docs (Swagger): http://localhost:8000/docs
- Health probe: http://localhost:8000/health

To use a different port, override the command:

```bash
uvicorn netdocs.api.main:app --reload --port 9000
```

### 2 — Start the Streamlit UI

```bash
make ui
```

This runs `streamlit run src/netdocs/ui/app.py`. Streamlit opens a browser tab automatically (default: **http://localhost:8501**).

If the API is on a non-default URL, set the environment variable before running:

```bash
NETDOCS_API_URL=http://localhost:9000 make ui
```

You can also change the API URL at any time in the sidebar of the running UI.

---

## UI features

| Feature | Description |
|---|---|
| **Chat panel** | Multi-turn conversation; each answer shows expandable source citation snippets |
| **Ask docs / Agent toggle** | Switch between single-shot RAG and the multi-step agent tool-calling loop |
| **Sidebar filters** | Filter by document type, site ID, and date range before each query |
| **Agent trace** | Toggle to show every tool call and its result for agent-mode responses |
| **API status panel** | Live reachability indicator and indexed-chunk count in the sidebar |
| **Degraded banner** | Prominent warning when the API returns a degraded response |
| **AI disclaimer footer** | Persistent reminder that answers must be verified against source documents |

---

## Other Makefile targets

| Target | Description |
|---|---|
| `make ingest` | Ingest `data/raw/` into the vector store |
| `make api` | Start the FastAPI service (alias: `make serve`) |
| `make ui` | Start the Streamlit UI |
| `make test` | Run pytest with coverage |
| `make eval` | Run retrieval benchmarks (MRR@10, Recall@10) — no LLM calls |
| `make lint` | Syntax-check all Python source files |

---

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design, including the ingestion pipeline, hybrid retrieval strategy, agent layer, and evaluation harness.

---

## Environment variables

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for GPT-4o answer generation |
| `GEMINI_API_KEY` | — | Optional Gemini fallback |
| `NETDOCS_LLM_PROVIDER` | `openai` | `openai`, `gemini`, `extractive`, or `fake` |
| `NETDOCS_API_URL` | `http://localhost:8000` | UI → API base URL |

---

*Answers produced by this tool are AI-generated and must be verified against authoritative source documents before acting on them.*
