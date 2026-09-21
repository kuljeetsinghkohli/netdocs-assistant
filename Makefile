.PHONY: ingest eval serve api ui lint test

# ---- Ingestion -------------------------------------------------------
ingest:
	python -m netdocs ingest

# ---- Evaluation (retrieval metrics only — no LLM calls) ---------------
eval:
	python -m netdocs.eval.retrieval_bench

# ---- API server -------------------------------------------------------
#  `api` is the canonical target; `serve` is kept as an alias.
api:
	uvicorn netdocs.api.main:app --reload --port 8000

serve: api

# ---- Streamlit UI -----------------------------------------------------
#  Reads NETDOCS_API_URL from the environment (default: http://localhost:8000).
ui:
	streamlit run src/netdocs/ui/app.py

# ---- Tests ------------------------------------------------------------
test:
	pytest tests/ -v --cov=src/netdocs --cov-report=term-missing

# ---- Lint -------------------------------------------------------------
lint:
	python -m py_compile src/netdocs/**/*.py && echo "Syntax OK"
