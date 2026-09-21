.PHONY: ingest eval serve lint test

# ---- Ingestion -------------------------------------------------------
ingest:
	python -m netdocs ingest

# ---- Evaluation (retrieval metrics only — no LLM calls) ---------------
eval:
	python -m netdocs.eval.retrieval_bench

# ---- API server -------------------------------------------------------
serve:
	uvicorn netdocs.api.main:app --reload --port 8000

# ---- Streamlit UI -----------------------------------------------------
ui:
	streamlit run app.py

# ---- Tests ------------------------------------------------------------
test:
	pytest tests/ -v --cov=src/netdocs --cov-report=term-missing

# ---- Lint -------------------------------------------------------------
lint:
	python -m py_compile src/netdocs/**/*.py && echo "Syntax OK"
