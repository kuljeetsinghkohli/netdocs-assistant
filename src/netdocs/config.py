"""
NetDocs configuration — loaded from environment / .env file.

All application-wide constants live here.  Import ``settings`` from this
module; never read ``os.environ`` directly elsewhere.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings resolved from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NETDOCS_",
        extra="ignore",
        # Allow OPENAI_API_KEY alias (no NETDOCS_ prefix)
        populate_by_name=True,
    )

    # --- Embedding backend ------------------------------------------------
    embed_backend: str = Field(
        default="sentence-transformers",
        description="'openai' or 'sentence-transformers'",
    )
    embed_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Sentence-transformers model name.",
    )
    openai_embed_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model (only used when embed_backend='openai').",
    )

    # --- Vector store -----------------------------------------------------
    chroma_path: Path = Field(
        default=Path("chroma_db"),
        description="Persistent ChromaDB directory.",
    )
    chroma_collection: str = Field(
        default="netdocs",
        description="ChromaDB collection name.",
    )

    # --- Ingestion --------------------------------------------------------
    data_dir: Path = Field(
        default=Path("data/raw"),
        description="Root directory for raw source documents.",
    )

    chunk_size_prose: int = Field(default=400)
    chunk_overlap_prose: int = Field(default=50)
    chunk_size_runbook: int = Field(default=300)
    chunk_size_ticket: int = Field(default=250)
    chunk_size_config: int = Field(default=600)

    # --- LLM --------------------------------------------------------------
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai' | 'ollama' | 'fake' (testing only).",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name passed to the LLM provider.",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Sampling temperature (0 = deterministic).",
    )
    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens in the LLM completion.",
    )
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for Ollama / custom OpenAI-compatible endpoint.",
    )
    # Read as OPENAI_API_KEY (no NETDOCS_ prefix) via alias
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
        description="OpenAI API key (required when llm_provider='openai').",
    )

    # --- Retrieval --------------------------------------------------------
    retrieval_top_k: int = Field(
        default=5,
        description="Number of final chunks returned to the generator.",
    )
    retrieval_candidate_k: int = Field(
        default=20,
        description="Candidates fetched before reranking.",
    )
    retrieval_confidence_threshold: float = Field(
        default=0.10,
        description=(
            "Minimum reranker score for the top result. "
            "Answers are refused when all results are below this value."
        ),
    )
    bm25_k1: float = Field(default=1.5)
    bm25_b: float = Field(default=0.75)
    rrf_k: int = Field(default=60, description="RRF constant k.")

    # --- API server -------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Logging ----------------------------------------------------------
    log_level: str = Field(default="INFO")


# Module-level singleton — import this everywhere.
settings = Settings()
