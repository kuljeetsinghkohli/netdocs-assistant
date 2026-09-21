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

    # --- Logging ----------------------------------------------------------
    log_level: str = Field(default="INFO")


# Module-level singleton — import this everywhere.
settings = Settings()
