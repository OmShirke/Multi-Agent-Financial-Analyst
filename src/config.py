"""
Centralised settings loaded once at startup via pydantic-settings.

pydantic-settings reads from environment variables and .env files automatically.
Importing `settings` anywhere in the codebase gives a single, validated config
object — no scattered os.getenv() calls.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is two levels up from this file (src/config.py → financial-analyst/)
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently ignore unknown env vars
    )

    # API keys
    google_api_key: str
    tavily_api_key: str

    # Gemini model names — kept in config so we can swap without touching agent code
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"

    # Chroma persistence directory
    chroma_persist_dir: str = str(PROJECT_ROOT / "chroma_db")
    chroma_collection_name: str = "sec_filings"

    # Tickers pre-loaded during ingestion — agents validate against this list
    supported_tickers: list[str] = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    # RAG retrieval — how many chunks to pull per query
    rag_top_k: int = 6


# Module-level singleton — import this, don't instantiate Settings yourself
settings = Settings()
