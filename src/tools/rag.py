"""
Chroma retriever wrapper for 10-K filings.

The Fundamentals agent calls `retrieve(ticker, query)` to get the most relevant
chunks from a company's SEC filings. We filter by `ticker` metadata so that an
AAPL query never retrieves Tesla chunks even though they live in the same
collection.

The Chroma client is cached at module level — opening the persistent store is
the slow part, and we want to pay that cost exactly once per process.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Cached embeddings client — same model used at ingest and query time."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Open the persistent Chroma collection.

    Cached so we don't reopen the SQLite-backed store on every retrieval.
    The collection is created lazily — first call will succeed even if the
    store is empty (useful for tests).
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def retrieve(ticker: str, query: str, k: int | None = None) -> List[Document]:
    """
    Return the top-k chunks for `query`, filtered to the given ticker.

    The metadata filter is crucial: without it, a question about Apple could
    pull in semantically similar text from Microsoft's 10-K and the citations
    would be wrong.
    """
    k = k or settings.rag_top_k
    store = get_vectorstore()

    try:
        docs = store.similarity_search(
            query=query,
            k=k,
            filter={"ticker": ticker.upper()},
        )
        logger.info("Retrieved %d chunks for ticker=%s query=%r", len(docs), ticker, query[:60])
        return docs
    except Exception:
        logger.exception("RAG retrieval failed for ticker=%s", ticker)
        return []


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Render retrieved docs as a single prompt-ready string with source tags.

    Each chunk is wrapped with [SOURCE i] markers so the LLM can cite by index
    when filling Citation fields in the structured output.
    """
    if not docs:
        return "(no documents retrieved)"

    parts = []
    for i, doc in enumerate(docs, start=1):
        section = doc.metadata.get("section", "unknown section")
        ticker = doc.metadata.get("ticker", "?")
        year = doc.metadata.get("year", "?")
        parts.append(
            f"[SOURCE {i}] {ticker} 10-K {year} — {section}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)
