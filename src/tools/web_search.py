"""
Tavily web search wrapper used by the News agent.

Tavily is purpose-built for LLM agents — it returns clean, summarised content
rather than raw HTML, which avoids a parsing step. We restrict results to
recent news domains to keep noise down.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, TypedDict

from langchain_tavily import TavilySearch

from src.config import settings

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    title: str
    url: str
    content: str
    score: float


@lru_cache(maxsize=1)
def _get_client() -> TavilySearch:
    return TavilySearch(
        tavily_api_key=settings.tavily_api_key,
        max_results=8,
        # 'advanced' depth surfaces richer summaries — worth the extra latency
        # for a news-sentiment task where context quality matters
        search_depth="advanced",
        topic="news",
    )


def search_company_news(company_name: str, ticker: str) -> List[SearchResult]:
    """
    Fetch recent news for a company.

    We query both name and ticker because some outlets use one but not the
    other — this widens recall without sacrificing precision (Tavily ranks
    by relevance, so duplicates fall to the bottom).
    """
    query = f"{company_name} ({ticker}) stock news recent earnings outlook"
    client = _get_client()

    try:
        response = client.invoke({"query": query})
        # langchain-tavily returns a dict with 'results' key
        results = response.get("results", []) if isinstance(response, dict) else []
        logger.info("Tavily returned %d results for %s", len(results), ticker)
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in results
        ]
    except Exception:
        logger.exception("Tavily search failed for ticker=%s", ticker)
        return []


def format_results_for_prompt(results: List[SearchResult]) -> str:
    """Render search results as numbered, prompt-ready text."""
    if not results:
        return "(no news results found)"

    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[ARTICLE {i}] {r['title']}\nURL: {r['url']}\n{r['content']}"
        )
    return "\n\n---\n\n".join(parts)
