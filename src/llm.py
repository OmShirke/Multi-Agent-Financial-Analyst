"""
LLM factory — one function, one provider, used by every agent.

Centralising model construction here means:
- API key is set in exactly one place
- Switching models (e.g. flash → pro) is a one-line change in config.py
- Agents never import langchain_google_genai directly
"""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return a cached Gemini 2.0 Flash instance.

    lru_cache ensures we construct the client once per process — avoids
    redundant auth round-trips when multiple agents call get_llm() at startup.

    temperature=0 for deterministic, citation-grounded outputs.
    Analyst reports should not be creative.
    """
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )
