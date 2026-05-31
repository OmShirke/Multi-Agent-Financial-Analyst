"""
News agent — recent news + sentiment via Tavily.

Flow:
  1. Search Tavily for recent news about the company.
  2. Ask the LLM to extract the most material developments and assign sentiment.
  3. Return a structured NewsAnalysis.

Sentiment labels are constrained by the Pydantic schema, so the LLM cannot
emit free-form sentiment strings.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.llm import get_llm
from src.models import NewsAnalysis
from src.state import AgentState
from src.tools.web_search import search_company_news, format_results_for_prompt

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a financial news analyst.

You will be given:
- A user question about a public company
- A set of recent news article summaries with URLs

Your job:
- Identify the 3–5 most material recent developments.
- For each, classify sentiment as 'positive', 'negative', or 'neutral'.
- Assign an overall sentiment ('positive', 'negative', 'mixed', or 'neutral')
  with brief reasoning tying it back to the specific articles.
- Every NewsItem must have a Citation pointing to the article URL.
- Do NOT speculate about anything not present in the provided articles.
"""


def news_node(state: AgentState) -> dict:
    company = state["target_company"]
    ticker = state["ticker"]
    query = state["query"]

    results = search_company_news(company_name=company, ticker=ticker)

    if not results:
        logger.warning("No news results for %s — returning empty analysis.", ticker)
        empty = NewsAnalysis(
            overall_sentiment="neutral",
            sentiment_reasoning="No recent news articles retrieved.",
            key_developments=[],
            data_available=False,
        )
        return {
            "news_analysis": empty,
            "messages": [AIMessage(content=f"[news] No results for {ticker}.")],
        }

    context = format_results_for_prompt(results)
    structured_llm = get_llm().with_structured_output(NewsAnalysis)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User question: {query}\n\n"
        f"Company: {company} ({ticker})\n\n"
        f"Recent articles:\n{context}\n"
    )

    try:
        analysis: NewsAnalysis = structured_llm.invoke(prompt)
        logger.info("News analysis complete for %s.", ticker)
        return {
            "news_analysis": analysis,
            "messages": [AIMessage(content=f"[news] Analysed {len(results)} articles for {ticker}.")],
        }
    except Exception:
        logger.exception("News agent failed for %s", ticker)
        return {
            "news_analysis": None,
            "messages": [AIMessage(content=f"[news] FAILED for {ticker}.")],
        }
