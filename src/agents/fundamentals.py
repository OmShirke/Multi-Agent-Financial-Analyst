"""
Fundamentals agent — RAG over 10-K filings.

Flow:
  1. Retrieve top-k chunks for the user's query, filtered by ticker.
  2. Ask the LLM to synthesise a FundamentalsAnalysis grounded in those chunks.
  3. Return the structured object as a state update.

The agent never fabricates numbers — if the retriever finds nothing, we return
a `data_available=False` analysis so Synthesis can gracefully skip the section.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.llm import get_llm
from src.models import FundamentalsAnalysis
from src.state import AgentState
from src.tools.rag import retrieve, format_docs_for_prompt

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior equity analyst specialising in SEC filings.

You will be given:
- A user question about a public company
- Retrieved excerpts from that company's most recent 10-K filing

Your job:
- Produce a structured FundamentalsAnalysis that directly addresses the question
  using ONLY the provided excerpts. Do not invent facts.
- Every claim in the output must be backed by a Citation that references the
  10-K source label shown in the [SOURCE i] markers.
- Be concise but specific — name real segments, real risks, real numbers from
  the filing.
"""


def fundamentals_node(state: AgentState) -> dict:
    """LangGraph node — reads state, returns partial state update."""
    ticker = state["ticker"]
    query = state["query"]

    docs = retrieve(ticker=ticker, query=query)

    if not docs:
        logger.warning("No 10-K chunks retrieved for %s — returning empty analysis.", ticker)
        empty = FundamentalsAnalysis(
            business_overview=f"No 10-K data available for {ticker} in the local store.",
            revenue_segments=[],
            key_risks=[],
            management_outlook="",
            citations=[],
            data_available=False,
        )
        return {
            "fundamentals_analysis": empty,
            "messages": [AIMessage(content=f"[fundamentals] No 10-K data for {ticker}.")],
        }

    context = format_docs_for_prompt(docs)
    structured_llm = get_llm().with_structured_output(FundamentalsAnalysis)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User question: {query}\n\n"
        f"Company: {state['target_company']} ({ticker})\n\n"
        f"Retrieved 10-K excerpts:\n{context}\n"
    )

    try:
        analysis: FundamentalsAnalysis = structured_llm.invoke(prompt)
        logger.info("Fundamentals analysis complete for %s.", ticker)
        return {
            "fundamentals_analysis": analysis,
            "messages": [AIMessage(content=f"[fundamentals] Analysed {len(docs)} chunks for {ticker}.")],
        }
    except Exception:
        logger.exception("Fundamentals agent failed for %s", ticker)
        return {
            "fundamentals_analysis": None,
            "messages": [AIMessage(content=f"[fundamentals] FAILED for {ticker}.")],
        }
