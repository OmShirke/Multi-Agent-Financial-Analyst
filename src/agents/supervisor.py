"""
Supervisor agent — entry point of the graph.

Responsibilities:
  1. Parse the user's query and extract the target company + ticker.
  2. Validate that the ticker is in the supported set (we only ingested 5).
  3. Write target_company + ticker into state so specialists can pick them up.

The Supervisor does NOT itself dispatch the specialists — dispatch happens
in graph.py via a conditional edge that uses the Send API to fan out three
parallel branches. Keeping that wiring in the graph (not the agent) makes
the parallelism explicit at the topology level rather than buried in a node.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.config import settings
from src.llm import get_llm
from src.models import SupervisorPlan
from src.state import AgentState

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the supervisor of a financial analyst team.

You receive a user question that mentions a public company. Your only job at
this stage is to identify the company and its stock ticker symbol.

Supported tickers (we have 10-K data for these only): {supported}

Rules:
- If the question references a supported company, return its full legal name
  (e.g. 'Apple Inc.') and ticker (e.g. 'AAPL').
- If the question references a company NOT in the supported list, still
  return your best guess at the legal name and ticker — downstream agents
  will degrade gracefully on missing 10-K data.
- Briefly explain how you identified the company.
"""


def supervisor_node(state: AgentState) -> dict:
    query = state["query"]

    structured_llm = get_llm().with_structured_output(SupervisorPlan)
    prompt = (
        SYSTEM_PROMPT.format(supported=", ".join(settings.supported_tickers))
        + f"\n\nUser question: {query}"
    )

    plan: SupervisorPlan = structured_llm.invoke(prompt)
    ticker = plan.ticker.upper()

    if ticker not in settings.supported_tickers:
        logger.warning(
            "Ticker %s is not in the pre-ingested set %s — fundamentals will be empty.",
            ticker, settings.supported_tickers,
        )

    logger.info("Supervisor identified %s (%s). Reason: %s", plan.target_company, ticker, plan.reasoning)
    return {
        "target_company": plan.target_company,
        "ticker": ticker,
        "messages": [AIMessage(content=f"[supervisor] Targeting {plan.target_company} ({ticker}).")],
    }
