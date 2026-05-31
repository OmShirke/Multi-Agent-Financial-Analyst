"""
Quantitative agent — market data + ratios via yfinance.

Flow:
  1. Fetch a snapshot for the target ticker + its peers.
  2. Hand the raw snapshot JSON to the LLM and ask it to faithfully populate
     a QuantitativeAnalysis, plus a plain-English summary.

We intentionally pass the numbers to the LLM rather than constructing the
model programmatically — the structured-output pattern keeps the contract
uniform across all four specialists, and at temperature=0 the LLM reliably
copies the provided values into the schema.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage

from src.llm import get_llm
from src.models import QuantitativeAnalysis
from src.state import AgentState
from src.tools.market_data import fetch_snapshot, fetch_peer_snapshots

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a quantitative equity analyst.

You will be given a JSON snapshot of market data for a target company and its
peers, all sourced from yfinance.

Your job:
- Populate the QuantitativeAnalysis schema using EXACTLY the values from the
  snapshot. Do not invent or round numbers beyond what is provided.
- Use null for any field that is missing in the snapshot.
- Write a 2–3 sentence price_performance_summary based on the provided
  price_history_summary.
- Add one Citation per material data point (source_type='market_data',
  label='yfinance', and the field name in the excerpt).
- Set data_available=true unless the snapshot is essentially empty.
"""


def quantitative_node(state: AgentState) -> dict:
    ticker = state["ticker"]

    target = fetch_snapshot(ticker)
    peers = fetch_peer_snapshots(ticker)

    # Detect total failure — if every numeric field is None, mark unavailable
    numeric_keys = ("current_price", "pe_ratio", "gross_margin_pct", "operating_margin_pct")
    has_any_data = any(target.get(k) is not None for k in numeric_keys)

    if not has_any_data:
        logger.warning("No market data for %s — returning empty analysis.", ticker)
        empty = QuantitativeAnalysis(
            price_performance_summary=f"Market data unavailable for {ticker}.",
            peer_comparisons=[],
            citations=[],
            data_available=False,
        )
        return {
            "quantitative_analysis": empty,
            "messages": [AIMessage(content=f"[quantitative] No data for {ticker}.")],
        }

    payload = {"target": target, "peers": peers}
    structured_llm = get_llm().with_structured_output(QuantitativeAnalysis)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Ticker: {ticker}\n\n"
        f"Snapshot JSON:\n{json.dumps(payload, indent=2, default=str)}\n"
    )

    try:
        analysis: QuantitativeAnalysis = structured_llm.invoke(prompt)
        logger.info("Quantitative analysis complete for %s.", ticker)
        return {
            "quantitative_analysis": analysis,
            "messages": [AIMessage(content=f"[quantitative] Snapshot + {len(peers)} peers analysed for {ticker}.")],
        }
    except Exception:
        logger.exception("Quantitative agent failed for %s", ticker)
        return {
            "quantitative_analysis": None,
            "messages": [AIMessage(content=f"[quantitative] FAILED for {ticker}.")],
        }
