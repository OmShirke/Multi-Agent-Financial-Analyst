"""
LangGraph state schema for the financial analyst graph.

TypedDict is the canonical way to define state in LangGraph — it acts as the
shared memory that every node reads from and writes to. Nodes return a dict
with only the keys they want to update; LangGraph merges it into the existing
state automatically.

The `Annotated[List, add_messages]` pattern on `messages` tells LangGraph to
*append* new messages rather than overwrite the whole list — this is the
standard reducer for message history.
"""

from __future__ import annotations

from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from src.models import FundamentalsAnalysis, NewsAnalysis, QuantitativeAnalysis, AnalystReport


class AgentState(TypedDict):
    # The raw question the user asked
    query: str

    # Extracted by the Supervisor before specialist agents are dispatched
    target_company: str  # e.g. "Apple Inc."
    ticker: str          # e.g. "AAPL"

    # Specialist outputs — Optional because they arrive asynchronously and any
    # one can be None if that agent failed (graceful partial-data synthesis)
    fundamentals_analysis: Optional[FundamentalsAnalysis]
    news_analysis: Optional[NewsAnalysis]
    quantitative_analysis: Optional[QuantitativeAnalysis]

    # Final output written by the Synthesis agent
    final_report: Optional[AnalystReport]

    # Accumulated message log — uses add_messages reducer so appends are safe
    # across concurrent branches (the parallel specialist agents all write here)
    messages: Annotated[List[BaseMessage], add_messages]
