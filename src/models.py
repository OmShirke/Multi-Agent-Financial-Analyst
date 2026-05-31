"""
Pydantic v2 schemas for all structured agent outputs.

Every agent uses `llm.with_structured_output(SomeModel)` so the LLM is forced
to return valid JSON that conforms to one of these schemas — no free-form
string parsing between agents.

Design rule: each model is self-contained and serialisable so it can be stored
in LangGraph state, logged, and rendered in the UI without extra processing.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A traceable source reference attached to a specific claim."""
    source_type: str = Field(
        description="One of: '10-K', 'news_article', 'market_data'"
    )
    label: str = Field(
        description="Human-readable label, e.g. 'AAPL 10-K 2023 — Risk Factors' or 'Reuters 2024-05-01'"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL if available (news articles, SEC EDGAR links)"
    )
    excerpt: Optional[str] = Field(
        default=None,
        description="Short verbatim quote or data point that supports the claim"
    )


# ---------------------------------------------------------------------------
# Supervisor output
# ---------------------------------------------------------------------------

class SupervisorPlan(BaseModel):
    """Structured output from the Supervisor's initial planning step."""
    target_company: str = Field(description="Full legal company name, e.g. 'Tesla, Inc.'")
    ticker: str = Field(description="Stock ticker symbol, e.g. 'TSLA'")
    reasoning: str = Field(description="Brief explanation of how the company was identified")


# ---------------------------------------------------------------------------
# Specialist agent outputs
# ---------------------------------------------------------------------------

class FundamentalsAnalysis(BaseModel):
    """RAG-grounded analysis drawn from 10-K filings."""
    business_overview: str = Field(description="What the company does and how it makes money")
    revenue_segments: List[str] = Field(description="Key revenue segments with approximate share")
    key_risks: List[str] = Field(description="Top risks disclosed in the filing")
    management_outlook: str = Field(description="What management says about the future (MD&A)")
    citations: List[Citation] = Field(description="Sources for the above claims")
    data_available: bool = Field(
        default=True,
        description="False if no 10-K data was found for this ticker in Chroma"
    )


class NewsItem(BaseModel):
    """A single news article summary."""
    headline: str
    summary: str
    sentiment: str = Field(description="One of: 'positive', 'negative', 'neutral'")
    citation: Citation


class NewsAnalysis(BaseModel):
    """Synthesis of recent news and market sentiment."""
    overall_sentiment: str = Field(description="One of: 'positive', 'negative', 'mixed', 'neutral'")
    sentiment_reasoning: str = Field(description="Why the overall sentiment was assigned")
    key_developments: List[NewsItem] = Field(description="The 3–5 most material recent stories")
    data_available: bool = Field(default=True)


class PeerComparison(BaseModel):
    """One row in the peer comparison table."""
    ticker: str
    pe_ratio: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    revenue_growth_yoy_pct: Optional[float] = None


class QuantitativeAnalysis(BaseModel):
    """Market data and financial ratios from yfinance."""
    current_price: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    pe_ratio: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    revenue_growth_yoy_pct: Optional[float] = None
    price_performance_summary: str = Field(
        description="Plain-English summary of price trend over the past 3–6 months"
    )
    peer_comparisons: List[PeerComparison] = Field(default_factory=list)
    citations: List[Citation] = Field(description="Data points with yfinance as source")
    data_available: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """A named section of the analyst report."""
    title: str
    content: str
    citations: List[Citation] = Field(default_factory=list)


class AnalystReport(BaseModel):
    """
    The final structured output delivered to the user.
    Every material claim must be backed by a Citation so the reader can
    verify the source independently.
    """
    company_name: str
    ticker: str
    query_addressed: str = Field(description="Restates the original user question")

    executive_summary: str = Field(description="3–5 sentence answer to the user's question")
    sections: List[ReportSection] = Field(
        description="Detailed sections: Fundamentals, Recent News, Quantitative, Risks"
    )
    investment_thesis: str = Field(
        description="Bull/bear framing — key upside and downside factors"
    )
    key_risks: List[str] = Field(description="Top 3–5 risks the reader should consider")

    # Required disclaimer — this field must never be omitted or emptied
    disclaimer: str = Field(
        default=(
            "DISCLAIMER: This report is generated by an AI system for educational "
            "purposes only. It does NOT constitute investment advice, a recommendation "
            "to buy or sell any security, or a solicitation of any kind. Always consult "
            "a licensed financial advisor before making investment decisions."
        )
    )
