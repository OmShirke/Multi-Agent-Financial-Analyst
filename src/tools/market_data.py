"""
yfinance wrapper for the Quantitative agent.

yfinance is unofficial and occasionally flaky — we wrap every call in a try/
except and return None for missing fields rather than raising. The agent's
QuantitativeAnalysis model has Optional fields everywhere precisely for this
reason.

Peer maps are hardcoded: deriving peers programmatically is its own research
problem, and a static map keeps the demo deterministic.
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict

import yfinance as yf

logger = logging.getLogger(__name__)


# Hand-picked peer groups — keeps results stable and explainable
PEER_MAP: dict[str, list[str]] = {
    "AAPL": ["MSFT", "GOOGL"],
    "MSFT": ["AAPL", "GOOGL"],
    "GOOGL": ["MSFT", "META"],
    "TSLA": ["F", "GM"],
    "NVDA": ["AMD", "INTC"],
}


class MarketSnapshot(TypedDict, total=False):
    ticker: str
    current_price: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    pe_ratio: Optional[float]
    gross_margin_pct: Optional[float]
    operating_margin_pct: Optional[float]
    revenue_growth_yoy_pct: Optional[float]
    price_history_summary: str


def _safe_get(info: dict, key: str) -> Optional[float]:
    """Return float(info[key]) or None — yfinance often returns None or missing keys."""
    val = info.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pct(val: Optional[float]) -> Optional[float]:
    """Convert a 0–1 ratio to a percentage; pass through None."""
    return round(val * 100, 2) if val is not None else None


def fetch_snapshot(ticker: str) -> MarketSnapshot:
    """
    Pull a full snapshot for one ticker.

    Returns a partial dict on failure (with whatever fields succeeded) rather
    than raising — the agent decides how to render gaps.
    """
    snapshot: MarketSnapshot = {"ticker": ticker}

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        snapshot["current_price"] = _safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice")
        snapshot["fifty_two_week_high"] = _safe_get(info, "fiftyTwoWeekHigh")
        snapshot["fifty_two_week_low"] = _safe_get(info, "fiftyTwoWeekLow")
        snapshot["pe_ratio"] = _safe_get(info, "trailingPE")
        snapshot["gross_margin_pct"] = _pct(_safe_get(info, "grossMargins"))
        snapshot["operating_margin_pct"] = _pct(_safe_get(info, "operatingMargins"))
        snapshot["revenue_growth_yoy_pct"] = _pct(_safe_get(info, "revenueGrowth"))

        # Price history — 6 months of daily closes, then a plain-English summary
        history = tk.history(period="6mo")
        if not history.empty:
            start = float(history["Close"].iloc[0])
            end = float(history["Close"].iloc[-1])
            high = float(history["Close"].max())
            low = float(history["Close"].min())
            change_pct = ((end - start) / start) * 100 if start else 0.0
            snapshot["price_history_summary"] = (
                f"Over the past 6 months, {ticker} moved from ${start:.2f} to ${end:.2f} "
                f"({change_pct:+.1f}%), with a high of ${high:.2f} and a low of ${low:.2f}."
            )
        else:
            snapshot["price_history_summary"] = f"No price history available for {ticker}."

        logger.info("Fetched snapshot for %s", ticker)
    except Exception:
        logger.exception("yfinance fetch failed for %s", ticker)
        snapshot.setdefault("price_history_summary", f"Market data unavailable for {ticker}.")

    return snapshot


def fetch_peer_snapshots(ticker: str) -> list[MarketSnapshot]:
    """Return snapshots for the predefined peer list — empty if ticker unknown."""
    peers = PEER_MAP.get(ticker.upper(), [])
    return [fetch_snapshot(p) for p in peers]
