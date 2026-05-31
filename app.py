"""
Streamlit UI for the multi-agent financial analyst.

Run with:
    uv run streamlit run app.py

We use graph.stream() rather than graph.invoke() so the UI can show
per-node progress as each agent completes — important for a multi-agent
demo where total latency is 15–30s and silent waits feel broken.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.config import settings
from src.graph import graph
from src.models import AnalystReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# --------------------------------------------------------------------------- #
# Page config + sidebar
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Multi-Agent Financial Analyst", page_icon="📊", layout="wide")

with st.sidebar:
    st.header("About")
    st.markdown(
        "A multi-agent system built with **LangGraph** that produces an analyst "
        "report by coordinating four specialist agents in parallel."
    )

    st.subheader("Supported companies")
    st.markdown("\n".join(f"- **{t}**" for t in settings.supported_tickers))

    st.subheader("Example queries")
    examples = [
        "Should I be concerned about Tesla's margin trajectory?",
        "How exposed is Apple to China-related risks?",
        "What's driving NVIDIA's recent performance?",
        "Is Microsoft's cloud strategy paying off?",
        "How is Google's ad business holding up?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["query"] = ex

    st.divider()
    st.caption("Stack: LangGraph · Gemini 2.0 Flash · Chroma · yfinance · Tavily")


# --------------------------------------------------------------------------- #
# Main: query input
# --------------------------------------------------------------------------- #

st.title("📊 Multi-Agent Financial Analyst")
st.markdown(
    "Ask a question about a public company. The supervisor agent identifies the "
    "ticker, then dispatches three specialists in parallel (Fundamentals · News · "
    "Quantitative). The synthesis agent composes the final report with citations."
)

query = st.text_area(
    "Your question",
    value=st.session_state.get("query", ""),
    placeholder="e.g. Should I be concerned about Tesla's margin trajectory?",
    height=80,
)

run = st.button("Run analysis", type="primary", disabled=not query.strip())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NODE_LABELS = {
    "supervisor": "🧭 Supervisor — identifying company",
    "fundamentals": "📄 Fundamentals — analysing 10-K filings",
    "news": "📰 News — scanning recent coverage",
    "quantitative": "📈 Quantitative — fetching market data",
    "synthesis": "✍️  Synthesis — composing report",
}


def render_report(report: AnalystReport) -> None:
    """Render an AnalystReport with structured sections and citation expanders."""
    st.success(f"Report ready for **{report.company_name} ({report.ticker})**")

    st.subheader("Executive Summary")
    st.markdown(report.executive_summary)

    st.subheader("Investment Thesis")
    st.markdown(report.investment_thesis)

    for section in report.sections:
        with st.expander(f"📑 {section.title}", expanded=True):
            st.markdown(section.content)
            if section.citations:
                st.markdown("**Sources:**")
                for c in section.citations:
                    label = f"`{c.source_type}` · {c.label}"
                    if c.url:
                        label += f" — [link]({c.url})"
                    st.markdown(f"- {label}")
                    if c.excerpt:
                        st.caption(f"> {c.excerpt[:280]}{'…' if len(c.excerpt) > 280 else ''}")

    st.subheader("Key Risks")
    for risk in report.key_risks:
        st.markdown(f"- {risk}")

    st.divider()
    st.warning(report.disclaimer)


# --------------------------------------------------------------------------- #
# Run the graph
# --------------------------------------------------------------------------- #

if run and query.strip():
    initial_state = {
        "query": query.strip(),
        "target_company": "",
        "ticker": "",
        "fundamentals_analysis": None,
        "news_analysis": None,
        "quantitative_analysis": None,
        "final_report": None,
        "messages": [],
    }

    progress_box = st.container()
    final_state: dict = {}
    seen_nodes: set[str] = set()

    with progress_box:
        st.markdown("### Progress")
        # graph.stream() yields dicts of {node_name: state_update} after each node fires.
        # We use it to give the user live feedback rather than a 20s blank wait.
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                # Track the merged final state ourselves — `updates` mode only
                # gives us per-node deltas, not the full accumulated state.
                final_state.update({k: v for k, v in update.items() if v is not None})
                if node_name not in seen_nodes:
                    seen_nodes.add(node_name)
                    st.markdown(f"✅ {NODE_LABELS.get(node_name, node_name)}")

    report = final_state.get("final_report")
    if isinstance(report, AnalystReport):
        st.divider()
        render_report(report)
    else:
        st.error("No final report was produced. Check the logs for details.")
