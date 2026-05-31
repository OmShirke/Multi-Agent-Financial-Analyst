"""
Synthesis agent — composes the final AnalystReport from specialist outputs.

Flow:
  1. Read the three specialist analyses from state (any may be None).
  2. Pass their JSON to the LLM with strict instructions about citations and
     graceful handling of missing data.
  3. Return a fully-structured AnalystReport including the mandatory disclaimer.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.llm import get_llm
from src.models import AnalystReport
from src.state import AgentState

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the senior analyst who composes the final report.

You will receive structured outputs from three specialists:
- Fundamentals (10-K based)
- News (recent articles)
- Quantitative (market data)

Any of these may be missing or marked data_available=false. Handle missing
data explicitly — do NOT fabricate substitute facts. If a specialist's data
is unavailable, state that plainly in the corresponding section.

Your job:
- Write a tight executive_summary (3–5 sentences) that directly answers the
  user's original question.
- Produce sections in this order: Fundamentals, Recent News, Quantitative,
  Risks. Each section's citations must come from the corresponding specialist.
- Write a balanced investment_thesis with bull and bear framing.
- List the top 3–5 key_risks (combining 10-K risks and news-based risks).
- ALWAYS include the disclaimer field — never omit it.

Tone: institutional, sober, no hype. This is for an educated reader.
"""


def synthesis_node(state: AgentState) -> dict:
    fundamentals = state.get("fundamentals_analysis")
    news = state.get("news_analysis")
    quant = state.get("quantitative_analysis")

    # Pydantic v2 .model_dump_json() — robust serialisation including Optional fields
    def dump(obj) -> str:
        return obj.model_dump_json(indent=2) if obj is not None else "null (specialist did not complete)"

    structured_llm = get_llm().with_structured_output(AnalystReport)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Original user question: {state['query']}\n\n"
        f"Company: {state['target_company']} ({state['ticker']})\n\n"
        f"--- FUNDAMENTALS ---\n{dump(fundamentals)}\n\n"
        f"--- NEWS ---\n{dump(news)}\n\n"
        f"--- QUANTITATIVE ---\n{dump(quant)}\n"
    )

    try:
        report: AnalystReport = structured_llm.invoke(prompt)
        # Belt-and-braces: enforce disclaimer even if the LLM emitted a weaker one
        if not report.disclaimer or "investment advice" not in report.disclaimer.lower():
            report.disclaimer = AnalystReport.model_fields["disclaimer"].default

        logger.info("Synthesis complete for %s.", state["ticker"])
        return {
            "final_report": report,
            "messages": [AIMessage(content=f"[synthesis] Final report ready for {state['ticker']}.")],
        }
    except Exception:
        logger.exception("Synthesis failed for %s", state["ticker"])
        fallback = AnalystReport(
            company_name=state["target_company"],
            ticker=state["ticker"],
            query_addressed=state["query"],
            executive_summary="The synthesis step failed. See logs for details.",
            sections=[],
            investment_thesis="Unavailable due to synthesis error.",
            key_risks=[],
        )
        return {
            "final_report": fallback,
            "messages": [AIMessage(content="[synthesis] FAILED — emitted fallback report.")],
        }
