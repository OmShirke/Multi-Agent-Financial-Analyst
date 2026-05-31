# Multi-Agent Financial Analyst

A LangGraph system that produces a cited analyst report on a public company by coordinating four specialist agents — **three running in parallel** via LangGraph's Send API — and synthesising their structured outputs into a single report.

Built to explore the supervisor pattern, parallel agent dispatch, and structured-output contracts between LLM nodes.

---

## What it does

Given a question like *"Should I be concerned about Tesla's margin trajectory?"*, the system:

1. **Supervisor** identifies the target company and ticker.
2. **Three specialists run concurrently:**
   - **Fundamentals** — RAG over the company's most recent 10-K filing
   - **News** — Tavily web search for recent coverage with sentiment
   - **Quantitative** — yfinance market data, ratios, peer comparisons
3. **Synthesis** composes the final report with citations on every claim.

Every inter-agent message is a typed Pydantic model — no free-form strings between nodes.

---

## Architecture

```mermaid
graph TD
    START([User Query]) --> SUP[Supervisor<br/>identifies ticker]
    SUP -->|Send API<br/>parallel fan-out| FUND[Fundamentals<br/>10-K RAG]
    SUP -->|Send API| NEWS[News<br/>Tavily search]
    SUP -->|Send API| QUANT[Quantitative<br/>yfinance]
    FUND --> SYN[Synthesis<br/>composes report]
    NEWS --> SYN
    QUANT --> SYN
    SYN --> END([AnalystReport<br/>with citations])

    style SUP fill:#e1f5ff,stroke:#0288d1
    style FUND fill:#fff4e1,stroke:#f57c00
    style NEWS fill:#fff4e1,stroke:#f57c00
    style QUANT fill:#fff4e1,stroke:#f57c00
    style SYN fill:#e8f5e9,stroke:#388e3c
```

The three specialists dispatch concurrently. Synthesis runs once, after all three complete — LangGraph's super-step semantics give us the fan-in barrier for free.

---

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph + LangChain |
| LLM | Gemini 2.5 Flash (`langchain-google-genai`) |
| Embeddings | `gemini-embedding-001` |
| Vector store | Chroma (in-process, persistent) |
| 10-K filings | `sec-edgar-downloader` |
| Market data | `yfinance` |
| Web search | `langchain-tavily` |
| Structured outputs | Pydantic v2 |
| Config | `pydantic-settings` |
| UI | Streamlit |
| Packaging | `uv` + `pyproject.toml` |

No Docker. No managed services. Five tickers pre-ingested: AAPL, MSFT, GOOGL, TSLA, NVDA.

---

## Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) installed
- API keys for **[Google AI Studio](https://aistudio.google.com/app/apikey)** and **[Tavily](https://app.tavily.com)** (both free tier)

### Install

```bash
git clone <repo-url>
cd financial-analyst

uv sync                       # creates .venv and installs all deps
cp .env.example .env          # then paste your GOOGLE_API_KEY and TAVILY_API_KEY
```

### Ingest 10-K filings (one-time)

```bash
uv run python -m data.ingest
```

Downloads the latest 10-K for each supported ticker, chunks, embeds, stores in `chroma_db/`. Idempotent. Free-tier embedding quota is 1000 requests/day — the script batches and respects rate limits, but a full ingest of all 5 tickers may need to be split across two days on free tier.

### Run

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`.

---

## Sample run

Query: *"What are the main risks Apple discloses, and how is the stock performing recently?"*

```
✅ Supervisor — identifying company           ~2s
✅ Quantitative — fetching market data       ~11s   ┐
✅ News — scanning recent coverage           ~12s   ├─ parallel
✅ Fundamentals — analysing 10-K filings     ~16s   ┘
✅ Synthesis — composing report             ~170s
```

Output: an `AnalystReport` containing executive summary, balanced bull/bear thesis, four sections (Fundamentals / Recent News / Quantitative / Risks) with citations, top key-risks list, and a not-investment-advice disclaimer.

The parallel specialists take only as long as the slowest one (~16s vs. ~39s sequential). Synthesis dominates total latency because Gemini 2.5 Flash is filling a large nested schema — switching that one call to `gemini-2.5-flash-lite` is a one-line change in `src/llm.py` if speed matters more than depth.

---

## Design decisions

### Supervisor pattern

A supervisor decouples *what to do* (the query) from *who does it* (the specialists). Adding a new specialist is one node + one entry in the dispatch list — no other agents need to know. The supervisor also handles input validation (ticker identification, supported-set check) so specialists can assume clean inputs.

### Send API vs. static edges

LangGraph offers two ways to fan out:

| Approach | What it gives you | What it costs |
|---|---|---|
| Static edges | Parallel execution | Targets fixed at compile time |
| `Send` via conditional edge | Parallel execution **+ runtime choice of targets + custom payload per branch** | One extra function in `graph.py` |

Static edges would behave identically today since we always dispatch the same three specialists. We chose `Send` because:
1. It's the idiomatic supervisor pattern.
2. `dispatch_specialists()` is the natural extension point for "skip News if the question is purely historical" or "spawn two Fundamentals branches with different sub-queries".
3. Understanding the runtime mechanics of parallel dispatch was the main learning goal.

See [`src/graph.py`](src/graph.py) for the heavily-commented topology.

### Structured outputs everywhere

Every agent uses `llm.with_structured_output(SomeModel)`:
- No parsing between nodes — `AgentState` carries typed objects, not strings.
- Schema constraints prevent classes of errors (e.g. sentiment can only be `'positive' | 'negative' | 'neutral'`).
- Validation happens at the LLM-call boundary — bad outputs are retried before reaching application code.

### Parallel state writes don't conflict

Each specialist writes to a **different** state key (`fundamentals_analysis`, `news_analysis`, `quantitative_analysis`), so there's no overwrite. All three append to `messages`, which uses the `add_messages` reducer — safe under concurrent writes.

If two parallel branches tried to write the same non-message key without a reducer, LangGraph would raise `InvalidUpdateError`. That constraint drove the per-specialist state-key design.

### Graceful per-agent failure

A `yfinance` blip shouldn't kill the report. Each specialist returns a `data_available=False` analysis on failure rather than raising. Synthesis is explicitly told to acknowledge missing data in the corresponding section rather than fabricate substitutes. The user always gets *some* report, with clear signposting of gaps.

### Citations as first-class

`Citation` is a shared Pydantic primitive used by all three specialists and threaded through into `AnalystReport.sections`. Every material claim is backed by a `Citation` with source type, label, optional URL, and excerpt. Streamlit renders these as clickable references.

---

## Project structure

```
financial-analyst/
├── data/
│   └── ingest.py            # SEC download + chunk + embed (one-shot)
├── src/
│   ├── config.py            # pydantic-settings singleton
│   ├── llm.py               # get_llm() factory
│   ├── state.py             # LangGraph state schema (TypedDict)
│   ├── models.py            # All Pydantic v2 output schemas
│   ├── graph.py             # Topology + Send API dispatch
│   ├── agents/
│   │   ├── supervisor.py
│   │   ├── fundamentals.py
│   │   ├── news.py
│   │   ├── quantitative.py
│   │   └── synthesis.py
│   └── tools/
│       ├── rag.py           # Chroma retriever wrapper
│       ├── web_search.py    # Tavily wrapper
│       └── market_data.py   # yfinance wrapper
├── app.py                   # Streamlit UI
├── pyproject.toml
└── .env.example
```

---

## Roadmap

Deliberately scoped out for v1:

- Section-level 10-K extraction (Items 1 / 1A / 7) for finer-grained citations
- Multi-year filing support (currently latest 10-K only)
- LangSmith tracing
- Eval harness with held-out questions and rubric-graded outputs
- Dynamic specialist routing via the `dispatch_specialists()` extension point

---

## Disclaimer

Educational project. Output is AI-generated and is **not** investment advice. Always consult a licensed financial advisor before making investment decisions.
