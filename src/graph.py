"""
LangGraph topology — wires the supervisor + 3 specialists + synthesis into a
graph with PARALLEL specialist execution via the Send API.

ASCII view of the flow:

         START
           │
           ▼
      ┌──────────┐
      │supervisor│   identifies company + ticker
      └────┬─────┘
           │
   ┌───────┼────────┐    Send API fan-out (PARALLEL)
   ▼       ▼        ▼
 ┌────┐ ┌────┐  ┌────────────┐
 │fund│ │news│  │quantitative│
 └─┬──┘ └─┬──┘  └─────┬──────┘
   │      │            │
   └──────┼────────────┘
          ▼                 LangGraph waits for ALL three (fan-in barrier)
      ┌─────────┐
      │synthesis│   composes final report
      └────┬────┘
           │
           ▼
          END

================================================================
THE SEND API — WHY AND HOW
================================================================

LangGraph offers two ways to fan out work from one node to many:

  (A) Multiple static edges:   supervisor → fundamentals
                               supervisor → news
                               supervisor → quantitative
      All three run in parallel automatically. Simple, but the targets
      are HARDCODED at graph-compile time.

  (B) Send API via conditional edge:
      A function returns a list of Send(target_node, payload) objects.
      LangGraph schedules one node invocation per Send, in parallel.
      The list is computed AT RUNTIME, so we can:
        - choose which specialists to invoke based on the query
        - pass a CUSTOMISED payload to each branch
        - even fan out to N copies of the same node (map-reduce style)

For this project we use (B) because it's the canonical "supervisor delegates
to workers" pattern and is the main thing the project is meant to teach.
Right now we always dispatch all three specialists; the extension point for
selective dispatch is the `dispatch_specialists` function below.

================================================================
HOW PARALLEL STATE MERGING WORKS
================================================================

When N branches run concurrently and all write back to the same AgentState,
LangGraph needs a deterministic merge rule for any key that gets written by
more than one branch. That rule is the REDUCER attached to the key.

In our state schema (src/state.py):
  - messages: Annotated[List, add_messages]   ← reducer: append safely
  - all other keys: default reducer (overwrite)

The three specialists each write to a DIFFERENT key
(fundamentals_analysis / news_analysis / quantitative_analysis), so there's
no overwrite conflict on those. They all write to `messages`, but the
add_messages reducer handles concurrent appends cleanly.

If two parallel branches tried to write the SAME non-message key with no
reducer, LangGraph would raise an InvalidUpdateError. That's the constraint
that drove our schema design.

================================================================
HOW THE FAN-IN BARRIER WORKS
================================================================

`synthesis` has three incoming edges (one from each specialist).
LangGraph's super-step semantics guarantee that a node only runs once ALL
of its incoming edges have produced their state updates. So `synthesis`
fires exactly once, after the slowest specialist finishes. No manual
synchronisation needed.
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.state import AgentState
from src.agents.supervisor import supervisor_node
from src.agents.fundamentals import fundamentals_node
from src.agents.news import news_node
from src.agents.quantitative import quantitative_node
from src.agents.synthesis import synthesis_node

logger = logging.getLogger(__name__)


# Names match the keys used in `builder.add_node(...)` below.
SPECIALIST_NODES: list[str] = ["fundamentals", "news", "quantitative"]


def dispatch_specialists(state: AgentState) -> list[Send]:
    """
    Conditional-edge function — runs AFTER supervisor, BEFORE specialists.

    It returns a list of `Send(node_name, payload)` objects. LangGraph treats
    each Send as an independent branch and runs them all in parallel.

    `state` is the full current state (after supervisor wrote ticker + company).
    We pass it through to every specialist — they each read what they need.

    EXTENSION POINT: this is the natural place to add per-query routing logic,
    e.g. skip the News agent if the user explicitly asks about historical
    fundamentals, or run two copies of `fundamentals` with different sub-queries.
    For now, we always dispatch all three.
    """
    logger.info("Dispatching %d specialists in parallel for %s", len(SPECIALIST_NODES), state.get("ticker"))
    return [Send(node_name, state) for node_name in SPECIALIST_NODES]


def build_graph():
    """Construct and compile the analyst graph."""
    builder = StateGraph(AgentState)

    # --- Register every node ---
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("fundamentals", fundamentals_node)
    builder.add_node("news", news_node)
    builder.add_node("quantitative", quantitative_node)
    builder.add_node("synthesis", synthesis_node)

    # --- Entry edge ---
    builder.add_edge(START, "supervisor")

    # --- Fan-out via Send API ---
    # add_conditional_edges signature: (source, fn, path_map)
    # When `fn` returns Send objects, LangGraph dispatches them concurrently.
    # The third arg is the list of POSSIBLE target nodes — used for graph
    # visualisation and static validation, not for runtime routing.
    builder.add_conditional_edges("supervisor", dispatch_specialists, SPECIALIST_NODES)

    # --- Fan-in: all specialists feed into synthesis ---
    # Three static edges, one per specialist. LangGraph's super-step semantics
    # mean `synthesis` only runs after all three branches have produced updates,
    # so we get a barrier for free.
    for node in SPECIALIST_NODES:
        builder.add_edge(node, "synthesis")

    # --- Exit edge ---
    builder.add_edge("synthesis", END)

    return builder.compile()


# Module-level compiled graph — import this directly from app.py / tests.
graph = build_graph()
