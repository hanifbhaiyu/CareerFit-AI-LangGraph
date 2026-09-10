"""The multi-agent workflow, assembled as a LangGraph state machine.

Shape of the graph:

                        ┌──────────────┐
                        │    router    │   supervisor: one cheap LLM call
                        └──────┬───────┘
             ┌────────────┬────┴─────┬─────────────┬──────────────┐
             ▼            ▼          ▼             ▼              ▼
        cv_analyst   job_scout   skill_gap   market_intel   general_advice
             │            │          │             │              │
             └────────────┴────┬─────┴─────────────┴──────────────┘
                               ▼
                        ┌──────────────┐
                        │  synthesiser │   merges findings into one answer
                        └──────┬───────┘
                               ▼
                              END
"""

from __future__ import annotations

import logging
import time
import uuid

from langgraph.graph import END, StateGraph

from backend.agents.advisor_agent import give_general_advice, synthesise_answer
from backend.agents.cv_analyst_agent import analyse_cv
from backend.agents.job_scout_agent import find_matching_jobs
from backend.agents.market_intel_agent import gather_market_intel
from backend.agents.router_agent import route_request, select_branch
from backend.agents.skill_gap_agent import analyse_skill_gap
from backend.agents.state import CareerState
from backend.core.tracing import run_config

logger = logging.getLogger(__name__)

_compiled_graph = None


def build_graph():
    """Construct and compile the workflow with throttled nodes."""
    workflow = StateGraph(CareerState)

    # Pauses execution for 5 seconds before running an agent node
    # to stay under the free tier rate limit
    def throttle(node_func):
        def wrapper(state: CareerState):
            time.sleep(30)
            return node_func(state)
        return wrapper

    workflow.add_node("router", throttle(route_request))
    workflow.add_node("cv_analyst", throttle(analyse_cv))
    workflow.add_node("job_scout", throttle(find_matching_jobs))
    workflow.add_node("skill_gap", throttle(analyse_skill_gap))
    workflow.add_node("market_intel", throttle(gather_market_intel))
    workflow.add_node("general_advice", throttle(give_general_advice))
    workflow.add_node("synthesiser", throttle(synthesise_answer))

    workflow.set_entry_point("router")

    # The router's returned route string selects the next node.
    workflow.add_conditional_edges(
        "router",
        select_branch,
        {
            "cv_review": "cv_analyst",
            "job_match": "job_scout",
            "skill_gap": "skill_gap",
            "market_intel": "market_intel",
            "general_advice": "general_advice",
        },
    )

    # A CV review continues into skill gap analysis
    workflow.add_edge("cv_analyst", "skill_gap")

    workflow.add_edge("skill_gap", "synthesiser")
    workflow.add_edge("job_scout", "synthesiser")
    workflow.add_edge("market_intel", "synthesiser")
    workflow.add_edge("general_advice", "synthesiser")
    workflow.add_edge("synthesiser", END)

    return workflow.compile()


def get_graph():
    """Return the compiled graph, building it once per process."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("Compiled the CareerFit agent graph.")
    return _compiled_graph


def run_workflow(
    query: str,
    session_id: str | None = None,
    cv_text: str = "",
    cv_profile: dict | None = None,
    target_role: str = "",
    location: str = "",
    ocr_engine: str = "",
) -> dict:
    """Execute the full workflow for one request."""
    session = session_id or str(uuid.uuid4())

    initial_state: CareerState = {
        "query": query,
        "session_id": session,
        "cv_text": cv_text,
        "cv_profile": cv_profile or {},
        "target_role": target_role,
        "location": location,
        "ocr_engine": ocr_engine,
        "findings": {},
        "retrieved_chunks": [],
        "search_results": [],
        "grounding_sources": [],
        "grounding_queries": [],
        "errors": [],
    }

    entrypoint = "cv_upload" if cv_text else "chat"
    config = run_config(session_id=session, entrypoint=entrypoint)

    logger.info("Running workflow for session %s", session)
    final_state = get_graph().invoke(initial_state, config=config)

    return final_state
