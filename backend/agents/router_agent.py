"""Router agent: decide which specialist should handle the request.

This is the supervisor of the system. It runs one cheap LLM call that returns a
route label, and the graph's conditional edge sends the request down that
branch. Routing with a model rather than keyword matching means the system
handles phrasings we never anticipated ("would I get hired at a fintech?"
routes to job_match without any rule mentioning fintech).

The route is stored on the state, so a LangSmith trace shows the decision and
its stated reasoning before any expensive work happens.
"""

from __future__ import annotations

import logging

from backend.agents.state import CareerState
from backend.config import get_settings
from backend.core.llm import get_chat_model
from backend.core.parsing import message_text, parse_json_response
from backend.prompts.templates import ROUTER_PROMPT

logger = logging.getLogger(__name__)

VALID_ROUTES = {"cv_review", "job_match", "skill_gap", "market_intel", "general_advice"}
DEFAULT_ROUTE = "general_advice"


def route_request(state: CareerState) -> dict:
    """Classify the request and record the decision on the state."""
    query = state.get("query", "")
    has_cv = bool(state.get("cv_text") or state.get("cv_profile"))

    # A freshly uploaded CV with no specific question means a review.
    if has_cv and not query.strip():
        return {
            "route": "cv_review",
            "route_reasoning": "A CV was uploaded with no specific question.",
        }

    try:
        model = get_chat_model(model=get_settings().router_model, temperature=0.0)
        prompt = ROUTER_PROMPT.format(query=query, has_cv="yes" if has_cv else "no")
        response = model.invoke(prompt)
        parsed = parse_json_response(message_text(response), fallback={}) or {}

        route = str(parsed.get("route", "")).strip()
        if route not in VALID_ROUTES:
            logger.warning("Router returned unknown route %r; using default.", route)
            route = DEFAULT_ROUTE

        # Routes that need a CV cannot run without one.
        if route in {"cv_review", "skill_gap"} and not has_cv:
            logger.info("Route %s needs a CV but none is present; answering generally.", route)
            return {
                "route": "general_advice",
                "route_reasoning": "No CV uploaded, so answering from the playbook instead.",
            }

        reasoning = str(parsed.get("reasoning", "")).strip()
        logger.info("Routed to %s (%s)", route, reasoning)
        return {"route": route, "route_reasoning": reasoning}

    except Exception as exc:  # noqa: BLE001 - routing must never hard-fail
        logger.error("Router failed (%s); falling back to %s.", exc, DEFAULT_ROUTE)
        return {
            "route": DEFAULT_ROUTE,
            "route_reasoning": f"Router error, defaulted to general advice ({exc}).",
            "errors": [f"router: {exc}"],
        }


def select_branch(state: CareerState) -> str:
    """Conditional-edge function. Returns the node name to jump to."""
    return state.get("route", DEFAULT_ROUTE)
