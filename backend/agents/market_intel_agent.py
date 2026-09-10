"""Market intelligence agent: salary bands, demand and hiring trends.

This is the only agent that uses Google Search grounding rather than the search
tool. The reason is accountability. Salary figures are the numbers a candidate
will actually act on, so the answer needs to be tied to sources we can display,
and grounding returns that citation metadata directly from Google.

If grounding is unavailable, the agent falls back to the ordinary search tool
and clearly labels the answer as ungrounded rather than silently degrading.
"""

from __future__ import annotations

import logging
from datetime import date

from backend.agents.state import CareerState
from backend.core.llm import get_chat_model
from backend.core.parsing import message_text
from backend.prompts.templates import MARKET_INTEL_GROUNDING_HINT, MARKET_INTEL_QUESTION
from backend.tools.google_grounding import grounded_answer
from backend.tools.web_search import format_results_for_prompt, search_web

logger = logging.getLogger(__name__)

DEFAULT_LOCATION = "Bangladesh"


def gather_market_intel(state: CareerState) -> dict:
    """Graph node: answer market questions with Google Search grounding."""
    profile = state.get("cv_profile") or {}
    target_role = state.get("target_role") or profile.get("headline") or "software developer"
    location = state.get("location") or DEFAULT_LOCATION
    user_query = state.get("query", "")

    # If the user asked something specific, ground that. Otherwise ask the
    # standard salary-and-demand question for their target role.
    question = user_query.strip() or MARKET_INTEL_QUESTION.format(
        target_role=target_role,
        location=location,
        today=date.today().strftime("%B %Y"),
    )

    result = grounded_answer(question, system_hint=MARKET_INTEL_GROUNDING_HINT)

    if result.grounded and result.text:
        logger.info(
            "Grounded answer used %d sources from %d searches.",
            len(result.sources),
            len(result.queries),
        )
        return {
            "findings": {"market_intel": result.text},
            "grounding_sources": result.sources,
            "grounding_queries": result.queries,
        }

    logger.warning("Grounding unavailable; falling back to the search tool.")
    return _fallback_to_search(question, target_role, location)


def _fallback_to_search(question: str, target_role: str, location: str) -> dict:
    """Answer from ordinary search results when grounding is not available."""
    results = search_web(f"{target_role} salary hiring demand {location} {date.today().year}")

    if not results:
        return {
            "findings": {
                "market_intel": (
                    "Live market data could not be retrieved, so no salary figures "
                    "are reported here rather than guessing at them."
                )
            }
        }

    try:
        model = get_chat_model(temperature=0.2)
        prompt = (
            f"{MARKET_INTEL_GROUNDING_HINT}\n\nQuestion: {question}\n\n"
            f"Search results:\n{format_results_for_prompt(results)}\n\n"
            "Answer using only these results. Note clearly that the figures come "
            "from general web search rather than verified grounding."
        )
        answer = message_text(model.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        logger.error("Market intel fallback failed: %s", exc)
        return {
            "errors": [f"market_intel: {exc}"],
            "findings": {"market_intel": "Market intelligence could not be gathered."},
        }

    return {
        "findings": {"market_intel": answer},
        "search_results": [result.to_dict() for result in results],
    }
