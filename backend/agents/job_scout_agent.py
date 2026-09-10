"""Job scout agent: find live openings and score the candidate against them.

This agent owns the internet-search tool. It builds its own query from the
candidate's profile rather than passing the user's words straight through,
because a good job search query needs the role, seniority and location in a
specific order, and users rarely phrase it that way.

Everything this agent reports must come from the search results. The prompt
forbids inventing listings, which matters: a hallucinated job opening wastes a
real person's afternoon.
"""

from __future__ import annotations

import json
import logging

from backend.agents.state import CareerState
from backend.core.llm import get_chat_model
from backend.core.parsing import message_text
from backend.prompts.templates import JOB_MATCH_PROMPT
from backend.tools.web_search import format_results_for_prompt, search_web

logger = logging.getLogger(__name__)

DEFAULT_LOCATION = "Bangladesh"


def build_search_query(state: CareerState) -> str:
    """Compose a job-board-friendly query from the profile and preferences."""
    profile = state.get("cv_profile") or {}
    target_role = state.get("target_role") or profile.get("headline") or "software developer"
    location = state.get("location") or DEFAULT_LOCATION

    years = profile.get("years_experience")
    if isinstance(years, (int, float)):
        seniority = "entry level" if years < 2 else "mid level" if years < 6 else "senior"
    else:
        seniority = ""

    parts = [target_role, seniority, "jobs", location, "hiring now"]
    return " ".join(part for part in parts if part).strip()


def find_matching_jobs(state: CareerState) -> dict:
    """Graph node: search for openings, then score fit against the candidate."""
    query = build_search_query(state)
    logger.info("Job search query: %r", query)

    results = search_web(query)
    profile = state.get("cv_profile") or {}
    target_role = state.get("target_role") or "the target role"
    location = state.get("location") or DEFAULT_LOCATION

    if not results:
        message = (
            "No live job listings came back from the search. This is usually a "
            "search-provider issue rather than an empty market."
        )
        return {"findings": {"job_matches": message}, "search_results": []}

    try:
        model = get_chat_model(temperature=0.3)
        prompt = JOB_MATCH_PROMPT.format(
            cv_profile=json.dumps(profile, indent=2, ensure_ascii=False) if profile
            else "No CV was uploaded; judge fit from the stated target role only.",
            target_role=target_role,
            location=location,
            search_results=format_results_for_prompt(results),
        )
        analysis = message_text(model.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        logger.error("Job matching failed: %s", exc)
        return {
            "search_results": [result.to_dict() for result in results],
            "errors": [f"job_match: {exc}"],
            "findings": {"job_matches": "Job matching could not be completed."},
        }

    return {
        "search_results": [result.to_dict() for result in results],
        "findings": {"job_matches": analysis},
    }


def collect_market_requirements(state: CareerState) -> str:
    """Search for the skills employers are currently asking for.

    Used by the skill-gap agent so that gaps are grounded in real listings
    rather than in the model's assumptions about a role.
    """
    target_role = state.get("target_role") or "software developer"
    location = state.get("location") or DEFAULT_LOCATION
    query = f"{target_role} job requirements skills required {location}"

    results = search_web(query)
    return format_results_for_prompt(results)
