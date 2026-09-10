"""Skill gap agent: compare the candidate against what the market asks for.

This agent is the clearest case in the project for combining both knowledge
sources. It needs two different kinds of information:

* **What employers want right now** - volatile, so it comes from live search.
* **How to actually learn a skill and prove it** - stable and opinionated, so it
  comes from the vetted playbook via RAG.

Answering from either source alone produces worse advice. Search alone gives a
list of missing keywords with no learning path; RAG alone gives a generic
roadmap disconnected from what is being hired for this quarter.
"""

from __future__ import annotations

import json
import logging

from backend.agents.job_scout_agent import collect_market_requirements
from backend.agents.state import CareerState
from backend.core.llm import get_chat_model
from backend.core.parsing import message_text
from backend.prompts.templates import SKILL_GAP_PROMPT
from backend.tools.retriever import format_chunks_for_prompt, retrieve

logger = logging.getLogger(__name__)


def analyse_skill_gap(state: CareerState) -> dict:
    """Graph node: identify gaps and produce a prioritised learning plan."""
    profile = state.get("cv_profile") or {}
    target_role = state.get("target_role") or profile.get("headline") or "the target role"

    # Live requirements from the market.
    market_requirements = collect_market_requirements(state)

    # Stable learning guidance from the knowledge base.
    chunks = retrieve(f"skill roadmap and learning path to become a {target_role}")

    try:
        model = get_chat_model(temperature=0.3)
        prompt = SKILL_GAP_PROMPT.format(
            cv_profile=json.dumps(profile, indent=2, ensure_ascii=False) if profile
            else "No CV uploaded.",
            target_role=target_role,
            market_requirements=market_requirements,
            playbook_context=format_chunks_for_prompt(chunks),
        )
        analysis = message_text(model.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        logger.error("Skill gap analysis failed: %s", exc)
        return {
            "errors": [f"skill_gap: {exc}"],
            "findings": {"skill_gap": "The skill gap analysis could not be generated."},
        }

    existing = state.get("retrieved_chunks") or []
    return {
        "retrieved_chunks": existing + [chunk.to_dict() for chunk in chunks],
        "findings": {"skill_gap": analysis},
    }
