"""CV analyst agent.

Two responsibilities, deliberately split into two LLM calls:

1. **Extraction** - turn messy OCR text into a structured profile. Run at
   temperature 0 because this is a parsing task where creativity is a bug. The
   structured profile is then reused by every other agent, so the OCR text is
   only interpreted once.
2. **Review** - critique the CV against retrieved playbook guidance. This is the
   RAG step: the reviewer's standards come from the knowledge base, not from the
   model's own opinions, which is what stops the feedback drifting into generic
   advice.
"""

from __future__ import annotations

import json
import logging

from backend.agents.state import CareerState
from backend.config import get_settings
from backend.core.llm import get_chat_model
from backend.core.parsing import message_text, parse_json_response
from backend.prompts.templates import CV_EXTRACTION_PROMPT, CV_REVIEW_PROMPT
from backend.tools.retriever import format_chunks_for_prompt, retrieve

logger = logging.getLogger(__name__)

EMPTY_PROFILE: dict = {
    "name": None,
    "headline": None,
    "years_experience": None,
    "education": [],
    "experience": [],
    "technical_skills": [],
    "soft_skills": [],
    "tools": [],
    "certifications": [],
    "languages": [],
    "contact": {},
}


def extract_profile(cv_text: str) -> dict:
    """Parse raw OCR text into a structured candidate profile."""
    if not cv_text.strip():
        return dict(EMPTY_PROFILE)

    try:
        model = get_chat_model(temperature=0.0)
        # Cap the input: CVs longer than this are almost always OCR noise, and
        # the tail contributes little while costing tokens.
        prompt = CV_EXTRACTION_PROMPT.format(cv_text=cv_text[:12000])
        response = model.invoke(prompt)
        profile = parse_json_response(message_text(response), fallback=None)

        if not isinstance(profile, dict):
            logger.warning("CV extraction did not return an object.")
            return dict(EMPTY_PROFILE)

        # Fill any key the model omitted so downstream agents can index safely.
        for key, default in EMPTY_PROFILE.items():
            profile.setdefault(key, default)

        logger.info(
            "Extracted profile: %s skills, %s roles",
            len(profile.get("technical_skills") or []),
            len(profile.get("experience") or []),
        )
        return profile

    except Exception as exc:  # noqa: BLE001
        logger.error("CV extraction failed: %s", exc)
        return dict(EMPTY_PROFILE)


def analyse_cv(state: CareerState) -> dict:
    """Graph node: ensure a profile exists, then produce a RAG-backed review."""
    cv_text = state.get("cv_text", "")
    profile = state.get("cv_profile") or extract_profile(cv_text)
    target_role = state.get("target_role") or profile.get("headline") or "the roles they are targeting"

    # Retrieval query is built from the target role plus the review intent, so
    # the ATS and CV-writing documents rank above interview or salary material.
    retrieval_query = (
        f"CV and resume writing rules, ATS formatting, bullet points for {target_role}"
    )
    chunks = retrieve(retrieval_query)

    try:
        model = get_chat_model(temperature=0.3)
        prompt = CV_REVIEW_PROMPT.format(
            playbook_context=format_chunks_for_prompt(chunks),
            cv_profile=json.dumps(profile, indent=2, ensure_ascii=False),
            target_role=target_role,
        )
        review = message_text(model.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        logger.error("CV review failed: %s", exc)
        return {
            "cv_profile": profile,
            "errors": [f"cv_review: {exc}"],
            "findings": {"cv_review": "The CV review could not be generated."},
        }

    return {
        "cv_profile": profile,
        "retrieved_chunks": [chunk.to_dict() for chunk in chunks],
        "findings": {"cv_review": review},
    }
