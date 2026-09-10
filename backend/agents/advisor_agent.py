"""Advisor agent: general questions, and the final synthesis step.

Two graph nodes live here.

`give_general_advice` handles questions the playbook can answer on its own,
such as interview preparation or negotiation. It is a plain RAG node.

`synthesise_answer` is the last node in every path. It takes whatever the
specialists produced and writes the single response the user reads. This step
exists because specialist output is verbose and overlapping: the CV reviewer and
the skill gap analyst will both mention the same missing skill, and nobody wants
to read it twice. Synthesis also enforces the closing "Do this next" section, so
every answer ends with something actionable.
"""

from __future__ import annotations

import json
import logging

from backend.agents.state import CareerState
from backend.core.llm import get_chat_model
from backend.core.parsing import message_text
from backend.prompts.templates import GENERAL_ADVICE_PROMPT, SYNTHESIS_PROMPT
from backend.tools.retriever import format_chunks_for_prompt, retrieve

logger = logging.getLogger(__name__)


def give_general_advice(state: CareerState) -> dict:
    """Graph node: answer a career question from the knowledge base."""
    query = state.get("query", "")
    profile = state.get("cv_profile") or {}

    chunks = retrieve(query)

    try:
        model = get_chat_model(temperature=0.4)
        prompt = GENERAL_ADVICE_PROMPT.format(
            playbook_context=format_chunks_for_prompt(chunks),
            cv_profile=json.dumps(profile, indent=2, ensure_ascii=False) if profile
            else "No CV uploaded.",
            query=query,
        )
        advice = message_text(model.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        logger.error("General advice failed: %s", exc)
        return {
            "errors": [f"general_advice: {exc}"],
            "findings": {"general_advice": "The advice could not be generated."},
        }

    return {
        "retrieved_chunks": [chunk.to_dict() for chunk in chunks],
        "findings": {"general_advice": advice},
    }


def synthesise_answer(state: CareerState) -> dict:
    """Graph node: merge specialist findings into the final user-facing answer."""
    findings = state.get("findings") or {}

    if not findings:
        return {
            "answer": (
                "No specialist produced a result for this request. Try rephrasing "
                "the question, or upload a CV so the analysis has something to work with."
            )
        }

    # A single specialist finding needs no merging; passing it through avoids a
    # second LLM call and keeps the specialist's specificity intact.
    if len(findings) == 1:
        only_finding = next(iter(findings.values()))
        return {"answer": _append_source_note(only_finding, state)}

    formatted = "\n\n".join(
        f"### {name.replace('_', ' ').title()}\n{content}" for name, content in findings.items()
    )

    try:
        model = get_chat_model(temperature=0.4)
        prompt = SYNTHESIS_PROMPT.format(
            agent_findings=formatted,
            query=state.get("query", "Review my career position."),
        )
        answer = message_text(model.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        logger.error("Synthesis failed (%s); returning raw findings.", exc)
        return {"answer": formatted, "errors": [f"synthesis: {exc}"]}

    return {"answer": _append_source_note(answer, state)}


def _append_source_note(answer: str, state: CareerState) -> str:
    """Add grounding citations so the user can verify market claims."""
    sources = state.get("grounding_sources") or []
    if not sources:
        return answer

    lines = ["\n\n**Sources checked**"]
    for source in sources[:5]:
        title = source.get("title") or "Source"
        url = source.get("url") or ""
        lines.append(f"- [{title}]({url})" if url else f"- {title}")

    return answer + "\n".join(lines)
