"""The state object passed between graph nodes.

LangGraph hands this dictionary to every node and merges whatever the node
returns back into it. Keeping one explicit, typed state object is what makes the
multi-agent flow debuggable: a LangSmith trace shows the state at the entry and
exit of each node, so you can see exactly which agent added which finding.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

Route = Literal["cv_review", "job_match", "skill_gap", "market_intel", "general_advice"]


def merge_findings(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer so specialist nodes can write findings without clobbering each other."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class CareerState(TypedDict, total=False):
    """Everything the graph knows about the current request."""

    # --- Input ---
    query: str
    session_id: str
    target_role: str
    location: str

    # --- CV context ---
    cv_text: str            # raw OCR output
    cv_profile: dict        # structured profile from the CV analyst
    ocr_engine: str         # which OCR engine produced cv_text

    # --- Routing ---
    route: Route
    route_reasoning: str

    # --- Specialist output, merged rather than overwritten ---
    findings: Annotated[dict[str, Any], merge_findings]

    # --- Evidence surfaced to the UI ---
    retrieved_chunks: list[dict]
    search_results: list[dict]
    grounding_sources: list[dict]
    grounding_queries: list[str]

    # --- Output ---
    answer: str
    errors: list[str]
