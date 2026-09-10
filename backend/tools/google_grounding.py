"""Google Search grounding.

Grounding is distinct from the search tool in `web_search.py`. Here the model is
handed the Google Search tool natively: Gemini decides what to look up, runs the
query inside Google's infrastructure, writes an answer that is constrained to
what it found, and returns `groundingMetadata` telling us which sources backed
the claim and which search queries it ran.

Why this project uses it: salary bands and hiring demand change every quarter,
and an ungrounded model will confidently state figures from its training data.
Grounding forces the answer to come from retrieved pages, and gives us citations
we can show the user.
"""

from __future__ import annotations

import logging

from backend.config import get_settings

logger = logging.getLogger(__name__)


class GroundedAnswer:
    """A grounded response plus the evidence Google returned with it."""

    def __init__(
        self,
        text: str,
        sources: list[dict] | None = None,
        queries: list[str] | None = None,
        grounded: bool = True,
    ) -> None:
        self.text = text
        self.sources = sources or []
        self.queries = queries or []
        self.grounded = grounded

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "sources": self.sources,
            "search_queries": self.queries,
            "grounded": self.grounded,
        }


def grounded_answer(question: str, system_hint: str = "") -> GroundedAnswer:
    """Answer a question with Google Search grounding enabled.

    Args:
        question: The user-facing question, e.g. a salary or demand query.
        system_hint: Extra instruction shaping the answer format.
    """
    settings = get_settings()
    if not settings.google_api_key:
        return GroundedAnswer(
            "Google Search grounding is unavailable because GOOGLE_API_KEY is not set.",
            grounded=False,
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai is not installed; grounding disabled.")
        return GroundedAnswer(
            "Grounding unavailable: install the google-genai package.", grounded=False
        )

    prompt = f"{system_hint}\n\n{question}".strip() if system_hint else question

    try:
        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model=settings.chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
            ),
        )
        return _parse_grounded_response(response)
    except Exception as exc:  # noqa: BLE001 - degrade instead of failing the request
        logger.error("Google Search grounding failed: %s", exc)
        return GroundedAnswer(
            f"Live market lookup could not be completed ({exc}).", grounded=False
        )


def _parse_grounded_response(response) -> GroundedAnswer:
    """Pull answer text, citations and executed queries out of the SDK response."""
    text = getattr(response, "text", "") or ""
    sources: list[dict] = []
    queries: list[str] = []

    try:
        candidate = response.candidates[0]
        metadata = getattr(candidate, "grounding_metadata", None)

        if metadata is not None:
            queries = list(getattr(metadata, "web_search_queries", None) or [])

            for chunk in getattr(metadata, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web is not None:
                    sources.append(
                        {
                            "title": getattr(web, "title", "") or "Source",
                            "url": getattr(web, "uri", "") or "",
                        }
                    )
    except (AttributeError, IndexError) as exc:
        logger.debug("No grounding metadata on response: %s", exc)

    # Drop duplicate URLs while preserving order.
    seen: set[str] = set()
    unique_sources = []
    for source in sources:
        if source["url"] and source["url"] not in seen:
            seen.add(source["url"])
            unique_sources.append(source)

    return GroundedAnswer(
        text=text,
        sources=unique_sources,
        queries=queries,
        grounded=bool(unique_sources or queries),
    )
