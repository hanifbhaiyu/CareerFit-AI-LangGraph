"""Internet search tool.

This is the *tool-call* style of search: our agent decides on a query, calls an
external search API, reads the returned snippets, and reasons over them. That is
different from Google Search grounding (see `google_grounding.py`), where the
model itself is bound to live search results and returns citation metadata.

Both are implemented in this project because they behave differently:
search-as-a-tool gives the agent control over the query and lets us keep the raw
result list; grounding gives higher factual reliability with less code.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from backend.config import get_settings

logger = logging.getLogger(__name__)


class SearchResult:
    """One search hit, normalised across providers."""

    def __init__(self, title: str, url: str, snippet: str, score: float | None = None) -> None:
        self.title = title
        self.url = url
        self.snippet = snippet
        self.score = score

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "score": self.score}

    def as_context_line(self) -> str:
        return f"- {self.title} ({self.url})\n  {self.snippet}"


def search_web(query: str, max_results: int | None = None) -> list[SearchResult]:
    """Run an internet search and return normalised results.

    Tries Tavily first (better snippets, built for LLM consumption), then falls
    back to DuckDuckGo, which needs no API key.
    """
    settings = get_settings()
    limit = max_results or settings.search_max_results

    results = _search_tavily(query, limit)
    if results:
        return results

    return _search_duckduckgo(query, limit)


def _search_tavily(query: str, limit: int) -> list[SearchResult]:
    settings = get_settings()
    if not settings.tavily_api_key:
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=limit, search_depth="basic")
        results = [
            SearchResult(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                snippet=(item.get("content") or "")[:500],
                score=item.get("score"),
            )
            for item in response.get("results", [])
        ]
        logger.info("Tavily returned %d results for %r", len(results), query)
        return results
    except Exception as exc:  # noqa: BLE001 - never let search kill the request
        logger.warning("Tavily search failed (%s); falling back to DuckDuckGo.", exc)
        return []


def _search_duckduckgo(query: str, limit: int) -> list[SearchResult]:
    """Keyless fallback so the project still runs without a Tavily key."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            logger.warning("No DuckDuckGo client installed; search returns nothing.")
            return []

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=limit))
        results = [
            SearchResult(
                title=hit.get("title", "Untitled"),
                url=hit.get("href", "") or hit.get("url", ""),
                snippet=(hit.get("body", "") or "")[:500],
            )
            for hit in hits
        ]
        logger.info("DuckDuckGo returned %d results for %r", len(results), query)
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []


def format_results_for_prompt(results: list[SearchResult]) -> str:
    """Render results as context text for an LLM prompt."""
    if not results:
        return "No search results were returned."
    return "\n".join(result.as_context_line() for result in results)


# --------------------------------------------------------------------------- #
# LangChain tool wrapper - this is what the agent binds to, and what shows up
# as a tool call inside a LangSmith trace.
# --------------------------------------------------------------------------- #

@tool("search_job_market", return_direct=False)
def search_job_market_tool(query: str) -> str:
    """Search the live internet for job listings, hiring trends or employer information.

    Use this for anything time-sensitive that a training corpus would not know:
    currently open roles, which companies are hiring, recent salary reports.
    Pass a natural search query, for example
    'junior python developer jobs Dhaka 2026'.
    """
    return format_results_for_prompt(search_web(query))
