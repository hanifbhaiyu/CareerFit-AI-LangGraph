"""LangSmith tracing setup.

LangChain and LangGraph pick up tracing from environment variables, so this
module's job is to translate our own settings into the variables the SDK reads,
and to expose helpers for tagging runs and returning trace URLs.

Once `configure_tracing()` has run, every LLM call, tool call, retrieval step
and graph node in this project is recorded as a nested run in LangSmith.
"""

from __future__ import annotations

import logging
import os

from backend.config import get_settings

logger = logging.getLogger(__name__)

_configured = False


def configure_tracing() -> bool:
    """Wire LangSmith into the process. Returns True if tracing is active."""
    global _configured
    settings = get_settings()

    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing disabled (set LANGSMITH_TRACING=true to enable).")
        return False

    if not settings.langsmith_api_key:
        logger.warning("LangSmith tracing requested but LANGSMITH_API_KEY is missing.")
        return False

    # Both the legacy LANGCHAIN_* and current LANGSMITH_* names are set so the
    # project works across langchain versions.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    _configured = True
    logger.info("LangSmith tracing enabled for project '%s'.", settings.langsmith_project)
    return True


def is_tracing_enabled() -> bool:
    return _configured


def run_config(session_id: str, entrypoint: str) -> dict:
    """Build a LangChain run config so traces are searchable in LangSmith.

    `run_name` becomes the trace title, tags allow filtering by entrypoint, and
    metadata carries the session id so a whole conversation can be reconstructed.
    """
    return {
        "run_name": f"careerfit:{entrypoint}",
        "tags": ["careerfit-ai", entrypoint],
        "metadata": {"session_id": session_id, "entrypoint": entrypoint},
    }
