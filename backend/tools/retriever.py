"""Retrieval tool over the curated career knowledge base.

This is the RAG half of the system. The knowledge base holds advice that is
stable and opinionated - ATS formatting rules, interview structures, skill
roadmaps, salary negotiation tactics - the kind of guidance that should come
from a vetted source rather than from whatever the model happens to recall.

Volatile facts (who is hiring right now, this quarter's salary bands) come from
search and grounding instead. Splitting the two is the core retrieval design
decision in this project.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

from backend.config import get_settings
from backend.rag.vector_store import load_vector_store

logger = logging.getLogger(__name__)


class RetrievedChunk:
    """A retrieved passage with its provenance and similarity score."""

    def __init__(self, content: str, title: str, source: str, score: float | None) -> None:
        self.content = content
        self.title = title
        self.source = source
        self.score = score

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "score": self.score,
            "preview": self.content[:220],
        }


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Semantic search over the knowledge base.

    Uses `similarity_search_with_score` rather than plain `similarity_search` so
    the API can report how confident retrieval was, which is useful both for the
    UI and for debugging a trace where the answer looks off-topic.
    """
    settings = get_settings()
    k = top_k or settings.retriever_top_k

    try:
        store = load_vector_store()
        pairs: list[tuple[Document, float]] = store.similarity_search_with_score(query, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed: %s", exc)
        return []

    chunks = [
        RetrievedChunk(
            content=document.page_content,
            title=document.metadata.get("title", "Career guide"),
            source=document.metadata.get("source", "unknown"),
            # FAISS returns L2 distance: lower is closer. Invert into a rough
            # 0-1 relevance figure so the UI can display something intuitive.
            score=round(1.0 / (1.0 + float(distance)), 3),
        )
        for document, distance in pairs
    ]

    logger.info("Retrieved %d chunks for %r", len(chunks), query)
    return chunks


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as labelled context for the LLM prompt."""
    if not chunks:
        return "No relevant guidance was found in the knowledge base."

    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {index}: {chunk.title}]\n{chunk.content}")
    return "\n\n---\n\n".join(blocks)


@tool("search_career_playbook", return_direct=False)
def career_playbook_tool(query: str) -> str:
    """Look up vetted career guidance from the internal knowledge base.

    Covers CV and ATS formatting rules, interview preparation frameworks, skill
    roadmaps for common tech roles, salary negotiation tactics, and guidance for
    the Bangladesh job market. Use this for advice and best practice, not for
    live job listings.
    """
    return format_chunks_for_prompt(retrieve(query))
