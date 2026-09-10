"""Helpers for reading structured data out of LLM responses.

Models are asked for bare JSON, but in practice they sometimes wrap it in
markdown fences or add a sentence of preamble. Rather than retrying the call,
these helpers recover the JSON, which is cheaper and faster.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_response(raw: str, fallback: Any = None) -> Any:
    """Extract a JSON object from a model response.

    Tries, in order: the raw string, the contents of a markdown fence, and the
    outermost brace-balanced span. Returns `fallback` if all three fail.
    """
    if not raw:
        return fallback

    text = raw.strip()

    for candidate in _candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    logger.warning("Could not parse JSON from model response: %s", text[:200])
    return fallback


def _candidates(text: str) -> list[str]:
    """Progressively more forgiving attempts at isolating the JSON."""
    candidates = [text]

    fenced = _FENCE_PATTERN.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    return candidates


def message_text(response: Any) -> str:
    """Get plain text out of a LangChain message, whatever its content shape."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Multimodal responses arrive as a list of typed blocks.
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)

    return str(content)
