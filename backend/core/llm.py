"""LLM and embedding model factories.

Every agent obtains its model through this module rather than constructing one
itself, so the model name, temperature and API key are configured in exactly one
place. `get_chat_model` is cached per (model, temperature) pair to avoid
rebuilding clients on every request.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from backend.config import get_settings
logger = logging.getLogger(__name__)
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from backend.config import get_settings
from tenacity import retry, stop_after_attempt, wait_exponential
logger = logging.getLogger(__name__)

class MissingCredentialsError(RuntimeError):
    """Raised when the app is started without an LLM API key."""


@lru_cache(maxsize=8)
def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Return a chat model client.

    Args:
        model: Override the configured model name, e.g. for a cheaper router.
        temperature: Override sampling temperature. Extraction agents use 0.0,
            advisory agents use a slightly higher value for readable prose.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise MissingCredentialsError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return ChatGoogleGenerativeAI(
        model=model or settings.chat_model,
        temperature=settings.temperature if temperature is None else temperature,
        google_api_key=settings.google_api_key,
        max_retries=2,
        timeout=180,
    )

def get_vision_model(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Return a chat model client.

    Args:
        model: Override the configured model name, e.g. for a cheaper router.
        temperature: Override sampling temperature. Extraction agents use 0.0,
            advisory agents use a slightly higher value for readable prose.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise MissingCredentialsError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return ChatGoogleGenerativeAI(
        model=model or settings.chat_model,
        temperature=settings.temperature if temperature is None else temperature,
        google_api_key=settings.google_api_key,
        max_retries=2,
        timeout=180,
    )
@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the local embedding model used for both ingestion and query time.
    
    This exclusively uses the HuggingFace local model defined in the environment.
    """
    settings = get_settings()

    logger.info("Loading local embedding model: %s on %s", 
                settings.local_embedding_model, 
                settings.embedding_device)
    
    return HuggingFaceEmbeddings(
        model_name=settings.local_embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
