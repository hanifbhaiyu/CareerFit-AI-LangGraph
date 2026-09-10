"""Central configuration.

Every secret and tunable lives here and is read from the environment, so no key
is ever hard-coded in application code. See `.env.example` for the full list.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of where the process was started.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Application settings resolved from environment variables."""

    def __init__(self) -> None:
        # --- LLM ---
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
        self.chat_model: str = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
        # Routing and CV parsing are short, mechanical calls. Sending them to a
        # smaller model keeps the main model's quota for the reasoning-heavy
        # agents, and free-tier quota is counted per model, so splitting the
        # traffic roughly doubles how many full runs are possible per day.
        self.router_model: str = os.getenv("ROUTER_MODEL", "")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_model: str = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        self.navyai_api_key: str = os.getenv("NAVYAI_API_KEY", "")
        self.navyai_model: str = os.getenv("NAVYAI_MODEL", "deepseek-v3.2")
        self.navyai_base_url: str = os.getenv("NAVYAI_BASE_URL", "https://api.navy/v1")
        # --- Web search (optional; grounding is used as fallback) ---
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
        self.search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
        self.use_local_embeddings: bool = _as_bool(os.getenv("USE_LOCAL_EMBEDDINGS", "true"))
        # Example: 'sentence-transformers/all-MiniLM-L6-v2' (384 dims) or 'BAAI/bge-small-en-v1.5'
        self.local_embedding_model: str = os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cuda" if GPU is available
        # --- Vector store ---
        self.vector_store_dir: Path = PROJECT_ROOT / os.getenv("VECTOR_STORE_DIR", "data/faiss_index")
        self.knowledge_base_dir: Path = PROJECT_ROOT / "backend" / "rag" / "knowledge_base"
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "4"))

        # --- OCR ---
        self.tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")
        self.ocr_fallback_to_vision: bool = _as_bool(os.getenv("OCR_FALLBACK_TO_VISION", "true"))

        # --- LangSmith tracing ---
        self.langsmith_tracing: bool = _as_bool(os.getenv("LANGSMITH_TRACING", "false"))
        self.langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
        self.langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "careerfit-ai")
        self.langsmith_endpoint: str = os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )

        # --- Server ---
        self.api_host: str = os.getenv("API_HOST", "127.0.0.1")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems, empty if configuration is sound."""
        problems: list[str] = []
        if not self.google_api_key:
            problems.append(
                "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey"
            )
        if self.langsmith_tracing and not self.langsmith_api_key:
            problems.append("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is missing.")
        return problems

    @property
    def search_provider(self) -> str:
        """Which internet-search backend is available."""
        return "tavily" if self.tavily_api_key else "grounding-only"


def _as_bool(raw: str | None) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
