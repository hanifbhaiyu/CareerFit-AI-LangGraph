"""End-to-end smoke test.

Run after setup to confirm every subsystem works before recording a demo:

    python scripts/smoke_test.py

Checks configuration, the vector store, retrieval, internet search, Google
Search grounding, and a full graph run. Each check reports independently, so a
missing optional key shows up as one failed check rather than a crash.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402
from backend.core.logging_config import setup_logging  # noqa: E402
from backend.core.tracing import configure_tracing  # noqa: E402

PASS, FAIL, WARN = "  PASS", "  FAIL", "  WARN"
results: list[tuple[str, str]] = []


def check(name: str):
    """Decorator that runs a check and records its outcome."""

    def wrapper(func):
        print(f"\n[{name}]")
        started = time.time()
        try:
            status, detail = func()
        except Exception as exc:  # noqa: BLE001
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started
        print(f"{status}  {detail}  ({elapsed:.1f}s)")
        results.append((name, status))
        return func

    return wrapper


def main() -> None:
    setup_logging("WARNING")
    settings = get_settings()

    @check("Configuration")
    def _config():
        problems = settings.validate()
        if problems:
            return FAIL, "; ".join(problems)
        return PASS, f"model={settings.chat_model}, search={settings.search_provider}"

    @check("LangSmith tracing")
    def _tracing():
        if configure_tracing():
            return PASS, f"project '{settings.langsmith_project}'"
        return WARN, "tracing is off; set LANGSMITH_TRACING=true and add a key"

    @check("Vector store")
    def _vectors():
        from backend.rag.vector_store import load_vector_store

        store = load_vector_store()
        return PASS, f"{store.index.ntotal} chunks indexed"

    @check("Retrieval (RAG)")
    def _retrieval():
        from backend.tools.retriever import retrieve

        chunks = retrieve("how should I write CV bullet points")
        if not chunks:
            return FAIL, "no chunks returned"
        top = chunks[0]
        return PASS, f"top hit '{top.title}' (relevance {top.score})"

    @check("Internet search")
    def _search():
        from backend.tools.web_search import search_web

        hits = search_web("backend developer jobs Dhaka")
        if not hits:
            return FAIL, "search returned nothing; check TAVILY_API_KEY or network"
        return PASS, f"{len(hits)} results, first: {hits[0].title[:50]}"

    @check("Google Search grounding")
    def _grounding():
        from backend.tools.google_grounding import grounded_answer

        answer = grounded_answer("What is the average salary for a backend developer in Bangladesh?")
        if not answer.grounded:
            return WARN, f"not grounded: {answer.text[:80]}"
        return PASS, f"{len(answer.sources)} sources, {len(answer.queries)} search queries"

    @check("OCR")
    def _ocr():
        sample = Path(__file__).resolve().parent.parent / "data" / "sample_cv.png"
        if not sample.exists():
            return WARN, "no sample_cv.png; run scripts/make_sample_cv.py first"

        from backend.tools.ocr import extract_text

        result = extract_text(sample.read_bytes(), "sample_cv.png")
        if not result.is_usable:
            return FAIL, f"only {len(result.text)} chars via {result.engine}"
        return PASS, f"{len(result.text)} chars via {result.engine}"

    @check("Full agent workflow")
    def _workflow():
        from backend.agents.graph import run_workflow

        state = run_workflow(
            query="What should I put in my CV summary line?",
            target_role="Backend Developer",
            location="Dhaka, Bangladesh",
        )
        answer = state.get("answer", "")
        if not answer:
            return FAIL, "workflow produced no answer"
        return PASS, f"route={state.get('route')}, {len(answer)} chars returned"

    print("\n" + "=" * 62)
    failed = [name for name, status in results if status == FAIL]
    warned = [name for name, status in results if status == WARN]
    print(f"{len(results) - len(failed) - len(warned)}/{len(results)} passed", end="")
    if warned:
        print(f", {len(warned)} warning(s): {', '.join(warned)}", end="")
    print()

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("System is ready. Check your traces at https://smith.langchain.com")


if __name__ == "__main__":
    main()
