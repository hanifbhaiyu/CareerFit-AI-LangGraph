"""FastAPI application: the HTTP surface of CareerFit AI.

Endpoints:

* `POST /api/upload`  - accept a CV file, OCR it, return a structured profile.
* `POST /api/analyse` - run the multi-agent workflow on a question.
* `GET  /api/health`  - report configuration status, used by the UI on startup.

The API layer stays thin on purpose. It validates input, calls into the agent
graph, and shapes the response. No prompting, retrieval or routing logic lives
here, which is what keeps the agents testable without a running server.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.cv_analyst_agent import extract_profile
from backend.agents.graph import get_graph, run_workflow
from backend.config import get_settings
from backend.core.logging_config import setup_logging
from backend.core.tracing import configure_tracing
from backend.schemas import (
    AnalyseRequest,
    AnalyseResponse,
    HealthResponse,
    UploadResponse,
)
from backend.tools.ocr import extract_text

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".md")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up and shut-down work."""
    setup_logging()
    settings = get_settings()

    for problem in settings.validate():
        logger.warning("Configuration: %s", problem)

    configure_tracing()

    # Compile the graph at boot so the first user request is not slowed by it.
    try:
        get_graph()
    except Exception as exc:  # noqa: BLE001 - surface but do not block startup
        logger.error("Could not compile the agent graph at startup: %s", exc)

    logger.info("CareerFit API ready on %s:%s", settings.api_host, settings.api_port)
    yield
    logger.info("CareerFit API shutting down.")


app = FastAPI(
    title="CareerFit AI",
    description="Multi-agent career assistant: CV analysis, skill gaps, live job matching.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local development; restrict before any real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the system is correctly configured.

    The Streamlit UI calls this on load so misconfiguration shows up as a clear
    message rather than as a failed request later on.
    """
    settings = get_settings()
    problems = settings.validate()

    indexed_chunks = None
    vector_store_ready = False
    try:
        from backend.rag.vector_store import load_vector_store

        store = load_vector_store()
        indexed_chunks = store.index.ntotal
        vector_store_ready = True
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Vector store unavailable: {exc}")

    from backend.core.tracing import is_tracing_enabled

    return HealthResponse(
        status="ok" if not problems else "degraded",
        llm_configured=bool(settings.google_api_key),
        vector_store_ready=vector_store_ready,
        indexed_chunks=indexed_chunks,
        search_provider=settings.search_provider,
        langsmith_tracing=is_tracing_enabled(),
        langsmith_project=settings.langsmith_project if is_tracing_enabled() else None,
        problems=problems,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_cv(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
) -> UploadResponse:
    """Accept a CV, run OCR, and return the structured profile.

    Extraction happens here rather than inside the graph so the user sees their
    parsed profile immediately and can correct the target role before any
    analysis runs.
    """
    settings = get_settings()
    filename = file.filename or "upload"

    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Upload one of: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f} MB; the limit is {settings.max_upload_mb} MB.",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    logger.info("Received CV upload: %s (%.2f MB)", filename, size_mb)

    ocr_result = extract_text(contents, filename)
    if not ocr_result.text:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be read from this file. "
                f"{ocr_result.notes or 'Try a clearer scan or a PDF export.'}"
            ),
        )

    profile = extract_profile(ocr_result.text)

    return UploadResponse(
        session_id=session_id or str(uuid.uuid4()),
        filename=filename,
        ocr_engine=ocr_result.engine,
        characters_extracted=len(ocr_result.text),
        cv_text=ocr_result.text,
        cv_profile=profile,
        notes=ocr_result.notes,
    )


@app.post("/api/analyse", response_model=AnalyseResponse)
def analyse(request: AnalyseRequest) -> AnalyseResponse:
    """Run the multi-agent workflow and return the answer with its evidence."""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        state = run_workflow(
            query=request.query,
            session_id=session_id,
            cv_text=request.cv_text,
            cv_profile=request.cv_profile,
            target_role=request.target_role,
            location=request.location,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow execution failed.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalyseResponse(
        answer=state.get("answer", "No answer was produced."),
        session_id=session_id,
        route=state.get("route", "unknown"),
        route_reasoning=state.get("route_reasoning", ""),
        cv_profile=state.get("cv_profile") or {},
        retrieved_chunks=state.get("retrieved_chunks") or [],
        search_results=state.get("search_results") or [],
        grounding_sources=state.get("grounding_sources") or [],
        grounding_queries=state.get("grounding_queries") or [],
        errors=state.get("errors") or [],
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
