"""Request and response models for the API.

Pydantic validates every payload at the boundary, so agent code can assume its
inputs are well-formed rather than defensively checking types.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyseRequest(BaseModel):
    """A chat-style request, optionally carrying CV context from a prior upload."""

    query: str = Field(..., min_length=1, max_length=2000, description="The user's question.")
    session_id: str | None = Field(None, description="Groups related requests in LangSmith.")
    cv_text: str = Field("", description="Raw CV text from a previous upload.")
    cv_profile: dict | None = Field(None, description="Structured profile from a previous upload.")
    target_role: str = Field("", max_length=200, description="Role the candidate is aiming for.")
    location: str = Field("", max_length=200, description="Preferred job location.")


class RetrievedChunkOut(BaseModel):
    title: str
    source: str
    score: float | None = None
    preview: str


class SearchResultOut(BaseModel):
    title: str
    url: str
    snippet: str
    score: float | None = None


class GroundingSourceOut(BaseModel):
    title: str
    url: str


class AnalyseResponse(BaseModel):
    """The final answer plus the evidence behind it."""

    answer: str
    session_id: str
    route: str
    route_reasoning: str = ""
    cv_profile: dict = Field(default_factory=dict)
    retrieved_chunks: list[RetrievedChunkOut] = Field(default_factory=list)
    search_results: list[SearchResultOut] = Field(default_factory=list)
    grounding_sources: list[GroundingSourceOut] = Field(default_factory=list)
    grounding_queries: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Result of a CV upload: OCR output plus the structured profile."""

    session_id: str
    filename: str
    ocr_engine: str
    characters_extracted: int
    cv_text: str
    cv_profile: dict
    notes: str = ""


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    vector_store_ready: bool
    indexed_chunks: int | None = None
    search_provider: str
    langsmith_tracing: bool
    langsmith_project: str | None = None
    problems: list[str] = Field(default_factory=list)
