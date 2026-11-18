"""Request and response models for the HTTP API."""

import os
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    # Stored as an ISO-8601 string (local JSON) or timestamptz (PostgreSQL);
    # pydantic parses either to a tz-aware datetime and FastAPI serializes
    # responses back to ISO-8601.
    created_at: datetime


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    created_at: str


class DocumentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    stream: bool = False
    mode: Literal["vector", "keyword", "hybrid"] = os.getenv("RAG_DEFAULT_MODE", "vector")
    rerank: bool = os.getenv("RAG_DEFAULT_RERANK", "false").lower() == "true"
    top_k: int = Field(default=4, ge=1, le=20)
    context_chars: int = Field(default=12000, ge=650, le=32000)


class DocumentQueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence_score: float

    citations: list[dict] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


# Development-only /ask demo engine.
RetrievalMode = Literal["basic", "multi_query", "hybrid", "agentic"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: RetrievalMode = "hybrid"
    stream: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    mode: RetrievalMode
    rewritten_query: str | None = None
    retrieved_documents: int
