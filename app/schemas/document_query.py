import os
from typing import Literal

from pydantic import BaseModel, Field


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
