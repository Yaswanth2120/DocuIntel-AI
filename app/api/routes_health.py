from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from app.api.dependencies import require_admin, require_operator

from app.core.config import settings
from app.schemas import HealthResponse
from app.rag import backends as rag_backends


router = APIRouter()


@router.get("/")
def root():
    return {
        "message": "DocuIntel AI is running",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
    )


@router.get("/features")
def features():
    return {
        "retrieval_modes": {
            "basic": "Vector similarity retrieval",
            "multi_query": "LLM-generated query variations for better recall",
            "hybrid": "Vector retrieval plus keyword retrieval",
            "agentic": "Hybrid retrieval with relevance check, rewrite, and retry",
        },
        "observability": "LangSmith tracing when LANGSMITH_TRACING=true",
        "optional_storage": "PostgreSQL/pgvector shared storage, or local development storage",
        "document_retrieval_modes": ["vector", "keyword", "hybrid"],
        "document_reranking": "Configurable neural cross-encoder or lexical reranking",
        "tenant_roles": ["reader", "writer", "admin"],
        "embedding_backend": rag_backends.embedding_backend_name(),
    }


@router.get("/ready")
def readiness():
    from app.core.runtime import readiness

    try:
        return readiness()
    except Exception:
        raise HTTPException(status_code=503, detail="Required dependencies are not ready")


@router.get("/metrics", dependencies=[Depends(require_operator)])
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/admin/audit", dependencies=[Depends(require_admin)])
def audit():
    from app.storage import postgres

    if not postgres.enabled():
        raise HTTPException(status_code=409, detail="Audit events require shared PostgreSQL storage")
    return postgres.audit_events()
