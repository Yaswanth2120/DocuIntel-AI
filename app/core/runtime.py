"""Production guardrails and dependency readiness; never silently downgrade."""

import os
from app.core.tenancy import credentials
from app.storage import postgres, cache
from app.rag import backends


def validate_production():
    if os.getenv("APP_ENV") != "production":
        return
    mapping = credentials()
    if not mapping or any(len(key) < 32 for key in mapping):
        raise RuntimeError("Production requires tenant API keys of at least 32 characters")
    if not postgres.enabled() or not os.getenv("REDIS_URL"):
        raise RuntimeError("Production requires DATABASE_URL and REDIS_URL")
    if backends.embedding_backend_name() == "local_hash":
        raise RuntimeError("Production requires semantic embeddings")
    if not backends.use_openai():
        raise RuntimeError("Production generation requires OPENAI_API_KEY")
    if os.getenv("RERANKER_BACKEND") != "cross_encoder":
        raise RuntimeError("Production requires the neural cross-encoder reranker")
    if os.getenv("GROUNDING_VERIFIER") != "nli":
        raise RuntimeError("Production requires the independent NLI grounding gate")
    postgres.readiness()
    cache.ready()
    # Load once before accepting concurrent requests. functools.lru_cache alone
    # can execute a cold model factory more than once across worker threads.
    from app.rag.models import encoder, escalation_nli_model, reranker, nli_model

    encoder()
    reranker()
    nli_model()
    if os.getenv("GROUNDING_ESCALATION", "nli") == "nli":
        escalation_nli_model()


def readiness():
    if postgres.enabled():
        postgres.readiness()
    if os.getenv("REDIS_URL"):
        cache.ready()
    return {
        "status": "ready",
        "storage": "postgres" if postgres.enabled() else "local",
        "cache": "redis" if os.getenv("REDIS_URL") else "process",
        "embedding_backend": backends.embedding_backend_name(),
    }


class BodyLimitMiddleware:
    """Reject excessive request bodies before multipart spooling."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse
        from starlette.exceptions import HTTPException

        maximum = int(
            os.getenv("MAX_REQUEST_BYTES", str(int(os.getenv("MAX_UPLOAD_BYTES", "10485760")) + 1048576))
        )
        length = dict(scope.get("headers", [])).get(b"content-length")
        if length is not None:
            try:
                excessive = int(length) > maximum
            except ValueError:
                return await JSONResponse({"detail": "Invalid content length"}, 400)(scope, receive, send)
            if excessive:
                return await JSONResponse({"detail": "Request body exceeds size limit"}, 413)(
                    scope, receive, send
                )
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > maximum:
                    raise HTTPException(status_code=413, detail="Request body exceeds size limit")
            return message

        await self.app(scope, bounded_receive, send)
