"""Request correlation and bounded-cardinality Prometheus metrics."""

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from prometheus_client import Counter, Histogram

request_id = ContextVar("request_id", default="")
REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"])
DURATION = Histogram("rag_http_request_seconds", "Full HTTP response duration", ["route"])
RETRIEVAL = Histogram("rag_retrieval_seconds", "Retrieval duration", ["mode", "reranker"])
CACHE = Counter("rag_retrieval_cache_total", "Retrieval cache operations", ["result"])
TOKENS = Counter("rag_model_tokens_total", "Provider-reported model tokens", ["kind"])


class ObservabilityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        header = dict(scope.get("headers", [])).get(b"x-request-id", b"").decode("ascii", errors="ignore")
        identifier = header if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", header) else uuid.uuid4().hex
        token = request_id.set(identifier)
        start, status = time.perf_counter(), 500

        async def wrapped(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", identifier.encode())
                ]
            await send(message)

        try:
            await self.app(scope, receive, wrapped)
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            elapsed = time.perf_counter() - start
            REQUESTS.labels(scope["method"], route, str(status)).inc()
            DURATION.labels(route).observe(elapsed)
            logging.getLogger("app.access").info(
                json.dumps(
                    {
                        "request_id": identifier,
                        "method": scope["method"],
                        "route": route,
                        "status_code": status,
                        "duration_ms": round(elapsed * 1000, 3),
                    }
                )
            )
            request_id.reset(token)
