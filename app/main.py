import logging
from psycopg import OperationalError
from psycopg_pool import PoolTimeout
from redis.exceptions import RedisError
from openai import OpenAIError

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import routes_documents, routes_health, routes_query
from app.core.config import settings
from app.core.tenancy import TenantMiddleware
from contextlib import asynccontextmanager
from app.core.observability import ObservabilityMiddleware
from app.core.runtime import validate_production, BodyLimitMiddleware
from app.core.rate_limit import limiter


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app):
    validate_production()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


app.add_middleware(TenantMiddleware)
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(ObservabilityMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_query.router)
app.include_router(routes_documents.router)


async def backend_unavailable(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": "A required backend is unavailable"})


for backend_error in (OperationalError, PoolTimeout, RedisError, OpenAIError):
    app.add_exception_handler(backend_error, backend_unavailable)
