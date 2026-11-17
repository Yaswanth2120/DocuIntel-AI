"""Per-IP rate limiting for the answer endpoints, backed by slowapi.

The limit is read through ``current_rate_limit`` (a callable) so it can be
adjusted at runtime — the routes evaluate it per request. It defaults to the
``RATE_LIMIT`` env var (via settings).
"""

import os
import hashlib

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings


def client_identity(request):
    if os.getenv("TENANT_API_KEYS"):
        from app.core.tenancy import tenant_id

        return hashlib.sha256(tenant_id.get().encode()).hexdigest()
    key = request.headers.get("X-API-Key")
    return hashlib.sha256(key.encode()).hexdigest() if key else get_remote_address(request)


limiter = Limiter(key_func=client_identity, storage_uri=os.getenv("REDIS_URL", "memory://"))

_rate_limit = settings.rate_limit


def current_rate_limit() -> str:
    """Return the active limit string (e.g. ``"30/minute"``)."""
    return _rate_limit


def set_rate_limit(value: str) -> None:
    """Override the active limit (used by tests)."""
    global _rate_limit
    _rate_limit = value


def reset_limits() -> None:
    """Clear accumulated per-IP counters (used by tests for isolation)."""
    try:
        limiter._storage.reset()
    except Exception:
        pass
