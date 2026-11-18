"""Server-bound tenant and role identity propagated through async/thread work."""

import hmac
import json
import os
from contextvars import ContextVar
from hashlib import sha256

tenant_id = ContextVar("tenant_id", default="default")
role = ContextVar("role", default="admin")


def namespace():
    return sha256(tenant_id.get().encode()).hexdigest()[:32]


def credentials():
    value = os.getenv("TENANT_API_KEYS", "")
    if not value:
        return {}
    mapping = json.loads(value)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("TENANT_API_KEYS must be a nonempty object")
    normalized = {}
    for secret, identity in mapping.items():
        identity = {"tenant_id": identity, "role": "admin"} if isinstance(identity, str) else identity
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("tenant_id"), str)
            or not identity["tenant_id"].strip()
        ):
            raise ValueError("Every credential requires a nonempty tenant_id")
        if identity.get("role", "reader") not in {"reader", "writer", "admin"} or not secret:
            raise ValueError("Invalid credential role or empty key")
        normalized[secret] = {"tenant_id": identity["tenant_id"], "role": identity.get("role", "reader")}
    return normalized


def authenticate(key):
    return next(
        (
            value
            for secret, value in credentials().items()
            if hmac.compare_digest(secret.encode(), key.encode())
        ),
        None,
    )


class TenantMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        tenant_token = role_token = None
        if os.getenv("TENANT_API_KEYS"):
            from starlette.responses import JSONResponse

            try:
                key = dict(scope.get("headers", [])).get(b"x-api-key", b"").decode("utf-8", errors="replace")
                identity = authenticate(key)
            except (ValueError, TypeError):
                return await JSONResponse({"detail": "Credential configuration unavailable"}, 503)(
                    scope, receive, send
                )
            if identity is None:
                if scope["path"].startswith(("/documents", "/query", "/admin")):
                    return await JSONResponse({"detail": "Invalid or missing API key"}, 401)(
                        scope, receive, send
                    )
            else:
                tenant_token = tenant_id.set(identity["tenant_id"])
                role_token = role.set(identity["role"])
        try:
            await self.app(scope, receive, send)
        finally:
            if tenant_token is not None:
                tenant_id.reset(tenant_token)
                role.reset(role_token)
