"""Shared bounded-lifetime retrieval cache. Tenant/model/revision bind every key."""

import hashlib
import json
import logging
import os
from functools import lru_cache
from app.core.observability import CACHE


@lru_cache(maxsize=2)
def client(url):
    from redis import Redis

    return Redis.from_url(
        url, socket_connect_timeout=1, socket_timeout=1, decode_responses=True, max_connections=20
    )


def key(configuration):
    return "rag:v1:" + hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()


def get(identifier):
    if not os.getenv("REDIS_URL"):
        return None
    try:
        value = client(os.environ["REDIS_URL"]).get(identifier)
        CACHE.labels("hit" if value else "miss").inc()
        return json.loads(value) if value else None
    except Exception:
        CACHE.labels("error").inc()
        logging.getLogger(__name__).warning("Retrieval cache unavailable; using source of truth")
        return None


def put(identifier, value):
    if os.getenv("REDIS_URL"):
        try:
            client(os.environ["REDIS_URL"]).setex(
                identifier, int(os.getenv("RETRIEVAL_CACHE_TTL", "60")), json.dumps(value)
            )
        except Exception:
            CACHE.labels("error").inc()


def ready():
    return bool(client(os.environ["REDIS_URL"]).ping())
