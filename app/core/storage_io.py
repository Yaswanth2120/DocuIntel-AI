"""Atomic file replacement and serialization for the single-process local backend."""

import json
import os
import tempfile
from functools import wraps
from threading import RLock

_lock = RLock()


def synchronized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from app.core import offline

        if os.getenv("DATABASE_URL") and not offline.is_offline():
            return function(*args, **kwargs)
        with _lock:
            return function(*args, **kwargs)

    return wrapped


def atomic_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
