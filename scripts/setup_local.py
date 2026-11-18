"""Generate local deployment secrets without overwriting existing configuration."""

import json
import os
from pathlib import Path
import secrets


def main():
    destination = Path(".env")
    if destination.exists():
        raise SystemExit(".env exists; preserve it and merge required settings from .env.example.")
    values = {
        "APP_ENV": "development",
        "METRICS_API_KEY": secrets.token_urlsafe(32),
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "APP_DB_PASSWORD": secrets.token_urlsafe(32),
        "REDIS_PASSWORD": secrets.token_urlsafe(32),
        "TENANT_API_KEYS": json.dumps({secrets.token_urlsafe(32): {"tenant_id": "local", "role": "admin"}}),
        "OPENAI_API_KEY": "",
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "CORS_ALLOW_ORIGINS": "http://localhost:5500,http://127.0.0.1:5500",
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        for key, value in values.items():
            handle.write(key + "=" + value + "\n")
    print("Created .env with random local secrets (mode 0600). Run docker compose up --build.")
    print("Read TENANT_API_KEYS in .env for your local admin credential; it is not printed here.")


if __name__ == "__main__":
    main()
