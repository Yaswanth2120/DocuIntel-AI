"""Idempotent schema installation with separate admin/app database credentials.

MIGRATION_DATABASE_URL is used only here, never by the API. APP_DB_PASSWORD is
passed as a bound SQL literal; secrets are neither printed nor committed.
"""

import os
from pathlib import Path
import psycopg
from psycopg import sql


def main():
    with psycopg.connect(os.environ["MIGRATION_DATABASE_URL"], autocommit=True) as connection:
        if not connection.execute("SELECT 1 FROM pg_roles WHERE rolname='docuintel_app'").fetchone():
            connection.execute(
                sql.SQL(
                    "CREATE ROLE docuintel_app LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE"
                ).format(sql.Literal(os.environ["APP_DB_PASSWORD"]))
            )
        connection.execute(Path("migrations/001_shared_storage.sql").read_text())
    print("Shared storage schema installed; API role has no RLS bypass privileges.")


if __name__ == "__main__":
    main()
