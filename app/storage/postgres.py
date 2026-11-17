"""Shared transactional document/vector/full-text storage with mandatory RLS.

The API login must not own tables or bypass row-level security. Every pooled
transaction sets a local tenant; pool reuse cannot carry identity to a caller.
"""

import os
from contextlib import contextmanager
from functools import lru_cache

from langchain_core.documents import Document
from app.core import offline
from app.core.tenancy import tenant_id


def enabled():
    return bool(os.getenv("DATABASE_URL")) and not offline.is_offline()


@lru_cache(maxsize=4)
def _pool(url):
    from psycopg_pool import ConnectionPool
    from psycopg.rows import dict_row

    pool = ConnectionPool(
        url,
        min_size=1,
        max_size=int(os.getenv("DB_POOL_SIZE", "10")),
        timeout=10,
        kwargs={"row_factory": dict_row, "connect_timeout": 5},
    )
    import atexit

    atexit.register(pool.close)
    return pool


@contextmanager
def transaction():
    pool = _pool(os.environ["DATABASE_URL"])
    with pool.connection() as connection:
        with connection.transaction():
            connection.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id.get(),))
            connection.execute("SET LOCAL statement_timeout = '15s'")
            yield connection


def metadata(row):
    if not row:
        return None
    return {key: row[key] for key in ("document_id", "filename", "file_type", "status", "created_at")}


def _audit(connection, action, document_id=None):
    from app.core.observability import request_id

    connection.execute(
        "INSERT INTO rag_audit (tenant_id, action, resource_id, request_id) VALUES (%s,%s,%s,%s)",
        (tenant_id.get(), action, document_id, request_id.get()),
    )


def _bump(connection):
    connection.execute(
        "INSERT INTO rag_revisions (tenant_id, revision) VALUES (%s, 1) "
        "ON CONFLICT (tenant_id) DO UPDATE SET revision = rag_revisions.revision + 1",
        (tenant_id.get(),),
    )


class TenantQuotaExceeded(ValueError):
    """A tenant would exceed its configured shared-storage quota."""


def _usage(connection):
    connection.execute(
        "INSERT INTO rag_usage (tenant_id) VALUES (%s) ON CONFLICT DO NOTHING", (tenant_id.get(),)
    )
    return connection.execute(
        "SELECT * FROM rag_usage WHERE tenant_id=%s FOR UPDATE", (tenant_id.get(),)
    ).fetchone()


def save(record, content, original):
    with transaction() as connection:
        usage = _usage(connection)
        if usage["documents"] + 1 > int(os.getenv("MAX_TENANT_DOCUMENTS", "10000")) or usage["bytes"] + len(
            original
        ) > int(os.getenv("MAX_TENANT_BYTES", "1073741824")):
            raise TenantQuotaExceeded("Tenant document or byte quota exceeded")
        connection.execute(
            "UPDATE rag_usage SET documents=documents+1,bytes=bytes+%s WHERE tenant_id=%s",
            (len(original), tenant_id.get()),
        )
        connection.execute(
            """INSERT INTO rag_documents
            (tenant_id,document_id,filename,file_type,status,created_at,content,original)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                tenant_id.get(),
                record["document_id"],
                record["filename"],
                record["file_type"],
                record["status"],
                record["created_at"],
                content,
                original,
            ),
        )
        _audit(connection, "document.upload", record["document_id"])


def list_documents(limit=100, offset=0):
    with transaction() as connection:
        return [
            metadata(row)
            for row in connection.execute(
                "SELECT document_id,filename,file_type,status,created_at FROM rag_documents WHERE tenant_id=%s ORDER BY created_at,document_id LIMIT %s OFFSET %s",
                (tenant_id.get(), limit, offset),
            ).fetchall()
        ]


def get_document(identifier):
    with transaction() as connection:
        return metadata(
            connection.execute(
                "SELECT document_id,filename,file_type,status,created_at FROM rag_documents WHERE tenant_id=%s AND document_id=%s",
                (tenant_id.get(), identifier),
            ).fetchone()
        )


def text(identifier):
    with transaction() as connection:
        row = connection.execute(
            "SELECT content FROM rag_documents WHERE tenant_id=%s AND document_id=%s",
            (tenant_id.get(), identifier),
        ).fetchone()
        if row is None:
            raise LookupError(identifier)
        return row["content"]


def replace_chunks(identifier, chunks, vectors, embedding_key):
    """Commit complete index replacement and revision increment atomically."""
    if len(chunks) != len(vectors):
        raise ValueError("Embedding count does not match chunks")
    with transaction() as connection:
        usage = _usage(connection)
        previous = connection.execute(
            "SELECT count(*) AS count FROM rag_chunks WHERE tenant_id=%s AND document_id=%s",
            (tenant_id.get(), identifier),
        ).fetchone()["count"]
        new_count = usage["chunks"] - previous + len(chunks)
        if new_count > int(os.getenv("MAX_TENANT_CHUNKS", "200000")):
            raise TenantQuotaExceeded("Tenant chunk quota exceeded")
        connection.execute("UPDATE rag_usage SET chunks=%s WHERE tenant_id=%s", (new_count, tenant_id.get()))
        row = connection.execute(
            "SELECT document_id FROM rag_documents WHERE tenant_id=%s AND document_id=%s FOR UPDATE",
            (tenant_id.get(), identifier),
        ).fetchone()
        if row is None:
            raise LookupError(identifier)
        connection.execute(
            "DELETE FROM rag_chunks WHERE tenant_id=%s AND document_id=%s", (tenant_id.get(), identifier)
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO rag_chunks
                (tenant_id,chunk_id,document_id,chunk_index,text,created_at,embedding_key,embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)""",
                [
                    (
                        tenant_id.get(),
                        c["chunk_id"],
                        identifier,
                        c["chunk_index"],
                        c["text"],
                        c["created_at"],
                        embedding_key,
                        str(vector),
                    )
                    for c, vector in zip(chunks, vectors)
                ],
            )
        connection.execute(
            "UPDATE rag_documents SET status='ingested' WHERE tenant_id=%s AND document_id=%s",
            (tenant_id.get(), identifier),
        )
        _bump(connection)
        _audit(connection, "document.ingest", identifier)


def list_chunks(identifier=None):
    # Retrieval uses indexed searches below, never a corpus scan.
    with transaction() as connection:
        return connection.execute(
            "SELECT chunk_id,document_id,chunk_index,text,created_at FROM rag_chunks WHERE tenant_id=%s "
            + ("AND document_id=%s " if identifier else "")
            + "ORDER BY document_id,chunk_index LIMIT 10000",
            (tenant_id.get(), identifier) if identifier else (tenant_id.get(),),
        ).fetchall()


def revision():
    with transaction() as connection:
        row = connection.execute(
            "SELECT revision FROM rag_revisions WHERE tenant_id=%s", (tenant_id.get(),)
        ).fetchone()
        return row["revision"] if row else 0


def _document(row):
    return Document(
        page_content=row["text"],
        metadata={key: row[key] for key in ("document_id", "chunk_id", "chunk_index")},
    )


def dense_search(vector, key, k):
    dimensions = len(vector)
    # Only predeclared indexed dimensions are interpolated; all data is bound.
    expression = f"embedding::halfvec({dimensions})" if dimensions in (384, 1536) else "embedding"
    parameter = f"%s::halfvec({dimensions})" if dimensions in (384, 1536) else "%s::vector"
    with transaction() as connection:
        connection.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        rows = connection.execute(
            f"""SELECT chunk_id,document_id,chunk_index,text,
            1 - ({expression} <=> {parameter}) AS score FROM rag_chunks
            WHERE tenant_id=%s AND embedding_key=%s AND embedding_dimensions=%s
            ORDER BY {expression} <=> {parameter} LIMIT %s""",
            (str(vector), tenant_id.get(), key, dimensions, str(vector), k),
        ).fetchall()
        return [(_document(row), float(row["score"])) for row in rows]


def keyword_search(query, key, k):
    import re

    # OR across normalized terms provides recall; websearch parsing and bound
    # parameters preserve safety, while English text search removes stopwords.
    query = " OR ".join(re.findall(r"[A-Za-z0-9]+", query)[:64])
    if not query:
        return []
    with transaction() as connection:
        rows = connection.execute(
            """SELECT chunk_id,document_id,chunk_index,text,
            ts_rank_cd(search_text, websearch_to_tsquery('english', %s)) AS score
            FROM rag_chunks WHERE tenant_id=%s AND embedding_key=%s
            AND search_text @@ websearch_to_tsquery('english', %s)
            ORDER BY score DESC, chunk_id LIMIT %s""",
            (query, tenant_id.get(), key, query, k),
        ).fetchall()
        return [(_document(row), float(row["score"])) for row in rows]


def delete_document(identifier):
    with transaction() as connection:
        _usage(connection)
        record = connection.execute(
            "SELECT octet_length(original) AS bytes FROM rag_documents WHERE tenant_id=%s AND document_id=%s FOR UPDATE",
            (tenant_id.get(), identifier),
        ).fetchone()
        if not record:
            return False
        chunks = connection.execute(
            "SELECT count(*) AS count FROM rag_chunks WHERE tenant_id=%s AND document_id=%s",
            (tenant_id.get(), identifier),
        ).fetchone()["count"]
        connection.execute(
            "DELETE FROM rag_documents WHERE tenant_id=%s AND document_id=%s", (tenant_id.get(), identifier)
        )
        connection.execute(
            "UPDATE rag_usage SET documents=documents-1,bytes=bytes-%s,chunks=chunks-%s WHERE tenant_id=%s",
            (record["bytes"], chunks, tenant_id.get()),
        )
        _bump(connection)
        _audit(connection, "document.delete", identifier)
        return True


def audit_events(limit=100):
    with transaction() as connection:
        return connection.execute(
            "SELECT action,resource_id,request_id,created_at FROM rag_audit WHERE tenant_id=%s ORDER BY id DESC LIMIT %s",
            (tenant_id.get(), limit),
        ).fetchall()


def readiness():
    with transaction() as connection:
        role = connection.execute(
            "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
        ).fetchone()
        if role["rolsuper"] or role["rolbypassrls"]:
            raise RuntimeError("API database role bypasses row-level security")
        tables = connection.execute(
            "SELECT relrowsecurity,relforcerowsecurity,pg_get_userbyid(relowner)=current_user AS owner FROM pg_class WHERE relname IN ('rag_documents','rag_chunks','rag_revisions','rag_audit','rag_usage') AND relnamespace='public'::regnamespace"
        ).fetchall()
        if len(tables) != 5 or any(
            not row["relrowsecurity"] or not row["relforcerowsecurity"] or row["owner"] for row in tables
        ):
            raise RuntimeError("Storage schema or row-level security is not ready")
        connection.execute("SELECT count(*) FROM rag_documents").fetchone()
    return True
