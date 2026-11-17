"""Real shared storage, isolation, concurrency, rollback, roles and quota tests."""

import os
import uuid
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import json
from fastapi.testclient import TestClient
from app.main import app
from app.core import offline, rate_limit
from app.core.tenancy import tenant_id
from app.storage import postgres
from app.rag import backends


@unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Shared storage requires TEST_DATABASE_URL")
class SharedStorageTests(unittest.TestCase):
    def setUp(self):
        self.a = "test-a-" + uuid.uuid4().hex
        self.b = "test-b-" + uuid.uuid4().hex
        self.environment = patch.dict(
            os.environ,
            {
                "DATABASE_URL": os.environ["TEST_DATABASE_URL"],
                "REDIS_URL": os.getenv("TEST_REDIS_URL", ""),
                "EMBEDDING_BACKEND": "local_hash",
                "TENANT_API_KEYS": json.dumps(
                    {
                        "key-a": {"tenant_id": self.a, "role": "admin"},
                        "key-b": {"tenant_id": self.b, "role": "reader"},
                        "key-reader": {"tenant_id": self.a, "role": "reader"},
                    }
                ),
            },
        )
        self.environment.start()
        self.previous_offline = offline.is_offline()
        offline.set_offline(False)
        self.client = TestClient(app)
        self.headers = {"X-API-Key": "key-a"}
        rate_limit.reset_limits()

    def tearDown(self):
        for tenant in (self.a, self.b):
            token = tenant_id.set(tenant)
            with postgres.transaction() as connection:
                for table in ("rag_documents", "rag_revisions", "rag_usage"):
                    connection.execute("DELETE FROM " + table + " WHERE tenant_id=%s", (tenant,))
            tenant_id.reset(token)
        offline.set_offline(self.previous_offline)
        self.environment.stop()

    def upload(self, text=b"Zephyr confidential launch code is violet."):
        response = self.client.post(
            "/documents/upload", headers=self.headers, files={"file": ("policy.txt", text, "text/plain")}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["document_id"]

    def test_rls_without_application_where_clause(self):
        identifier = self.upload()
        token = tenant_id.set(self.b)
        try:
            with postgres.transaction() as connection:
                self.assertEqual(connection.execute("SELECT * FROM rag_documents").fetchall(), [])
                import psycopg

                with self.assertRaises(psycopg.errors.InsufficientPrivilege):
                    connection.execute("INSERT INTO rag_revisions VALUES (%s,1)", (self.a,))
        finally:
            tenant_id.reset(token)
        self.assertEqual(
            self.client.get(f"/documents/{identifier}/chunks", headers={"X-API-Key": "key-b"}).status_code,
            404,
        )
        self.assertEqual(self.client.get("/ready").status_code, 200)

    def test_concurrent_uploads_do_not_lose_records(self):
        with ThreadPoolExecutor(max_workers=6) as executor:
            identifiers = list(
                executor.map(lambda i: self.upload(f"Concurrent record {i}".encode()), range(18))
            )
        self.assertEqual(
            {r["document_id"] for r in self.client.get("/documents", headers=self.headers).json()},
            set(identifiers),
        )

    def test_atomic_index_replace_rolls_back_on_failure(self):
        identifier = self.upload()
        self.assertEqual(
            self.client.post(f"/documents/{identifier}/ingest", headers=self.headers).status_code, 200
        )
        token = tenant_id.set(self.a)
        try:
            original = postgres.list_chunks(identifier)
            revision = postgres.revision()
            with self.assertRaises(Exception):
                postgres.replace_chunks(identifier, original, [[float("nan")]], backends.embedding_key())
            self.assertEqual(postgres.list_chunks(identifier), original)
            self.assertEqual(postgres.revision(), revision)
        finally:
            tenant_id.reset(token)

    def test_roles_audit_delete_and_cache_invalidation(self):
        identifier = self.upload()
        self.client.post(f"/documents/{identifier}/ingest", headers=self.headers)
        query = {"question": "What is the Zephyr confidential launch code?", "mode": "hybrid"}
        first = self.client.post("/query/documents", headers=self.headers, json=query)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertIn("violet", first.text)
        second = self.client.post("/query/documents", headers=self.headers, json=query)
        if os.getenv("TEST_REDIS_URL"):
            self.assertTrue(second.json()["diagnostics"]["cache_hit"])
        reader = {"X-API-Key": "key-reader"}
        self.assertEqual(self.client.delete(f"/documents/{identifier}", headers=reader).status_code, 403)
        for route in ("/admin/audit", "/metrics"):
            self.assertEqual(self.client.get(route, headers=reader).status_code, 403)
        self.assertEqual(self.client.get("/admin/audit", headers=self.headers).status_code, 200)
        self.assertEqual(
            self.client.delete(f"/documents/{identifier}", headers=self.headers).status_code, 204
        )
        result = self.client.post("/query/documents", headers=self.headers, json=query).json()
        self.assertEqual(result["sources"], [])
        self.assertFalse(result["diagnostics"]["cache_hit"])
        self.assertEqual(self.client.get("/documents", headers=self.headers).json(), [])

    def test_upload_size_and_request_correlation(self):
        with patch.dict(os.environ, {"MAX_UPLOAD_BYTES": "4"}):
            response = self.client.post(
                "/documents/upload",
                headers=self.headers,
                files={"file": ("large.txt", b"12345", "text/plain")},
            )
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.headers["x-request-id"])

    def test_shared_quota_and_released_capacity(self):
        with patch.dict(os.environ, {"MAX_TENANT_DOCUMENTS": "1"}):
            identifier = self.upload()
            response = self.client.post(
                "/documents/upload",
                headers=self.headers,
                files={"file": ("second.txt", b"more", "text/plain")},
            )
            self.assertEqual(response.status_code, 409)
            self.assertEqual(
                self.client.delete("/documents/" + identifier, headers=self.headers).status_code, 204
            )
            self.upload()
