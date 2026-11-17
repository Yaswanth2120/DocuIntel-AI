import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.core import offline, rate_limit
from app.documents import service, chunking
from app.storage import vector_store
from evals.metrics import ranking_metrics
from app.rag.grounding import grounding_proxy, judge_grounding


class EnterpriseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original = service.STORAGE_DIR
        service.configure_storage(Path(self.temporary.name))
        self.was_offline = offline.is_offline()
        offline.set_offline(True)
        rate_limit.reset_limits()
        self.environment = patch.dict(
            os.environ, {"TENANT_API_KEYS": json.dumps({"secret-a": "acme", "secret-b": "bravo"})}
        )
        self.environment.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.environment.stop()
        offline.set_offline(self.was_offline)
        vector_store.reset()
        service.configure_storage(self.original)
        self.temporary.cleanup()

    def test_tenant_isolation_all_paths_and_stream(self):
        a = {"X-API-Key": "secret-a"}
        b = {"X-API-Key": "secret-b", "X-Tenant-ID": "acme"}
        response = self.client.post(
            "/documents/upload",
            headers=a,
            files={"file": ("secret.txt", b"Zephyr launch code is violet.", "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        identifier = response.json()["document_id"]
        self.assertEqual(self.client.post(f"/documents/{identifier}/ingest", headers=a).status_code, 200)
        self.assertEqual(self.client.get("/documents", headers=b).json(), [])
        for suffix in ("chunks", "ingest"):
            method = self.client.get if suffix == "chunks" else self.client.post
            self.assertEqual(method(f"/documents/{identifier}/{suffix}", headers=b).status_code, 404)
        question = {"question": "What is the Zephyr launch code?"}
        self.assertIn("violet", self.client.post("/query/documents", headers=a, json=question).text)
        for stream in (False, True):
            response = self.client.post("/query/documents", headers=b, json={**question, "stream": stream})
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("violet", response.text)
        self.assertIn(
            "violet", self.client.post("/query/documents", headers=a, json={**question, "stream": True}).text
        )
        self.assertEqual(self.client.get("/documents", headers={"X-API-Key": "bad"}).status_code, 401)

    def test_metrics_and_grounding_proxy(self):
        metrics = ranking_metrics(["distractor", "relevant"], ["relevant", "other"])
        for key in ("precision", "recall", "mrr"):
            self.assertEqual(metrics[key], 0.5)
        self.assertEqual(
            grounding_proxy("Earth is round. Mars is purple.", "Earth is round.")[
                "unsupported_claim_rate_proxy"
            ],
            0.5,
        )

    def test_chunking_validation_and_budget(self):
        for strategy in ("fixed", "recursive"):
            pieces = chunking.split_text("Alpha sentence.\n\nBeta sentence. " * 40, 100, 10, strategy)
            self.assertTrue(pieces)
            self.assertTrue(all(len(p) <= 100 for p in pieces))
        with self.assertRaises(ValueError):
            chunking.split_text("hello", 100, 100)
        self.assertEqual(
            self.client.post(
                "/query/documents",
                headers={"X-API-Key": "secret-a"},
                json={"question": "test", "top_k": 1000},
            ).status_code,
            422,
        )

    def test_judge_requires_verbatim_evidence(self):
        class FakeJudge:
            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, messages):
                return self.schema(
                    claims=[{"claim": "Mars is purple", "supported": True, "evidence": "fabricated quote"}]
                )

        self.assertEqual(
            judge_grounding("Mars is purple", "Mars is red", FakeJudge())["unsupported_claim_rate_judge"], 1.0
        )
