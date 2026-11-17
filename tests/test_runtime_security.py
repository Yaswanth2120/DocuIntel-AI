import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from app.main import app
from app.core.runtime import validate_production
from app.core import rate_limit
from app.rag import answering


class RuntimeSecurityTests(unittest.TestCase):
    def test_production_refuses_missing_credentials(self):
        with patch.dict(os.environ, {"APP_ENV": "production", "TENANT_API_KEYS": ""}):
            with self.assertRaisesRegex(RuntimeError, "tenant API keys"):
                validate_production()

    def test_invalid_tenant_configuration_fails_closed(self):
        with patch.dict(os.environ, {"TENANT_API_KEYS": '{"secret":{"tenant_id":"acme","role":"owner"}}'}):
            response = TestClient(app).get("/documents", headers={"X-API-Key": "secret"})
        self.assertEqual(response.status_code, 503)

    def test_grounding_gate_does_not_stream_unsupported_claims(self):
        rate_limit.reset_limits()
        doc = Document(
            page_content="Earth is round.", metadata={"document_id": "earth", "chunk_id": "earth-1"}
        )
        with (
            patch.dict(os.environ, {"GROUNDING_VERIFIER": "nli"}),
            patch("app.rag.retrieval.retrieve", return_value=([(doc, 0.8)], {})),
            patch.object(answering, "_generate_answer", return_value="Mars is purple."),
            patch(
                "app.rag.grounding.nli_grounding",
                return_value={
                    "claims": [{"text": "Mars is purple.", "supported": False}],
                    "unsupported_claim_rate_nli": 1.0,
                },
            ),
        ):
            for stream in (False, True):
                response = TestClient(app).post(
                    "/query/documents",
                    headers={"X-API-Key": "test-api-key"},
                    json={"question": "What shape is Earth?", "stream": stream},
                )
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("purple", response.text)
                self.assertIn(answering.NO_CONTEXT_MESSAGE, response.text)

    def test_public_demo_disabled_in_production(self):
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            self.assertEqual(TestClient(app).post("/ask", json={"question": "hello"}).status_code, 404)

    def test_request_size_checked_before_multipart_parsing(self):
        with patch.dict(os.environ, {"MAX_REQUEST_BYTES": "64"}):
            response = TestClient(app).post(
                "/documents/upload",
                headers={"X-API-Key": "test-api-key"},
                files={"file": ("oversized.txt", b"x" * 1000, "text/plain")},
            )
        self.assertEqual(response.status_code, 413)

    def test_global_metrics_require_operator_credential(self):
        with patch.dict(os.environ, {"METRICS_API_KEY": "operator-only-secret"}):
            client = TestClient(app)
            self.assertEqual(client.get("/metrics", headers={"X-API-Key": "test-api-key"}).status_code, 403)
            self.assertEqual(
                client.get("/metrics", headers={"X-Metrics-Key": "operator-only-secret"}).status_code, 200
            )

    def test_backend_failure_is_sanitized_and_returns_503(self):
        from redis.exceptions import ConnectionError

        rate_limit.reset_limits()
        with patch(
            "app.rag.answering.answer_question", side_effect=ConnectionError("private connection details")
        ):
            response = TestClient(app).post(
                "/query/documents", headers={"X-API-Key": "test-api-key"}, json={"question": "test"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)

    def test_local_generation_retains_question_with_long_context(self):
        import sys
        from contextlib import nullcontext
        from types import SimpleNamespace
        from app.rag.models import local_answer

        received = {}

        class Tokenizer:
            def __call__(self, text, **options):
                return {"input_ids": text[: options["max_length"]]}

            def decode(self, tokens, **options):
                return "answer"

        class Model:
            def generate(self, **inputs):
                received["input"] = inputs["input_ids"]
                return [[1]]

        question = "Which security mechanism is required?"
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}),
            patch("app.rag.models.local_generator", return_value=(Tokenizer(), Model())),
        ):
            local_answer(question, "unrelated long context " * 1000)
        self.assertIn(question, received["input"])

    def test_citation_punctuation_is_not_an_unsupported_claim(self):
        from app.rag.grounding import split_claims

        self.assertEqual(
            split_claims("Production access requires hardware MFA. [" + "a" * 32 + "]."),
            ["Production access requires hardware MFA."],
        )
        self.assertEqual(split_claims("- The limit is -5 degrees.\n**"), ["The limit is -5 degrees."])

    def test_prefixed_citations_are_metadata_and_unknown_ids_are_rejected(self):
        from app.rag.grounding import split_claims

        identifier = "a" * 32
        self.assertEqual(
            split_claims("Production access requires hardware MFA. [chunk_id: " + identifier + "]."),
            ["Production access requires hardware MFA."],
        )
        document = Document(
            page_content="Earth is round.", metadata={"document_id": "earth", "chunk_id": identifier}
        )
        rate_limit.reset_limits()
        with (
            patch.dict(os.environ, {"GROUNDING_VERIFIER": "nli"}),
            patch("app.rag.retrieval.retrieve", return_value=([(document, 0.8)], {})),
            patch.object(answering, "_generate_answer", return_value="Earth is round. [" + "b" * 32 + "]"),
        ):
            response = TestClient(app).post(
                "/query/documents",
                headers={"X-API-Key": "test-api-key"},
                json={"question": "What shape is Earth?"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], [])
        self.assertEqual(response.json()["diagnostics"]["abstention_reason"], "invalid_generated_citations")

    def test_short_source_labels_map_to_retrieved_chunks(self):
        from app.rag.grounding import cited_references, evidence_premises, split_claims

        self.assertEqual(cited_references("A [S1]. B [S1, S3]. C [source: S2]."), {"s1", "s2", "s3"})
        self.assertEqual(split_claims("Oxygen has atomic number 8 [S2]."), ["Oxygen has atomic number 8."])
        premises = evidence_premises(["One fact. Two fact. Three fact. Four fact."])
        self.assertIn("Two fact.", premises)
        self.assertIn("Two fact. Three fact. Four fact.", premises)
        self.assertEqual(premises[-1], "One fact. Two fact. Three fact. Four fact.")

        documents = [
            Document(page_content="Earth is round.", metadata={"document_id": "earth", "chunk_id": "c" * 64}),
            Document(page_content="Mars is red.", metadata={"document_id": "mars", "chunk_id": "d" * 64}),
        ]
        supported = {
            "claims": [{"text": "Earth is round.", "supported": True}],
            "unsupported_claim_rate_nli": 0.0,
        }
        for answer, accepted in (("Earth is round. [S1]", True), ("Earth is round. [S3]", False)):
            rate_limit.reset_limits()
            with (
                patch.dict(os.environ, {"GROUNDING_VERIFIER": "nli"}),
                patch("app.rag.retrieval.retrieve", return_value=([(d, 0.8) for d in documents], {})),
                patch.object(answering, "_generate_answer", return_value=answer),
                patch("app.rag.grounding.nli_grounding", return_value=supported),
            ):
                body = (
                    TestClient(app)
                    .post(
                        "/query/documents",
                        headers={"X-API-Key": "test-api-key"},
                        json={"question": "What shape is Earth?"},
                    )
                    .json()
                )
            if accepted:
                self.assertEqual(body["answer"], answer)
                self.assertEqual([c["label"] for c in body["citations"]], ["S1", "S2"])
                self.assertEqual([c["cited"] for c in body["citations"]], [True, False])
            else:
                self.assertEqual(body["diagnostics"]["abstention_reason"], "invalid_generated_citations")

    def test_refusals_are_abstentions_not_claims(self):
        from app.rag.grounding import is_abstention

        for answer in (
            "You do not know.",
            "I don't know.",
            "The retrieved evidence does not specify any other factor.",
            "This is not mentioned in the sources [S2].",
        ):
            self.assertTrue(is_abstention(answer), answer)
        for answer in ("Peyton Manning threw 17 interceptions [S3].", "No smoking is allowed on site [S1]."):
            self.assertFalse(is_abstention(answer), answer)
