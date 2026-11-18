"""Public-data evaluation through the real PostgreSQL retrieval adapter.

python -m evals.public_benchmark --dataset scifact --queries 100
python -m evals.public_benchmark --dataset squad --queries 100 --generate 30

Uses complete corpora, fixed seeded query samples, human qrels/answer labels,
real semantic embeddings and a neural reranker. Never substitutes hash vectors.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import unicodedata
import re
from statistics import mean, median
from time import perf_counter
import uuid

from app.core import offline
from app.core.tenancy import tenant_id
from app.documents.chunking import split_text
from app.rag import backends, models
from app.rag.retrieval import retrieve
from app.rag.grounding import CITATION_PATTERN
from app.storage import postgres
from evals.datasets import load


def metrics(actual, expected, k):
    actual = list(dict.fromkeys(actual))[:k]
    expected = set(expected)
    hits = [int(identifier in expected) for identifier in actual]
    dcg = sum(value / math.log2(i + 2) for i, value in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(expected))))
    return {
        "precision_at_1": float(bool(hits and hits[0])),
        "precision_at_k": sum(hits) / k,
        "recall_at_k": sum(hits) / len(expected),
        "mrr_at_k": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
    }


def normalize(text):
    # NFKC folds superscripts/compatibility forms (m³ -> m3); dropping combining
    # marks folds diacritics (Temür -> Temur) so formatting is not scored as error.
    text = unicodedata.normalize("NFKC", CITATION_PATTERN.sub("", text))
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(answer, references):
    scores = []
    for reference in references:
        actual, expected = normalize(answer).split(), normalize(reference).split()
        common = sum((Counter(actual) & Counter(expected)).values())
        scores.append(2 * common / (len(actual) + len(expected)) if actual or expected else 1.0)
    return {
        "answer_exact_match": float(any(normalize(answer) == normalize(ref) for ref in references)),
        "reference_span_hit": float(
            any(
                " " + normalize(ref) + " " in " " + normalize(answer) + " "
                for ref in references
                if normalize(ref)
            )
        ),
        "answer_token_f1": max(scores, default=0.0),
    }


def embedding_signature(chunks, key):
    digest = hashlib.sha256(key.encode())
    for chunk in sorted(chunks, key=lambda chunk: chunk["chunk_id"]):
        digest.update(chunk["chunk_id"].encode())
        digest.update(chunk["text"].encode())
    return digest.hexdigest()


def seed(corpus, strategy):
    started = perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    prepared, records = [], []
    for identifier, text in corpus.items():
        records.append((identifier, text))
        pieces = split_text(text, 650, 80, strategy)
        prepared.extend(
            {
                "document_id": identifier,
                "chunk_id": hashlib.sha256((identifier + ":" + str(index)).encode()).hexdigest(),
                "chunk_index": index,
                "text": piece,
                "created_at": now,
            }
            for index, piece in enumerate(pieces)
        )
    print(f"Embedding {len(prepared)} {strategy} chunks from {len(corpus)} documents", flush=True)
    import numpy as np

    cache_path = Path("data/embeddings") / (embedding_signature(prepared, backends.embedding_key()) + ".npz")
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            positions = {identifier: index for index, identifier in enumerate(cached["ids"].tolist())}
            embeddings = cached["vectors"][[positions[chunk["chunk_id"]] for chunk in prepared]]
        print("Reusing content/model-addressed benchmark embeddings", flush=True)
    else:
        backend = backends.get_embeddings()
        embeddings = np.concatenate(
            [
                np.asarray(
                    backend.embed_documents([chunk["text"] for chunk in prepared[start : start + 256]]),
                    dtype="float32",
                )
                for start in range(0, len(prepared), 256)
            ]
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            ids=np.asarray([chunk["chunk_id"] for chunk in prepared]),
            vectors=np.asarray(embeddings, dtype="float32"),
        )
    # Bulk transactions make corpus setup reproducible and avoid per-document
    # model calls. The schema and vector indexing are the same as live ingestion.
    with postgres.transaction() as connection:
        connection.execute("SET LOCAL statement_timeout = '300s'")
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO rag_documents (tenant_id,document_id,filename,file_type,status,created_at,content,original) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    (
                        tenant_id.get(),
                        identifier,
                        identifier + ".txt",
                        "txt",
                        "ingested",
                        now,
                        text,
                        text.encode(),
                    )
                    for identifier, text in records
                ],
            )
            for start in range(0, len(prepared), 512):
                batch = prepared[start : start + 512]
                cursor.executemany(
                    "INSERT INTO rag_chunks (tenant_id,chunk_id,document_id,chunk_index,text,created_at,embedding_key,embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                    [
                        (
                            tenant_id.get(),
                            chunk["chunk_id"],
                            chunk["document_id"],
                            chunk["chunk_index"],
                            chunk["text"],
                            now,
                            backends.embedding_key(),
                            str(vector.tolist()),
                        )
                        for chunk, vector in zip(batch, embeddings[start : start + 512])
                    ],
                )
        postgres._bump(connection)
    return {"chunks": len(prepared), "ingestion_seconds": perf_counter() - started}


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)]


def run(
    dataset="scifact",
    queries=100,
    generate=0,
    seed_value=42,
    exclude_report=None,
    k=10,
    output=None,
    generation_backend="local",
    generation_only=False,
):
    import torch

    torch.set_num_threads(int(os.getenv("MODEL_THREADS", "4")))
    offline.set_offline(False)
    if generation_backend == "configured" and generate and not backends.use_openai():
        raise ValueError("Configured generation evaluation requires OPENAI_API_KEY")
    os.environ["EMBEDDING_BACKEND"] = "sentence_transformers"
    os.environ["RERANKER_BACKEND"] = "cross_encoder"
    if not postgres.enabled():
        raise ValueError("Public benchmark requires DATABASE_URL with migrated shared storage")
    postgres.readiness()
    corpus, all_cases, manifest = load(dataset)
    reports = [exclude_report] if isinstance(exclude_report, str) else (exclude_report or [])
    excluded = {qid for path in reports for qid in json.loads(Path(path).read_text())["query_ids"]}
    pool = sorted((case for case in all_cases if case["id"] not in excluded), key=lambda case: case["id"])
    sample = random.Random(seed_value).sample(pool, min(queries, len(pool)))
    suffix = "-production" if generation_backend == "configured" else ""
    report_path = Path(output or f"evals/reports/{dataset}{suffix}.json")
    previous = None
    # A separate --output report evaluates generation on its own (larger) sample
    # with the production retrieval configuration; it does not inherit ablations.
    if generation_only and not output:
        previous_path = report_path
        if not previous_path.exists() and generation_backend == "configured":
            previous_path = Path("evals/reports/squad.json")
        if dataset != "squad" or not generate or not previous_path.exists():
            raise ValueError("Generation-only requires an existing SQuAD retrieval report and --generate")
        previous = json.loads(previous_path.read_text())
        if (
            previous["query_ids"] != [case["id"] for case in sample]
            or previous["dataset"]["sha256"] != manifest["sha256"]
        ):
            raise ValueError("Generation-only must reuse the exact dataset and query sample")
    # Load weights outside timed query intervals.
    models.encoder()
    models.reranker()
    runs, generation, pending_generation = [], [], []
    for strategy in ("recursive",) if generation_only else ("fixed", "recursive"):
        token = tenant_id.set("benchmark-" + uuid.uuid4().hex)
        try:
            indexing = seed(corpus, strategy)
            for mode, rerank in (
                (("hybrid", True),)
                if generation_only
                else (("vector", False), ("keyword", False), ("hybrid", False), ("hybrid", True))
            ):
                cases = []
                for number, case in enumerate(sample):
                    # Shared-cache keys include this unique benchmark tenant,
                    # mode and corpus revision, so the first request is cold.
                    results, diagnostics = retrieve(case["question"], mode, rerank, k, 12000)
                    _, warm = retrieve(case["question"], mode, rerank, k, 12000)
                    actual = [document.metadata["document_id"] for document, _ in results]
                    row = {
                        "id": case["id"],
                        "expected": case["sources"],
                        "retrieved": list(dict.fromkeys(actual)),
                        **metrics(actual, case["sources"], k),
                        **diagnostics,
                        "warm_retrieval_ms": warm["retrieval_ms"],
                    }
                    cases.append(row)
                    if (
                        dataset == "squad"
                        and strategy == "recursive"
                        and mode == "hybrid"
                        and rerank
                        and number < generate
                    ):
                        pending_generation.append((case, [document for document, _ in results]))
                summary = {
                    metric: mean(case[metric] for case in cases)
                    for metric in ("precision_at_1", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")
                }
                summary.update(
                    cold_p50_ms=percentile([case["retrieval_ms"] for case in cases], 0.5),
                    cold_p95_ms=percentile([case["retrieval_ms"] for case in cases], 0.95),
                    warm_p95_ms=percentile([case["warm_retrieval_ms"] for case in cases], 0.95),
                    mean_context_tokens_estimate=mean(case["estimated_context_tokens"] for case in cases),
                )
                runs.append(
                    {
                        "strategy": strategy,
                        "mode": mode,
                        "neural_rerank": rerank,
                        **indexing,
                        **summary,
                        "cases": cases,
                    }
                )
                print(strategy, mode, rerank, json.dumps(summary), flush=True)
            if pending_generation and not generation:
                # Runs while this strategy's corpus is still indexed, so answer
                # recovery can re-retrieve with a rewritten query.
                generation.extend(_generate(pending_generation, generation_backend, k))
        finally:
            with postgres.transaction() as connection:
                connection.execute("SET LOCAL statement_timeout = '300s'")
                connection.execute("DELETE FROM rag_documents WHERE tenant_id=%s", (tenant_id.get(),))
                connection.execute("DELETE FROM rag_revisions WHERE tenant_id=%s", (tenant_id.get(),))
            tenant_id.reset(token)
    report = {
        "dataset": manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed_value,
        "excluded_query_ids_from": exclude_report,
        "query_sample_size": len(sample),
        "query_ids": [case["id"] for case in sample],
        "k": k,
        "models": {
            "embedding": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": "cross-encoder/ms-marco-TinyBERT-L2-v2",
            "generator": (
                backends.settings.chat_model if generation_backend == "configured" else "google/flan-t5-small"
            )
            if generate
            else None,
            "verifier": "cross-encoder/nli-deberta-v3-xsmall" if generate else None,
            "revisions": {name: models.revision(name) for name in models.REVISIONS},
        },
        "model_device": os.getenv("MODEL_DEVICE", "cpu"),
        "storage": "PostgreSQL pgvector + indexed full-text search + Redis retrieval cache",
        "latency_scope": "Query-level cold/warm; weights preloaded, query cache cold first and warm second; no concurrent load",
        "generation_backend": generation_backend,
        "api_cost_usd": None if generation_backend == "configured" and generate else 0,
        "cost_scope": "Provider usage is recorded for configured generation. Default local inference has no API charge. Hardware/infrastructure and embedding costs are excluded.",
        "runs": runs,
        "generation": generation,
    }
    if generation:
        report["generation_summary"] = {
            "cases": len(generation),
            "exact_match": mean(row["answer_exact_match"] for row in generation),
            "token_f1": mean(row["answer_token_f1"] for row in generation),
            "reference_span_hit": mean(row["reference_span_hit"] for row in generation),
            "abstention_rate": mean(row["abstention_reason"] == "model_abstained" for row in generation),
            # Over answers the model actually attempted; refusals carry no claims.
            "unsupported_claim_rate_nli": mean(
                row["unsupported_claim_rate_nli"]
                for row in generation
                if row["abstention_reason"] != "model_abstained"
            )
            if any(row["abstention_reason"] != "model_abstained" for row in generation)
            else 0.0,
            "grounding_gate_coverage": mean(row["accepted_by_grounding_gate"] for row in generation),
            "previous_release_coverage": mean(row["accepted_by_previous_release"] for row in generation),
            "recovered_by_salvage": mean(row["recovery_path"] == "salvage" for row in generation),
            "recovered_by_retry": mean(row["recovery_path"] == "retry" for row in generation),
            "declined_low_rank_evidence": mean(
                row["abstention_reason"] == "low_rank_evidence_only" for row in generation
            ),
            "partial_delivery_rate": mean(row["partial_delivery"] for row in generation),
            "escalation_rescue_rate": mean(row["escalation_rescued_claims"] > 0 for row in generation),
            # Outcomes of the delivered (post-gate) text against human references.
            "delivered_with_reference_answer": mean(
                row["accepted_by_grounding_gate"] and row["delivered_reference_span_hit"] == 1.0
                for row in generation
            ),
            "delivered_without_reference_answer": mean(
                row["accepted_by_grounding_gate"] and row["delivered_reference_span_hit"] == 0.0
                for row in generation
            ),
            "median_generation_and_verification_ms": median(
                row["generation_and_verification_ms"] for row in generation
            ),
            "invalid_citation_rate": mean(row["invalid_citations"] > 0 for row in generation),
            "limitations": "Generator backend is explicit; NLI is an imperfect automated contextual-support measurement, not a truth oracle.",
        }
    if previous:
        report["runs"] = previous["runs"]
        report["generation_scope"] = (
            "Generation refreshed using the same data/query sample and fresh recursive hybrid neural retrieval. Original full retrieval comparisons retained."
        )
    path = report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _generate(pending, generation_backend, k):
    from app.rag.answering import grounded_answer

    rows = []
    for case, documents in pending:
        started = perf_counter()
        usage = {}
        if generation_backend == "configured":

            def retrieve_more(query):
                return [document for document, _ in retrieve(query, "hybrid", True, k, 12000)[0]]

            result = grounded_answer(case["question"], documents, usage, retrieve_more)
            verdict, recovery = result["verdict"], result["recovery"]
            answer = recovery["initial_answer"]
        else:
            from app.rag.answering import verify_answer

            answer = models.local_answer(case["question"], "\n\n".join(d.page_content for d in documents))
            verdict, recovery = verify_answer(answer, documents), {"path": "initial"}
        if verdict.get("abstention_reason") == "model_abstained":
            assessment = {"claims": [], "unsupported_claim_rate_nli": 0.0}
        else:
            assessment = verdict.get("assessment") or {"claims": [], "unsupported_claim_rate_nli": 0.0}
        delivered = verdict.get("answer") if verdict["accepted"] else None
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": answer,
                "reference_answers": case["answers"],
                "provider_usage": usage,
                "delivered_answer": delivered,
                **verdict["citation_validation"],
                **answer_metrics(answer, case["answers"]),
                "delivered_reference_span_hit": answer_metrics(delivered, case["answers"])[
                    "reference_span_hit"
                ]
                if delivered
                else 0.0,
                **assessment,
                "generation_and_verification_ms": (perf_counter() - started) * 1000,
                "accepted_by_grounding_gate": verdict["accepted"],
                "partial_delivery": bool(verdict.get("partial")),
                "dropped_claims": verdict.get("dropped_claims", []),
                "escalated_claims": sum("escalation_score" in c for c in assessment["claims"]),
                "escalation_rescued_claims": sum(
                    c.get("verified_by") == "escalation" for c in assessment["claims"]
                ),
                "min_cited_rank": verdict.get("min_cited_rank"),
                "recovery_path": recovery.get("path"),
                "recovery": {key: value for key, value in recovery.items() if key != "initial_answer"},
                # Counterfactual: what the previous release (no recovery, no rank
                # guard) would have delivered for this same first answer.
                "accepted_by_previous_release": recovery.get("initial_reason")
                in (None, "low_rank_evidence_only"),
                "previous_release_answer": recovery.get("initial_delivered"),
                "abstention_reason": verdict.get("abstention_reason"),
            }
        )
        print(len(rows), recovery.get("path"), verdict.get("abstention_reason"), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scifact", "squad"), default="scifact")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-report",
        action="append",
        help="Held-out evaluation: never sample query ids from this report (repeatable)",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Refresh generation for an existing report without repeating all retrieval ablations",
    )
    parser.add_argument(
        "--generation-backend",
        choices=("local", "configured"),
        default="local",
        help="configured calls the production provider and incurs API charges",
    )
    args = parser.parse_args()
    if args.queries < 1 or args.generate < 0:
        parser.error("queries must be positive and generate must be nonnegative")
    run(
        args.dataset,
        args.queries,
        args.generate,
        output=args.output,
        seed_value=args.seed,
        exclude_report=args.exclude_report,
        generation_backend=args.generation_backend,
        generation_only=args.generation_only,
    )
