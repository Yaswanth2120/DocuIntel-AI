"""Concurrent API read/query load smoke test with percentile/error reporting.

RAG_BASE_URL and RAG_TEST_API_KEY target an explicitly configured deployment.
This never uploads or deletes user data. Query workload is a benign fixed policy
question; customize it for domain SLO testing through RAG_TEST_QUESTION.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
from time import perf_counter
import httpx


def run(requests=100, concurrency=8, output="evals/reports/load.json"):
    base = os.environ["RAG_BASE_URL"].rstrip("/")
    key = os.environ["RAG_TEST_API_KEY"]
    question = os.getenv("RAG_TEST_QUESTION", "What authentication is required for production access?")
    with httpx.Client(base_url=base, headers={"X-API-Key": key}, timeout=60) as client:
        client.get("/ready").raise_for_status()

        def request(number):
            start = perf_counter()
            response = client.post(
                "/query/documents", json={"question": question, "mode": "hybrid", "rerank": True}
            )
            return {
                "status": response.status_code,
                "latency_ms": (perf_counter() - start) * 1000,
                "request_id": response.headers.get("x-request-id"),
            }

        started = perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(request, range(requests)))
        elapsed = perf_counter() - started
    latencies = sorted(result["latency_ms"] for result in results)
    report = {
        "requests": requests,
        "concurrency": concurrency,
        "duration_seconds": elapsed,
        "requests_per_second": requests / elapsed,
        "errors": sum(result["status"] != 200 for result in results),
        "p50_ms": latencies[math.ceil(len(latencies) * 0.5) - 1],
        "p95_ms": latencies[math.ceil(len(latencies) * 0.95) - 1],
        "scope": "Repeated-query cache/load smoke test, not a representative mixed production workload",
        "results": results,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if report["errors"]:
        raise SystemExit("Deployment load gate failed: HTTP errors observed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.requests < 1 or not 1 <= args.concurrency <= 64:
        parser.error("Invalid request count or concurrency")
    run(args.requests, args.concurrency)
