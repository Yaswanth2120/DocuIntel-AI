"""Seed a tenant with attributed public SQuAD passages through the real API.

Set RAG_BASE_URL and RAG_TEST_API_KEY for the intended tenant. Existing
customer documents are never modified or deleted.
"""

import argparse
import json
import os
import httpx
from evals.datasets import load


def main(count=10):
    documents, cases, manifest = load("squad")
    selected = []
    seen = set()
    for case in cases:
        identifier = case["sources"][0]
        if identifier not in seen:
            seen.add(identifier)
            selected.append(case)
        if len(selected) >= count:
            break
    examples = []
    with httpx.Client(
        base_url=os.environ["RAG_BASE_URL"].rstrip("/"),
        headers={"X-API-Key": os.environ["RAG_TEST_API_KEY"]},
        timeout=120,
    ) as client:
        for case in selected:
            identifier = case["sources"][0]
            text = (
                documents[identifier]
                + "\n\nPublic source: Stanford SQuAD 1.1 development dataset, CC BY-SA 4.0.\n"
                + manifest["url"]
            )
            uploaded = client.post(
                "/documents/upload",
                files={"file": ("squad-" + identifier + ".txt", text.encode(), "text/plain")},
            )
            uploaded.raise_for_status()
            document_id = uploaded.json()["document_id"]
            client.post(
                "/documents/" + document_id + "/ingest", params={"strategy": "recursive"}
            ).raise_for_status()
            examples.append(
                {
                    "document_id": document_id,
                    "question": case["question"],
                    "reference_answers": case["answers"],
                }
            )
    print(json.dumps({"source": manifest["url"], "seeded": len(examples), "examples": examples}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("count must be between 1 and 100")
    main(args.count)
