"""Nonzero exit on missing/invalid public evidence or retrieval SLO failures."""

import json
from pathlib import Path


def check(directory=Path("evals/reports")):
    errors = []
    for name in ("scifact", "squad"):
        path = directory / (name + ".json")
        if not path.exists():
            errors.append(f"{name}: missing report")
            continue
        report = json.loads(path.read_text())
        if report["query_sample_size"] < 100 or len(report["runs"]) != 8:
            errors.append(f"{name}: insufficient evaluation coverage")
        production = [
            row
            for row in report["runs"]
            if row["strategy"] == "recursive" and row["mode"] == "hybrid" and row["neural_rerank"]
        ]
        if len(production) != 1:
            errors.append(f"{name}: missing production retrieval configuration")
            continue
        row = production[0]
        for metric, minimum in [("recall_at_k", 0.75), ("ndcg_at_k", 0.45)]:
            if row[metric] < minimum:
                errors.append(f"{name}: {metric}={row[metric]:.3f} below {minimum}")
        if row["cold_p95_ms"] > 2000 or row["warm_p95_ms"] > 100:
            errors.append(f"{name}: query latency exceeds the configured local benchmark gate")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        "Public retrieval quality/latency gates passed. These gates do not certify generation quality or production capacity."
    )


if __name__ == "__main__":
    check()
