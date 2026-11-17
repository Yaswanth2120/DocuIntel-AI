"""Ranking metrics shared by smoke tests and public evaluations."""

import math


def ranking_metrics(actual, expected, k=None):
    actual = list(dict.fromkeys(actual))
    cutoff = k if k is not None else len(actual)
    actual = actual[:cutoff]
    expected = set(expected)
    hits = [int(identifier in expected) for identifier in actual]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(expected), cutoff)))
    return {
        "precision": sum(hits) / cutoff if cutoff else 0.0,
        "recall": sum(hits) / len(expected) if expected else float(not actual),
        "mrr": next((1 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0),
        "ndcg": dcg / ideal if ideal else 0.0,
    }
