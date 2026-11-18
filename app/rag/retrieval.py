"""Bounded hybrid retrieval with reciprocal rank fusion and lexical reranking.

Scores are ranking signals, not calibrated probabilities of answer correctness.
"""

import math
import os
import re
from collections import Counter
from time import perf_counter

from langchain_core.documents import Document
from app.core.config import settings
from app.rag.models import revision
from app.core.storage_io import synchronized
from app.storage import vector_store
from app.documents import chunking as chunk_service
from app.rag import backends as rag_backends
from app.storage import postgres, cache
from app.core.tenancy import tenant_id
from app.core.observability import RETRIEVAL


STOPWORDS = set(
    "a an the is are was were be been to of in on at for from with and or by as it this that what which how do does can when who within".split()
)


def tokens(text):
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOPWORDS]


def bm25(query, documents):
    terms = [Counter(tokens(text)) for text in documents]
    lengths = [sum(c.values()) for c in terms]
    average = sum(lengths) / max(1, len(lengths)) or 1
    scores = []
    for counts, length in zip(terms, lengths):
        score = 0.0
        for word in set(tokens(query)):
            df = sum(word in row for row in terms)
            idf = math.log(1 + (len(terms) - df + 0.5) / (df + 0.5))
            frequency = counts[word]
            score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
        scores.append(score)
    return scores


@synchronized
def _retrieve(question, mode="hybrid", rerank=True, top_k=None, context_chars=12000):
    start = perf_counter()
    k = top_k or settings.top_k
    candidates = min(100, max(k * 4, 20))
    dense = (
        [(d, s) for d, s in vector_store.search(question, candidates) if s >= settings.relevance_threshold]
        if mode != "keyword"
        else []
    )
    dense_scores = {d.metadata["chunk_id"]: float(score) for d, score in dense}
    docs = {d.metadata["chunk_id"]: d for d, _ in dense}
    rankings = [[d.metadata["chunk_id"] for d, _ in dense]] if dense else []
    if mode != "vector" and postgres.enabled():
        lexical = postgres.keyword_search(question, rag_backends.embedding_key(), candidates)
        rankings.append([d.metadata["chunk_id"] for d, _ in lexical])
        docs.update({d.metadata["chunk_id"]: d for d, _ in lexical})
    elif mode != "vector":
        chunks = chunk_service.all_chunks()
        scores = bm25(question, [c["text"] for c in chunks])
        lexical = sorted(zip(chunks, scores), key=lambda row: row[1], reverse=True)
        lexical = [(c, s) for c, s in lexical[:candidates] if s > 0]
        rankings.append([c["chunk_id"] for c, _ in lexical])
        for c, _ in lexical:
            docs[c["chunk_id"]] = Document(
                page_content=c["text"], metadata={"chunk_id": c["chunk_id"], "document_id": c["document_id"]}
            )
    fusion = Counter()
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, 1):
            fusion[identifier] += 1 / (60 + rank)
    ordered = sorted(docs, key=lambda identifier: fusion[identifier], reverse=True)
    # An inexpensive second-stage lexical reranker over fused candidates.
    # This is deliberately labeled lexical; it is not a neural cross-encoder.
    if rerank:
        ordered = ordered[: max(k, min(100, int(os.getenv("RAG_RERANK_CANDIDATES", "40"))))]
    neural = rerank and os.getenv("RERANKER_BACKEND", "lexical") == "cross_encoder"
    if neural and ordered:
        from app.rag.models import reranker

        scores = reranker().predict(
            [(question, docs[i].page_content) for i in ordered], batch_size=32, show_progress_bar=False
        )
        neural_scores = dict(zip(ordered, map(float, scores)))
        ordered.sort(key=lambda identifier: neural_scores[identifier], reverse=True)
    elif rerank and ordered:
        scores = bm25(question, [docs[i].page_content for i in ordered])
        maximum = max(scores) or 1
        reranked = {
            i: 0.7 * fusion[i] * 61 / max(1, len(rankings)) + 0.3 * score / maximum
            for i, score in zip(ordered, scores)
        }
        ordered.sort(key=lambda i: reranked[i], reverse=True)
    selected = []
    per_document = Counter()
    maximum_per_document = int(os.getenv("RAG_MAX_CHUNKS_PER_DOCUMENT", "2"))
    remaining = context_chars
    for identifier in ordered:
        doc = docs[identifier]
        if len(selected) >= k:
            break
        if per_document[doc.metadata["document_id"]] >= maximum_per_document:
            continue
        if len(doc.page_content) > remaining:
            continue
        selected.append(
            (
                doc,
                dense_scores[identifier]
                if mode == "vector"
                else min(1.0, fusion[identifier] * 61 / max(1, len(rankings))),
            )
        )
        remaining -= len(doc.page_content)
        per_document[doc.metadata["document_id"]] += 1
    return selected, {
        "retrieval_ms": round((perf_counter() - start) * 1000, 3),
        "candidate_count": len(docs),
        "context_chars": context_chars - remaining,
        "estimated_context_tokens": math.ceil((context_chars - remaining) / 4),
        "mode": mode,
        "reranker": "cross_encoder" if neural else ("lexical_bm25" if rerank else "none"),
    }


def retrieve(question, mode="hybrid", rerank=True, top_k=None, context_chars=12000):
    start = perf_counter()
    identifier = None
    if postgres.enabled():
        configuration = {
            "tenant": tenant_id.get(),
            "revision": postgres.revision(),
            "embedding": rag_backends.embedding_key(),
            "query": question,
            "mode": mode,
            "rerank": rerank,
            "reranker": os.getenv("RERANKER_BACKEND", "lexical"),
            "model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L2-v2"),
            "model_revision": revision("RERANKER_REVISION"),
            "rerank_candidates": os.getenv("RAG_RERANK_CANDIDATES", "40"),
            "max_chunks_per_document": os.getenv("RAG_MAX_CHUNKS_PER_DOCUMENT", "2"),
            "top_k": top_k,
            "context_chars": context_chars,
            "threshold": settings.relevance_threshold,
        }
        identifier = cache.key(configuration)
        cached = cache.get(identifier)
        if cached:
            results = [
                (Document(page_content=row["text"], metadata=row["metadata"]), row["score"])
                for row in cached["results"]
            ]
            diagnostics = {
                **cached["diagnostics"],
                "cache_hit": True,
                "retrieval_ms": round((perf_counter() - start) * 1000, 3),
            }
            RETRIEVAL.labels(mode, diagnostics["reranker"]).observe((perf_counter() - start))
            return results, diagnostics
    results, diagnostics = _retrieve(question, mode, rerank, top_k, context_chars)
    diagnostics["cache_hit"] = False
    if identifier:
        cache.put(
            identifier,
            {
                "results": [{"text": d.page_content, "metadata": d.metadata, "score": s} for d, s in results],
                "diagnostics": diagnostics,
            },
        )
    diagnostics["retrieval_ms"] = round((perf_counter() - start) * 1000, 3)
    RETRIEVAL.labels(mode, diagnostics["reranker"]).observe((perf_counter() - start))
    return results, diagnostics
