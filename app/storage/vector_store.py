"""Chroma vector store for uploaded-document retrieval.

Uploaded document chunks are embedded into a persistent Chroma collection at
ingest time; ``/query/documents`` retrieves from it via similarity search.
The collection lives under the active storage root, so tests that repoint
storage (``document_service.configure_storage``) stay fully isolated.

Embeddings come from ``rag_backends`` (OpenAI in production, a deterministic
local backend offline).
"""

from functools import lru_cache
from time import monotonic

from app.documents import service as document_service
from app.rag import backends as rag_backends
from app.storage import postgres


COLLECTION_NAME = "documents"

# Cache Chroma handles by persist path so ingest and query in the same process
# share one collection without reopening it on every call.
_stores: dict[str, object] = {}


def _persist_dir() -> str:
    return str(document_service.storage_root() / "chroma")


def _collection_name() -> str:
    # Namespace the collection by embedding backend. OpenAI embeddings (1536-d)
    # and the local hashing embeddings (4096-d) have different dimensions; a
    # single Chroma collection fixes its dimension at creation, so mixing them
    # would raise a dimension-mismatch error. Separate collections keep a
    # persisted store valid across backend switches (each backend re-ingests
    # into its own collection).
    return f"{COLLECTION_NAME}_{rag_backends.embedding_key()}"


def _get_store():
    from langchain_chroma import Chroma

    persist_dir = _persist_dir()
    collection = _collection_name()
    cache_key = f"{persist_dir}::{collection}"
    store = _stores.get(cache_key)
    if store is None:
        store = Chroma(
            collection_name=collection,
            embedding_function=rag_backends.get_embeddings(),
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        _stores[cache_key] = store
    return store


def index_document(document_id: str, chunks: list[dict]) -> None:
    """Embed a document's chunks into the collection, replacing any prior ones."""
    if postgres.enabled():
        vectors = (
            rag_backends.get_embeddings().embed_documents([chunk["text"] for chunk in chunks])
            if chunks
            else []
        )
        postgres.replace_chunks(document_id, chunks, vectors, rag_backends.embedding_key())
        return
    store = _get_store()

    # Idempotent re-ingest: drop this document's existing vectors first.
    store._collection.delete(where={"document_id": document_id})

    if not chunks:
        _cached_search.cache_clear()
        return

    store.add_texts(
        texts=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "document_id": document_id,
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
        ids=[chunk["chunk_id"] for chunk in chunks],
    )
    _cached_search.cache_clear()


def search(query: str, k: int) -> list[tuple[object, float]]:
    """Return up to ``k`` (Document, relevance_score) pairs for the query."""
    if postgres.enabled():
        return postgres.dense_search(
            rag_backends.get_embeddings().embed_query(query), rag_backends.embedding_key(), k
        )
    # Namespace by full tenant storage root, backend, and a short TTL. Ingest
    # clears entries so changed corpora cannot serve stale retrieval results.
    return list(_cached_search(_persist_dir(), _collection_name(), query, k, int(monotonic() // 30)))


@lru_cache(maxsize=256)
def _cached_search(persist_dir, collection, query, k, time_bucket):
    return tuple(_get_store().similarity_search_with_relevance_scores(query, k=k))


def reset() -> None:
    """Drop cached store handles (used by tests between storage swaps)."""
    _stores.clear()
    _cached_search.cache_clear()


def clear_search_cache():
    """Clear bounded query cache, e.g. for cold-versus-warm benchmarks."""
    _cached_search.cache_clear()
