from app.core.storage_io import synchronized, atomic_json
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.documents import service as document_service
from app.storage import postgres


def _chunks_file() -> Path:
    # Reuse the document service storage root so a single configure_storage()
    # call (e.g. in tests) covers documents and chunks alike.
    return document_service.storage_root() / "chunks.json"


def _read_chunks() -> list[dict]:
    chunks_file = _chunks_file()
    if not chunks_file.exists():
        return []
    with chunks_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_chunks(records: list[dict]) -> None:
    chunks_file = _chunks_file()
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(chunks_file, records)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows (pure Python, no LLM deps)."""
    if not text.strip():
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        window = text[start : start + chunk_size].strip()
        if window:
            chunks.append(window)
        start += step
    return chunks


@synchronized
def ingest_document(document_id: str, strategy="fixed", chunk_size=None, overlap=None) -> list[dict]:
    """Chunk a previously uploaded document's text and persist the chunks.

    Re-ingesting replaces any existing chunks for the same document. Returns
    the chunk records created for this document.
    """
    document = document_service.get_document(document_id)
    if document is None:
        raise document_service.DocumentNotFoundError(document_id)

    # Local file location is derived from the id (not stored in metadata),
    # so this works whether the metadata came from PostgreSQL or local JSON.
    text = document_service.document_text(document_id)
    pieces = split_text(
        text,
        chunk_size or settings.chunk_size,
        settings.chunk_overlap if overlap is None else overlap,
        strategy,
    )

    import os

    if len(pieces) > int(os.getenv("MAX_DOCUMENT_CHUNKS", "5000")):
        raise postgres.TenantQuotaExceeded("Document chunk count exceeds processing limit")

    created_at = datetime.now(timezone.utc).isoformat()
    new_chunks = [
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": document_id,
            "chunk_index": index,
            "text": piece,
            "created_at": created_at,
        }
        for index, piece in enumerate(pieces)
    ]

    from app.storage import vector_store

    vector_store.index_document(document_id, new_chunks)
    if not postgres.enabled():
        others = [c for c in _read_chunks() if c["document_id"] != document_id]
        _write_chunks(others + new_chunks)

    return new_chunks


def list_chunks(document_id: str) -> list[dict]:
    """Return chunks for a single document, ordered by chunk_index."""
    if postgres.enabled():
        return postgres.list_chunks(document_id)
    chunks = [c for c in _read_chunks() if c["document_id"] == document_id]
    return sorted(chunks, key=lambda c: c["chunk_index"])


def all_chunks() -> list[dict]:
    """Return every stored chunk across all documents."""
    return postgres.list_chunks() if postgres.enabled() else _read_chunks()


def split_text(text, chunk_size=650, overlap=80, strategy="fixed"):
    """Compare fixed windows with recursive paragraph/sentence boundaries."""
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    if strategy == "fixed":
        return _split_text(text, chunk_size, overlap)
    if strategy != "recursive":
        raise ValueError("Unknown chunking strategy")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    ).split_text(text)
