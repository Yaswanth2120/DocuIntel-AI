"""Backend selection for document-QA embeddings and answer generation.

When ``OPENAI_API_KEY`` is configured (and offline mode is not forced) the
production backend is used: OpenAI embeddings (matching ``AdvancedRAGEngine``'s
embedding config) and an OpenAI chat model for answer generation.

Otherwise a local, deterministic, offline fallback is used: hashing embeddings
plus an extractive answer. This fallback has materially lower retrieval quality
than real embeddings, so its use is made loud rather than silent: a warning is
logged and ``/features`` reports ``embedding_backend: "local_hash"``.
"""

import logging
import os
import hashlib

from app.core import offline
from app.core.config import settings
from app.rag.hash_embeddings import LocalHashingEmbeddings


logger = logging.getLogger(__name__)

_warned_local = False


def use_openai() -> bool:
    """True when the OpenAI backend should be used (key set, not forced offline)."""
    return (not offline.is_offline()) and bool(settings.openai_api_key)


def embedding_backend_name() -> str:
    """Name of the active embedding backend: ``"openai"`` or ``"local_hash"``."""
    if offline.is_offline():
        return "local_hash"
    selected = os.getenv("EMBEDDING_BACKEND", "auto")
    if selected not in {"auto", "openai", "sentence_transformers", "local_hash"}:
        raise ValueError("Unsupported EMBEDDING_BACKEND")
    return ("openai" if use_openai() else "local_hash") if selected == "auto" else selected


def _warn_local_once() -> None:
    global _warned_local
    if not _warned_local:
        logger.warning(
            "Document-QA is using the LOCAL hashing embedding fallback "
            "(OPENAI_API_KEY not set or offline mode forced). Retrieval quality "
            "is degraded compared to OpenAI embeddings. Set OPENAI_API_KEY for "
            "production use."
        )
        _warned_local = True


def get_embeddings():
    """Return the embeddings object for the document-QA vector store."""
    if embedding_backend_name() == "sentence_transformers":
        from app.rag.models import SentenceEmbeddings

        return SentenceEmbeddings()
    if embedding_backend_name() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)

    _warn_local_once()
    return LocalHashingEmbeddings()


def get_llm():
    """Return the chat model used to generate grounded answers, or None locally.

    When the OpenAI backend is not active there is no LLM; the document-QA
    service falls back to an extractive answer built from the retrieved chunks.
    """
    if not use_openai():
        return None

    from langchain_openai import ChatOpenAI

    options = {}
    if os.getenv("MODEL_REASONING_EFFORT"):
        options["reasoning_effort"] = os.environ["MODEL_REASONING_EFFORT"]
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "512")),
        timeout=30,
        max_retries=2,
        **options,
    )


def embedding_key():
    from app.rag.models import revision

    backend = embedding_backend_name()
    model = (
        settings.embedding_model
        if backend == "openai"
        else (
            os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            + ":"
            + revision("LOCAL_EMBEDDING_REVISION")
            if backend == "sentence_transformers"
            else "hash4096-v1"
        )
    )
    return backend + "_" + hashlib.sha256(model.encode()).hexdigest()[:16]
