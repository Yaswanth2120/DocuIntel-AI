"""Lazy, reusable local neural models. No remote custom model code is executed."""

import os
from functools import lru_cache


REVISIONS = {
    "LOCAL_EMBEDDING_REVISION": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "RERANKER_REVISION": "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc",
    "NLI_REVISION": "a150876415327c80daeff35ca6f68f5ed8cf5c24",
    "ESCALATION_NLI_REVISION": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
    "LOCAL_GENERATION_REVISION": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
}


def revision(variable):
    return os.getenv(variable) or REVISIONS[variable]


@lru_cache(maxsize=1)
def encoder():
    import torch

    torch.set_num_threads(int(os.getenv("MODEL_THREADS", "4")))
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        revision=revision("LOCAL_EMBEDDING_REVISION"),
        device=os.getenv("MODEL_DEVICE", "cpu"),
        trust_remote_code=False,
    )


class SentenceEmbeddings:
    def embed_documents(self, texts):
        return (
            encoder()
            .encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            .tolist()
        )

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L2-v2"),
        revision=revision("RERANKER_REVISION"),
        device=os.getenv("MODEL_DEVICE", "cpu"),
        max_length=512,
        trust_remote_code=False,
    )


@lru_cache(maxsize=1)
def nli_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall"),
        revision=revision("NLI_REVISION"),
        device=os.getenv("MODEL_DEVICE", "cpu"),
        max_length=512,
        trust_remote_code=False,
    )


@lru_cache(maxsize=1)
def escalation_nli_model():
    """Stronger NLI (MNLI+FEVER+ANLI, MIT) consulted only for rejected claims."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        os.getenv("ESCALATION_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
        revision=revision("ESCALATION_NLI_REVISION"),
        device=os.getenv("MODEL_DEVICE", "cpu"),
        max_length=512,
        trust_remote_code=False,
    )


@lru_cache(maxsize=1)
def local_generator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model = os.getenv("LOCAL_GENERATION_MODEL", "google/flan-t5-small")
    return (
        AutoTokenizer.from_pretrained(model, revision=revision("LOCAL_GENERATION_REVISION")),
        AutoModelForSeq2SeqLM.from_pretrained(model, revision=revision("LOCAL_GENERATION_REVISION")).eval(),
    )


def local_answer(question, context):
    """Small local generative baseline for reproducible no-key evaluation."""
    import torch

    tokenizer, model = local_generator()
    # Keep the question and instruction before the bounded context. Truncation
    # must never silently turn QA into an unrelated context-completion task.
    prompt = "Answer this question using the context. Question: " + question + " Context: " + context
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)
