# DocuIntel AI

A multi-tenant RAG API with shared PostgreSQL storage, pgvector semantic search,
indexed full-text search, neural reranking, independent grounding verification,
and reproducible public-data evaluations.

## Start locally

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker Compose.

```bash
uv sync --frozen --extra dev --python 3.12
uv run python scripts/setup_local.py
docker compose up --build -d
curl http://127.0.0.1:8000/ready
```

`setup_local.py` creates a private `.env` with random database, Redis, and tenant
credentials. It never overwrites an existing file. Read your admin API key from
`TENANT_API_KEYS` in that file. The first neural-model download takes time.
The default local setup uses real semantic retrieval and an extractive answer
when no generation API key is configured.

To enable production generation, set `OPENAI_API_KEY` and `APP_ENV=production`
in `.env`, then recreate the API container. `OPENAI_BASE_URL` can point at a
private OpenAI-compatible inference service. Production startup requires real
embeddings, the neural reranker, grounding verification, restricted database
permissions, Redis, and strong tenant credentials; it refuses silent fallback.

An existing private Ollama endpoint also works through this interface. For a
host service, use `OPENAI_BASE_URL=http://127.0.0.1:11434/v1` and
`DOCKER_OPENAI_BASE_URL=http://host.docker.internal:11434/v1`; set the installed
chat model and the local protocol token. `MODEL_REASONING_EFFORT=none` can disable
thinking for compatible models. This workspace's private `.env` is configured
for `qwen3:4b-instruct-2507-q4_K_M` (`ollama pull qwen3:4b-instruct-2507-q4_K_M`).

The static frontend remains available with:

```bash
uv run python -m http.server 5500 --directory frontend
```

Connect to `http://127.0.0.1:8000` and use your tenant API key. The frontend is a
development client; production identity and TLS termination belong at the edge.

## Measured results

Public data, full corpora, 100 seeded queries per dataset, top-10, CPU-only
(2 vCPU / 1.2 GB container). Full reports: [`evals/reports/`](evals/reports/);
method: [evaluation](docs/evaluation.md).

**Retrieval: chunking × search mode × reranking** (recall@10 / nDCG@10)

| Configuration | SciFact (5,183 docs) | SQuAD (2,067 passages) |
| --- | --- | --- |
| Fixed chunks, vector | 0.810 / 0.698 | 0.940 / 0.843 |
| Fixed chunks, keyword (full-text) | 0.613 / 0.435 | 0.810 / 0.630 |
| Fixed chunks, hybrid | 0.761 / 0.663 | 0.960 / 0.842 |
| Fixed chunks, hybrid + reranker | 0.801 / 0.711 | 0.970 / 0.888 |
| Recursive chunks, vector | 0.824 / 0.718 | 0.940 / 0.862 |
| Recursive chunks, keyword | 0.583 / 0.394 | 0.850 / 0.650 |
| Recursive chunks, hybrid | 0.775 / 0.674 | 0.960 / 0.846 |
| **Recursive chunks, hybrid + reranker (default)** | **0.854 / 0.716** | **0.970 / 0.894** |

Default retrieval latency: p95 486 ms cold / 4 ms warm (SciFact), 181 ms / <1 ms
(SQuAD). Redis caches results per tenant, configuration, and corpus revision.
Recursive chunking cuts SciFact prompt context from ~1,530 to ~1,240 estimated
tokens versus fixed chunks at higher recall.

**Generation and hallucination** (SQuAD questions with human reference answers,
Qwen3 4B-Instruct on CPU, full corpus retrieval; [report](evals/reports/generation_quality.json))

| Outcome | Dev baseline (200) | Dev improved (200) | **Held-out final (200)** |
| --- | --- | --- | --- |
| Delivered and correct | 76.5% | 80.0% | **75.5%** |
| Delivered but wrong | 4.0% | 5.0% | **5.5%** |
| Correct answer withheld by the gate | 8.0% | 3.5% | 3.5% |
| Model refused although gold passage retrieved | 8.0% | 9.5% | 11.5% |
| Gold passage not retrieved | 1.5% | 1.5% | 2.5% |

The held-out set (200 new questions, never inspected while tuning) is the
number to quote. On the same held-out answers, the improved gate (stronger NLI
second opinion for rejected claims, supported-sentence delivery, refusal
detection) delivered 14 more answers, 13 of them correct, versus the previous
all-claims policy (69.0% -> 75.5% correct). 93% of delivered answers are correct.
Wrong answers are almost all supported by a retrieved passage but answer a
different reading of a context-dependent question; the gate verifies support,
not intent. Median answer time is ~10 s on 2 CPU cores; API cost is $0.

**Operations:** 100 concurrent requests at concurrency 8 with 0 errors; tenant
isolation enforced by PostgreSQL row-level security (verified even without the
application filter); backup restored into a separate database with policies intact.

## Workflow

All document routes require `X-API-Key`.

1. `POST /documents/upload` — upload `.txt`, `.md`, or `.pdf`.
2. `POST /documents/{id}/ingest?strategy=recursive` — atomically index chunks.
3. `POST /query/documents` — retrieve evidence and answer.
4. `DELETE /documents/{id}` — delete document, chunks, and vectors together.

Example query body:

```json
{
  "question": "What are the incident response requirements?",
  "mode": "hybrid",
  "rerank": true,
  "top_k": 4,
  "context_chars": 12000
}
```

The response includes source IDs, exact retrieved passages, ranking scores,
cache/latency/context diagnostics, and grounding measurements. A retrieval score
is not a probability that an answer is correct. With verification enabled,
streaming buffers generated text until the grounding check passes.

## Verify the system

```bash
make test
make lint
make eval
```

For real shared-storage integration tests, set `TEST_DATABASE_URL` to the
restricted API database URL and `TEST_REDIS_URL` to a disposable Redis database,
then run `make test-shared`. The migration administrator must first run
`scripts/bootstrap_db.py` with `MIGRATION_DATABASE_URL` and `APP_DB_PASSWORD`.

Public evaluation downloads complete corpora from the original providers:

```bash
# Start the Compose stack first; the wrapper keeps database/cache ports private.
make eval-public
make gates
```

The public runner compares two chunking strategies across vector, keyword,
hybrid, and neural-reranked hybrid retrieval, using fixed seeded query samples.
It measures precision/recall, MRR, nDCG, cold/warm latency, and context size.
SQuAD also provides reference answers for a local generation/NLI experiment.
See [evaluation methodology and results](docs/evaluation.md).

## Repository map

```text
app/
  api/          HTTP endpoints and authorization dependencies
  core/         Configuration, identity, quotas, runtime checks, metrics
  documents/    Upload, extraction, chunking, document lifecycle
  rag/          Retrieval, neural models, generation, grounding
  storage/      PostgreSQL/pgvector, Redis, local/legacy adapters
evals/          Public benchmarks, smoke evaluation, metrics, release gates
  fixtures/     Small deterministic offline test questions
  reports/      Measured public-data and operational validation results
frontend/       Static development UI and browser tests
migrations/     Shared-storage schema, indexes, RLS, permissions
scripts/        Local setup, migrations, load verification, backups
tests/          Unit, API, isolation, concurrency, and integration tests
docs/           Architecture, operations, evaluation, cleanup record
```

Tutorial examples and historical planning files were removed. Production source
was reorganized without changing the document API paths. See the
[cleanup record](docs/cleanup.md).

Read [architecture](docs/architecture.md) and [operations](docs/operations.md)
for security boundaries, deployment, scaling, backups, and release checks.

To seed attributed public examples into a tenant, set `RAG_BASE_URL` and
`RAG_TEST_API_KEY`, then run `python -m scripts.seed_public_demo --count 10`.
The script uploads new public passages and never replaces existing documents.
