# Deployment and operations

## Local deployment

Run `scripts/setup_local.py` once and `docker compose up --build -d`. PostgreSQL
and Redis have no published ports. The API is bound to loopback. Named volumes
persist the database, cache, and downloaded models. The container runs as a
non-root user. The migration job uses administrator credentials, while the API
receives only a restricted login. `docker compose down` preserves volumes;
never add `-v` unless intentionally deleting the deployment's data.

Use `APP_ENV=production` with a configured generation provider for the production
path. Startup validates dependencies and refuses fallback to hashing or missing
security controls. For a remote deployment, configure TLS at the ingress and
SSL/TLS database and Redis connections, rotate generated secrets, restrict
network access, and manage credentials with the platform's secret store.
The local Compose stack is not a multi-region highly available cluster.

This workspace evaluates a private Ollama service through the same
OpenAI-compatible production interface. The default is Qwen3 4B-Instruct (Q4_K_M,
~2.5 GB, ~7 s median CPU answer); Qwen3 1.7B is the lower-latency alternative
(~3 s) with lower verified coverage. See `model_selection` in `evals/reports/generation_quality.json`. The smaller
FLAN-T5 baseline is reported separately. Local inference incurs no commercial
API charge; hardware costs are separate. `OPENAI_BASE_URL` selects the provider;
`DOCKER_OPENAI_BASE_URL` overrides it inside Docker when the service runs on the
host. The loopback Ollama token is an SDK protocol placeholder, not a cloud key.
Use real authentication/TLS for inference services outside the trusted host.
Model choice is configurable; rerun provider and domain-specific checks whenever
changing it. Public benchmark results are not a universal quality certificate.

## Health, observability, and limits

- `/health`: process liveness.
- `/ready`: dependency readiness and database-role/RLS checks; failure is 503.
- `/metrics`: operator-only `X-Metrics-Key` access (configure `METRICS_API_KEY`) to Prometheus request, retrieval,
  token, and cache metrics. Metrics are process-local; scrape each replica.
- `/admin/audit`: authenticated admin access to append-only tenant events.

Every response carries a bounded `X-Request-ID`. Logs record method, matched
route, status, and duration, without questions, evidence, credentials, or raw
path IDs. Configure log shipping and retention outside the process.

`RATE_LIMIT` uses Redis in the shared deployment. Requests from different keys
of the same tenant share a budget. Set application quotas through
`MAX_TENANT_DOCUMENTS`, `MAX_TENANT_BYTES`, and `MAX_TENANT_CHUNKS`.
`MAX_UPLOAD_BYTES`, `MAX_EXTRACTED_CHARS`, and `MAX_DOCUMENT_CHUNKS` bound
individual ingestion jobs. Quotas return 409; oversized input returns 413.
For sustained ingestion, provision worker capacity and an external durable job
queue; this API performs bounded ingestion synchronously in its thread pool.

## Scale and model rollout

Use one worker per memory-constrained container and scale replicas behind a load
balancer. Each worker loads its own neural weights. Redis and PostgreSQL are
shared; size `DB_POOL_SIZE` times replicas within the database connection budget.
The Compose deployment uses local persistent volumes; managed database/cache
services are needed for host-independent availability. Shared Redis is a
production dependency for quotas/rate limits; monitor its memory and persistence.

Before changing an embedding model, provision its vector index, reingest the
corpus, run public and organization-specific checks, and then route traffic.
Old model indexes remain logically isolated. API credentials are not a substitute
for an organization's SSO, user provisioning, or per-document ACL policy; add
those integrations where required. Current access control is tenant-level RBAC.

## Backups and restore

```bash
scripts/backup.sh /secure/location/docuintel.dump
```

The script uses `pg_dump -Fc`, mode 0600 through a private umask, and refuses to
overwrite existing backups. Encrypt and copy backups outside the hosting machine.
Schedule retention and monitor backup failures. A backup is useful only after a
restore test. Restore into a **new** database using `createdb` and `pg_restore`,
connect with the restricted API role, and verify document counts and cross-tenant
RLS before switching traffic. Never test restoration over the live database.
The `backup_restore` section of `evals/reports/operations.json` records a successful isolated local restore.

## Release checks

```bash
make lint
make test
make test-shared  # TEST_DATABASE_URL / TEST_REDIS_URL configured
make eval-public # migrated disposable DATABASE_URL / REDIS_URL configured
make gates
```

For concurrent API validation, set `RAG_BASE_URL`, `RAG_TEST_API_KEY`, and
optionally `RAG_TEST_QUESTION`, then run:

```bash
python scripts/validate_deployment.py --requests 100 --concurrency 8
```

The load script reports errors, throughput, p50, and p95. Its repeated-query
workload tests the cache path; it does not establish mixed-workload production
capacity. Set the test tenant's rate limit deliberately rather than disabling
production rate limits globally. Validate recovery, failover, latency, generation
quality, and privacy requirements against your actual hosting environment before
routing customer traffic.
