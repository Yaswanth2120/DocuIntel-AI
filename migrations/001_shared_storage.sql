-- Run with a migration administrator, never the API role.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_documents (
    tenant_id text NOT NULL,
    document_id text NOT NULL,
    filename text NOT NULL,
    file_type text NOT NULL,
    status text NOT NULL DEFAULT 'uploaded',
    created_at timestamptz NOT NULL DEFAULT now(),
    original bytea NOT NULL,
    content text NOT NULL,
    PRIMARY KEY (tenant_id, document_id)
);
CREATE TABLE IF NOT EXISTS rag_chunks (
    tenant_id text NOT NULL,
    chunk_id text NOT NULL,
    document_id text NOT NULL,
    chunk_index integer NOT NULL,
    text text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    embedding_key text NOT NULL,
    embedding vector NOT NULL,
    embedding_dimensions integer GENERATED ALWAYS AS (vector_dims(embedding)) STORED,
    search_text tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (tenant_id, chunk_id),
    FOREIGN KEY (tenant_id, document_id) REFERENCES rag_documents ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS rag_chunks_document ON rag_chunks (tenant_id, document_id);
CREATE INDEX IF NOT EXISTS rag_chunks_search ON rag_chunks USING gin(search_text);
CREATE INDEX IF NOT EXISTS rag_chunks_vector384 ON rag_chunks USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops) WHERE embedding_dimensions=384;
CREATE INDEX IF NOT EXISTS rag_chunks_vector1536 ON rag_chunks USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops) WHERE embedding_dimensions=1536;
CREATE TABLE IF NOT EXISTS rag_revisions (
    tenant_id text PRIMARY KEY,
    revision bigint NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS rag_usage (
    tenant_id text PRIMARY KEY,
    documents bigint NOT NULL DEFAULT 0 CHECK (documents >= 0),
    bytes bigint NOT NULL DEFAULT 0 CHECK (bytes >= 0),
    chunks bigint NOT NULL DEFAULT 0 CHECK (chunks >= 0)
);
CREATE TABLE IF NOT EXISTS rag_audit (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id text NOT NULL,
    action text NOT NULL,
    resource_id text,
    request_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);
DO $$
DECLARE tab text;
BEGIN
    FOREACH tab IN ARRAY ARRAY['rag_documents','rag_chunks','rag_revisions','rag_audit','rag_usage'] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tab);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', tab);
        IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename=tab AND policyname='tenant_isolation') THEN
            EXECUTE format('CREATE POLICY tenant_isolation ON %I USING (tenant_id = nullif(current_setting(''app.tenant_id'', true), '''')) WITH CHECK (tenant_id = nullif(current_setting(''app.tenant_id'', true), ''''))', tab);
        END IF;
    END LOOP;
END $$;
-- The bootstrap creates this restricted login before applying the migration.
GRANT USAGE ON SCHEMA public TO docuintel_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON rag_documents, rag_chunks, rag_revisions, rag_usage TO docuintel_app;
GRANT SELECT, INSERT ON rag_audit TO docuintel_app;
GRANT USAGE ON SEQUENCE rag_audit_id_seq TO docuintel_app;

-- Backfill counters when upgrading an existing database. Existing counter rows
-- remain authoritative; normal mutations update them in the same transaction.
INSERT INTO rag_usage (tenant_id,documents,bytes,chunks)
SELECT d.tenant_id,count(*),sum(octet_length(d.original)),
       (SELECT count(*) FROM rag_chunks c WHERE c.tenant_id=d.tenant_id)
FROM rag_documents d GROUP BY d.tenant_id
ON CONFLICT (tenant_id) DO NOTHING;
