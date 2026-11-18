ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS documents_tenant_id_idx ON public.documents (tenant_id, document_id);
