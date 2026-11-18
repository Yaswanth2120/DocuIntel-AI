from app.core.storage_io import synchronized, atomic_json
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from app.storage import postgres


ALLOWED_TYPES = {".txt", ".md", ".pdf"}

# Base storage location. Overridable at runtime (e.g. in tests) via
# ``configure_storage`` so uploads stay out of the repo tree.
STORAGE_DIR = Path("storage")


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file has an extension we do not accept."""


class DocumentTooLarge(ValueError):
    """Extracted document content exceeds configured processing limits."""


class DocumentNotFoundError(LookupError):
    """Raised when a document_id does not match any stored document."""


def configure_storage(base_dir: Path) -> None:
    """Point the service at a different storage root (used by tests)."""
    global STORAGE_DIR
    STORAGE_DIR = Path(base_dir)


def storage_root() -> Path:
    from app.core.tenancy import tenant_id, namespace

    return STORAGE_DIR if tenant_id.get() == "default" else STORAGE_DIR / "tenants" / namespace()


def _uploads_dir() -> Path:
    return storage_root() / "uploads"


def stored_path_for(document_id: str, file_type: str) -> Path:
    """Local path of the original uploaded file for a document."""
    return _uploads_dir() / f"{document_id}.{file_type}"


def text_path_for(document_id: str) -> Path:
    """Local path of the extracted plain text for a document.

    Paths are always DERIVED from document_id; they are never persisted.
    The live Supabase table's legacy ``text_path`` column stays NULL and is
    ignored on read (see app/db/document_repository.py).
    """
    return _uploads_dir() / f"{document_id}.txt"


def _metadata_file() -> Path:
    return storage_root() / "documents.json"


def _file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def _read_metadata() -> list[dict]:
    metadata_file = _metadata_file()
    if not metadata_file.exists():
        return []
    with metadata_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_metadata(records: list[dict]) -> None:
    metadata_file = _metadata_file()
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(metadata_file, records)


def _extract_text(path: Path, extension: str) -> str:
    if extension in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    # PDF
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


@synchronized
def save_upload(filename: str, content: bytes) -> dict:
    """Validate, persist, and extract text from an uploaded document.

    Returns the stored metadata record. Raises ``UnsupportedFileTypeError``
    for disallowed extensions.
    """
    import os

    extension = _file_extension(filename)

    def validate_text(text):
        if len(text) > int(os.getenv("MAX_EXTRACTED_CHARS", "2000000")):
            raise DocumentTooLarge("Extracted document exceeds processing size limit")
        return text

    if extension not in ALLOWED_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or filename}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_TYPES))}."
        )

    document_id = str(uuid.uuid4())
    if postgres.enabled():
        import io

        if extension == ".pdf":
            from pypdf import PdfReader

            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
        else:
            text = content.decode("utf-8", errors="replace")
        record = {
            "document_id": document_id,
            "filename": filename,
            "file_type": extension.lstrip("."),
            "status": "uploaded",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        validate_text(text)
        postgres.save(record, text, content)
        return record
    uploads_dir = _uploads_dir()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_type = extension.lstrip(".")
    stored_path = stored_path_for(document_id, file_type)
    stored_path.write_bytes(content)

    text = validate_text(_extract_text(stored_path, extension))
    text_path_for(document_id).write_text(text, encoding="utf-8")

    # The persisted metadata record is exactly the DocumentMetadata schema
    # (app/schemas.py) and the Supabase ``documents`` table columns.
    # The on-disk file locations are derived from document_id/file_type
    # (see stored_path_for/text_path_for), not stored, so this record matches
    # the table field-for-field whether it goes to Supabase or local JSON.
    record = {
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "status": "uploaded",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    from app.storage import supabase_metadata as document_repository

    if document_repository.enabled():
        document_repository.insert(record)
    else:
        records = _read_metadata()
        records.append(record)
        _write_metadata(records)

    return record


def list_documents(limit=100, offset=0) -> list[dict]:
    """Return all stored document metadata records."""
    if postgres.enabled():
        return postgres.list_documents(limit, offset)
    from app.storage import supabase_metadata as document_repository

    records = document_repository.list_all() if document_repository.enabled() else _read_metadata()
    return records[offset : offset + limit]


def get_document(document_id: str) -> dict | None:
    """Return a single document metadata record, or None if not found."""
    if postgres.enabled():
        return postgres.get_document(document_id)
    from app.storage import supabase_metadata as document_repository

    if document_repository.enabled():
        return document_repository.get(document_id)

    for record in _read_metadata():
        if record["document_id"] == document_id:
            return record
    return None


def document_text(document_id):
    if postgres.enabled():
        return postgres.text(document_id)
    return text_path_for(document_id).read_text(encoding="utf-8", errors="replace")


@synchronized
def delete_document(identifier):
    if postgres.enabled():
        return postgres.delete_document(identifier)
    record = get_document(identifier)
    if record is None:
        return False
    from app.storage import supabase_metadata, vector_store
    from app.documents import chunking

    if supabase_metadata.enabled():
        supabase_metadata.delete(identifier)
    else:
        _write_metadata([r for r in _read_metadata() if r["document_id"] != identifier])
    chunking._write_chunks([c for c in chunking._read_chunks() if c["document_id"] != identifier])
    vector_store._get_store()._collection.delete(where={"document_id": identifier})
    vector_store.clear_search_cache()
    text_path_for(identifier).unlink(missing_ok=True)
    stored_path_for(identifier, record["file_type"]).unlink(missing_ok=True)
    return True
