import os
from starlette.concurrency import run_in_threadpool

from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query

from app.api.dependencies import require_api_key, require_writer
from app.schemas.chunk import Chunk
from app.storage.postgres import TenantQuotaExceeded
from app.schemas.document import DocumentMetadata
from app.documents import chunking as chunk_service
from app.documents import service as document_service


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/upload", response_model=DocumentMetadata, dependencies=[Depends(require_writer)])
async def upload_document(file: UploadFile = File(...)):
    maximum = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))
    content = await file.read(maximum + 1)
    if len(content) > maximum:
        raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
    if not file.filename or len(file.filename) > 255:
        raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        record = await run_in_threadpool(document_service.save_upload, file.filename, content)
    except document_service.DocumentTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except TenantQuotaExceeded as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except document_service.UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document upload failed") from exc

    return DocumentMetadata(**record)


@router.get("", response_model=list[DocumentMetadata])
def list_documents(limit: int = Query(default=100, ge=1, le=1000), offset: int = Query(default=0, ge=0)):
    return [DocumentMetadata(**record) for record in document_service.list_documents(limit, offset)]


@router.post("/{document_id}/ingest", response_model=list[Chunk], dependencies=[Depends(require_writer)])
def ingest_document(
    document_id: str,
    strategy: Literal["fixed", "recursive"] = "fixed",
    chunk_size: int = Query(default=650, ge=100, le=4000),
    overlap: int = Query(default=80, ge=0, le=1000),
):
    if overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="overlap must be smaller than chunk_size")
    try:
        chunks = chunk_service.ingest_document(document_id, strategy, chunk_size, overlap)
    except TenantQuotaExceeded as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found") from exc

    return [Chunk(**record) for record in chunks]


@router.get("/{document_id}/chunks", response_model=list[Chunk])
def list_document_chunks(document_id: str):
    if document_service.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    return [Chunk(**record) for record in chunk_service.list_chunks(document_id)]


@router.delete("/{document_id}", status_code=204, dependencies=[Depends(require_writer)])
def delete_document(document_id: str):
    if not document_service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
