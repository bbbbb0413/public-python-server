from fastapi import APIRouter, Depends, HTTPException

from ai_service.core.security import require_admin_api_key
from ai_service.knowledge.dependencies import DocumentRepositoryDep, VectorStoreDep
from ai_service.knowledge.schemas import ChunkOut, DocumentOut

router = APIRouter(
    prefix="/knowledge/documents", tags=["knowledge"], dependencies=[Depends(require_admin_api_key)]
)


@router.get("", response_model=list[DocumentOut], response_model_by_alias=True)
async def list_documents(repo: DocumentRepositoryDep) -> list[DocumentOut]:
    documents = await repo.find_all()
    return [DocumentOut.from_domain(d) for d in documents]


@router.get("/{document_id}", response_model=DocumentOut, response_model_by_alias=True)
async def get_document(document_id: str, repo: DocumentRepositoryDep) -> DocumentOut:
    document = await repo.find_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {document_id}")
    return DocumentOut.from_domain(document)


@router.get("/{document_id}/chunks", response_model=list[ChunkOut], response_model_by_alias=True)
async def list_chunks(document_id: str, vector_store: VectorStoreDep) -> list[ChunkOut]:
    chunks = await vector_store.find_chunks_by_document_id(document_id)
    return [ChunkOut(chunkIndex=c.metadata.chunk_index, text=c.text) for c in chunks]


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str, repo: DocumentRepositoryDep, vector_store: VectorStoreDep
) -> None:
    await vector_store.delete_by_document_id(document_id)
    await repo.remove(document_id)


__all__ = ["router"]
