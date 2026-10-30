from typing import Annotated, Any

from fastapi import Depends, Request

from ai_service.core.database import MongoDbDep
from ai_service.knowledge.application.ingest_document_use_case import IngestDocumentUseCase
from ai_service.knowledge.repository import DocumentRepository, QdrantVectorAdapter


def get_document_repository(db: MongoDbDep) -> DocumentRepository:
    return DocumentRepository(db)


def get_vector_store(request: Request) -> QdrantVectorAdapter:
    return request.app.state.vector_store  # type: ignore[no-any-return]


def get_ingest_use_case(request: Request) -> IngestDocumentUseCase:
    return request.app.state.ingest_use_case  # type: ignore[no-any-return]


DocumentRepositoryDep = Annotated[Any, Depends(get_document_repository)]
VectorStoreDep = Annotated[Any, Depends(get_vector_store)]
IngestUseCaseDep = Annotated[Any, Depends(get_ingest_use_case)]

__all__ = [
    "DocumentRepositoryDep",
    "IngestUseCaseDep",
    "VectorStoreDep",
    "get_document_repository",
    "get_ingest_use_case",
    "get_vector_store",
]
