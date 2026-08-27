from typing import Annotated, Any

from fastapi import Depends, Request

from ai_service.core.database import MongoDbDep
from ai_service.knowledge.repository import DocumentRepository, QdrantVectorAdapter


def get_document_repository(db: MongoDbDep) -> DocumentRepository:
    return DocumentRepository(db)


def get_vector_store(request: Request) -> QdrantVectorAdapter:
    return request.app.state.vector_store  # type: ignore[no-any-return]


DocumentRepositoryDep = Annotated[Any, Depends(get_document_repository)]
VectorStoreDep = Annotated[Any, Depends(get_vector_store)]

__all__ = [
    "DocumentRepositoryDep",
    "VectorStoreDep",
    "get_document_repository",
    "get_vector_store",
]
