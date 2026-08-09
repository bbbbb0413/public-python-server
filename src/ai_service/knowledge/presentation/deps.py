from typing import Annotated

from fastapi import Depends, Request

from ai_service.config.dependencies import MongoDbDep
from ai_service.knowledge.domain.port.vector_store_port import IVectorStorePort
from ai_service.knowledge.domain.repository.document_repository import IDocumentRepository
from ai_service.knowledge.infrastructure.persistence.document_repository_impl import (
    DocumentRepositoryImpl,
)


def get_document_repository(db: MongoDbDep) -> IDocumentRepository:
    return DocumentRepositoryImpl(db)


def get_vector_store(request: Request) -> IVectorStorePort:
    return request.app.state.vector_store  # type: ignore[no-any-return]


DocumentRepositoryDep = Annotated[IDocumentRepository, Depends(get_document_repository)]
VectorStoreDep = Annotated[IVectorStorePort, Depends(get_vector_store)]
