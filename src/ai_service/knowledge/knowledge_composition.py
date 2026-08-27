from dataclasses import dataclass
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from qdrant_client import AsyncQdrantClient

from ai_service.core.config import Settings
from ai_service.knowledge.application.ingest_document_use_case import IngestDocumentUseCase
from ai_service.knowledge.infrastructure.providers.embedding_factory import (
    build_embedding_provider,
)
from ai_service.knowledge.repository import DocumentRepository, QdrantVectorAdapter
from ai_service.llm_gateway.infrastructure.providers.factory import build_llm_provider
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator


@dataclass
class KnowledgeComposition:
    ingest_use_case: IngestDocumentUseCase
    document_repo: DocumentRepository | Any
    vector_store: QdrantVectorAdapter | Any


async def build_knowledge_composition(
    settings: Settings,
    mongo_db: AsyncIOMotorDatabase[dict[str, object]],
) -> KnowledgeComposition:
    llm_provider = build_llm_provider(settings)
    embedding_provider = build_embedding_provider(settings)

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    vector_store = QdrantVectorAdapter(qdrant_client, settings.embedding_dimension)
    await vector_store.ensure_collection()

    document_repo = DocumentRepository(mongo_db)
    rag_validator = RagContentValidator()

    ingest_use_case = IngestDocumentUseCase(
        document_repo=document_repo,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        rag_validator=rag_validator,
        contextual_embeddings_enabled=settings.contextual_embeddings_enabled,
    )

    return KnowledgeComposition(
        ingest_use_case=ingest_use_case,
        document_repo=document_repo,
        vector_store=vector_store,
    )


__all__ = ["KnowledgeComposition", "build_knowledge_composition"]
