from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VectorDocumentMetadata:
    document_id: str
    file_name: str
    chunk_index: int
    char_count: int | None = None
    parent_text: str | None = None
    parent_chunk_id: str | None = None


@dataclass(frozen=True)
class VectorDocument:
    id: str
    text: str
    embedding: list[float]
    metadata: VectorDocumentMetadata


@dataclass(frozen=True)
class SimilaritySearchResult:
    text: str
    score: float
    metadata: VectorDocumentMetadata


class IVectorStorePort(Protocol):
    async def upsert(self, documents: list[VectorDocument]) -> None: ...

    async def similarity_search(
        self, query_embedding: list[float], top_k: int
    ) -> list[SimilaritySearchResult]: ...

    async def find_by_parent_chunk_ids(
        self, parent_chunk_ids: list[str]
    ) -> list[SimilaritySearchResult]: ...

    async def find_chunks_by_document_id(
        self, document_id: str
    ) -> list[SimilaritySearchResult]: ...

    async def delete_by_document_id(self, document_id: str) -> None: ...
