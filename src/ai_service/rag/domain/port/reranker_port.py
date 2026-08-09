from typing import Protocol

from ai_service.knowledge.domain.port.vector_store_port import SimilaritySearchResult


class IRerankerPort(Protocol):
    async def rerank(
        self, query: str, chunks: list[SimilaritySearchResult], top_n: int
    ) -> list[SimilaritySearchResult]: ...
