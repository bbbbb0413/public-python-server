from typing import Protocol

from ai_service.knowledge.domain.port.vector_store_port import SimilaritySearchResult


class ILexicalSearchPort(Protocol):
    async def search(self, query: str, top_k: int) -> list[SimilaritySearchResult]: ...
