from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SemanticCacheHit:
    answer: str
    score: float


class ISemanticCachePort(Protocol):
    async def find_similar(
        self, embedding: list[float], threshold: float, tenant: str
    ) -> SemanticCacheHit | None: ...

    async def store(
        self,
        embedding: list[float],
        question: str,
        answer: str,
        ttl_seconds: int,
        tenant: str,
    ) -> None: ...
