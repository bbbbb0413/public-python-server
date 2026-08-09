from typing import Protocol


class IEmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
