import asyncio

from google import genai
from google.genai import types


class GoogleEmbeddingProvider:
    """Gemini `embed_content` 기반 IEmbeddingProvider 구현체.

    SDK 호출이 동기(sync)라 asyncio.to_thread 로 감싸서 노출한다.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return list(await asyncio.gather(*(self._embed_one(text) for text in texts)))

    async def _embed_one(self, text: str) -> list[float]:
        response = await asyncio.to_thread(
            self._client.models.embed_content,
            model=self._model,
            contents=text,
            config=self._config,
        )
        if response.embeddings and response.embeddings[0].values:
            return list(response.embeddings[0].values)
        return []

