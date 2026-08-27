from pydantic import SecretStr

from ai_service.config.settings import Settings
from ai_service.knowledge.domain.port.embedding_provider_port import IEmbeddingProvider


class LangChainEmbeddingProvider:
    """LangChain `Embeddings` 구현체를 감싸는 IEmbeddingProvider 어댑터."""

    def __init__(self, embeddings: object) -> None:
        self._embeddings = embeddings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = await self._embeddings.aembed_documents(  # type: ignore[attr-defined]
            texts
        )
        return result


def build_embedding_provider(settings: Settings) -> IEmbeddingProvider:
    provider = settings.embedding_provider

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return LangChainEmbeddingProvider(
            OllamaEmbeddings(
                base_url=settings.ollama_base_url, model=settings.ollama_embedding_model
            )
        )

    if provider == "groq":
        # TODO(Phase 2 후속): langchain-groq는 임베딩 모델을 제공하지 않는다.
        # NestJS 쪽 GroqEmbeddingProvider는 별도 REST 엔드포인트를 직접 호출하므로
        # 필요 시 동일한 방식으로 별도 구현해야 한다.
        raise NotImplementedError("Groq 임베딩 프로바이더는 아직 이식되지 않았습니다.")

    if provider == "gemini":
        from ai_service.knowledge.infrastructure.providers.google_embedding_provider import (
            GoogleEmbeddingProvider,
        )

        return GoogleEmbeddingProvider(
            api_key=settings.google_api_key,
            model=settings.google_embedding_model,
            output_dimensionality=settings.embedding_dimension,
        )

    from langchain_openai import OpenAIEmbeddings

    model = settings.embedding_model or "text-embedding-3-small"
    api_key = SecretStr(settings.openai_api_key) if settings.openai_api_key else None
    return LangChainEmbeddingProvider(OpenAIEmbeddings(model=model, api_key=api_key))
