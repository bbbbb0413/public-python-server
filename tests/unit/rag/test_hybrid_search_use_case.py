from unittest.mock import AsyncMock

import pytest

from ai_service.core.config import Settings
from ai_service.knowledge.repository import QdrantVectorAdapter
from ai_service.knowledge.schemas import SimilaritySearchResult, VectorDocumentMetadata
from ai_service.rag.application.command.hybrid_search_command import HybridSearchCommand
from ai_service.rag.application.hybrid_search_use_case import HybridSearchUseCase
from ai_service.rag.application.hyde_service import HydeService
from ai_service.rag.application.query_decomposer_service import QueryDecomposer
from ai_service.rag.application.rrf_fusion_service import RrfFusionService
from ai_service.rag.infrastructure.search.qdrant_text_search_adapter import (
    QdrantTextSearchAdapter,
)


def _make_result(document_id: str, chunk_index: int, score: float) -> SimilaritySearchResult:
    return SimilaritySearchResult(
        text=f"chunk-{document_id}-{chunk_index}",
        score=score,
        metadata=VectorDocumentMetadata(
            document_id=document_id, file_name="doc.txt", chunk_index=chunk_index
        ),
    )


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    # spec=QdrantVectorAdapter여야 실제 어댑터에 없는 메서드를 호출하면 AttributeError로 걸러진다.
    mock = AsyncMock(spec=QdrantVectorAdapter)
    mock.find_similar.return_value = [_make_result("doc-1", 0, 0.9)]
    return mock


@pytest.fixture
def mock_lexical_search() -> AsyncMock:
    mock = AsyncMock(spec=QdrantTextSearchAdapter)
    mock.search_by_text.return_value = [_make_result("doc-2", 0, 0.5)]
    return mock


@pytest.fixture
def mock_embedding_provider() -> AsyncMock:
    mock = AsyncMock()
    mock.embed.return_value = [[0.1, 0.2, 0.3]]
    return mock


@pytest.fixture
def mock_reranker() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_hyde_service() -> AsyncMock:
    mock = AsyncMock(spec=HydeService)
    mock.generate_hypothetical_document.return_value = None
    return mock


@pytest.fixture
def mock_query_decomposer() -> AsyncMock:
    mock = AsyncMock(spec=QueryDecomposer)
    mock.decompose.return_value = []
    return mock


def _make_use_case(
    mock_vector_store: AsyncMock,
    mock_lexical_search: AsyncMock,
    mock_embedding_provider: AsyncMock,
    mock_reranker: AsyncMock,
    mock_hyde_service: AsyncMock,
    mock_query_decomposer: AsyncMock,
    *,
    reranker_enabled: bool = False,
) -> HybridSearchUseCase:
    settings = Settings(reranker_enabled=reranker_enabled)
    return HybridSearchUseCase(
        embedding_provider=mock_embedding_provider,
        vector_store=mock_vector_store,
        lexical_search=mock_lexical_search,
        reranker=mock_reranker,
        rrf_fusion=RrfFusionService(),
        hyde_service=mock_hyde_service,
        query_decomposer=mock_query_decomposer,
        settings=settings,
    )


class TestHybridSearchUseCase:
    async def test_실제_어댑터_메서드명과_일치하는_이름으로_검색을_호출한다(
        self,
        mock_vector_store: AsyncMock,
        mock_lexical_search: AsyncMock,
        mock_embedding_provider: AsyncMock,
        mock_reranker: AsyncMock,
        mock_hyde_service: AsyncMock,
        mock_query_decomposer: AsyncMock,
    ) -> None:
        use_case = _make_use_case(
            mock_vector_store,
            mock_lexical_search,
            mock_embedding_provider,
            mock_reranker,
            mock_hyde_service,
            mock_query_decomposer,
        )

        result = await use_case.execute(HybridSearchCommand(question="질문입니다"))

        mock_vector_store.find_similar.assert_awaited_once()
        mock_lexical_search.search_by_text.assert_awaited_once()
        assert len(result.chunks) > 0

    async def test_HyDE_가설문서가_생성되면_임계값과_함께_추가_벡터검색을_호출한다(
        self,
        mock_vector_store: AsyncMock,
        mock_lexical_search: AsyncMock,
        mock_embedding_provider: AsyncMock,
        mock_reranker: AsyncMock,
        mock_hyde_service: AsyncMock,
        mock_query_decomposer: AsyncMock,
    ) -> None:
        mock_hyde_service.generate_hypothetical_document.return_value = "가설 문서"
        mock_embedding_provider.embed.side_effect = [
            [[0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6]],
        ]
        use_case = _make_use_case(
            mock_vector_store,
            mock_lexical_search,
            mock_embedding_provider,
            mock_reranker,
            mock_hyde_service,
            mock_query_decomposer,
        )

        await use_case.execute(HybridSearchCommand(question="질문입니다", use_hyde=True))

        assert mock_vector_store.find_similar.await_count == 2
        hyde_call_args = mock_vector_store.find_similar.await_args_list[1].args
        assert hyde_call_args == ([0.4, 0.5, 0.6], 40, 0.5)
