import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.config.settings import Settings
from ai_service.knowledge.domain.port.vector_store_port import (
    SimilaritySearchResult,
    VectorDocumentMetadata,
)
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.prompt.application.get_active_prompt_use_case import GetActivePromptUseCase
from ai_service.prompt.domain.model.prompt_template import PromptTemplate
from ai_service.rag.application.ask_command import AskCommand
from ai_service.rag.application.ask_use_case import AskUseCase
from ai_service.rag.application.conversational_query_rewriter_service import (
    ConversationalQueryRewriter,
)
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.hybrid_search_use_case import (
    HybridSearchResult,
    HybridSearchUseCase,
)
from ai_service.rag.domain.port.llm_cache_port import ILlmCachePort
from ai_service.rag.domain.port.semantic_cache_port import ISemanticCachePort
from ai_service.rag.domain.repository.conversation_session_repository import (
    IConversationSessionRepository,
)


def _make_prompt_template(name: str = "rag-qa-system") -> PromptTemplate:
    return PromptTemplate.create(
        name,
        "Context: {{context}}\nDate: {{currentDate}}",
        ["context", "currentDate"],
    )


@pytest.fixture
def mock_llm_gateway() -> MagicMock:
    mock = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for token in ["테스트", " ", "답변입니다."]:
            yield token

    mock.stream.side_effect = fake_stream
    return mock


@pytest.fixture
def mock_hybrid_search() -> AsyncMock:
    mock = AsyncMock(spec=HybridSearchUseCase)
    mock.execute.return_value = HybridSearchResult(chunks=[], query_embedding=[0.1, 0.2])
    return mock


@pytest.fixture
def mock_get_active_prompt() -> AsyncMock:
    mock = AsyncMock(spec=GetActivePromptUseCase)
    mock.execute.return_value = _make_prompt_template()
    return mock


@pytest.fixture
def mock_llm_cache() -> AsyncMock:
    mock = AsyncMock(spec=ILlmCachePort)
    mock.get.return_value = None
    mock.set_with_ttl.return_value = None
    return mock


@pytest.fixture
def mock_semantic_cache() -> AsyncMock:
    mock = AsyncMock(spec=ISemanticCachePort)
    mock.find_similar.return_value = None
    mock.store.return_value = None
    return mock


@pytest.fixture
def mock_session_repo() -> AsyncMock:
    mock = AsyncMock(spec=IConversationSessionRepository)
    mock.find_by_id.return_value = None
    return mock


@pytest.fixture
def mock_query_rewriter() -> AsyncMock:
    mock = AsyncMock(spec=ConversationalQueryRewriter)
    mock.is_follow_up.return_value = False
    return mock


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_cache_ttl_seconds=3600,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.85,
        semantic_cache_ttl_seconds=3600,
    )


@pytest.fixture
def rag_validator() -> RagContentValidator:
    return RagContentValidator()


@pytest.fixture
def secret_pii_scanner() -> SecretPiiScanner:
    return SecretPiiScanner()


@pytest.mark.asyncio
async def test_ask_use_case_sources_include_snippet_and_mask_pii(
    mock_llm_gateway: MagicMock,
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_llm_cache: AsyncMock,
    mock_semantic_cache: AsyncMock,
    settings: Settings,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
    mock_session_repo: AsyncMock,
    mock_query_rewriter: AsyncMock,
):
    chunk1_text = "이것은 일반적인 문서 본문입니다."
    chunk2_text = "사용자 연락처는 010-1234-5678 이며 개인정보입니다."
    chunk3_text = "A" * 500

    chunks = [
        SimilaritySearchResult(
            text=chunk1_text,
            score=0.9,
            metadata=VectorDocumentMetadata(
                document_id="doc-1",
                file_name="guide.pdf",
                chunk_index=0,
            ),
        ),
        SimilaritySearchResult(
            text=chunk2_text,
            score=0.85,
            metadata=VectorDocumentMetadata(
                document_id="doc-2",
                file_name="privacy.pdf",
                chunk_index=1,
            ),
        ),
        SimilaritySearchResult(
            text=chunk3_text,
            score=0.8,
            metadata=VectorDocumentMetadata(
                document_id="doc-3",
                file_name="long_doc.pdf",
                chunk_index=2,
            ),
        ),
    ]

    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    use_case = AskUseCase(
        llm_gateway=mock_llm_gateway,
        hybrid_search=mock_hybrid_search,
        get_active_prompt=mock_get_active_prompt,
        llm_cache=mock_llm_cache,
        semantic_cache=mock_semantic_cache,
        settings=settings,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        session_repo=mock_session_repo,
        query_rewriter=mock_query_rewriter,
    )

    command = AskCommand(question="출처 확인 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    assert len(results) > 0
    assert results[0].startswith("__SOURCES:")

    sources = json.loads(results[0][len("__SOURCES:") :])
    assert len(sources) == 3

    assert sources[0]["fileName"] == "guide.pdf"
    assert sources[0]["chunkIndex"] == 0
    assert sources[0]["documentId"] == "doc-1"
    assert sources[0]["snippet"] == chunk1_text

    assert sources[1]["fileName"] == "privacy.pdf"
    assert "010-1234-5678" not in sources[1]["snippet"]
    assert "[REDACTED_KR_PHONE]" in sources[1]["snippet"]

    assert sources[2]["fileName"] == "long_doc.pdf"
    assert len(sources[2]["snippet"]) == 303
    assert sources[2]["snippet"] == "A" * 300 + "..."
