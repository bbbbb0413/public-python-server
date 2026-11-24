import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.core.config import Settings
from ai_service.knowledge.schemas import (
    SimilaritySearchResult,
    VectorDocumentMetadata,
)
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.prompt.schemas import PromptTemplate
from ai_service.prompt.service import PromptService
from ai_service.rag.application.ask_command import AskCommand
from ai_service.rag.application.ask_use_case import AskUseCase
from ai_service.rag.application.conversational_query_rewriter_service import (
    ConversationalQueryRewriter,
)
from ai_service.rag.application.critique_generator_service import CritiqueGeneratorService
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.hybrid_search_use_case import (
    HybridSearchResult,
    HybridSearchUseCase,
)
from ai_service.rag.infrastructure.cache.redis_semantic_cache_adapter import (
    SemanticCacheHit,
)
from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import Critique


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
    mock = AsyncMock(spec=PromptService)
    mock.execute.return_value = _make_prompt_template()
    return mock


@pytest.fixture
def mock_llm_cache() -> AsyncMock:
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set_with_ttl.return_value = None
    return mock


@pytest.fixture
def mock_semantic_cache() -> AsyncMock:
    mock = AsyncMock()
    mock.find_similar.return_value = None
    mock.store.return_value = None
    return mock


@pytest.fixture
def mock_session_repo() -> AsyncMock:
    mock = AsyncMock(spec=ConversationSessionRepository)
    mock.find_by_id.return_value = None
    mock.persist.return_value = None
    mock.update.return_value = None
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


@pytest.mark.asyncio
async def test_ask_use_case_yields_confidence_metadata_when_chunks_exist(
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
    chunks = [
        SimilaritySearchResult(
            text="문서 본문 1",
            score=0.72,
            metadata=VectorDocumentMetadata(
                document_id="doc-1",
                file_name="guide.pdf",
                chunk_index=0,
            ),
        ),
        SimilaritySearchResult(
            text="문서 본문 2",
            score=0.88,
            metadata=VectorDocumentMetadata(
                document_id="doc-2",
                file_name="privacy.pdf",
                chunk_index=1,
            ),
        ),
    ]

    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)
    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.85
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
        critique_generator=mock_critique_generator,
    )

    command = AskCommand(question="신뢰도 테스트 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    # 1. __SOURCES가 가장 먼저 방출되고, 조기 __CONFIDENCE는 방출되지 않음
    assert results[0].startswith("__SOURCES:")
    assert not results[1].startswith("__CONFIDENCE:")

    # 2. 토큰 스트리밍 후 마지막에 비평 기반 __CONFIDENCE 방출
    assert results[-1].startswith("__CONFIDENCE:")
    confidence_meta = json.loads(results[-1][len("__CONFIDENCE:") :])
    # 검색 점수 0.88이 아니라 실제 비평 신뢰도 0.85가 반영됨
    assert confidence_meta["confidence"] == 0.85
    assert confidence_meta["missing"] == []

    # 3. critique_generator.generate가 호출되었는지 검증
    mock_critique_generator.generate.assert_awaited_once_with(
        "신뢰도 테스트 질문", "테스트 답변입니다.", chunks, "default"
    )


@pytest.mark.asyncio
async def test_ask_use_case_yields_low_confidence_and_missing_when_critique_detects_hallucination(
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
    chunks = [
        SimilaritySearchResult(
            text="정확한 사실 정보가 담긴 문서 청크",
            score=0.88,
            metadata=VectorDocumentMetadata(
                document_id="doc-1",
                file_name="fact.pdf",
                chunk_index=0,
            ),
        ),
    ]

    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)
    # LLM이 환각 답변을 생성하여 비평 점수가 낮고 결여 항목이 발견된 상황 모킹
    mock_critique_generator.generate.return_value = Critique.of(
        answered=False,
        missing=["핵심 수치 정보 불일치"],
        next_query="구체적 수치 재검색",
        confidence=0.35,
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
        critique_generator=mock_critique_generator,
    )

    command = AskCommand(question="환각 유발 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    assert results[0].startswith("__SOURCES:")
    assert results[-1].startswith("__CONFIDENCE:")

    confidence_meta = json.loads(results[-1][len("__CONFIDENCE:") :])
    # 검색 점수 0.88이 아닌 비평 결과(0.35 및 missing 항목)가 반영됨
    assert confidence_meta["confidence"] == 0.35
    assert confidence_meta["missing"] == ["핵심 수치 정보 불일치"]


@pytest.mark.asyncio
async def test_ask_use_case_critique_fallback_on_exception(
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
    chunks = [
        SimilaritySearchResult(
            text="문서 본문",
            score=0.9,
            metadata=VectorDocumentMetadata(
                document_id="doc-1",
                file_name="guide.pdf",
                chunk_index=0,
            ),
        ),
    ]

    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)
    mock_critique_generator.generate.side_effect = RuntimeError("LLM Gateway timeout")

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
        critique_generator=mock_critique_generator,
    )

    command = AskCommand(question="예외 테스트 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    # 비평 실패 시 안전 폴백(confidence=0.0, missing=[]) 방출
    assert results[-1].startswith("__CONFIDENCE:")
    confidence_meta = json.loads(results[-1][len("__CONFIDENCE:") :])
    assert confidence_meta["confidence"] == 0.0
    assert confidence_meta["missing"] == []


@pytest.mark.asyncio
async def test_ask_use_case_fallback_when_critique_generator_is_none(
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
    chunks = [
        SimilaritySearchResult(
            text="문서 본문",
            score=0.95,
            metadata=VectorDocumentMetadata(
                document_id="doc-1",
                file_name="guide.pdf",
                chunk_index=0,
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
        critique_generator=None,
    )

    command = AskCommand(question="생성기 미주입 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    assert results[-1].startswith("__CONFIDENCE:")
    confidence_meta = json.loads(results[-1][len("__CONFIDENCE:") :])
    assert confidence_meta["confidence"] == 0.0
    assert confidence_meta["missing"] == []


@pytest.mark.asyncio
async def test_ask_use_case_skips_critique_when_no_chunks(
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
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=[],
        query_embedding=[0.1, 0.2],
    )

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)

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
        critique_generator=mock_critique_generator,
    )

    command = AskCommand(question="청크 없음 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    # 청크가 없으므로 비평 호출 건너뜀
    mock_critique_generator.generate.assert_not_called()

    assert results[0].startswith("__CONFIDENCE:")
    confidence_meta = json.loads(results[0][len("__CONFIDENCE:") :])
    assert confidence_meta["confidence"] == 0.0
    assert confidence_meta["missing"] == []


@pytest.mark.asyncio
async def test_ask_use_case_yields_zero_confidence_and_missing_when_no_chunks(
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
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=[],
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

    command = AskCommand(question="문서 없는 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    assert len(results) >= 1
    # 첫 번째 chunk가 __CONFIDENCE 여야 함 (__SOURCES는 청크가 없으므로 방출되지 않음)
    assert results[0].startswith("__CONFIDENCE:")

    confidence_meta = json.loads(results[0][len("__CONFIDENCE:") :])
    assert confidence_meta["confidence"] == 0.0
    assert confidence_meta["missing"] == []


@pytest.mark.asyncio
async def test_ask_use_case_yields_confidence_metadata_on_llm_cache_hit_when_chunks_exist(
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
    chunks = [
        SimilaritySearchResult(
            text="캐시 테스트 문서 본문",
            score=0.9,
            metadata=VectorDocumentMetadata(
                document_id="doc-cache-1",
                file_name="cache_guide.pdf",
                chunk_index=0,
            ),
        ),
    ]
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    mock_llm_cache.get.return_value = "캐시된 LLM 답변입니다."

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)
    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.95
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
        critique_generator=mock_critique_generator,
    )

    command = AskCommand(question="캐시 적중 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    assert len(results) == 3
    assert results[0].startswith("__SOURCES:")
    assert results[1] == "캐시된 LLM 답변입니다."
    assert results[2].startswith("__CONFIDENCE:")

    confidence_meta = json.loads(results[2][len("__CONFIDENCE:") :])
    assert confidence_meta["confidence"] == 0.95
    assert confidence_meta["missing"] == []

    mock_critique_generator.generate.assert_awaited_once_with(
        "캐시 적중 질문", "캐시된 LLM 답변입니다.", chunks, "default"
    )
    mock_llm_gateway.stream.assert_not_called()


@pytest.mark.asyncio
async def test_ask_use_case_yields_confidence_metadata_on_semantic_cache_hit_when_chunks_exist(
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
    chunks = [
        SimilaritySearchResult(
            text="시맨틱 캐시 테스트 문서 본문",
            score=0.88,
            metadata=VectorDocumentMetadata(
                document_id="doc-sem-1",
                file_name="sem_guide.pdf",
                chunk_index=0,
            ),
        ),
    ]
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    mock_llm_cache.get.return_value = None

    settings.semantic_cache_enabled = True
    mock_semantic_cache.find_similar.return_value = SemanticCacheHit(
        answer="시맨틱 캐시된 답변입니다.",
        score=0.92,
    )

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)
    mock_critique_generator.generate.return_value = Critique.of(
        answered=False, missing=["결여 항목 A"], next_query="", confidence=0.35
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
        critique_generator=mock_critique_generator,
    )

    command = AskCommand(question="시맨틱 캐시 적중 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    assert len(results) == 3
    assert results[0].startswith("__SOURCES:")
    assert results[1] == "시맨틱 캐시된 답변입니다."
    assert results[2].startswith("__CONFIDENCE:")

    confidence_meta = json.loads(results[2][len("__CONFIDENCE:") :])
    assert confidence_meta["confidence"] == 0.35
    assert confidence_meta["missing"] == ["결여 항목 A"]

    mock_critique_generator.generate.assert_awaited_once_with(
        "시맨틱 캐시 적중 질문", "시맨틱 캐시된 답변입니다.", chunks, "default"
    )
    mock_llm_gateway.stream.assert_not_called()


@pytest.mark.asyncio
async def test_ask_use_case_llm_cache_hit_skips_duplicate_confidence_when_no_chunks(
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
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=[],
        query_embedding=[0.1, 0.2],
    )
    mock_llm_cache.get.return_value = "문서 없이 캐시된 답변"

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)

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
        critique_generator=mock_critique_generator,
    )

    command = AskCommand(question="청크 없는 캐시 질문")
    results = [chunk async for chunk in use_case.execute(command)]

    assert len(results) == 2
    assert results[0].startswith("__CONFIDENCE:")
    assert results[1] == "문서 없이 캐시된 답변"
    mock_critique_generator.generate.assert_not_called()
