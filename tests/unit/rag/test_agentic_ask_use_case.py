import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.knowledge.schemas import (
    SimilaritySearchResult,
    VectorDocumentMetadata,
)
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.prompt.schemas import PromptTemplate
from ai_service.prompt.service import PromptService
from ai_service.rag.application.agentic_ask_use_case import AgenticAskUseCase
from ai_service.rag.application.command.agentic_ask_command import AgenticAskCommand
from ai_service.rag.application.critique_generator_service import CritiqueGeneratorService
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.hybrid_search_use_case import (
    HybridSearchResult,
    HybridSearchUseCase,
)
from ai_service.rag.application.query_refiner_service import QueryRefinerService
from ai_service.rag.schemas import Critique, IterationBudget


def _make_prompt_template(name: str = "rag-qa-system") -> PromptTemplate:
    return PromptTemplate.create(
        name,
        "Context: {{context}}\nDate: {{currentDate}}",
        ["context", "currentDate"],
    )


@pytest.fixture
def mock_hybrid_search() -> AsyncMock:
    mock = AsyncMock(spec=HybridSearchUseCase)
    mock.execute.return_value = HybridSearchResult(chunks=[], query_embedding=[0.1, 0.2])
    return mock


@pytest.fixture
def mock_llm_gateway() -> MagicMock:
    mock = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for token in ["테스트", " ", "답변입니다."]:
            yield token

    mock.stream.side_effect = fake_stream
    return mock


@pytest.fixture
def mock_get_active_prompt() -> AsyncMock:
    mock = AsyncMock(spec=PromptService)
    mock.execute.return_value = _make_prompt_template()
    return mock


@pytest.fixture
def mock_critique_generator() -> AsyncMock:
    mock = AsyncMock(spec=CritiqueGeneratorService)
    return mock


@pytest.fixture
def mock_query_refiner() -> MagicMock:
    mock = MagicMock(spec=QueryRefinerService)
    mock.refine.return_value = "개선된 질문"
    return mock


@pytest.fixture
def rag_validator() -> RagContentValidator:
    return RagContentValidator()


@pytest.fixture
def secret_pii_scanner() -> SecretPiiScanner:
    return SecretPiiScanner()


@pytest.mark.asyncio
async def test_agentic_ask_single_iteration_progress_events(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
    )

    events: list[dict] = []

    async def on_progress(data: dict) -> None:
        events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        on_progress=on_progress,
    )

    command = AgenticAskCommand(
        question="환불 정책이 어떻게 되나요?",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."

    # 1회 반복에서 searching -> generating -> critiquing -> 완료 후 최종 이벤트
    assert len(events) == 4
    assert events[0] == {"iteration": 1, "phase": "searching", "confidence": 0.0, "missing": []}
    assert events[1] == {"iteration": 1, "phase": "generating", "confidence": 0.0, "missing": []}
    assert events[2] == {"iteration": 1, "phase": "critiquing", "confidence": 0.0, "missing": []}
    assert events[3] == {"iteration": 1, "phase": "critiquing", "confidence": 0.9, "missing": []}


@pytest.mark.asyncio
async def test_agentic_ask_multi_iteration_progress_events(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: 만족 못함 (missing: ["결제 취소 정책"], confidence: 0.5)
    # 2회차: 만족함 (missing: [], confidence: 0.95)
    mock_critique_generator.generate.side_effect = [
        Critique.of(
            answered=False,
            missing=["결제 취소 정책"],
            next_query="결제 취소 정책",
            confidence=0.5,
        ),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.95),
    ]

    events: list[dict] = []

    async def on_progress(data: dict) -> None:
        events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    command = AgenticAskCommand(
        question="결제 및 환불 안내",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
        on_progress=on_progress,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."

    # 1회차: searching -> generating -> critiquing -> refining
    # 2회차: searching -> generating -> critiquing -> 최종 완료
    assert len(events) == 8
    assert events[0] == {"iteration": 1, "phase": "searching", "confidence": 0.0, "missing": []}
    assert events[1] == {"iteration": 1, "phase": "generating", "confidence": 0.0, "missing": []}
    assert events[2] == {"iteration": 1, "phase": "critiquing", "confidence": 0.0, "missing": []}
    assert events[3] == {
        "iteration": 1,
        "phase": "refining",
        "confidence": 0.5,
        "missing": ["결제 취소 정책"],
    }
    assert events[4] == {
        "iteration": 2,
        "phase": "searching",
        "confidence": 0.5,
        "missing": ["결제 취소 정책"],
    }
    assert events[5] == {
        "iteration": 2,
        "phase": "generating",
        "confidence": 0.5,
        "missing": ["결제 취소 정책"],
    }
    assert events[6] == {
        "iteration": 2,
        "phase": "critiquing",
        "confidence": 0.5,
        "missing": ["결제 취소 정책"],
    }
    assert events[7] == {"iteration": 2, "phase": "critiquing", "confidence": 0.95, "missing": []}


@pytest.mark.asyncio
async def test_agentic_ask_budget_exhausted_progress_events(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차 완료 후 2회차에서 max_iterations=2로 인해 generating 직후 budget 소진되는 경우
    mock_critique_generator.generate.return_value = Critique.of(
        answered=False,
        missing=["남은 정보"],
        next_query="남은 정보 검색",
        confidence=0.4,
    )

    events: list[dict] = []

    async def on_progress(data: dict) -> None:
        events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        on_progress=on_progress,
    )

    command = AgenticAskCommand(
        question="질문",
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."

    assert len(events) == 7
    assert events[0] == {"iteration": 1, "phase": "searching", "confidence": 0.0, "missing": []}
    assert events[1] == {"iteration": 1, "phase": "generating", "confidence": 0.0, "missing": []}
    assert events[2] == {"iteration": 1, "phase": "critiquing", "confidence": 0.0, "missing": []}
    assert events[3] == {
        "iteration": 1,
        "phase": "refining",
        "confidence": 0.4,
        "missing": ["남은 정보"],
    }
    assert events[4] == {
        "iteration": 2,
        "phase": "searching",
        "confidence": 0.4,
        "missing": ["남은 정보"],
    }
    assert events[5] == {
        "iteration": 2,
        "phase": "generating",
        "confidence": 0.4,
        "missing": ["남은 정보"],
    }
    assert events[6] == {
        "iteration": 2,
        "phase": "generating",
        "confidence": 0.4,
        "missing": ["남은 정보"],
    }


@pytest.mark.asyncio
async def test_agentic_ask_progress_callback_error_does_not_break_generation(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
    )

    async def failing_on_progress(_data: dict) -> None:
        raise ConnectionError("Redis connection lost")

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        on_progress=failing_on_progress,
    )

    command = AgenticAskCommand(
        question="질문",
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."


@pytest.mark.asyncio
async def test_agentic_ask_sources_include_snippet_and_mask_pii(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
    )

    chunk1_text = "이것은 일반적인 문서 본문입니다."
    chunk2_text = "사용자 연락처는 010-1234-5678 이며 개인정보입니다."
    chunk3_text = "B" * 500

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

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    command = AgenticAskCommand(
        question="출처 확인 질문",
        budget=IterationBudget.of(max_iterations=1, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    results = [c async for c in use_case.execute(command)]

    assert len(results) > 0
    assert results[0].startswith("__SOURCES:")

    sources = json.loads(results[0][len("__SOURCES:") :])
    assert len(sources) == 3

    # chunk 1: 300자 이하 본문이 그대로 snippet에 들어있음
    assert sources[0]["fileName"] == "guide.pdf"
    assert sources[0]["chunkIndex"] == 0
    assert sources[0]["documentId"] == "doc-1"
    assert sources[0]["snippet"] == chunk1_text

    # chunk 2: PII (전화번호) 마스킹 확인
    assert sources[1]["fileName"] == "privacy.pdf"
    assert "010-1234-5678" not in sources[1]["snippet"]
    assert "[REDACTED_KR_PHONE]" in sources[1]["snippet"]

    # chunk 3: 300자 초과 본문은 300자 + '...' 로 잘림
    assert sources[2]["fileName"] == "long_doc.pdf"
    assert len(sources[2]["snippet"]) == 303
    assert sources[2]["snippet"] == "B" * 300 + "..."


@pytest.mark.asyncio
async def test_agentic_ask_fallback_critique_triggers_refining(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: 파싱 실패 폴백 (answered=False, missing=["비평 파싱 실패"], confidence=0.0)
    # 2회차: 정상 비평 (answered=True, missing=[], confidence=0.9)
    mock_critique_generator.generate.side_effect = [
        Critique.of(answered=False, missing=["비평 파싱 실패"], next_query="", confidence=0.0),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.9),
    ]

    events: list[dict] = []

    async def on_progress(data: dict) -> None:
        events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        on_progress=on_progress,
    )

    command = AgenticAskCommand(
        question="비평 실패 시 재시도 확인 질문",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.6,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."
    assert mock_critique_generator.generate.call_count == 2
    assert mock_query_refiner.refine.call_count == 1


@pytest.mark.asyncio
async def test_agentic_ask_multi_iteration_accumulates_sources(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: doc-1 (chunk 0) 검색 -> 비평 미흡 (confidence 0.4)
    # 2회차: doc-2 (chunk 1) 추가 검색 -> 비평 만족 (confidence 0.9)
    chunk1 = SimilaritySearchResult(
        text="문서 1 내용",
        score=0.8,
        metadata=VectorDocumentMetadata(
            document_id="doc-1",
            file_name="doc1.pdf",
            chunk_index=0,
        ),
    )
    chunk2 = SimilaritySearchResult(
        text="문서 2 내용",
        score=0.85,
        metadata=VectorDocumentMetadata(
            document_id="doc-2",
            file_name="doc2.pdf",
            chunk_index=1,
        ),
    )

    mock_hybrid_search.execute.side_effect = [
        HybridSearchResult(chunks=[chunk1], query_embedding=[0.1, 0.2]),
        HybridSearchResult(chunks=[chunk2], query_embedding=[0.1, 0.2]),
    ]

    mock_critique_generator.generate.side_effect = [
        Critique.of(answered=False, missing=["추가 정보"], next_query="추가 질문", confidence=0.4),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.9),
    ]

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    command = AgenticAskCommand(
        question="다회차 출처 누적 테스트",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    yielded_items = [item async for item in use_case.execute(command)]
    source_events = [
        json.loads(item[len("__SOURCES:") :])
        for item in yielded_items
        if item.startswith("__SOURCES:")
    ]

    assert len(source_events) == 2

    # 1회차 방출된 출처: doc-1
    assert len(source_events[0]) == 1
    assert source_events[0][0]["documentId"] == "doc-1"
    assert source_events[0][0]["chunkIndex"] == 0

    # 2회차 방출된 출처: doc-1 + doc-2 누적
    assert len(source_events[1]) == 2
    assert {s["documentId"] for s in source_events[1]} == {"doc-1", "doc-2"}


@pytest.mark.asyncio
async def test_agentic_ask_multi_iteration_deduplicates_sources_and_updates_higher_score(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: doc-1 (chunk 0, score=0.7)
    # 2회차: doc-1 (chunk 0, score=0.95) 동일 청크 더 높은 점수로 재검색
    chunk1_v1 = SimilaritySearchResult(
        text="문서 1 공통 내용",
        score=0.7,
        metadata=VectorDocumentMetadata(
            document_id="doc-1",
            file_name="doc1.pdf",
            chunk_index=0,
        ),
    )
    chunk1_v2 = SimilaritySearchResult(
        text="문서 1 공통 내용",
        score=0.95,
        metadata=VectorDocumentMetadata(
            document_id="doc-1",
            file_name="doc1.pdf",
            chunk_index=0,
        ),
    )

    mock_hybrid_search.execute.side_effect = [
        HybridSearchResult(chunks=[chunk1_v1], query_embedding=[0.1, 0.2]),
        HybridSearchResult(chunks=[chunk1_v2], query_embedding=[0.1, 0.2]),
    ]

    mock_critique_generator.generate.side_effect = [
        Critique.of(answered=False, missing=["추가 정보"], next_query="추가 질문", confidence=0.4),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.9),
    ]

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    command = AgenticAskCommand(
        question="중복 청크 병합 테스트",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    yielded_items = [item async for item in use_case.execute(command)]
    source_events = [
        json.loads(item[len("__SOURCES:") :])
        for item in yielded_items
        if item.startswith("__SOURCES:")
    ]

    # 1회차 방출, 2회차 점수 갱신으로 2회차도 방출됨
    assert len(source_events) == 2
    assert len(source_events[1]) == 1
    assert source_events[1][0]["documentId"] == "doc-1"
    assert source_events[1][0]["chunkIndex"] == 0
    assert source_events[1][0]["score"] == 0.95


@pytest.mark.asyncio
async def test_agentic_ask_multi_iteration_first_empty_second_found_sources(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: 검색 결과 0건
    # 2회차: doc-1 (chunk 0) 검색됨
    chunk1 = SimilaritySearchResult(
        text="문서 1 내용",
        score=0.88,
        metadata=VectorDocumentMetadata(
            document_id="doc-1",
            file_name="doc1.pdf",
            chunk_index=0,
        ),
    )

    mock_hybrid_search.execute.side_effect = [
        HybridSearchResult(chunks=[], query_embedding=[0.1, 0.2]),
        HybridSearchResult(chunks=[chunk1], query_embedding=[0.1, 0.2]),
    ]

    mock_critique_generator.generate.side_effect = [
        Critique.of(answered=False, missing=["정보 없음"], next_query="새 질문", confidence=0.2),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.9),
    ]

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    command = AgenticAskCommand(
        question="1회차 빈 검색 결과 후 2회차 출처 방출 테스트",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    yielded_items = [item async for item in use_case.execute(command)]
    source_events = [
        json.loads(item[len("__SOURCES:") :])
        for item in yielded_items
        if item.startswith("__SOURCES:")
    ]

    assert len(source_events) == 1
    assert len(source_events[0]) == 1
    assert source_events[0][0]["documentId"] == "doc-1"
    assert source_events[0][0]["score"] == 0.88


@pytest.mark.asyncio
async def test_agentic_ask_multi_iteration_pii_masked_in_later_rounds(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: 일반 텍스트
    # 2회차: PII(전화번호) 포함 텍스트
    chunk1 = SimilaritySearchResult(
        text="1회차 일반 정보",
        score=0.8,
        metadata=VectorDocumentMetadata(
            document_id="doc-1",
            file_name="general.pdf",
            chunk_index=0,
        ),
    )
    chunk2 = SimilaritySearchResult(
        text="담당자 연락처: 010-9876-5432 로 연락 바랍니다.",
        score=0.85,
        metadata=VectorDocumentMetadata(
            document_id="doc-2",
            file_name="contact.pdf",
            chunk_index=0,
        ),
    )

    mock_hybrid_search.execute.side_effect = [
        HybridSearchResult(chunks=[chunk1], query_embedding=[0.1, 0.2]),
        HybridSearchResult(chunks=[chunk2], query_embedding=[0.1, 0.2]),
    ]

    mock_critique_generator.generate.side_effect = [
        Critique.of(answered=False, missing=["연락처"], next_query="담당자 연락처", confidence=0.5),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.9),
    ]

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    command = AgenticAskCommand(
        question="다회차 PII 마스킹 테스트",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    yielded_items = [item async for item in use_case.execute(command)]
    source_events = [
        json.loads(item[len("__SOURCES:") :])
        for item in yielded_items
        if item.startswith("__SOURCES:")
    ]

    assert len(source_events) == 2
    # 2회차 출처 중 doc-2의 snippet에 전화번호가 마스킹되었는지 확인
    doc2_source = next(s for s in source_events[1] if s["documentId"] == "doc-2")
    assert "010-9876-5432" not in doc2_source["snippet"]
    assert "[REDACTED_KR_PHONE]" in doc2_source["snippet"]



