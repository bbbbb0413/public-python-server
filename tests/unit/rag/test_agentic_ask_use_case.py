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
from ai_service.rag.application.query_refiner_service import QueryRefinerService
from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import ConversationSession, Critique, IterationBudget


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
    assert "".join(chunks) == "테스트 답변입니다.\n\n테스트 답변입니다."

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
    assert "".join(chunks) == "테스트 답변입니다.\n\n테스트 답변입니다."

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
    assert "".join(chunks) == "테스트 답변입니다.\n\n테스트 답변입니다."
    assert mock_critique_generator.generate.call_count == 2
    assert mock_query_refiner.refine.call_count == 1


@pytest.mark.asyncio
async def test_agentic_ask_streams_tokens_immediately_as_they_arrive(
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    tokens = ["안녕\n", "하세요\n", ", \n", "반갑\n", "습니다.\n"]
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for t in tokens:
            yield t

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
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
        question="인사 질문",
        budget=IterationBudget.of(max_iterations=1, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    streamed_tokens: list[str] = []
    async for chunk in use_case.execute(command):
        streamed_tokens.append(chunk)

    # 전체 한 번에 합쳐서 오는 것이 아니라, 각 토큰이 개별적으로 yield되어야 함
    assert streamed_tokens == tokens


@pytest.mark.asyncio
async def test_agentic_ask_masks_pii_during_token_streaming(
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    tokens = ["연락처는 ", "010-1234-5678", " 입니다."]
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for t in tokens:
            yield t

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
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
        question="개인정보 질문",
        budget=IterationBudget.of(max_iterations=1, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    streamed_tokens = [c async for c in use_case.execute(command)]
    assert "010-1234-5678" not in streamed_tokens
    assert any("[REDACTED_KR_PHONE]" in t for t in streamed_tokens)


@pytest.mark.asyncio
async def test_agentic_ask_stops_streaming_when_token_budget_exhausted(
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 각 청크당 40자 -> APPROX_CHARS_PER_TOKEN(4) 기준 청크당 10토큰
    long_tokens = ["A" * 40 for _ in range(10)]
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for t in long_tokens:
            yield t

    mock_llm_gateway.stream.side_effect = fake_stream

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    # token_budget=25 -> 3번째 토큰(30토큰) 도달 시 중단되어야 함
    command = AgenticAskCommand(
        question="긴 답변 질문",
        budget=IterationBudget.of(max_iterations=1, token_budget=25, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    streamed_tokens = [c async for c in use_case.execute(command)]
    # 전체 10개가 아니라 버짓 소진 시점에서 스트림이 조기 종료되어야 함
    assert len(streamed_tokens) < len(long_tokens)
    assert len(streamed_tokens) <= 3


@pytest.mark.asyncio
async def test_agentic_ask_empty_chunks_handles_gracefully(
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 검색된 청크가 0건인 상황
    mock_hybrid_search.execute.return_value = HybridSearchResult(chunks=[], query_embedding=[0.1])

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for t in ["관련", " 정보를", " 찾을 수 없습니다."]:
            yield t

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
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
        question="없는 정보 질문",
        budget=IterationBudget.of(max_iterations=1, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    streamed_tokens = [c async for c in use_case.execute(command)]
    # __SOURCES: 없이 토큰만 정상 방출됨
    assert not any(t.startswith("__SOURCES:") for t in streamed_tokens)
    assert "".join(streamed_tokens) == "관련 정보를 찾을 수 없습니다."


async def test_agentic_ask_multi_turn_sources_accumulate_and_deduplicate(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: 만족 못함 -> 문서 A(청크 0, 점수 0.75) 검색
    # 2회차: 만족 못함 -> 문서 A(점수 0.88 갱신) + 문서 B(청크 1, 0.80 신규) 검색
    # 3회차: 만족함 -> 검색 결과 없음 (chunks=[])
    mock_critique_generator.generate.side_effect = [
        Critique.of(answered=False, missing=["추가 정보 1"], next_query="질의 2", confidence=0.4),
        Critique.of(answered=False, missing=["추가 정보 2"], next_query="질의 3", confidence=0.7),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.95),
    ]

    chunk_a1 = SimilaritySearchResult(
        text="문서 A 본문입니다. 연락처: 010-9999-8888",
        score=0.75,
        metadata=VectorDocumentMetadata(
            document_id="doc-A",
            file_name="doc_a.pdf",
            chunk_index=0,
        ),
    )
    chunk_a2_higher_score = SimilaritySearchResult(
        text="문서 A 본문입니다. 연락처: 010-9999-8888",
        score=0.88,
        metadata=VectorDocumentMetadata(
            document_id="doc-A",
            file_name="doc_a.pdf",
            chunk_index=0,
        ),
    )
    chunk_b = SimilaritySearchResult(
        text="문서 B 본문입니다.",
        score=0.80,
        metadata=VectorDocumentMetadata(
            document_id="doc-B",
            file_name="doc_b.pdf",
            chunk_index=1,
        ),
    )

    mock_hybrid_search.execute.side_effect = [
        HybridSearchResult(chunks=[chunk_a1], query_embedding=[0.1]),
        HybridSearchResult(chunks=[chunk_a2_higher_score, chunk_b], query_embedding=[0.2]),
        HybridSearchResult(chunks=[], query_embedding=[0.3]),
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
        budget=IterationBudget.of(max_iterations=4, token_budget=2000, timeout_ms=5000),
        confidence_threshold=0.9,
    )

    yielded_items = [item async for item in use_case.execute(command)]

    # __SOURCES: 이벤트 추출
    sources_events = [
        json.loads(item[len("__SOURCES:") :])
        for item in yielded_items
        if item.startswith("__SOURCES:")
    ]

    # 1회차: doc-A (score: 0.75) 방출
    # 2회차: doc-A (score: 0.88로 갱신) + doc-B (신규) 방출
    # 3회차: chunks=[] 이므로 새로운 __SOURCES 방출 없음 (총 2회 방출)
    assert len(sources_events) == 2

    # 1회차 검증
    assert len(sources_events[0]) == 1
    assert sources_events[0][0]["documentId"] == "doc-A"
    assert sources_events[0][0]["chunkIndex"] == 0
    assert sources_events[0][0]["score"] == 0.75
    assert "[REDACTED_KR_PHONE]" in sources_events[0][0]["snippet"]

    # 2회차 누적 및 중복 제거(최고점 반영) 검증
    assert len(sources_events[1]) == 2
    doc_a_source = next(s for s in sources_events[1] if s["documentId"] == "doc-A")
    doc_b_source = next(s for s in sources_events[1] if s["documentId"] == "doc-B")

    assert doc_a_source["score"] == 0.88
    assert doc_a_source["chunkIndex"] == 0
    assert doc_b_source["score"] == 0.80
    assert doc_b_source["chunkIndex"] == 1
    assert doc_b_source["fileName"] == "doc_b.pdf"


async def test_agentic_ask_multi_turn_lower_score_does_not_override_or_emit(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: 문서 A(청크 0, score 0.9)
    # 2회차: 문서 A(청크 0, score 0.7) - 점수가 더 낮음
    mock_critique_generator.generate.side_effect = [
        Critique.of(answered=False, missing=["추가 정보"], next_query="질의 2", confidence=0.5),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.95),
    ]

    chunk_a_high = SimilaritySearchResult(
        text="문서 A 본문",
        score=0.9,
        metadata=VectorDocumentMetadata(
            document_id="doc-A",
            file_name="doc_a.pdf",
            chunk_index=0,
        ),
    )
    chunk_a_low = SimilaritySearchResult(
        text="문서 A 본문",
        score=0.7,
        metadata=VectorDocumentMetadata(
            document_id="doc-A",
            file_name="doc_a.pdf",
            chunk_index=0,
        ),
    )

    mock_hybrid_search.execute.side_effect = [
        HybridSearchResult(chunks=[chunk_a_high], query_embedding=[0.1]),
        HybridSearchResult(chunks=[chunk_a_low], query_embedding=[0.2]),
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
        question="동일 문서 낮은 점수 재검색 테스트",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    yielded_items = [item async for item in use_case.execute(command)]

    sources_events = [
        json.loads(item[len("__SOURCES:") :])
        for item in yielded_items
        if item.startswith("__SOURCES:")
    ]

    # 1회차만 방출되고 2회차에서는 점수가 낮아 갱신되지 않으므로 방출되지 않음 (총 1회)
    assert len(sources_events) == 1
    assert sources_events[0][0]["score"] == 0.9


@pytest.mark.asyncio
async def test_agentic_ask_restores_session_history_and_rewrites_follow_up_query(
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    session = ConversationSession.create("user-1", "프리미엄 요금제 안내해 줘").append_turn(
        "프리미엄 요금제 안내해 줘", "월 29,000원입니다."
    )
    session_repo = MagicMock(spec=ConversationSessionRepository)
    session_repo.find_by_id = AsyncMock(return_value=session)

    rewritten_question = "프리미엄 요금제의 환불 조건과 해지 절차를 비교해 줘"
    query_rewriter = MagicMock(spec=ConversationalQueryRewriter)
    query_rewriter.is_follow_up.return_value = True
    query_rewriter.rewrite = AsyncMock(return_value=rewritten_question)

    mock_critique_generator.generate.side_effect = [
        Critique.of(
            answered=False,
            missing=["해지 위약금"],
            next_query="프리미엄 요금제 해지 위약금",
            confidence=0.5,
        ),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.95),
    ]

    captured_gateway_commands = []
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(gw_cmd) -> AsyncIterator[str]:
        captured_gateway_commands.append(gw_cmd)
        yield "비교 답변입니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        session_repo=session_repo,
        query_rewriter=query_rewriter,
    )

    original_question = "거기서 환불 조건과 해지 절차를 비교해 줘"
    command = AgenticAskCommand(
        question=original_question,
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
        session_id=session.get_session_id(),
        conversation_history=None,
    )

    _ = [c async for c in use_case.execute(command)]

    session_repo.find_by_id.assert_awaited_once_with(session.get_session_id())
    expected_history = [
        {"role": "user", "content": "프리미엄 요금제 안내해 줘"},
        {"role": "assistant", "content": "월 29,000원입니다."},
    ]
    query_rewriter.is_follow_up.assert_called_once_with(original_question, expected_history)
    query_rewriter.rewrite.assert_awaited_once_with(original_question, expected_history)

    first_search_cmd = mock_hybrid_search.execute.await_args_list[0].args[0]
    assert first_search_cmd.query == rewritten_question

    first_critique_args = mock_critique_generator.generate.await_args_list[0].args
    assert first_critique_args[0] == rewritten_question

    second_critique_args = mock_critique_generator.generate.await_args_list[1].args
    assert second_critique_args[0] == rewritten_question

    refine_args = mock_query_refiner.refine.call_args_list[0].args
    assert refine_args[0] == rewritten_question

    assert len(captured_gateway_commands) == 2
    for gw_cmd in captured_gateway_commands:
        roles_and_contents = [(m.role, m.content) for m in gw_cmd.messages[1:]]
        assert roles_and_contents == [
            ("user", "프리미엄 요금제 안내해 줘"),
            ("assistant", "월 29,000원입니다."),
            ("user", original_question),
        ]


@pytest.mark.asyncio
async def test_agentic_ask_restores_session_history_without_rewrite_for_independent_question(
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    session = ConversationSession.create("user-1", "프리미엄 요금제 안내해 줘").append_turn(
        "프리미엄 요금제 안내해 줘", "월 29,000원입니다."
    )
    session_repo = MagicMock(spec=ConversationSessionRepository)
    session_repo.find_by_id = AsyncMock(return_value=session)

    query_rewriter = MagicMock(spec=ConversationalQueryRewriter)
    query_rewriter.is_follow_up.return_value = False
    query_rewriter.rewrite = AsyncMock()

    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
    )

    captured_gateway_commands = []
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(gw_cmd) -> AsyncIterator[str]:
        captured_gateway_commands.append(gw_cmd)
        yield "독립 질문 답변입니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        session_repo=session_repo,
        query_rewriter=query_rewriter,
    )

    independent_question = "기본 요금제와 엔터프라이즈 요금제의 차이를 비교해 줘"
    command = AgenticAskCommand(
        question=independent_question,
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
        session_id=session.get_session_id(),
    )

    _ = [c async for c in use_case.execute(command)]

    query_rewriter.rewrite.assert_not_awaited()
    first_search_cmd = mock_hybrid_search.execute.await_args_list[0].args[0]
    assert first_search_cmd.query == independent_question

    first_critique_args = mock_critique_generator.generate.await_args_list[0].args
    assert first_critique_args[0] == independent_question

    assert len(captured_gateway_commands) == 1
    roles_and_contents = [(m.role, m.content) for m in captured_gateway_commands[0].messages[1:]]
    assert roles_and_contents == [
        ("user", "프리미엄 요금제 안내해 줘"),
        ("assistant", "월 29,000원입니다."),
        ("user", independent_question),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("session_state", ["missing", "empty_turns"])
async def test_agentic_ask_handles_missing_or_empty_session_gracefully(
    session_state: str,
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    session_repo = MagicMock(spec=ConversationSessionRepository)
    if session_state == "missing":
        session_repo.find_by_id = AsyncMock(return_value=None)
    else:
        empty_session = ConversationSession.create("user-1", "첫 질문")
        session_repo.find_by_id = AsyncMock(return_value=empty_session)

    query_rewriter = MagicMock(spec=ConversationalQueryRewriter)
    query_rewriter.is_follow_up.return_value = True
    query_rewriter.rewrite = AsyncMock(return_value="호출되면 안 되는 재작성 질문")

    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
    )

    captured_gateway_commands = []
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(gw_cmd) -> AsyncIterator[str]:
        captured_gateway_commands.append(gw_cmd)
        yield "첫 질문 답변입니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        session_repo=session_repo,
        query_rewriter=query_rewriter,
    )

    question = "거기서 환불 조건과 해지 절차를 비교해 줘"
    command = AgenticAskCommand(
        question=question,
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
        session_id="session-empty-or-missing",
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "첫 질문 답변입니다."

    query_rewriter.is_follow_up.assert_not_called()
    query_rewriter.rewrite.assert_not_awaited()
    first_search_cmd = mock_hybrid_search.execute.await_args_list[0].args[0]
    assert first_search_cmd.query == question

    first_critique_args = mock_critique_generator.generate.await_args_list[0].args
    assert first_critique_args[0] == question

    assert len(captured_gateway_commands) == 1
    roles_and_contents = [(m.role, m.content) for m in captured_gateway_commands[0].messages[1:]]
    assert roles_and_contents == [("user", question)]


@pytest.mark.asyncio
async def test_agentic_ask_uses_provided_conversation_history_without_session_lookup(
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    session_repo = MagicMock(spec=ConversationSessionRepository)
    session_repo.find_by_id = AsyncMock()

    provided_history = [
        {"role": "user", "content": "엔터프라이즈 보안 정책 알려줘"},
        {"role": "assistant", "content": "SSO와 감사 로그를 지원합니다."},
    ]
    rewritten_question = "엔터프라이즈 보안 정책의 SSO 설정 절차와 감사 로그 보관 주기를 비교해 줘"
    query_rewriter = MagicMock(spec=ConversationalQueryRewriter)
    query_rewriter.is_follow_up.return_value = True
    query_rewriter.rewrite = AsyncMock(return_value=rewritten_question)

    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.92
    )

    captured_gateway_commands = []
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(gw_cmd) -> AsyncIterator[str]:
        captured_gateway_commands.append(gw_cmd)
        yield "직접 전달된 이력 기반 답변입니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        session_repo=session_repo,
        query_rewriter=query_rewriter,
    )

    original_question = "그것들의 설정 절차와 보관 주기를 비교해 줘"
    command = AgenticAskCommand(
        question=original_question,
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
        session_id="session-should-not-be-queried",
        conversation_history=provided_history,
    )

    _ = [c async for c in use_case.execute(command)]

    session_repo.find_by_id.assert_not_awaited()
    query_rewriter.is_follow_up.assert_called_once_with(original_question, provided_history)
    query_rewriter.rewrite.assert_awaited_once_with(original_question, provided_history)

    first_search_cmd = mock_hybrid_search.execute.await_args_list[0].args[0]
    assert first_search_cmd.query == rewritten_question

    assert len(captured_gateway_commands) == 1
    roles_and_contents = [(m.role, m.content) for m in captured_gateway_commands[0].messages[1:]]
    assert roles_and_contents == [
        ("user", "엔터프라이즈 보안 정책 알려줘"),
        ("assistant", "SSO와 감사 로그를 지원합니다."),
        ("user", original_question),
    ]


@pytest.mark.asyncio
async def test_agentic_ask_filters_cot_inference_lines_and_masks_pii(
    mock_hybrid_search: AsyncMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 수용 기준 1 검증:
    # LLM Gateway가 Step 1: 분석\n\n문서 내용에 따르면 정답입니다.\nThe final answer is 정답 방출 시
    # CoT 라인은 방출되지 않고 정제된 본문만 방출되며 PII 마스킹 유지
    raw_tokens = [
        "Step 1: 분석\n",
        "\n",
        "문서 내용에 따르면 담당자 연락처는 010-1234-5678 입니다.\n",
        "The final answer is 정답",
    ]
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for t in raw_tokens:
            yield t

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
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
        question="복합 질문",
        budget=IterationBudget.of(max_iterations=1, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    streamed_chunks = [c async for c in use_case.execute(command)]
    full_text = "".join(streamed_chunks)

    assert "Step 1: 분석" not in full_text
    assert "The final answer is 정답" not in full_text
    assert "문서 내용에 따르면 담당자 연락처는" in full_text
    assert "010-1234-5678" not in full_text
    assert "[REDACTED_KR_PHONE]" in full_text


@pytest.mark.asyncio
async def test_agentic_ask_fallback_confidence_when_budget_exhausted_with_chunks(
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 수용 기준 2 검증:
    # 1회차 실행 중 청크 2건(0.85, 0.72)이 검색되었으나 예산 소진으로 조기 종료될 때
    # confidence=0.85 (청크 최고점), missing=["답변 검증 예산 소진으로 상세 비평 생략"]
    chunks = [
        SimilaritySearchResult(
            text="청크 내용 1",
            score=0.72,
            metadata=VectorDocumentMetadata(document_id="doc-1", file_name="f1.pdf", chunk_index=0),
        ),
        SimilaritySearchResult(
            text="청크 내용 2",
            score=0.85,
            metadata=VectorDocumentMetadata(document_id="doc-2", file_name="f2.pdf", chunk_index=1),
        ),
    ]
    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks, query_embedding=[0.1, 0.2]
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

    # 1회차 답변 생성 도중(또는 직후) 예산 소진 유도 (token_budget=1)
    command = AgenticAskCommand(
        question="1회차 예산 소진 질문",
        budget=IterationBudget.of(max_iterations=1, token_budget=1, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    _ = [c async for c in use_case.execute(command)]

    # 비평(critique)은 예산 소진으로 호출되지 않아야 함
    mock_critique_generator.generate.assert_not_awaited()

    # 마지막 progress 이벤트 확인
    assert len(events) >= 2
    final_event = events[-1]
    assert final_event["iteration"] == 1
    assert final_event["phase"] == "generating"
    assert final_event["confidence"] == 0.85
    assert final_event["missing"] == ["답변 검증 예산 소진으로 상세 비평 생략"]


@pytest.mark.asyncio
async def test_agentic_ask_fallback_confidence_when_budget_exhausted_with_no_chunks(
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 수용 기준 3 검증:
    # 1회차에 검색된 청크가 0건(chunks=[])이고 예산 소진 시
    # confidence=0.0, missing=["관련 지식베이스 문서를 찾지 못함"]
    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=[], query_embedding=[0.1, 0.2]
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
        question="문서 없는 예산 소진 질문",
        budget=IterationBudget.of(max_iterations=1, token_budget=1, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    _ = [c async for c in use_case.execute(command)]

    assert len(events) >= 2
    final_event = events[-1]
    assert final_event["iteration"] == 1
    assert final_event["phase"] == "generating"
    assert final_event["confidence"] == 0.0
    assert final_event["missing"] == ["관련 지식베이스 문서를 찾지 못함"]


@pytest.mark.asyncio
async def test_agentic_ask_preserves_previous_confidence_when_exhausted_in_subsequent_iteration(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 수용 기준 4 검증:
    # 1회차 비평에서 신뢰도 0.65가 산출된 후 2회차 생성 중 예산 소진 시
    # 이전 회차에서 검증된 신뢰도 0.65와 missing 정보 유지
    mock_critique_generator.generate.return_value = Critique.of(
        answered=False,
        missing=["추가 보충 필요 항목"],
        next_query="보충 검색",
        confidence=0.65,
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

    # max_iterations=2로 2회차 generating 직후 소진 유도
    command = AgenticAskCommand(
        question="2회차 예산 소진 질문",
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    _ = [c async for c in use_case.execute(command)]

    final_event = events[-1]
    assert final_event["iteration"] == 2
    assert final_event["phase"] == "generating"
    assert final_event["confidence"] == 0.65
    assert final_event["missing"] == ["추가 보충 필요 항목"]



